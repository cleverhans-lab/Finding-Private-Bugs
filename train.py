import numpy as np
import os
import torch
import copy
import utils
import model
import shutil
import warnings
torch.backends.cudnn.benchmark = True


class train_fn():
    def __init__(self, lr=0.01, batch_size=128, dataset='CIFAR10', architecture=model.resnet20_gn,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), correct_clipping=0,
                 noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5, physical_batch_size=32, batch_clipping=0,
                 no_clipping=0, wrong_noise_calibration=0, seed=0, optimizer="sgd", shuffle_train=True):
        self.device = device
        self.dataset = dataset
        if correct_clipping or batch_clipping or no_clipping or wrong_noise_calibration:
            import opacus
            import opacus_wrong
        if correct_clipping:
            if batch_size > physical_batch_size:
                assert batch_size % physical_batch_size == 0
                self.batch_size = batch_size
                self.physical_batch_size = physical_batch_size
                self.n_accumulation_steps = batch_size // physical_batch_size
            else:
                self.batch_size = batch_size
                self.physical_batch_size = batch_size
                self.n_accumulation_steps = 1
        else:
            self.batch_size = batch_size
            self.physical_batch_size = batch_size
            self.n_accumulation_steps = None

        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.correct_clipping = correct_clipping
        self.batch_clipping = batch_clipping
        self.no_clipping = no_clipping
        if self.batch_clipping:
            self.no_clipping = 0
        if self.batch_clipping or self.no_clipping:
            self.correct_clipping = 0
        self.wrong_noise_calibration = wrong_noise_calibration
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta

        self.trainset = utils.load_dataset(self.dataset, True, download=True,)

        testset = utils.load_dataset(self.dataset, False, download=True,)
        self.testset = testset

        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.physical_batch_size,
                                                        shuffle=shuffle_train, num_workers=0, pin_memory=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=0, pin_memory=True)

        self.net = architecture()

        self.net.to(self.device)
        sample_size = self.trainset.__len__() # since we don't care about budget calculation. This can be anything
        if self.correct_clipping:
            self.net.train()
            if not self.wrong_noise_calibration:
                self.privacy_engine = opacus.PrivacyEngine(self.net, batch_size=self.batch_size,
                                           sample_size=sample_size,
                                           alphas=[1 + x / 10. for x in range(1, 100)] + list(range(12, 64)),
                                           # alphas is the orders for renyi DP
                                           noise_multiplier=noise_multiplier,
                                           max_grad_norm=max_grad_norm, )  # max_grad_norm can be changed.
            else:
                self.privacy_engine = opacus_wrong.PrivacyEngine(self.net, batch_size=self.batch_size,
                                           sample_size=self.trainset.__len__(),
                                           alphas=[1 + x / 10. for x in range(1, 100)] + list(range(12, 64)),
                                           # alphas is the orders for renyi DP
                                           noise_multiplier=noise_multiplier,
                                           max_grad_norm=max_grad_norm, )  # max_grad_norm can be changed.
        else:
            self.privacy_engine = None
        self.optimizer = utils.get_optimizer(dataset, self.net, lr,
                                             privacy_engine=self.privacy_engine,
                                             optimizer=optimizer)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def predict(self, inputs):
        outputs = self.net(inputs)
        return outputs

    def update(self, bs, batch_idx):
        if self.batch_clipping or self.no_clipping:
            if self.batch_clipping:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.max_grad_norm)
            for p in self.net.parameters():
                # correct DP: self.noise_multiplier * self.max_grad_norm / bs
                if not self.wrong_noise_calibration:
                    p.grad.detach().add_(1. / bs * torch.normal(0, self.noise_multiplier * self.max_grad_norm,
                                                                p.grad.shape, device=self.device, ))
                # wrong DP noise
                else:
                    p.grad.detach().add_(1. / bs * torch.normal(0, self.noise_multiplier,
                                                                p.grad.shape, device=self.device, ))

            self.optimizer.step()
            self.optimizer.zero_grad()

        elif self.correct_clipping:
            if ((batch_idx + 1) % self.n_accumulation_steps == 0) or ((batch_idx + 1) == len(self.train_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad()

            else:
                self.optimizer.virtual_step()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def compute_loss(self, data):
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        outputs = self.predict(inputs.contiguous())
        loss = self.criterion(outputs, labels)
        bs = len(labels)
        return loss, bs

    def train_step(self, batch_idx, data):
        loss, bs = self.compute_loss(data)
        loss.backward()
        self.update(bs, batch_idx)
        return loss.item()

    def check_clip(self, bs_list=list(range(10))):
        net_state = copy.deepcopy(self.net.state_dict())
        opt_state = copy.deepcopy(self.optimizer.state_dict())
        norm_state = copy.deepcopy(self.max_grad_norm)
        loss_reduction_state = self.criterion.reduction

        self.net.eval()
        for batch_idx, data in enumerate(self.train_loader, 0):
            if batch_idx == 0:
                inputs = data[0][:2]
            if inputs.shape[0] >= 2:
                break
            else:
                inputs = torch.cat([inputs, data[0][:1]])

        inputs[1] = torch.rand(inputs[1].shape)
        outputs = self.predict(inputs.contiguous().to(self.device)).detach().cpu()
        labels = copy.deepcopy(outputs)
        labels[1] = - 10 * labels[1]
        cur_loss = self.criterion(outputs[-1].unsqueeze(0).to(self.device),
                                  labels[-1].unsqueeze(0).softmax(dim=1).to(self.device)).item()
        labels = labels.softmax(dim=1)
        self.net.train()

        loss_changes = []
        # assert self.max_grad_norm <= 1e-3
        for i in bs_list:
            torch.cuda.empty_cache()
            self.optimizer.load_state_dict(opt_state)
            self.optimizer.zero_grad()
            self.net.load_state_dict(net_state)

            temp_inputs = torch.cat([inputs[0].unsqueeze(0)] * i + [inputs[-1].unsqueeze(0)], 0)
            temp_labels = torch.cat([labels[0].unsqueeze(0)] * i + [labels[-1].unsqueeze(0)], 0)
            temp_inputs = temp_inputs.detach()
            outputs = self.predict(temp_inputs.contiguous().to(self.device))
            loss = self.criterion(outputs, temp_labels.to(self.device))
            loss.backward()
            self.update(1, -1)
            torch.cuda.empty_cache()
            with torch.no_grad():
                outputs = self.predict(temp_inputs.contiguous().to(self.device)).detach()
                new_loss = self.criterion(outputs[-1].unsqueeze(0),
                                          temp_labels[-1].unsqueeze(0).to(self.device)).detach().item()
                del temp_labels, outputs
                loss_changes.append(cur_loss - new_loss)
            del temp_inputs, new_loss

        self.optimizer.load_state_dict(opt_state)
        self.optimizer.zero_grad()
        self.net.load_state_dict(net_state)

        self.max_grad_norm = norm_state
        self.criterion.reduction = loss_reduction_state

        return loss_changes





