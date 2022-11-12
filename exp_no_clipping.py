import argparse
import train
import numpy as np
import model
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--arch', type=str, default="resnet20_gn")
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--train-lr', type=float, default=0.01)
parser.add_argument('--train-steps', type=int, default=1)
parser.add_argument("--cn-list", type=float, nargs="+", default=[1e-9, 0.01, 0.1, 1.0, 10.0, 100.0,
                                                                 1000.0, 10000.0, 100000.0, 1000000.0],
                    help="the clipping norm for Gaussian Mechanism for differential privacy")
parser.add_argument('--delta', type=float, default=1e-5,
                        help='value of delta for DP (1/num-datapoint ^ 1.1)')
parser.add_argument('--no-clipping', type=int, default=0)
parser.add_argument('--seed', type=int, default=2022)
arg = parser.parse_args()

arch = eval(f"model.{arg.arch}")

for cn in arg.cn_list:
    train_fn1 = train.train_fn(arg.train_lr, 1, arg.dataset, arch,
                               correct_clipping=0, noise_multiplier=0.0, max_grad_norm=cn, delta=arg.delta,
                               physical_batch_size=1, batch_clipping=not arg.no_clipping,
                               no_clipping=arg.no_clipping, wrong_noise_calibration=0,
                               seed=arg.seed, optimizer=arg.optim,
                               )

    train_fn1.optimizer.zero_grad()
    for batch_idx, data in enumerate(train_fn1.train_loader, 0):
        if train_fn1.n_accumulation_steps is not None:
            arg.train_steps = arg.train_steps * train_fn1.n_accumulation_steps
        for i in range(arg.train_steps):
            train_fn1.train_step(i, data)
        cur_loss = train_fn1.train_step(batch_idx, data)
        new_loss = train_fn1.compute_loss(data)[0].item()
        loss_change = np.abs(new_loss - cur_loss) / np.abs(cur_loss)
        break

    print(arg.no_clipping, cn, loss_change)
