import argparse
import copy
from statsmodels.formula.api import ols
import train
import model
import torch
import utils


def train_model(arg, arch, cn, seed=0, initial_state=None):
    train_fn1 = train.train_fn(arg.train_lr, arg.batch_size, arg.dataset, arch,
                               correct_clipping=arg.correct_clipping,
                               noise_multiplier=arg.noise_multiplier,
                               max_grad_norm=cn, delta=arg.delta,
                               batch_clipping=arg.batch_clipping,
                               no_clipping=arg.no_clipping,
                               wrong_noise_calibration=arg.wrong_noise_calibration, seed=seed,
                               optimizer=arg.optim, shuffle_train=arg.shuffle)

    if initial_state is None:
        initial_state = copy.deepcopy(train_fn1.net.state_dict())
    else:
        train_fn1.net.load_state_dict(initial_state)

    train_fn1.optimizer.zero_grad()
    for batch_idx, data in enumerate(train_fn1.train_loader, 0):
        train_fn1.train_step(batch_idx, data)
        if (batch_idx + 1) == arg.train_steps:
            break

    return initial_state, utils.consistent_type(train_fn1.net, )


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument('--arch', type=str, default="resnet20_gn")
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--train-lr', type=float, default=0.1)
    parser.add_argument('--train-steps', type=int, default=10)
    parser.add_argument('--noise-multiplier', type=float, default=0.001, help='ratio of Gaussian noise to add for DPSGD')
    parser.add_argument("--list-cn", type=float, nargs='+', default=list(range(100,1001,100)),
                        help="the clipping norm for Gaussian Mechanism for differential privacy")
    parser.add_argument('--delta', type=float, default=1e-5, help='value of delta for DP (1/num-datapoint ^ 1.1)')
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--correct-clipping', type=int, default=1, help="0 or 1 for whether correct clipping")
    parser.add_argument('--no-clipping', type=int, default=0)
    parser.add_argument('--batch-clipping', type=int, default=0)
    parser.add_argument('--wrong-noise-calibration', type=int, default=0,
                        help='set to 1 to simulate the case that the developer forgets to calibrate the noise')
    parser.add_argument('--no-noise', type=int, default=0,
                        help='set to 1 to simulate the case that the developer forgets to add noise')
    parser.add_argument('--seed', type=int, default=2022)
    arg = parser.parse_args()

    assert arg.correct_clipping + arg.no_clipping + arg.batch_clipping == 1, \
        "One and only one type of clipping needs to be specified."

    if arg.no_noise:
        if arg.wrong_noise_calibration:
            arg.wrong_noise_calibration = 0
            raise Warning("Setting no noise and wrong noise calibration at the same time")
        arg.noise_multipler = 0

    arch = eval(f"model.{arg.arch}")
    dist_list = []

    for cn in arg.list_cn:
        initial, model1 = train_model(arg, arch, cn, seed=arg.seed)
        _, model2 = train_model(arg, arch, cn, seed=int(arg.seed + 1), initial_state=initial)
        distance = utils.parameter_distance(model1, model2, order=['2'])[0]
        dist_list.append(distance)

    stats_model = ols(f"dist ~ cn", {"dist": dist_list, "cn": arg.list_cn}).fit()
    p_value = stats_model.pvalues["cn"]
    print(f"p-value={p_value :.4f} for whether pairwise model distance depends on the clipping norm "
          f"(>0.05 means there is something wrong)")
