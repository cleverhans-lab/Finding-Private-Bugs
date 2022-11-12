import argparse
from statsmodels.formula.api import ols
import train
import model
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--arch', type=str, default="resnet20_gn")
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--train-lr', type=float, default=0.1)
parser.add_argument("--max-grad-norm", type=float, default=1e-3,
                    help="the clipping norm for Gaussian Mechanism for differential privacy")
parser.add_argument('--batch-clipping', type=int, default=0)
parser.add_argument('--correct-clipping', type=int, default=0, help="0 or 1 for whether to use correct dpsgd or not")
parser.add_argument('--delta', type=float, default=1e-5,
                        help='value of delta for DP (1/num-datapoint ^ 1.1)')
parser.add_argument('--bs',  type=int, nargs='+', default=list(range(100)))
parser.add_argument('--seed', type=int, default=2022)
arg = parser.parse_args()

assert arg.correct_clipping + arg.batch_clipping == 1, "One and only one type of clipping needs to be specified."

arch = eval(f"model.{arg.arch}")

train_fn1 = train.train_fn(arg.train_lr, 32, arg.dataset, arch, device=torch.device('cpu'),
                           correct_clipping=arg.correct_clipping,
                           noise_multiplier=0, max_grad_norm=arg.max_grad_norm,
                           delta=arg.delta,
                           batch_clipping=arg.batch_clipping,
                           no_clipping=0, wrong_noise_calibration=0, seed=arg.seed, optimizer=arg.optim)

res = train_fn1.check_clip(arg.bs)

stats_model = ols(f"change_in_loss ~ bs", {"change_in_loss": res, "bs": arg.bs}).fit()
p_value = stats_model.pvalues["bs"]
print(f"p-value={p_value :.4f} for whether change in loss values depends on the batch size "
      f"(>0.05 means there is something wrong)")
