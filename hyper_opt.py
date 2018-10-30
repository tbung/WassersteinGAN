import argparse

import numpy as np

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

from main import main


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True,
                    help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True,
                    help='path to dataset')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25,
                    help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true',
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--netG', default='',
                    help="path to netG (to continue training)")
parser.add_argument('--netD', default='',
                    help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5,
                    help='number of D iters per each G iter')
args = parser.parse_args()

args = {
    "dataset": "cifar10",
    "dataroot": "./data",
    "workers": 4,
    "imageSize": 32,
    "niter": 25,
    "cuda": True,
    "ngpu": 1,
    "adam": True,
    "n_extra_layers": 0,
    "var_constraint": True,
}

config = {
    "batchSize": 64,
    "nc": 3,
    "nz": 100,
    "ngf": 64,
    "ndf": 64,
    "lrD": 1e-4,
    "lrG": 1e-4,
    "beta1": 0.5,
    "beta2": 0.9,
    "Diters": 5,
    "noBN": False,
    "type": "dcgan",
}

ray.init()
sched = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    reward_attr="neg_mean_loss",
    max_t=400,
    grace_period=20)
tune.register_trainable(
    "train_mnist",
    lambda cfg, rprtr: main(args.update(cfg), rprtr)
)
tune.run_experiments(
    {
        "var_constraint": {
            "stop": {
                "inception": 7,
                "training_iteration": 10000
            },
            "trial_resources": {
                "cpu": 16,
                "gpu": 1
            },
            "run": "main",
            "num_samples": 10,
            "config": {
                "lr": lambda spec: np.random.uniform(1e-7, 0.1),
                "beta1": lambda spec: np.random.uniform(0, 0.9),
            }
        }
    },
    verbose=0,
    scheduler=sched
)
