import argparse

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import HyperOptSearch
from hyperopt import hp

from main import main


args = {
    "experiment": "hyper_opt",
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
    "l_var": 1
}

space = {
    "batchSize": hp.choice('batchSize', [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
    "nc": 3,
    "nz": 100,
    "ngf": 64,
    "ndf": 64,
    "lrD": hp.loguniform('lrD', -8, -1),
    "lrG": hp.loguniform('lrG', -8, -1),
    "beta1": hp.uniform('beta1', 0, 1),
    "beta2": hp.uniform('beta2', 0, 1),
    "Diters": 5,
    "noBN": False,
    "type": hp.choice('type', ["dcgan", "mlp", "resnet"]),
}

ray.init()

algo = HyperOptSearch(space, max_concurrent=4, reward_attr="inception")

sched = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    reward_attr="inception",
    max_t=8,
    grace_period=2
)


def train(config, reporter):
    args.update(config)
    main(args, reporter)


tune.register_trainable(
    "main",
    train
)

tune.run_experiments(
    {
        "var_constraint": {
            "stop": {
                "inception": 6,
            },
            "trial_resources": {
                "cpu": 16,
                "gpu": 1
            },
            "run": "main",
            "num_samples": 100,
        }
    },
    verbose=True,
    scheduler=sched,
    search_alg=algo
)
