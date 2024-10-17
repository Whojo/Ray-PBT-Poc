#! /usr/bin/env python3

import matplotlib.pyplot as plt


import ray
from ray import tune, train as ray_train
from ray.tune.schedulers import PopulationBasedTraining
from ray.air import RunConfig

from my_train import train


PARAMS_RANGE = {"x": tune.uniform(0, 10)}


def get_pbt_scheduler():
    return PopulationBasedTraining(
        time_attr="epoch",
        metric="score",
        mode="min",
        perturbation_interval=1,
        hyperparam_mutations=PARAMS_RANGE,
    )


def get_tuner(scheduler):
    return tune.Tuner(
        train,
        tune_config=tune.TuneConfig(
            num_samples=30,  # Number of individuals in the population
            scheduler=scheduler,
        ),
        run_config=RunConfig(
            name="pbt_example_tuner",
        ),
    )


if __name__ == "__main__":
    ray.init(num_cpus=30)

    pbt = get_pbt_scheduler()
    tuner = get_tuner(scheduler=pbt)
    results_grid = tuner.fit()

    best_result = results_grid.get_best_result(metric="score", mode="min")
    print(f"Best result: {best_result}")
    print(f"Best config: {best_result.config}")

    # Propagation of a seed in the population indicates a successful
    # checkpointing mechanism
    print(results_grid.get_dataframe()["seed"])
