# This file simulates a real training process.

import os
import random
import tempfile

from ray import train as ray_train


def eval(config: dict):
    """
    Evaluate the objective function.
    """
    x = config["x"]
    return (x - 5) ** 2  # Example objective function


def train(config: dict):
    """
    Simulate a training process.

    Args:
        config (dict): A dictionary containing configuration parameters for the training process.

    Returns:
        None
    """
    seed = random.randint(0, 100)

    checkpoint = ray_train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, "config.txt"), "r") as file:
                seed = int(file.read())

    score = eval(config)
    for epoch in range(5):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "config.txt"), "w") as file:
                file.write(str(seed))

            checkpoint = ray_train.Checkpoint.from_directory(tmp_dir)

            ray_train.report(
                {"score": score, "epoch": epoch, "seed": seed},
                checkpoint=checkpoint,
            )
