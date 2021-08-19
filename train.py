import os
import gym
import torch
import argparse

from A3C import A3C
from A3C import SaveOnBestTrainingRewardCallback

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    "--seed",
    type=int,
    default=2021,
    help="Random seed (default: None)"
)
parser.add_argument(
    "--nprocs",
    type=int,
    default=4,
    help="how many training processes to use (default: 4)"
)
parser.add_argument(
    "--base_log_dir",
    type=str,
    default=os.path.join(
        os.path.dirname(
            os.path.realpath(__file__)
        ),
        "Result"
    ),
    help="how many training processes to use (default: 4)"
)
parser.add_argument(
    "--shared-momentum",
    dest="shared_momentum",
    action="store_true",
    help="Use an optimizer without shared momentum. (default: True)"
)
parser.add_argument(
    "--no-shared-momentum",
    dest="shared_momentum",
    action="store_false",
    help="Don't use an optimizer without shared momentum."
)
parser.set_defaults(shared_momentum=True)


def main(args):
    torch.manual_seed(args.seed)

    def make_a3c_dir(repetition):
        base_log_dir = os.path.join(args.base_log_dir, f"rep_{repetition}")
        os.makedirs(base_log_dir, exist_ok=True)
        env = gym.make('LunarLander-v2')
        model = A3C('MlpPolicy', env, verbose=0, base_log_dir=base_log_dir, shared_momentum=args.shared_momentum, seed=args.seed)

        return base_log_dir, model

    def train_a3c_dir(repetitions):
        for repetition in range(repetitions):
            base_log_dir, model = make_a3c_dir(repetition=repetition)
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=None)
            model.learn(total_timesteps=30000, log_interval=10, callback=callback)
            model.save(os.path.join(
                base_log_dir,
                "a3c_lunar"
            ))
            del model

    train_a3c_dir(repetitions=10)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
