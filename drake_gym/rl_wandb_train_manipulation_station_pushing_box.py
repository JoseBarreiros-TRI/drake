import argparse
import gym
import os
import pdb

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from pydrake.geometry import Meshcat, Cylinder, Rgba, Sphere, StartMeshcat
from stable_baselines3.common.env_checker import check_env

import wandb
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--train_single', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

gym.envs.register(id="ManipulationStationBoxPushing-v0",
                  entry_point="envs.manipulation_station_pushing_box:ManipulationStationBoxPushingEnv")

config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 10e6,
        "env_name": "ManipulationStationBoxPushing-v0",
        "num_workers": 21,
        "env_time_limit": 7,
        "local_log_dir": "/home/josebarreiros/rl/tmp/ManipulationStationBoxPushing/",
        "observations_set": "state",
        "model_save_freq": 10000,
    }

if __name__ == '__main__':

    num_cpu = config["num_workers"] if not args.test else 2
    time_limit = config["env_time_limit"] if not args.test else 0.5
    observations=config["observations_set"]
    log_dir=config["local_log_dir"]
    policy_type=config["policy_type"]
    total_timesteps=config["total_timesteps"] if not args.test else 3

    run = wandb.init(
        project="sb3_test",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    if not args.train_single:
        env = make_vec_env("ManipulationStationBoxPushing-v0",
                        n_envs=num_cpu,
                        seed=0,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs={
                            'observations': observations,
                            'time_limit': time_limit,
                        })
    else:
        config["num_workers"]=1
        meshcat = StartMeshcat()
        env = gym.make("ManipulationStationBoxPushing-v0", meshcat=meshcat, 
            observations=observations,time_limit=time_limit, debug=args.debug)
        print("Open tensorboard in another terminal. tensorboard --logdir ",log_dir+f"runs/{run.id}")
        input("Press Enter to continue...")
    
    if args.debug or args.test:
        check_env(env)

    if args.test:
        model = PPO(policy_type, env, n_steps=4, n_epochs=2, batch_size=8)
    else:
        model = PPO(policy_type, env, verbose=1, tensorboard_log=log_dir+f"runs/{run.id}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=1000,
            model_save_path=log_dir+f"models/{run.id}",
            verbose=2,
            model_save_freq=config["model_save_freq"],
        ),
    )
    run.finish()
