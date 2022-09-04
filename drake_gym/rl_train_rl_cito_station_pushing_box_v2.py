import argparse
import gym
import os
import pdb

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from pydrake.geometry import StartMeshcat
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)

import torch as th
import wandb
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--train_single_env', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--log_path', help="path to the logs directory.")
parser.add_argument('--notes', help="log extra notes to wandb.")
args = parser.parse_args()

gym.envs.register(id="RlCitoStationBoxPushing-v2",
                  entry_point="envs.rl_cito_station_pushing_box_v2:RlCitoStationBoxPushingEnv")


config = {
    "task": "reach",
    "policy_type": "MlpPolicy",
    "total_timesteps": 1e7,
    "env_name": "RlCitoStationBoxPushing-v2",
    "num_workers": 80,
    "env_time_limit": 7,
    "local_log_dir":
        args.log_path if args.log_path is not None else os.environ['HOME']+"/rl/tmp/RlCitoStationBoxPushing_v2/",
    "model_save_freq": 1e5 if not args.train_single_env else 1e4,
    "policy_kwargs": dict(activation_fn=th.nn.ReLU,
                    net_arch=[dict(pi=[128, 128,128], vf=[128,128,128])]),
    "observation_noise": True,
    "disturbances": True,
    # valid observation types are:
    # "state", or "distances" and any (or none) of the following: "buffer_10", "buffer_20"
    "observation_type": ["actions","distances"],
    # valid reward types are:
    # "sparse", or any combination of the following: "rew_success","cost_goal","cost_effort","cost_energy"
    "reward_type": ["cost_goal","cost_energy"],
    "eval_reward_type": ["cost_goal"],
    #valid termination types are:
    # 'out_of_range', "success"
    "termination_type": ["out_of_range"],
    # valid reset types are:
    # "home" or a combination of the following "random_positions",
    # "random_velocities", "random_mass","random_friction"
    "reset_type": ["random_positions","random_velocities",
                "random_mass","random_friction"],
    "notes": args.notes,
    }

if __name__ == '__main__':
    task=config["task"]
    env_name = config["env_name"]
    if args.test:
        num_env = 2
    elif args.train_single_env:
        num_env = 1
    else:
        num_env = config["num_workers"]
    time_limit = config["env_time_limit"] if not args.test else 0.5
    log_dir = config["local_log_dir"]
    policy_type = config["policy_type"]
    total_timesteps = config["total_timesteps"] if not args.test else 5
    policy_kwargs = config["policy_kwargs"] if not args.test else None
    eval_freq = config["model_save_freq"]
    obs_noise = config["observation_noise"]
    add_disturbances = config["disturbances"]
    obs_type=config["observation_type"]
    rew_type=config["reward_type"]
    eval_rew_type=config["eval_reward_type"]
    reset_type=config["reset_type"]
    termination_type=config["termination_type"]

    if args.test:
        run = wandb.init(mode="disabled")
    else:
        run = wandb.init(
            project="sb3_tactile",
            config=config,
            sync_tensorboard=True,  # Auto-upload sb3's tensorboard metrics.
            monitor_gym=True,  # Auto-upload the videos of agents playing.
            save_code=True,
        )

    if not args.train_single_env:
        env = make_vec_env(env_name,
                           n_envs=num_env,
                           seed=0,
                           vec_env_cls=SubprocVecEnv,
                           env_kwargs={
                               'time_limit': time_limit,
                               'obs_noise': obs_noise,
                               'add_disturbances': add_disturbances,
                               'observation_type': obs_type,
                               'reward_type': rew_type,
                               'reset_type': reset_type,
                               'termination_type': termination_type,
                        })
    else:
        meshcat = StartMeshcat()
        env = gym.make(env_name,
                       meshcat=meshcat,
                       time_limit=time_limit,
                       debug=args.debug,
                       obs_noise=obs_noise,
                       observation_type=obs_type,
                       reward_type=rew_type,
                       reset_type=reset_type,
                       termination_type=termination_type,
                       )
        check_env(env)
        if args.debug:
            env.simulator.set_target_realtime_rate(1.0)
        input("Open meshcat (optional). Press Enter to continue...")
    
    if args.test:
        model = PPO(policy_type, env, n_steps=4, n_epochs=2, 
                    batch_size=8, policy_kwargs=policy_kwargs)
    else:
        model = PPO(policy_type, env, n_steps=int(2048/num_env), n_epochs=10,
                    # In SB3, this is the mini-batch size.
                    # https://github.com/DLR-RM/stable-baselines3/blob/master/docs/modules/ppo.rst
                    batch_size=64,#*num_env,
                    verbose=1, tensorboard_log=log_dir +
                    f"runs/{run.id}", policy_kwargs=policy_kwargs)

    # Separate evaluation env.
    eval_env = gym.make(env_name,
                        time_limit=time_limit,
                        obs_noise=obs_noise,
                        observation_type=obs_type,
                        reward_type=eval_rew_type,
                        monitoring_camera=True,
                        )
    eval_env = DummyVecEnv([lambda: eval_env])
    # Record a video every 1 evaluation rollouts.
    eval_env = VecVideoRecorder(
                        eval_env,
                        log_dir+f"videos/{run.id}",
                        record_video_trigger=lambda x: x % 1 == 0,
                        video_length=200)

    eval_dir=log_dir+f'eval_logs/{run.id}'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    # Use deterministic actions for evaluation.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir+f'eval_logs/{run.id}',
        log_path=eval_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False)

    # Log best model to wandb
    artifact=wandb.Artifact(name='best-model', type='model')
    artifact.add_dir(eval_dir)
    wandb.log_artifact(artifact)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[WandbCallback(
            gradient_save_freq=1e3,
            model_save_path=log_dir+f"models/{run.id}",
            verbose=2,
            model_save_freq=config["model_save_freq"],
            ),
            eval_callback]
    )
    run.finish()