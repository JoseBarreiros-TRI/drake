import argparse
import gym
import os
import pdb
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pydrake.geometry import StartMeshcat
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

try:
    import wandb
except ImportError:
    raise ImportError(
        "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
    )
from wandb.integration.sb3 import WandbCallback


parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--train_single_env', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

gym.envs.register(id="Cartpole-v0",
                  entry_point="envs.cart_pole:CartpoleEnv")

config = {
        "policy_type": "MlpPolicy", 
        "total_timesteps": 1e7,
        "env_name": "Cartpole-v0",
        "num_workers": 10,
        "env_time_limit": 7,
        "local_log_dir": "/home/josebarreiros/rl/tmp/Cartpole/",
        "model_save_freq": 1e4,
        "policy_kwargs": dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[64, 64, 64], vf=[64,64,64])]),
        "observation_noise": True,
    }

if __name__ == '__main__':
    env_name=config["env_name"]
    num_env = 2 if args.test else 1 if args.train_single_env else config["num_workers"]
    time_limit = config["env_time_limit"] if not args.test else 0.5
    log_dir=config["local_log_dir"]
    policy_type=config["policy_type"]
    total_timesteps=config["total_timesteps"] if not args.test else 3
    policy_kwargs=config["policy_kwargs"] if not args.test else None
    eval_freq=config["model_save_freq"]
    obs_noise=config["observation_noise"]

    run = wandb.init(
        project="sb3_test",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    if not args.train_single_env:
        env = make_vec_env(env_name,
                        n_envs=num_env,
                        seed=0,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs={
                            'time_limit': time_limit,
                            'obs_noise': obs_noise,
                        })
    else:
        meshcat = StartMeshcat()
        env = gym.make(env_name, 
                meshcat=meshcat, 
                time_limit=time_limit, 
                debug=args.debug,
                obs_noise=obs_noise,
                )
        check_env(env)
        input("Open meshcat (optional). Press Enter to continue...")
    
    if args.test:
        model = PPO(policy_type, env, n_steps=4, n_epochs=2, batch_size=8,policy_kwargs=policy_kwargs)
    else:
        model = PPO(policy_type, env, verbose=1, tensorboard_log=log_dir+f"runs/{run.id}",policy_kwargs=policy_kwargs)

    # Separate evaluation env
    eval_env = gym.make(env_name,  
                time_limit=time_limit, 
                obs_noise=obs_noise,
                monitoring_camera=True,
                )
    eval_env = DummyVecEnv([lambda: eval_env])
    #record a video every 2 evaluation rollouts
    eval_env = VecVideoRecorder(eval_env, log_dir+f"videos/{run.id}", record_video_trigger=lambda x: x % 2 == 0, video_length=200)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir+f'eval_logs/{run.id}',
                                log_path=log_dir+f'eval_logs/{run.id}', eval_freq=eval_freq,
                                deterministic=True, render=False)

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
