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

gym.envs.register(id="ManipulationStationBoxPushing-v0",
                  entry_point="envs.manipulation_station_pushing_box:ManipulationStationBoxPushingEnv")

parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--train_single', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

observations = "state"
time_limit = 7 if not args.test else 0.5
zip = "/home/josebarreiros/rl/data/ManipulationStationBoxPushing_ppo_{observations}.zip"
log = "/home/josebarreiros/rl/tmp/ManipulationStationBoxPushing/"
checkpoint_callback = CheckpointCallback(save_freq=20000, save_path='/home/josebarreiros/rl/tmp/ManipulationStationBoxPushing/model_checkpoints/')
debug=args.debug #True

if __name__ == '__main__':
    num_cpu = 12 if not args.test else 2
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
        meshcat = StartMeshcat()
        env = gym.make("ManipulationStationBoxPushing-v0", meshcat=meshcat, 
            observations=observations,time_limit=time_limit, debug=debug)
        print("Open tensorboard in another terminal. tensorboard --logdir ",log)
        input("Press Enter to continue...")
    
    if args.debug:
        check_env(env)

    if args.test:
        model = PPO('MlpPolicy', env, n_steps=4, n_epochs=2, batch_size=8)
    else:
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log)

    new_log = True
    while True:
        model.learn(total_timesteps=10000 if not args.test else 4,
                    reset_num_timesteps=new_log, callback=[checkpoint_callback] if not args.test else [])
        if args.test:
            break
        model.save(zip)
        print("\nModel saved")
        new_log = False