import argparse
import gym
import os
import pdb

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pydrake.geometry import Meshcat, Cylinder, Rgba, Sphere, StartMeshcat

gym.envs.register(id="BoxFlipUp-v0",
                  entry_point="envs.box_flipup:BoxFlipUpEnv")

observations = "state"
zip = "/home/josebarreiros/rl/data/box_flipup_ppo_{observations}.zip"
log = "/home/josebarreiros/rl/tmp/ppo_box_flipup/"


if __name__ == '__main__':

    # Make a version of the env with meshcat.
    meshcat = StartMeshcat()
    env = gym.make("BoxFlipUp-v0", meshcat=meshcat, observations=observations)
    env.simulator.set_target_realtime_rate(1.0)
    model = PPO.load(zip, env, verbose=1, tensorboard_log=log)
    
    input("Press Enter to continue...")
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
