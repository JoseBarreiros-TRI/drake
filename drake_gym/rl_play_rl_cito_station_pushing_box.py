import argparse
from cgitb import handler
from locale import ABDAY_1
import gym
import os
import pdb
import numpy as np
import time
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pydrake.all import StartMeshcat
from stable_baselines3.common.env_checker import check_env
from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewVelocities,
        MakeNamedViewState,
        MakeNamedViewActuation)

gym.envs.register(id="RlCitoStationBoxPushing-v0",
                  entry_point="envs.rl_cito_station_pushing_box:RlCitoStationBoxPushingEnv")

parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--task', help="tasks: [reach,push]", type=str, default="reach")
parser.add_argument('--hardware', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--model_path', help="path to the policy")
args = parser.parse_args()

observations = "state"

#pdb.set_trace()
if args.model_path != None:
    zip=args.model_path
    log = "/home/josebarreiros/rl/tmp/RlCitoStationBoxPushing/"
else:
    zip = "/home/josebarreiros/rl/data/RlCitoStationBoxPushing_ppo_{observations}.zip"
    log = "/home/josebarreiros/rl/tmp/RlCitoStationBoxPushing/"


if __name__ == '__main__':

    # Make a version of the env with meshcat.
    meshcat = StartMeshcat()
    env = gym.make("RlCitoStationBoxPushing-v0", 
            meshcat=meshcat, 
            time_limit=7,
            observations=observations,
            debug=args.debug,
            hardware=args.hardware,
            task=args.task)
    env.simulator.set_target_realtime_rate(1.0)

    if not args.hardware:  
        check_env(env)
    
    
    if args.test:
        # the expected behavior is arms down and then arms up
        Na=7#env.plant.num_actuators()

        obs = env.reset()
        for i in range(1000):
            action=np.random.rand(Na)
            input("Press Enter to continue...")
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()

    input("Press Enter to continue...")   

    model = PPO.load(zip, env, verbose=1, tensorboard_log=log)
    obs = env.reset()
    for i in range(100000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if args.debug:
            input("Press Enter to continue...")
        env.render()
        if done:
            input("If continue the environment will reset. Press Enter to continue...")   
            obs = env.reset()
            if not args.hardware:
                #wait for meshcat to load env
                time.sleep(0.7)
