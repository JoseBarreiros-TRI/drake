import argparse
from locale import ABDAY_1
import gym
import os
import pdb

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pydrake.all import StartMeshcat
from stable_baselines3.common.env_checker import check_env
from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewVelocities,
        MakeNamedViewState,
        MakeNamedViewActuation)

gym.envs.register(id="PunyoidBoxLifting-v0",
                  entry_point="envs.punyoid_lifting_box:PunyoidBoxLiftingEnv")

parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

observations = "state"
zip = "/home/josebarreiros/rl/data/PunyoidBoxLifting_ppo_{observations}.zip"
log = "/home/josebarreiros/rl/tmp/PunyoidBoxLifting/"
debug=True

if __name__ == '__main__':

    # Make a version of the env with meshcat.
    meshcat = StartMeshcat()
    env = gym.make("PunyoidBoxLifting-v0", meshcat=meshcat, observations=observations,debug=debug)
    env.simulator.set_target_realtime_rate(1.0)
      
    check_env(env)
    
    #pdb.set_trace()
    
    if args.test:
        # the expected behavior is arms down and then arms up
         
        Na=env.plant.num_actuators()
        Np=env.plant.num_positions()
        ActuationView=MakeNamedViewActuation(env.plant, "Actuation")
        PositionView=MakeNamedViewPositions(env.plant,"Positions")
        actuation_matrix=env.plant.MakeActuationMatrix()
        # standing straigth
        a1=PositionView([0]*Np) 
        # arms up
        a2=PositionView([0]*Np) 
        a2.shoulderR_joint1=1.5
        a2.shoulderL_joint1=1.5
        a2.prismatic_z=0.3
        actions=[a1.__array__()[:Na],a2.__array__()[:Na]]

        for action in actions:
            input("Press Enter to continue...")
            obs = env.reset()
            for i in range(1000):
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
        env.render()
        if done:
            obs = env.reset()
