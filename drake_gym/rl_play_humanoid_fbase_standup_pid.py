import argparse
import gym
import os
import pdb

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pydrake.geometry import Meshcat, Cylinder, Rgba, Sphere, StartMeshcat
from stable_baselines3.common.env_checker import check_env

gym.envs.register(id="Humanoid_fbase_StandUpPID-v0",
                  entry_point="envs.humanoid_fbase_standup_pid:NoodlemanStandUpEnv")

observations = "state"
zip = "/home/josebarreiros/rl/data/humanoid_fbase_StandUpPID_ppo_{observations}.zip"
log = "/home/josebarreiros/rl/tmp/humanoid_fbase_StandUpPID/"
debug=True

if __name__ == '__main__':

    # Make a version of the env with meshcat.
    meshcat = StartMeshcat()
    env = gym.make("Humanoid_fbase_StandUpPID-v0", meshcat=meshcat, observations=observations,debug=debug)
    env.simulator.set_target_realtime_rate(1.0)
    pdb.set_trace()
    #obs=env.reset()
    #print(env.observation_space.contains(obs))
    
    
    check_env(env)
    pdb.set_trace()
    
    model = PPO.load(zip, env, verbose=1, tensorboard_log=log)
    
    input("Press Enter to continue...")
    obs = env.reset()
    for i in range(100000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
