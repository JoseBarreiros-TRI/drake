import argparse
import gym
import os
import pdb

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pydrake.geometry import Meshcat, Cylinder, Rgba, Sphere, StartMeshcat
from stable_baselines3.common.env_checker import check_env

gym.envs.register(id="NoodlemanStandUpTorque-v0",
                  entry_point="envs.noodleman_fbase_standup_torque:NoodlemanStandUpEnv")

observations = "state"
zip = "/home/josebarreiros/rl/data/noodlemanStandUpTorque_ppo_{observations}.zip"
log = "/home/josebarreiros/rl/tmp/noodlemanStandUpTorque/"
debug=True

if __name__ == '__main__':

    # Make a version of the env with meshcat.
    meshcat = StartMeshcat()
    env = gym.make("NoodlemanStandUpTorque-v0", meshcat=meshcat, observations=observations,debug=debug)
    env.simulator.set_target_realtime_rate(1.0)
    
    pdb.set_trace()
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
