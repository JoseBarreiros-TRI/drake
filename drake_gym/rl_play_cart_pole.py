import argparse
import gym
import os
import pdb
import time

from stable_baselines3 import PPO
from pydrake.all import StartMeshcat
from stable_baselines3.common.env_checker import check_env


gym.envs.register(id="Cartpole-v0",
                  entry_point="envs.cart_pole:CartpoleEnv")

parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--model_path', help="path to the policy zip file.")
args = parser.parse_args()

if args.model_path != None:
    zip=args.model_path
    log = "/home/josebarreiros/rl/tmp/Cartpole/play_runs/"
else:
    zip = "/home/josebarreiros/rl/tmp/Cartpole/models/{model_id}/model.zip"
    log = "/home/josebarreiros/rl/tmp/Cartpole/play_runs/"

if __name__ == '__main__':

    # Make a version of the env with meshcat.
    meshcat = StartMeshcat()
    env = gym.make("Cartpole-v0", 
                    meshcat=meshcat, 
                    time_limit=7,
                    debug=args.debug,
                    obs_noise=True,
                    add_disturbances=True)
    
    if args.test:
        check_env(env)

    env.simulator.set_target_realtime_rate(1.0)
    max_num_episodes=1e5 if args.test else 1e3
    
    if not args.test:
        model = PPO.load(zip, env, verbose=1, tensorboard_log=log)
        
    input("Press Enter to continue...")   
    obs = env.reset()
    for i in range(100000):
        if args.test:
            # plays a random policy
            action=env.action_space.sample()
        else:
            action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if args.debug:
            # this will play the policy step by step
            input("Press Enter to continue...")
        env.render()
        if done:
            #if args.debug:
            input("If continue the environment will reset. Press Enter to continue...")   
            obs = env.reset()
            #wait for meshcat to load env
            time.sleep(0.7)
