import argparse

import gym
import os
import pdb
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pydrake.all import StartMeshcat
from stable_baselines3.common.env_checker import check_env

gym.envs.register(id="RlCitoStationBoxPushing-v2",
                  entry_point="envs.rl_cito_station_pushing_box_v2:RlCitoStationBoxPushingEnv")

parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--task', help="tasks: [reach, push]", type=str, default="reach")
parser.add_argument('--hardware', action='store_true')
parser.add_argument('--mock_hardware', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--model_path', help="path to the policy")
parser.add_argument('--log_path', help="path to the logs directory.")
args = parser.parse_args()

if args.model_path is not None:
    zip = args.model_path
else:
    zip = os.environ['HOME']+"/rl/tmp/RlCitoStationBoxPushing_v2/models/{model_id}/model.zip"

if args.log_path is None:
    log = args.log_path
else:
    log = os.environ['HOME']+"/rl/tmp/RlCitoStationBoxPushing_v2/play_runs/"

if __name__ == '__main__':

    if args.hardware:
        meshcat=None
    else:
        # Make a version of the env with meshcat.
        meshcat = StartMeshcat()

    env = gym.make("RlCitoStationBoxPushing-v2",
                   meshcat=meshcat,
                   time_limit=7,
                   debug=args.debug,
                   obs_noise=True,
                   add_disturbances=True,
                   termination_type=["box_off_table","success","collision_w_table"],
                   reward_type=["cost_goal"],
                   observation_type=["state","EE_box_target_xyz","distances"],
                   reset_type=["random_positions_limited"],
                   hardware=args.hardware,
                   task=args.task,
                   mock_hardware=args.mock_hardware,
                   control_mode="EE_delta_pose",
                   )

    if args.test and not args.hardware:
        #pdb.set_trace()
        check_env(env)
        #pass

    env.simulator.set_target_realtime_rate(1.0)
    max_num_episodes = 1e5 if args.test else 1e3

    if not args.test:
        model = PPO.load(zip, env, verbose=1, tensorboard_log=log)

    input("Press Enter to continue...")
    obs = env.reset()
    for i in range(100000):
        if args.test:
            # Plays a random policy.
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        #pdb.set_trace()
        obs, reward, done, info = env.step(action)

        if args.debug:
            # This will play the policy step by step.
            input("Press Enter to continue...")

        env.render()
        if done:
            input("The environment will reset. Press Enter to continue...")
            obs = env.reset()
            # Wait for meshcat to load the env.
            # TODO(JoseBarreiros-TRI) Replace sleep() with a readiness signal
            # from meshcat.
            time.sleep(0.7)
