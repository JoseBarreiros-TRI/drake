import argparse
from locale import ABDAY_1
import gym
import os
import pdb

from pydrake.all import StartMeshcat

import ray
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print

parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--hardware', action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

observations = "state"
checkpoint_path="/home/josebarreiros/ray_results/PPOTrainer_manipulation_station_env_for_ray_2022-08-09_22-19-408n7oszpc/checkpoint_002000/checkpoint-2000"

time_limit = 10 

def env_creator(env_config):
    meshcat = StartMeshcat()
    gym.envs.register(id="ManipulationStationBoxPushing-v0",
                  entry_point="envs.manipulation_station_pushing_box:ManipulationStationBoxPushingEnv")

    env = gym.make("ManipulationStationBoxPushing-v0",
        observations=observations,
        time_limit=time_limit,
        debug=args.debug,
        meshcat=meshcat)
    env.simulator.set_target_realtime_rate(1.0)

    return env  

if __name__ == '__main__':

    current_directory=os.path.dirname(os.path.realpath(__file__))
    ray.init(runtime_env={"working_dir": current_directory})
    register_env("manipulation_station_env_for_ray", env_creator)

    config= {
        "env": "manipulation_station_env_for_ray",
        "num_workers": 1,
        "framework": "torch",
    }
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(checkpoint_path)

    input("Press Enter to continue...")   

    env=env_creator(config)

    obs = env.reset()
    cumulative_reward=0

    for i in range(100000):
        action = trainer.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        cumulative_reward += reward
        env.render()
        if done:
            print("Cumulative reward you've received is: {}. Congratulations!".format(cumulative_reward))
            input("If continue the environment will reset. Press Enter to continue...")   
            obs = env.reset()
            cumulative_reward=0
