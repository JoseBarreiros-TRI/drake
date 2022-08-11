import argparse
from threading import current_thread
import gym
import os
import pdb
import wandb

from pydrake.geometry import Meshcat, Cylinder, Rgba, Sphere, StartMeshcat


import ray
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from ray.tune.integration.wandb import wandb_mixin, WandbLogger, WandbLoggerCallback
from ray.tune.logger import DEFAULT_LOGGERS

parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()

observations = "state"
time_limit = 7 if not args.test else 0.5
api_key_file = "/.wandb_api_key" #only if using wandb

def env_creator(env_config):
    # meshcat = StartMeshcat()
    gym.envs.register(id="ManipulationStationBoxPushing-v0",
                  entry_point="envs.manipulation_station_pushing_box:ManipulationStationBoxPushingEnv")

    env = gym.make("ManipulationStationBoxPushing-v0",
        observations=observations,
        time_limit=time_limit,
        debug=args.debug)
    return env  


if __name__ == '__main__':

    current_directory=os.path.dirname(os.path.realpath(__file__))
    ray.init(runtime_env={"working_dir": current_directory})
    register_env("manipulation_station_env_for_ray", env_creator)

    # Configure the algorithm.
    if args.test:
        config= {
            "env": "manipulation_station_env_for_ray",
            "num_workers": 2,
            "framework": "torch",
        }
    else:
        config = {
            # Environment (RLlib understands openAI gym registered strings).
            "env": "manipulation_station_env_for_ray",
            # Use 2 environment workers (aka "rollout workers") that parallelly
            # collect samples from their own environment clone(s).
            "num_workers": 4,
            # Change this to "framework: torch", if you are using PyTorch.
            # Also, use "framework: tf2" for tf2.x eager execution.
            "framework": "torch",
            # Tweak the default model provided automatically by RLlib,
            # given the environment's observation- and action spaces.
            # "model": {
            #     "custom_model": "SmallCNN",
            #     "custom_model_config": {
            #     }
            # },
        }

    # With Tune. 
    # Tune is a Ray tool for easy experiment management and visualization of the results.
    # The logs will be uploaded to wandb
    analysis=ray.tune.run(
        "PPO",
        stop={"episode_reward_mean": 1000},
        config=config,
        callbacks=[WandbLoggerCallback(
            project="Test_RL",
            api_key_file=current_directory+api_key_file,
            log_config=True)]
    )

    # if there are multiple trials, select a specific trial or automatically
    # choose the best one according to a given metric
    last_checkpoint = analysis.get_last_checkpoint(
        metric="episode_reward_mean", mode="max"
    )
    print(last_checkpoint)


    # # Uncomment to run without tune. 
    # # The results will be in ~/ray_results and can be checked with tensorboard
    # trainer = ppo.PPOTrainer(config=config)
    # i=0
    # while True:
    #     result=trainer.train()
    #     print(pretty_print(result))
    #     i+=1
    #     if args.test and i>2:
    #         break
    #     if i % 1000 == 0:
    #         checkpoint = trainer.save()
    #         print("checkpoint saved at", checkpoint)
