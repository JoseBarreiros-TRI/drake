import argparse
import gym
import os
import pdb

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from pydrake.geometry import Meshcat, Cylinder, Rgba, Sphere, StartMeshcat

gym.envs.register(id="Humanoid_fbase_StandUpPID-v0",
                  entry_point="envs.humanoid_fbase_standup_pid:NoodlemanStandUpEnv")

parser = argparse.ArgumentParser(
    description='Install ToC and Navigation into book html files.')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

observations = "state"
time_limit = 5 if not args.test else 0.5
zip = "/home/josebarreiros/rl/data/humanoid_fbase_StandUpPID_ppo_{observations}.zip"
log = "/home/josebarreiros/rl/tmp/humanoid_fbase_StandUpPID/"
checkpoint_callback = CheckpointCallback(save_freq=20000, save_path='/home/josebarreiros/tmp/humanoid_fbase_StandUpPID/model_checkpoints/')
debug=True

if __name__ == '__main__':
    num_cpu = 12 if not args.test else 2
    if not debug:
        env = make_vec_env("Noodleman_fbase_StandUpPID-v0",
                        n_envs=num_cpu,
                        seed=0,
                        vec_env_cls=SubprocVecEnv,
                        env_kwargs={
                            'observations': observations,
                            'time_limit': time_limit,
                        })
    else:
    #env = "NoodlemanStandUp-v0"
        meshcat = StartMeshcat()
        env = gym.make("Humanoid_fbase_StandUpPID-v0", meshcat=meshcat, 
            observations=observations,time_limit=time_limit, debug=debug)
        input("Press Enter to continue...")
        
    #pdb.set_trace()
    if args.test:
        model = PPO('MlpPolicy', env, n_steps=4, n_epochs=2, batch_size=8)
    #elif os.path.exists(zip):
    #    model = PPO.load(zip, env, verbose=1, tensorboard_log=log)
    else:
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log)

    new_log = True
    while True:
        model.learn(total_timesteps=100000 if not args.test else 4,
                    reset_num_timesteps=new_log, callback=[checkpoint_callback] if not args.test else [])
        if args.test:
            break
        model.save(zip)
        print("\nModel saved")
        new_log = False