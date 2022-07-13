from stable_baselines3 import PPO
import gym
from envs.pendulum import PendulumEnv
from stable_baselines3.common.env_checker import check_env

from pydrake.all import (
    Meshcat,
)

from pydrake.geometry import Meshcat, Cylinder, Rgba, Sphere, StartMeshcat

meshcat = StartMeshcat()
#meshcat=Meshcat()
env = PendulumEnv(meshcat=meshcat,time_limit=2)

check_env(env)
meshcat.Delete()

model = PPO(policy = "MlpPolicy",env =  env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)
    env.render(mode='human')
    if done:
      obs = env.reset()
print("END")