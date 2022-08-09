## Drake Gym
This is a fork of [DrakeGym](https://github.com/RussTedrake/manipulation/tree/master/manipulation) with aditional examples. 

Setting up your environment:

- Build Drake from this branch. The steps are explained [here](https://drake.mit.edu/from_source.html) (make sure you clone this branch and not RobotLocomotion/drake.git)
- Install OpenAI Gym ```pip install gym ```
- Install StableBaselines ```pip install stable-baselines3```

Training the policy

```bazel run //drake_gym:rl_train_punyoid_lifting_box```

Playing the policy

```bazel run //drake_gym:rl_play_punyoid_lifting_box```
