import argparse
import gym
import os
import pdb
import time
import numpy as np

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from pydrake.geometry import StartMeshcat
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
)

import pickle
from pydrake.math import RigidTransform, RotationMatrix

from pydrake.multibody import inverse_kinematics as ik
import pydrake.solvers.mathematicalprogram as mp
from drake.examples.rl_cito_station.trajectory_planner import TrajectoryPlanner

parser = argparse.ArgumentParser(
    description=' ')
parser.add_argument('--test', action='store_true')
parser.add_argument('--train_single_env', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--log_path', help="path to the logs directory.")
parser.add_argument('--algo', help="training algorithm.  Valid options are [PPO, DDPG, SAC, TD3]")
parser.add_argument('--notes', help="log extra notes to wandb.")
args = parser.parse_args()

gym.envs.register(id="RlCitoStationBoxPushing-v2",
                  entry_point="envs.rl_cito_station_pushing_box_v2:RlCitoStationBoxPushingEnv")


config = {
    "task": "reach",
    "total_episodes": 1e8,
    "env_name": "RlCitoStationBoxPushing-v2",
    "env_time_limit": 7,
    "local_log_dir":
        args.log_path if args.log_path is not None else os.environ['HOME']+
        "/rl/tmp/RlCitoStationBoxPushing_v2/",
    "observation_noise": False,
    "disturbances": False,
    # valid control modes are:
    # joint_positions, EE_pose, EE_delta_pose.
    "control_mode": "EE_pose",
    # valid observation types are:
    # "state", "actions", "distances","EE_box_target_xyz",
    # "torques", and any (or none) of the following:
    # "buffer_10", "buffer_20" for a history
    # of observations (n= 10, 20).
    "observation_type": ["state",
                        "actions",
                        "EE_box_target_xyz",
                        "distances",
                        "torques",
                        ],
    # valid reward types are:
    # "cost_goal", "cost_goal_normalized",
    # "cost_effort","cost_energy","cost_collision",
    # "bonus_success"
    "reward_type": ["cost_paper",
                    "bonus_success",
                    ],
    #valid termination types are:
    # 'box_off_table', "success", "collision_w_table",
    # "velocity_limits"
    "termination_type": ["box_off_table",
                        "success",
                         ],
    # valid reset types are:
    # "home" or a combination of the following "random_positions",
    # "random_velocities", "random_mass","random_friction"
    "reset_type": ["random_positions_diffik"],
    }

if __name__ == '__main__':
    task=config["task"]
    env_name = config["env_name"]
    time_limit = config["env_time_limit"]
    log_dir = config["local_log_dir"]
    total_episodes = int(config["total_episodes"]) if not args.test else 2
    obs_noise = config["observation_noise"]
    add_disturbances = config["disturbances"]
    obs_type=config["observation_type"]
    rew_type=config["reward_type"]
    reset_type=config["reset_type"]
    termination_type=config["termination_type"]
    control_mode=config["control_mode"]


    meshcat = StartMeshcat()
    env = gym.make(env_name,
                    meshcat=meshcat,
                    debug=args.debug,
                    time_limit=time_limit,
                    task=task,
                    obs_noise=obs_noise,
                    add_disturbances=add_disturbances,
                    observation_type=obs_type,
                    reward_type=rew_type,
                    reset_type=reset_type,
                    termination_type=termination_type,
                    control_mode=control_mode,
                    monitoring_camera=True,
                    )

    if args.debug:
        check_env(env)
        env.simulator.set_target_realtime_rate(1.0)
    input("Open meshcat (optional). Press Enter to continue...")



    diagram = env.simulator.get_system()
    diagram_context=env.simulator.get_mutable_context()
    station = diagram.GetSubsystemByName("rl_cito_station")
    controller_plant=station.get_controller_plant()
    station_context=diagram.GetMutableSubsystemContext(station,diagram_context)
    controller_plant_context = controller_plant.CreateDefaultContext()
    TOLERANCE_TRANSLATION=0.02
    TOLERANCE_ROTATION=0.1
    frame_EE=controller_plant.GetFrameByName("iiwa_link_7")

    POS_LIMIT_FACTOR = 0.7
    VEL_LIMIT_FACTOR = 0.5
    EFFORT_LIMIT_FACTOR = 0.5
    # Set up a trajectory planner.
    q0_arm=station.GetOutputPort("iiwa_position_measured").Eval(station_context)
    planner = TrajectoryPlanner(
        initial_pose=q0_arm,
        preview=False,
        position_limit_factor=POS_LIMIT_FACTOR,
        velocity_limit_factor=VEL_LIMIT_FACTOR,
        effort_limit_factor=EFFORT_LIMIT_FACTOR,
        additional_constraints=None,#[EE_orientation_constraint],
        )


    filename=log_dir+"/data/data.data"
    done=False
    time_tracker=0
    time_step=env.gym_time_step
    data=[]
    for i in range(total_episodes):
        if done or i==0:
            input("The environment will reset. Press Enter to continue...")
            obs = env.reset()

            # Observation mapping
            # obs_type (size):
            # 1. state (2*Na),
            # 2. actions (joint_positions: Na, EE_pose or EE_delta_pose: 6),
            # 3. distances (3),
            # 4. EE_box_target_xyz (9),
            # 5. torques (Na).

            # Extract box position from observations
            Na=7 #number of actuators of iiwa14
            idx_EE_box_target_xyz=2*Na+(Na if control_mode=="joint_positions" else 6) + 3
            EE_box_target_xyz=obs[idx_EE_box_target_xyz:idx_EE_box_target_xyz+9]
            box_xyz=EE_box_target_xyz[3:6]

            # Get iiwa state
            q0_arm=station.GetOutputPort("iiwa_position_measured").Eval(station_context)

            # Get desired joint positions when EE is on top of the box
            my_ik = ik.InverseKinematics(
                plant=controller_plant, plant_context=controller_plant_context, with_joint_limits=True,
            )
            prog = my_ik.get_mutable_prog()
            q = my_ik.q()
            xyz_desired=box_xyz
            p_lower=xyz_desired-np.ones(3)*TOLERANCE_TRANSLATION
            p_upper=xyz_desired+np.ones(3)*TOLERANCE_TRANSLATION

            # update initial position for the planner and ik
            prog.SetInitialGuess(q, q0_arm)
            my_ik.AddPositionConstraint(
                frameB=frame_EE,
                p_BQ=np.array([0,0,0]),
                frameA=controller_plant.world_frame(),
                p_AQ_lower=p_lower,
                p_AQ_upper=p_upper,
            )
            my_ik.AddOrientationConstraint(
                frameAbar=frame_EE,
                R_AbarA=RotationMatrix(),
                frameBbar=controller_plant.world_frame(),
                R_BbarB=RotationMatrix(),
                theta_bound=TOLERANCE_ROTATION,
            )
            result = mp.Solve(prog)

            # Get the solution.
            q_res = result.GetSolution(q)
            iiwa = controller_plant.GetModelInstanceByName("iiwa")
            q_desired = controller_plant.GetPositionsFromArray(iiwa, q_res)
            # Evaluate forward dynamics for the IK solution.
            controller_plant.SetPositions(controller_plant_context, iiwa, q_desired)
            X_WE = controller_plant.CalcRelativeTransform(
                controller_plant_context,
                controller_plant.world_frame(),
                frame_EE)
            position_error = X_WE.translation() - xyz_desired
            # Compare the desired and found Cartesian poses.
            print(f"Desired Cartesian pose: {xyz_desired}")
            print(f"           IK solution: {q_desired}")
            print(f"        Cartesian pose: {X_WE.translation()}")
            print(f"                 Error: {position_error}, norm: {np.linalg.norm(position_error)}")
            # Confirm that the resulting Cartesian position is close enough.
            assert np.linalg.norm(X_WE.translation() - xyz_desired) < 2 * TOLERANCE_TRANSLATION

            # Get the plan to go from q0 to q_desired
            if task=="reach":
                planner.q0=q0_arm
                plan = planner.plan_to_joint_pose(q_goal=q_desired)
            else:
                print("Not Implemented")

            #save episode data
            with open(filename, 'a+') as fp:
                pickle.dump(episode_data,fp)
            data.append(episode_data)

            # clean the variable for a new episode
            episode_data=[]
            #pdb.set_trace()

            # Wait for meshcat to load the env.
            # TODO(JoseBarreiros-TRI) Replace sleep() with a readiness signal
            # from meshcat.
            time.sleep(0.7)

        if control_mode=="EE_pose":
            action=plan.value(time_tracker)
        time_tracker += time_step
        print(f"\tt: {time_tracker}, qpos: {action.T}")
        obs, reward, done, info = env.step(action)

        step_data=[obs,reward, action, done]
        episode_data.append(step_data)

        if args.debug:
            # This will play the policy step by step.
            input("Press Enter to continue...")

        env.render()


