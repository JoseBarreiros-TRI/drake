import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_checker import check_env

from pydrake.all import (
    DiagramBuilder,
    Meshcat,
    MeshcatVisualizerCpp,
    MeshcatVisualizerParams,
    RigidTransform,
    RotationMatrix,
    RollPitchYaw,
    SceneGraph,
    Simulator,
    EventStatus,
)
from pydrake.examples.pendulum import PendulumPlant, PendulumGeometry

from drake_gym import DrakeGymEnv
from scenarios import AddRgbdSensor

import pdb


def PendulumEnv(meshcat=None,time_limit=1):
    builder = DiagramBuilder()
    plant = builder.AddSystem(PendulumPlant())
    scene_graph = builder.AddSystem(SceneGraph())
    PendulumGeometry.AddToBuilder(builder, plant.get_state_output_port(),
                                  scene_graph)
    MeshcatVisualizerCpp.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(publish_period=np.inf))

    builder.ExportInput(plant.get_input_port(), "torque")
    # Note: The Pendulum-v0 in gym outputs [cos(theta), sin(theta), thetadot]
    builder.ExportOutput(plant.get_state_output_port(), "state")

    # Add a camera (I have sugar for this in the manip repo)
    X_WC = RigidTransform(RotationMatrix.MakeXRotation(-np.pi / 2),
                          [0, -1.5, 0])
    rgbd = AddRgbdSensor(builder, scene_graph, X_WC)
    builder.ExportOutput(rgbd.color_image_output_port(), "camera")

    diagram = builder.Build()
    simulator = Simulator(diagram)

    # Termination conditions:
    def monitor(context):
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    def reward(system, context):
        plant_context = plant.GetMyContextFromRoot(context)
        state = plant_context.get_continuous_state_vector()
        u = plant.get_input_port().Eval(plant_context)[0]
        theta = state[0] % 2 * np.pi  # Wrap to 2*pi
        theta_dot = state[1]
        return (theta - np.pi)**2 + 0.1 * theta_dot**2 + 0.001 * u

    max_torque = 3
    env = DrakeGymEnv(simulator,
                      time_step=0.05,
                      action_space=gym.spaces.Box(low=np.array([-max_torque]),
                                                  high=np.array([max_torque])),
                      observation_space=gym.spaces.Box(
                          low=np.array([-np.inf, -np.inf]),
                          high=np.array([np.inf, np.inf]),
                          dtype=np.float64,
                          ),
                      reward=reward,
                      render_rgb_port_id="camera")
    
    
    return env