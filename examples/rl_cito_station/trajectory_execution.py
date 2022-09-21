"""
This example plans a trajectory to a desired joint pose and executes it either
in simulation or in hardware.
"""
import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np

from pydrake.geometry import (
    CollisionFilterDeclaration,
    GeometrySet,
    Meshcat,
    MeshcatVisualizer,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.analysis import (
    ApplySimulatorConfig,
    Simulator,
    SimulatorConfig,
)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import PassThrough

from pydrake.examples import (
    RlCitoStation,
    RlCitoStationHardwareInterface,
)
from drake.examples.rl_cito_station.trajectory_planner import TrajectoryPlanner

# Set the numpy print precision.
np.set_printoptions(precision=5)

# Desired joint pose of the arm.
desired_arm_pos = np.array([0, 0, 0, np.pi/2, 0, -np.pi/2, 0])

# A scale in [0.5, 1] to cut down the joint position,
# velocity, and effort limits.
arm_limit_scale = 0.7

# Environment parameters
desired_box_pos = np.array([1, 0, 0, 0, 1, 0, 0.075])
initial_arm_pos = np.zeros(7)
initial_box_pos = np.array([1, 0, 0, 0, 0.6, 0, 0.075])
time_step = 1e-3
box_height = 0.15
contact_model = 'point'
contact_solver = 'sap'


# Filter collisison between parent and child of each joint.
def add_collision_filters(scene_graph, plant):
    filter_manager = scene_graph.collision_filter_manager()
    body_pairs = [
        ["iiwa_link_1", "iiwa_link_2"],
        ["iiwa_link_2", "iiwa_link_3"],
        ["iiwa_link_3", "iiwa_link_4"],
        ["iiwa_link_4", "iiwa_link_5"],
        ["iiwa_link_5", "iiwa_link_6"],
        ["iiwa_link_6", "iiwa_link_7"],
    ]

    for pair in body_pairs:
        parent = plant.GetBodyByName(pair[0])
        child = plant.GetBodyByName(pair[1])

        set = GeometrySet(
            plant.GetCollisionGeometriesForBody(parent) +
            plant.GetCollisionGeometriesForBody(child))
        filter_manager.Apply(
            declaration=CollisionFilterDeclaration().ExcludeWithin(
                set))


def make_environment(meshcat=None, hardware=False, args=None):
    builder = DiagramBuilder()

    if hardware:
        camera_ids = []
        station = builder.AddSystem(RlCitoStationHardwareInterface(
            has_optitrack=True))
        station.Connect(wait_for_optitrack=False)
        controller_plant = station.get_controller_plant()
        plant = None
    else:
        station = builder.AddSystem(
            RlCitoStation(
                time_step=time_step,
                contact_model=contact_model,
                contact_solver=contact_solver))
        station.SetupCitoRlStation()

        station.AddManipulandFromFile(
            "drake/examples/rl_cito_station/models/custom_box.sdf",
            RigidTransform(RotationMatrix.Identity(), np.zeros(3)), "box")

        controller_plant = station.get_controller_plant()
        plant = station.get_multibody_plant()

        station.Finalize()

        if meshcat:
            geometry_query_port = station.GetOutputPort("geometry_query")
            MeshcatVisualizer.AddToBuilder(
                builder=builder,
                query_object_port=geometry_query_port,
                meshcat=meshcat)

    # Connect iiwa_position to the commanded pose.
    iiwa_position = builder.AddSystem(
        PassThrough(controller_plant.num_actuators()))
    builder.Connect(iiwa_position.get_output_port(),
                    station.GetInputPort("iiwa_position"))
    builder.ExportInput(iiwa_position.get_input_port(),
                        "iiwa_position_commanded")

    # Build (and plot) the diagram.
    diagram = builder.Build()
    if args.plot_diagram:
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.plot(1)
        plt.show(block=False)

    return diagram, plant, station


def simulate_diagram(diagram,
                     plant,
                     station,
                     simulation_time,
                     target_realtime_rate,
                     hardware=False,
                     mocap=False,
                     preview=False):
    # Create context for the diagram.
    diagram_context = diagram.CreateDefaultContext()

    # Setup the simulator.
    simulator_config = SimulatorConfig(
        target_realtime_rate=target_realtime_rate,
        publish_every_time_step=False)

    simulator = Simulator(diagram, diagram_context)

    ApplySimulatorConfig(config=simulator_config, simulator=simulator)
    station_context = diagram.GetMutableSubsystemContext(
        station, simulator.get_mutable_context())

    if not hardware:
        plant_context = diagram.GetMutableSubsystemContext(plant,
                                                           diagram_context)
        print("Initial state variables: ",
              plant.GetPositionsAndVelocities(plant_context))

    # Get the simulator data.
    context = simulator.get_mutable_context()
    context.SetTime(0)

    # Advance the simulation for handling messages in the hardware case.
    time_tracker = 0
    if hardware:
        time_tracker += 1e-6
        simulator.AdvanceTo(time_tracker)
    else:
        # Set the system pose to the prescribed values.
        plant.SetPositions(plant_context, np.hstack(
            (initial_arm_pos, initial_box_pos)))

    # Get the initial pose from the robot.
    q0 = []
    q0_arm = station.GetOutputPort("iiwa_position_measured").Eval(
        station_context)
    # Keep the arm at the measured pose.
    diagram.GetInputPort("iiwa_position_commanded").FixValue(
        diagram_context, np.array(q0_arm))
    time_tracker += 1
    simulator.AdvanceTo(time_tracker)
    # Get the box pose from the mo-cap.
    if mocap:
        q0_box = station.GetOutputPort(
            "optitrack_manipuland_pose").Eval(station_context)
        # Set the initial pose of the system.
        q0 = np.hstack((q0_arm, 1, 0, 0, 0, q0_box.translation()[0],
                        q0_box.translation()[1], box_height/2))
    else:
        q0 = np.hstack((q0_arm, 1, 0, 0, 0, 1, 0, 0.075))

    # Plan a trajectory.
    POS_LIMIT_FACTOR=0.7
    VEL_LIMIT_FACTOR=0.7
    EFFORT_LIMIT_FACTOR=0.7

    planner = TrajectoryPlanner(
        initial_pose=q0,
        preview=preview,
        position_limit_factor=POS_LIMIT_FACTOR,
        velocity_limit_factor=VEL_LIMIT_FACTOR,
        effort_limit_factor=EFFORT_LIMIT_FACTOR,
        )
    plan = planner.plan_to_joint_pose(q_goal=desired_arm_pos)

    # Check for user permit.
    user_permit = ''
    while user_permit != 'y':
        user_permit = input("\n\tWould you like to run this trajectory? (y/N) ")
        if user_permit == 'N':
            return 0

    # Run a simulation executing the planned trajectory.
    if not hardware:
        simulator.Initialize()
        input("\n\nPress Enter to run the simulation...")
        for _ in range(int(simulation_time/time_step)):
            time_tracker += time_step
            q_cmd = plan.value(time_tracker)
            diagram.GetInputPort("iiwa_position_commanded").FixValue(
                diagram_context, q_cmd)
            print(f"\tt: {time_tracker}, qpos: {q_cmd.T}")
            simulator.AdvanceTo(time_tracker)
        print("\nThe simulation has been completed")
        return 1

    # Otherwise, command joint poses based on the time.
    input("Press Enter to start the execution...\n\n")
    iiwa_time = -1
    time_tracker += 3
    simulator.AdvanceTo(time_tracker)
    start_time = copy.copy(station.GetOutputPort(
        "iiwa_time_measured").Eval(station_context))
    print(start_time)
    while iiwa_time < 5:
        time_tracker += time_step
        simulator.AdvanceTo(time_tracker)
        # Get the current time.
        cur_time = copy.copy(station.GetOutputPort(
            "iiwa_time_measured").Eval(station_context))
        # Evaluate the time from the start in sec.
        iiwa_time = cur_time - start_time
        # Evaluate the corresponding joint pose command.
        q_cmd = plan.value(iiwa_time)
        print(f"time: {iiwa_time[0]}, cmd: {q_cmd[:, 0].T}")
        # Send the command.
        station.GetInputPort("iiwa_position").FixValue(
            station_context, q_cmd)
    print("\nCompleted the execution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meshcat", action="store_true",
        help="If set, visualize in meshcat. Use DrakeVisualizer otherwise")
    parser.add_argument(
        "--hardware", action="store_true",
        help="Use the RlCitoStationHardwareInterface instead of an "
             "in-process simulation.")
    parser.add_argument(
        "--mocap", action="store_true",
        help="Use the Optitrack detections instead of hard-coded box pose.")
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview the planned trajectory before execution.")
    parser.add_argument(
        "--plot_diagram", action="store_true",
        help="Plot the diagram flowchart.")
    args = parser.parse_args()

    if args.meshcat:
        meshcat_server = Meshcat()
        visualizer = meshcat_server
    else:
        visualizer = None

    diagram, plant, station = make_environment(
        meshcat=visualizer, hardware=args.hardware, args=args)

    simulate_diagram(
        diagram,
        plant,
        station,
        simulation_time=5.0,
        target_realtime_rate=1.0,
        hardware=args.hardware,
        mocap=args.mocap,
        preview=args.preview)
