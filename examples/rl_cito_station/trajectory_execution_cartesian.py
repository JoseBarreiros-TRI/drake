"""
This example plans a trajectory to a desired joint pose and executes it either
in simulation or in hardware.
"""
import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np
import pdb
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
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import PassThrough

from anzu.tactile.rl_cito_station.cc import (
    RlCitoStation,
    RlCitoStationHardwareInterface,
)
from anzu.tactile.rl_cito_station.trajectory_planner import TrajectoryPlanner
from anzu.tactile.rl_cito_station.differential_ik import DifferentialIK
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.manipulation.planner import DoDifferentialInverseKinematics
from pydrake.multibody import inverse_kinematics as ik
import pydrake.solvers.mathematicalprogram as mp

from pydrake.all import (
    OrientationConstraint,
)
# Set the numpy print precision.
np.set_printoptions(precision=5)

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
            has_optitrack=args.mocap))
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
            "models/tactile/custom_box.sdf",
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
        #pdb.set_trace()

    return diagram, plant, controller_plant, station


def simulate_diagram(diagram,
                     plant,
                     controller_plant,
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

    controller_plant_context = controller_plant.CreateDefaultContext()

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
    q0_arm = station.GetOutputPort("iiwa_position_measured").Eval(
        station_context)
    # Keep the arm at the measured pose.
    diagram.GetInputPort("iiwa_position_commanded").FixValue(
        diagram_context, np.array(q0_arm))
    time_tracker += 3
    simulator.AdvanceTo(time_tracker)

    # Keypoints in cartesian space
    points=[
        [0.3,0.3,0.5],
        [0.3,-0.3,0.5],
        [0.6,-0.3,0.5],
        [0.6,0.3,0.5],
    ]
    # points=[
    #     [0.0,0.3,0.5],
    #     [0.4,0.3,0.5],
    #     [0.4,-0.3,0.5],
    #     [0.0,-0.3,0.5],
    #     [-0.4,-0.3,0.5],
    #     [-0.4,0.3,0.5],
    # ]


   # Arguments for IK
    frame_EE=controller_plant.GetFrameByName("iiwa_link_7")
    TOLERANCE_TRANSLATION=0.02
    TOLERANCE_ROTATION=0.1

    # A scale in [0.5, 1] to cut down the joint position,
    # velocity, and effort limits.
    POS_LIMIT_FACTOR = 0.7
    VEL_LIMIT_FACTOR = 0.5
    EFFORT_LIMIT_FACTOR = 0.5

    def EE_orientation_constraint(program, plant, context):
        program.AddConstraint(
                OrientationConstraint(
                    plant,
                    frameAbar=plant.GetFrameByName("iiwa_link_7"),
                    R_AbarA=RotationMatrix(),
                    frameBbar=plant.world_frame(),
                    R_BbarB=RotationMatrix(),
                    theta_bound=TOLERANCE_ROTATION,
                    plant_context=context,
                )
            )


    # Set up a trajectory planner.
    planner = TrajectoryPlanner(
        initial_pose=q0_arm,
        preview=False,
        position_limit_factor=POS_LIMIT_FACTOR,
        velocity_limit_factor=VEL_LIMIT_FACTOR,
        effort_limit_factor=EFFORT_LIMIT_FACTOR,
        additional_constraints=None,#[EE_orientation_constraint],
        )

    simulator.Initialize()

    input("\n\nPress Enter to run the simulation...")
    while simulator.get_context().get_time()<simulation_time:

        for xyz_desired in points:
            # setup an ik problem
            my_ik = ik.InverseKinematics(
                plant=controller_plant, plant_context=controller_plant_context, with_joint_limits=True,
            )
            prog = my_ik.get_mutable_prog()
            q = my_ik.q()
            xyz_desired=np.array(xyz_desired)
            p_lower=xyz_desired-np.ones(3)*TOLERANCE_TRANSLATION
            p_upper=xyz_desired+np.ones(3)*TOLERANCE_TRANSLATION

            # update initial position for the planner and ik
            q0_arm = station.GetOutputPort("iiwa_position_measured").Eval(
                                station_context)
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

            #Find the plan.
            planner.q0=q0_arm
            plan = planner.plan_to_joint_pose(q_goal=q_desired)

            # Execute the plan.
            if not hardware:
                for _ in range(int(2/time_step)):
                    time_tracker += time_step
                    q_cmd = plan.value(time_tracker)
                    diagram.GetInputPort("iiwa_position_commanded").FixValue(
                        diagram_context, q_cmd)
                    print(f"\tt: {time_tracker}, qpos: {q_cmd.T}")
                    simulator.AdvanceTo(time_tracker)
                print("\nThe simulation has been completed")
            else:
                iiwa_time = -1
                start_time = copy.copy(station.GetOutputPort(
                    "iiwa_time_measured").Eval(station_context))
                print("iiwa_start_time: ", start_time)

                while iiwa_time<1.5:
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
                print("\nThe simulation has been completed")

    return 1

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
        "--plot_diagram", action="store_true",
        help="Plot the diagram flowchart.")
    parser.add_argument(
        "--simulation_time", type=float, default=30,
        help="Desired duration of the simulation in seconds. "
             "Default 30.0.")
    args = parser.parse_args()

    if args.meshcat:
        meshcat_server = Meshcat()
        visualizer = meshcat_server
    else:
        visualizer = None

    diagram, plant, controller_plant,station = make_environment(
        meshcat=visualizer, hardware=args.hardware, args=args)

    simulate_diagram(
        diagram,
        plant,
        controller_plant,
        station,
        simulation_time=args.simulation_time,
        target_realtime_rate=1.0,
        hardware=args.hardware,
        mocap=args.mocap,
        )
