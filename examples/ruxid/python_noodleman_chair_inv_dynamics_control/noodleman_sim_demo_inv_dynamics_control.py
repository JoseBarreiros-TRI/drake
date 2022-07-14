"""
This is an example for simulating a simplified humanoid (aka. noodleman) through pydrake.
It reads three simple SDFormat files of a hydroelastic humanoid,
a rigid chair, and rigid floor.
It uses an inverse dynamics controller to bring the noodleman from a sitting to standing up position.
"""
import argparse
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import DrakeVisualizer
from pydrake.math import RigidTransform
from pydrake.math import RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant
from pydrake.multibody.plant import ConnectContactResultsToDrakeVisualizer
from pydrake.multibody.plant import MultibodyPlantConfig
from pydrake.systems.analysis import ApplySimulatorConfig
from pydrake.systems.analysis import Simulator
from pydrake.systems.analysis import SimulatorConfig
from pydrake.systems.analysis import PrintSimulatorStatistics
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import VectorLogSink
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.all import (DiagramBuilder,Parser,
                         RigidTransform, Simulator)
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.multibody.tree import WeldJoint

def make_noodleman_chair(contact_model, contact_surface_representation,
                     time_step):
    multibody_plant_config = \
        MultibodyPlantConfig(
            time_step=time_step,
            contact_model=contact_model,
            contact_surface_representation=contact_surface_representation)

    p_WChair_fixed = RigidTransform(RollPitchYaw(0, 0, 0),
                                     np.array([0, 0, 0]))
    p_WFloor_fixed = RigidTransform(RollPitchYaw(0, 0, 0),
                                     np.array([0, 0, 0]))
                                    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)

    parser = Parser(plant)

    floor_sdf_file_name = \
        FindResourceOrThrow("drake/examples/ruxid/python_noodleman_chair_inv_dynamics_control/models"
                            "/floor.sdf")
    floor=parser.AddModelFromFile(floor_sdf_file_name, model_name="floor")
    plant.WeldFrames(
        frame_on_parent_P=plant.world_frame(),
        frame_on_child_C=plant.GetFrameByName("floor", floor),
        X_PC=p_WFloor_fixed
    )

    chair_sdf_file_name = \
        FindResourceOrThrow("drake/examples/ruxid/python_noodleman_chair_inv_dynamics_control/models"
                            "/chair_v1.sdf")
    chair = parser.AddModelFromFile(chair_sdf_file_name, model_name="chair_v1")
    plant.WeldFrames(
        frame_on_parent_P=plant.world_frame(),
        frame_on_child_C=plant.GetFrameByName("chair", chair),
        X_PC=p_WChair_fixed
    )
    noodleman_sdf_file_name = \
        FindResourceOrThrow("drake/examples/ruxid/python_noodleman_chair_inv_dynamics_control/models"
                            "/noodleman_v1.sdf")
    noodleman=parser.AddModelFromFile(noodleman_sdf_file_name, model_name="noodleman")
    p_WNoodleman_fixed = RigidTransform(RollPitchYaw(0, 0, 0),
                                     np.array([0, 0.6, 0.35]))

    # weld the lower leg of the noodleman to the world frame. 
    # The inverse dynamic controller does not work with floating base
    joint=WeldJoint(
          name="weld_lower_leg",
          frame_on_parent_P=plant.world_frame(),
          frame_on_child_C=plant.GetFrameByName("lower_leg", noodleman),
          X_PC=p_WNoodleman_fixed
        )
    plant.AddJoint(joint)
    plant.Finalize()

    print("\nnumber of position: ",plant.num_positions(),
        ", number of velocities: ",plant.num_velocities(),
        ", number of actuators: ",plant.num_actuators(),
        ", number of multibody states: ",plant.num_multibody_states(),'\n')

    #Desired state corresponding to a standing up position [tetha0,tetha1,tetha1_dot,tetha2_dot].
    desired_state=np.array([0,0,0,0])
    print("desired state:",desired_state)
    desired_state_source=builder.AddSystem(ConstantVectorSource(desired_state))

    ##Create inverse dynamics controller
    U=plant.num_actuators()

    kp = 2.8
    ki = 0.01
    kd = 2.8

    IDC = builder.AddSystem(InverseDynamicsController(robot=plant,
                                        kp=np.ones(U)*kp,
                                        ki=np.ones(U)*ki,
                                        kd=np.ones(U)*kd,
                                        has_reference_acceleration=False))
                                               
    builder.Connect(IDC.get_output_port_control(),plant.get_applied_generalized_force_input_port())
    builder.Connect(plant.get_state_output_port(),IDC.get_input_port_estimated_state())
    builder.Connect(desired_state_source.get_output_port(),IDC.get_input_port_desired_state())

    constant_zero_torque=builder.AddSystem(ConstantVectorSource(np.zeros(U)))
    builder.Connect(constant_zero_torque.get_output_port(),plant.get_actuation_input_port())

    DrakeVisualizer.AddToBuilder(builder=builder, scene_graph=scene_graph)
    ConnectContactResultsToDrakeVisualizer(builder=builder, plant=plant,
                                           scene_graph=scene_graph)

    nx = plant.num_positions() + plant.num_velocities()
    state_logger = builder.AddSystem(VectorLogSink(nx))
    builder.Connect(plant.get_state_output_port(),
                    state_logger.get_input_port())

    diagram = builder.Build()
    return diagram, plant, state_logger, noodleman


def simulate_diagram(diagram, chair_noodleman_plant, state_logger,
                     noodleman_init_position, noodleman_init_velocity,
                     simulation_time, target_realtime_rate):

    #initial position and velocities
    #qv_init_val = np.array([1.95, -1.87, 0, 0])
    qv_init_val = np.concatenate((noodleman_init_position, noodleman_init_velocity))

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(chair_noodleman_plant,
                                                diagram_context)

    chair_noodleman_plant.SetPositionsAndVelocities(plant_context,
                                                 qv_init_val)
    
    print("Initial state variables: ", chair_noodleman_plant.GetPositionsAndVelocities(plant_context))
    
    #setup the simulator
    simulator_config = SimulatorConfig(
                           target_realtime_rate=target_realtime_rate,
                           publish_every_time_step=True)
    simulator = Simulator(diagram,diagram_context)
    ApplySimulatorConfig(simulator, simulator_config)
    
    #simulator.get_mutable_context().SetTime(0)
    state_log = state_logger.FindMutableLog(simulator.get_mutable_context())
    state_log.Clear()
    simulator.Initialize()
    simulator.AdvanceTo(boundary_time=simulation_time)
    PrintSimulatorStatistics(simulator)
    return state_log.sample_times(), state_log.data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulation_time", type=float, default=8,
        help="Desired duration of the simulation in seconds. "
             "Default 8.0.")
    parser.add_argument(
        "--contact_model", type=str, default="hydroelastic_with_fallback",
        help="Contact model. Options are: 'point', 'hydroelastic', "
             "'hydroelastic_with_fallback'. "
             "Default 'hydroelastic_with_fallback'")
    parser.add_argument(
        "--contact_surface_representation", type=str, default="polygon",
        help="Contact-surface representation for hydroelastics. "
             "Options are: 'triangle' or 'polygon'. Default 'polygon'.")
    parser.add_argument(
        "--time_step", type=float, default=0.001,
        help="The fixed time step period (in seconds) of discrete updates "
             "for the multibody plant modeled as a discrete system. "
             "If zero, we will use an integrator for a continuous system. "
             "Non-negative. Default 0.001.")
    parser.add_argument(
        "--noodleman_initial_position", nargs=2, metavar=('tetha1', 'theta2'),
        default=[1.95, -1.87],
        help="Noodleman's initial joint position: tetha1, theta2 (in rad). "
             "Default: 1.95 -1.87. It correspond to a sitting position")
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Target realtime rate. Default 1.0.")
    args = parser.parse_args()

    diagram, noodleman_chair_plant, state_logger,noodleman_idx = make_noodleman_chair(
        args.contact_model, args.contact_surface_representation,
        args.time_step)
    time_samples, state_samples = simulate_diagram(
        diagram, noodleman_chair_plant, state_logger,
        np.array(args.noodleman_initial_position),
        np.array([0., 0.]),
        args.simulation_time, args.target_realtime_rate)
    print("\nFinal state variables:")
    print(state_samples[:, -1])
