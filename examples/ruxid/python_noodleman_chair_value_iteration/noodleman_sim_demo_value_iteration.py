"""
This is an example for simulating a simplified humanoid (aka. noodleman) through pydrake.
It reads three simple SDFormat files of a hydroelastic humanoid,
a rigid chair, and rigid floor.
It uses a value iteration policy to bring the noodleman from a sitting to standing up position.
"""
import argparse
import numpy as np
from matplotlib import cm
from time import sleep
import matplotlib.pyplot as plt
import pdb

from pydrake.common import FindResourceOrThrow
from pydrake.geometry import DrakeVisualizer
from pydrake.math import RigidTransform
from pydrake.math import RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant
from pydrake.multibody.plant import MultibodyPlantConfig
from pydrake.systems.analysis import ApplySimulatorConfig
from pydrake.systems.analysis import Simulator
from pydrake.systems.analysis import SimulatorConfig
from pydrake.systems.analysis import PrintSimulatorStatistics
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import VectorLogSink
from pydrake.all import (DiagramBuilder,Parser,
                         RigidTransform, Simulator)
from pydrake.multibody.tree import WeldJoint
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.plant import ConnectContactResultsToDrakeVisualizer
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz


from pydrake.all import (DiagramBuilder, DiscreteAlgebraicRiccatiEquation,
                         DynamicProgrammingOptions, FittedValueIteration,
                         InputPortIndex, LeafSystem, LinearSystem,
                         LogVectorOutput, MathematicalProgram,
                         MeshcatVisualizerCpp, MultilayerPerceptron,
                         PerceptronActivationType, PeriodicBoundaryCondition,
                         Polynomial, RandomGenerator, Rgba, RigidTransform,
                         RotationMatrix, SceneGraph, Simulator, Solve,
                         StartMeshcat, SymbolicVectorSystem, Variable,
                         Variables, WrapToSystem, ZeroOrderHold)

def noodleman_chair_plant(contact_model, contact_surface_representation,
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
    #plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    parser = Parser(plant)

    floor_sdf_file_name = \
        FindResourceOrThrow("drake/examples/ruxid/python_noodleman_chair_value_iteration/models"
                            "/floor.sdf")
    floor=parser.AddModelFromFile(floor_sdf_file_name, model_name="floor")
    plant.WeldFrames(
        frame_on_parent_P=plant.world_frame(),
        frame_on_child_C=plant.GetFrameByName("floor", floor),
        X_PC=p_WFloor_fixed
    )

    chair_sdf_file_name = \
        FindResourceOrThrow("drake/examples/ruxid/python_noodleman_chair_value_iteration/models"
                            "/chair_v1.sdf")
    chair = parser.AddModelFromFile(chair_sdf_file_name, model_name="chair_v1")
    plant.WeldFrames(
        frame_on_parent_P=plant.world_frame(),
        frame_on_child_C=plant.GetFrameByName("chair", chair),
        X_PC=p_WChair_fixed
    )
    noodleman_sdf_file_name = \
        FindResourceOrThrow("drake/examples/ruxid/python_noodleman_chair_value_iteration/models"
                            "/noodleman_v1.sdf")
    noodleman_idx=parser.AddModelFromFile(noodleman_sdf_file_name, model_name="noodleman")
    p_WNoodleman_fixed = RigidTransform(RollPitchYaw(0, 0, 0),
                                     np.array([0, 0.6, 0.35]))

    #weld the lower leg of the noodleman to the world frame. 
    #The inverse dynamic controller does not work with floating base
    joint=WeldJoint(
          name="weld_lower_leg",
          frame_on_parent_P=plant.world_frame(),
          frame_on_child_C=plant.GetFrameByName("lower_leg", noodleman_idx),
          X_PC=p_WNoodleman_fixed
        )
    plant.AddJoint(joint)
    plant.Finalize()

    print("\nnumber of position: ",plant.num_positions(),
        ", number of velocities: ",plant.num_velocities(),
        ", number of actuators: ",plant.num_actuators(),
        ", number of multibody states: ",plant.num_multibody_states(),'\n')

    DrakeVisualizer.AddToBuilder(builder=builder, scene_graph=scene_graph)
    ConnectContactResultsToDrakeVisualizer(builder=builder, plant=plant,
                                           scene_graph=scene_graph)

    nx = plant.num_positions() + plant.num_velocities()
    state_logger = builder.AddSystem(VectorLogSink(nx))
    builder.Connect(plant.get_state_output_port(),
                    state_logger.get_input_port())
    
    builder.ExportInput(plant.GetInputPort('noodleman_actuation'))

    diagram = builder.Build()

    return diagram, plant, state_logger, noodleman_idx


def noodleman_standUp_example(contact_model, contact_surface_representation,
                     time_step,min_time=True, animate=True):

    diagram, plant, state_logger, noodleman_idx = noodleman_chair_plant(contact_model, contact_surface_representation,
                     time_step)

    diagram_context = diagram.CreateDefaultContext()
    context = diagram.GetMutableSubsystemContext(plant,
                                                diagram_context)

    #print('cont: ',context.has_only_continuous_state()) 
    
    simulator = Simulator(diagram)#,plant.CreateDefaultContext())

    options = DynamicProgrammingOptions()

    q1s = np.linspace(0., 2. * np.pi, 5)
    q2s = np.linspace(0., 2. * np.pi, 5)
    q1dots = np.linspace(-10., 10., 5)
    q2dots = np.linspace(-10., 10., 5)

    state_grid = [set(q1s),set(q2s),set(q1dots),set(q2dots)]
    options.periodic_boundary_conditions = [
        PeriodicBoundaryCondition(0, 0., 2. * np.pi),
        PeriodicBoundaryCondition(1, 0., 2. * np.pi)
    ]
    options.discount_factor = .999
    
    
    print('joints:')
    for joint_idx in plant.GetJointIndices(noodleman_idx):
        joint_name=plant.get_joint(joint_idx).name()
        print(joint_name)

    options.input_port_index=InputPortIndex(0) #plant.GetInputPort('noodleman_actuation').get_index() #InputPortIndex(5) #

    #print(plant.GetInputPort('noodleman_actuation').size())
    #input_port =system.get_input_port_selection(options.in)
    #print('input port: ',system.get_input_port(options.input_port_index)) 

    input_limit = np.pi*1.5
    input_grid = [set(np.linspace(-input_limit, input_limit, 5)),set(np.linspace(-input_limit, input_limit, 5))]
    timestep = 0.01

    Q1, Q2, Q1dot,Q2dot = np.meshgrid(q1s, q2s, q1dots, q2dots)
    
    #visualize plant and diagram
    plt.figure()
    plot_graphviz(plant.GetTopologyGraphvizString())
    plt.figure()
    plot_system_graphviz(diagram, max_depth=2)
    #plt.show()

    def simulate(policy):
        # Animate the resulting policy.
        pass

    def min_time_cost(diagram_context):
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        #context_=context#.GetSubsystemContext('')
        x = plant_context.get_continuous_state_vector().CopyToVector()
        if x.dot(x) < .05:
            return 0.
        return 1.

    def quadratic_regulator_cost(diagram_context):
        #pdb.set_trace()
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        #context_=context#.GetSubsystemContext('')
        x = plant_context.get_continuous_state_vector().CopyToVector()
        #print(x)
        idx=plant.GetInputPort('noodleman_actuation').get_index()
        #idx=InputPortIndex(5) #
        u = plant.EvalVectorInput(plant_context, idx).CopyToVector()
        #print(u)
        return 2 * x.dot(x) + u.dot(u)

    if min_time:
        cost_function = min_time_cost
        options.convergence_tol = 0.001
    else:
        cost_function = quadratic_regulator_cost
        options.convergence_tol = 0.1

    policy, cost_to_go = FittedValueIteration(simulator, cost_function,
                                              state_grid, input_grid, timestep,
                                              options)

    print(cost_to_go)
    
    #J = np.reshape(cost_to_go, Q.shape)

    #if animate:
    #    print('Simulating...')
    #    simulate(policy)
    # pdb.set_trace()
    # fig = plt.figure(figsize=(9, 4))
    # ax1, ax2 = fig.subplots(1, 2)
    # ax1.set_xlabel("q")
    # ax1.set_ylabel("qdot")
    # ax1.set_title("Cost-to-Go")
    # ax2.set_xlabel("q")
    # ax2.set_ylabel("qdot")
    # ax2.set_title("Policy")
    # ax1.imshow(J,
    #            cmap=cm.jet, aspect='auto',
    #            extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]))
    # ax1.invert_yaxis()
    # Pi = np.reshape(policy.get_output_values(), Q.shape)
    # ax2.imshow(Pi,
    #            cmap=cm.jet, aspect='auto',
    #            extent=(qbins[0], qbins[-1], qdotbins[-1], qdotbins[0]))
    # ax2.invert_yaxis()
    # plt.show()
    # plt.pause(1)


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
        "--time_step", type=float, default=0,
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
    #pdb.set_trace()
    noodleman_standUp_example(args.contact_model, args.contact_surface_representation,
        args.time_step,min_time=False, animate=True)

