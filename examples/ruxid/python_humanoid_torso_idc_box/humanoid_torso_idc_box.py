"""
This is an example for simulating a simplified humanoid (aka. noodleman) through pydrake.
It reads three simple SDFormat files of a hydroelastic humanoid,
a rigid chair, and rigid floor.
It uses an inverse dynamics controller to bring the noodleman from a sitting to standing up position.
"""
import argparse
import numpy as np
import pdb
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import DrakeVisualizer
from pydrake.math import RigidTransform
from pydrake.math import RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, AddMultibodyPlantSceneGraph
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
from pydrake.multibody.tree import WeldJoint, RevoluteJoint, PrismaticJoint
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
import matplotlib.pyplot as plt
import pydrake.geometry as mut
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ConstantVectorSource,
    ContactVisualizer,
    ContactVisualizerParams,
    DiagramBuilder,
    EventStatus,
    FixedOffsetFrame,
    InverseDynamicsController,
    InverseDynamics,
    PidController,
    LeafSystem,
    MeshcatVisualizerCpp,
    MeshcatVisualizerParams,
    MultibodyPlant,
    MultibodyPositionToGeometryPose,
    Multiplexer,
    Parser,
    PassThrough,
    PlanarJoint,
    ContactModel,
    PrismaticJoint,
    RandomGenerator,
    Rgba,
    RigidTransform,
    RotationMatrix,
    SceneGraph,
    Simulator,
    SpatialInertia,
    Sphere,
    UnitInertia,
    Variable,
    JointIndex,
    RandomGenerator,
    PositionConstraint,
    MultibodyForces,
    Box,
    Meshcat,
)

box_size=[0.2+0.1*(np.random.random()-0.5),
            0.2+0.1*(np.random.random()-0.5),
            0.2+0.1*(np.random.random()-0.5),
            ]

from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewVelocities,
        MakeNamedViewState,
        MakeNamedViewActuation,
        AddShape,
        SetColor
        )

def AddAgent(plant):
    parser = Parser(plant)
    agent = parser.AddModelFromFile(FindResource("models/humanoid_torso_v2_noball_noZeroBodies_spring_prismatic.sdf"))
    #noodleman = parser.AddModelFromFile(FindResource("models/humanoid_v2_noball.sdf"))
    p_WAgent_fixed = RigidTransform(RollPitchYaw(0, 0, 0),
                                     np.array([0, 0, 0])) #0.25
    # weld the lower leg of the noodleman to the world frame. 
    # The inverse dynamic controller does not work with floating base
    weld=WeldJoint(
          name="weld_base",
          frame_on_parent_P=plant.world_frame(),
          frame_on_child_C=plant.GetFrameByName("base", agent), # "waist"
          X_PC=p_WAgent_fixed
        )
    plant.AddJoint(weld)
    return agent

def AddFloor(plant):
    parser = Parser(plant)
    floor = parser.AddModelFromFile(FindResource("models/floor.sdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("floor", floor),
        RigidTransform(RollPitchYaw(0, 0, 0),
                        np.array([0, 0, 0.0]))
                    )
    return floor

def AddBox(plant, welded=False):
    w= box_size[0]
    d= box_size[1]
    h= box_size[2]
    mass=1 + 1*(np.random.random()-0.5)
    mu=0.5 + 0.5*(np.random.random()-0.5)
    box=AddShape(plant, Box(w,d,h), name="box",mass=mass,mu=mu)
    # parser = Parser(plant)
    # box = parser.AddModelFromFile(FindResource("models/box.sdf"))
    if welded:
        plant.WeldFrames(
            plant.world_frame(), plant.GetFrameByName("box", box),
            RigidTransform(RollPitchYaw(0, 0, 0),
                            np.array([0, 0, 0.0]))
                        )

    return box

def add_collision_filters(scene_graph, plant):
    filter_manager=scene_graph.collision_filter_manager()
    body_pairs=[
        ["head","torso"],
        ["torso","waist"],
        ["torso","arm_L"],
        ["arm_L","forearm_L"],
        ["forearm_L","hand_L"],
        ["torso","arm_R"],
        ["arm_R","forearm_R"],
        ["forearm_R","hand_R"]
    ]

    for pair in body_pairs:
        parent=plant.GetBodyByName(pair[0])
        child=plant.GetBodyByName(pair[1])
        
        set=mut.GeometrySet(
            plant.GetCollisionGeometriesForBody(parent)+
            plant.GetCollisionGeometriesForBody(child))
        filter_manager.Apply(
            declaration=mut.CollisionFilterDeclaration().ExcludeWithin(
                set))

def make_environment(contact_model, contact_surface_representation,
                     time_step, meshcat=None):

    multibody_plant_config = \
        MultibodyPlantConfig(
            time_step=time_step,
            contact_model=contact_model,
            contact_surface_representation=contact_surface_representation)

    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    #plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    contact_model=ContactModel.kPoint
    plant.set_contact_model(contact_model) 
    agent = AddAgent(plant)
    AddFloor(plant)
    box = AddBox(plant,welded=False)
    plant.Finalize()
    plant.set_name("plant")
    # filter collisison between parent and child of each joint.
    add_collision_filters(scene_graph,plant)
    
    use_idc=True

    if use_idc:
        #controller plant
        controller_plant = MultibodyPlant(time_step=time_step)
        controller_plant.set_contact_model(contact_model)     
        AddAgent(controller_plant)
    else:
        controller_plant=plant

    if meshcat:
        MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
        ContactVisualizer.AddToBuilder(
            builder, plant, meshcat,
            ContactVisualizerParams(radius=0.005, newtons_per_meter=40.0))

        controller_scene_graph = builder.AddSystem(SceneGraph())
        controller_plant.RegisterAsSourceForSceneGraph(controller_scene_graph)
        SetColor(controller_scene_graph,
                    color=[1.0, 165.0 / 255, 0.0, 1.0],
                    source_id=controller_plant.get_source_id())
        controller_vis = DrakeVisualizer.AddToBuilder(builder=builder, 
        scene_graph=controller_scene_graph)
        controller_vis.set_name("controller vis")
    
    if use_idc:
        controller_plant.Finalize()
        controller_plant.set_name("controller_plant")
        add_collision_filters(scene_graph,controller_plant)

    Ns = controller_plant.num_multibody_states()
    Nv = controller_plant.num_velocities()
    Na = controller_plant.num_actuators()
    Nj = controller_plant.num_joints()
    Np = controller_plant.num_positions()

    print("\nnumber of position: ",Np,
        ", number of velocities: ",Nv,
        ", number of actuators: ",Na,
        ", number of multibody states: ",Ns,'\n')
    
    print('Joints:')
    for joint_idx in controller_plant.GetJointIndices(agent):
            print(controller_plant.get_joint(joint_idx).name())

    #Desired state corresponding to a standing up position [tetha0,tetha1,tetha1_dot,tetha2_dot].
    StateView=MakeNamedViewState(controller_plant, "States")
    PositionView=MakeNamedViewPositions(controller_plant, "Position")
    ActuationView=MakeNamedViewActuation(controller_plant, "Actuation")

    desired_q=PositionView([0]*Np)
    desired_q.shoulderR_joint1=np.pi/2
    desired_q.shoulderL_joint1=np.pi/2

    print("\nState view: ", StateView(np.ones(Ns)))
    print("\nActuation view: ", ActuationView(np.ones(Na)))
    print("Desired joint position:",desired_q)
    
    desired_state_source=builder.AddSystem(ConstantVectorSource(desired_q))

    
    if use_idc:
            ##Create inverse dynamics controller
        kp = [10] * Na
        ki = [0] * Na
        kd = [5] * Na
        IDC = builder.AddSystem(InverseDynamicsController(robot=controller_plant,
                                            kp=kp,
                                            ki=ki,
                                            kd=kd,
                                            has_reference_acceleration=False))                                  

        builder.Connect(plant.get_state_output_port(agent),
                        IDC.get_input_port_estimated_state())

        positions_to_state = builder.AddSystem(Multiplexer([Na, Na]))
        zeros_v = builder.AddSystem(ConstantVectorSource([0] * Na))
        builder.Connect( desired_state_source.get_output_port(),
                        positions_to_state.get_input_port(0))
        builder.Connect(zeros_v.get_output_port(),
                        positions_to_state.get_input_port(1))
    
    if use_idc:
        builder.Connect(positions_to_state.get_output_port(),
                        IDC.get_input_port_desired_state())
    
    class gate_controller_system(LeafSystem):

        def __init__(self):
            LeafSystem.__init__(self)
            Na = controller_plant.num_actuators()
            Ns = controller_plant.num_multibody_states()
            self.DeclareVectorInputPort("control_input", Na)
            #self.DeclareVectorInputPort("desired_state",Ns)
            self.DeclareVectorOutputPort("gated_control_output", Na, self.CalcControl)
            self.plant=controller_plant
            self.plant_context = controller_plant.CreateDefaultContext()
            self.actuation_matrix=controller_plant.MakeActuationMatrix()
            self.StateView=MakeNamedViewState(controller_plant, "States")
            self.ActuationView=MakeNamedViewActuation(controller_plant, "Actuator")

        def CalcControl(self, context,output):
            control_input = self.get_input_port(0).Eval(context)
            #state = self.get_input_port(1).Eval(context)
            #control_output=np.zeros(control_input.shape)
            #pdb.set_trace()
            control_output=control_input.dot(self.actuation_matrix)       
            print("control_output: ",control_output)  
            print("control_input: ",control_input)       
            #pdb.set_trace()
            output.set_value(control_output)

    gate_controller=builder.AddSystem(gate_controller_system())

    if use_idc:
        builder.Connect(IDC.get_output_port(),
                        gate_controller.get_input_port(0))
        # builder.Connect(IDC.get_output_port(),
        #                 plant.get_actuation_input_port(agent))       
    else:
        zero_actuation = builder.AddSystem(ConstantVectorSource([0] * Na))
        builder.Connect(zero_actuation.get_output_port(),
                        gate_controller.get_input_port(0))

    builder.Connect(gate_controller.get_output_port(),
                    plant.get_actuation_input_port(agent))  
                 
    if meshcat:
        positions_to_poses = builder.AddSystem(
            MultibodyPositionToGeometryPose(controller_plant))
        builder.Connect(
            positions_to_poses.get_output_port(),
            controller_scene_graph.get_source_pose_port(
                controller_plant.get_source_id()))
    else:
        DrakeVisualizer.AddToBuilder(builder=builder, scene_graph=scene_graph)
        ConnectContactResultsToDrakeVisualizer(builder=builder, plant=plant,
                                            scene_graph=scene_graph)

    nx = plant.num_positions() + plant.num_velocities()
    state_logger = builder.AddSystem(VectorLogSink(nx))
    builder.Connect(plant.get_state_output_port(),
                    state_logger.get_input_port())




    diagram = builder.Build()

    #visualize plant and diagram
    plt.figure()
    plot_graphviz(plant.GetTopologyGraphvizString())
    plt.figure()
    plot_system_graphviz(diagram, max_depth=2)
    plt.plot(1)
    plt.show(block=False)
    #pdb.set_trace()

    return diagram, plant, controller_plant, state_logger, agent

def set_home(plant,plant_context):

    home_positions=[
        ('shoulderR_joint1',0.3*(np.random.random()-0.5)+np.pi/4),
        ('shoulderL_joint1',0.3*(np.random.random()-0.5)),
        ('shoulderR_joint2',0.3*(np.random.random()-0.5)+np.pi/4),
        ('shoulderR_joint2',0.3*(np.random.random()-0.5)),
        ('elbowR_joint',0.3*(np.random.random()-0.5)+np.pi/4),
        ('elbowL_joint',0.3*(np.random.random()-0.5)+np.pi/4),
        ('torso_joint1',0.3*(np.random.random()-0.5)),
        ('torso_joint2',0.2*(np.random.random()-0.5)),
        ('torso_joint3',0.2*(np.random.random()-0.5)),
        ('torso_joint2',0.6*(np.random.random()-0.5)),
        ('prismatic_z',0.2*(np.random.random()-0.5)+0.35),
    ]

    for pair in home_positions:
        joint = plant.GetJointByName(pair[0])
        if joint.type_name()=="prismatic":
            joint.set_translation(plant_context,pair[1])
        elif joint.type_name()=="revolute":
            joint.set_angle(plant_context,pair[1])

    box=plant.GetBodyByName("box")
    #pdb.set_trace()
    box_pose = RigidTransform(
                    RollPitchYaw(0, 0, 0),
                    np.array(
                        [
                            0+0.3*(np.random.random()-0.5), 
                            0.4+0.15*(np.random.random()-0.5), 
                            box_size[2]/2+0.005,
                        ])
                    )
    plant.SetFreeBodyPose(plant_context,box,box_pose)

    # Np = plant.num_positions()
    # PositionView=MakeNamedViewPositions(plant, "Positions")

    # default_position=PositionView([0]*Np)
    # default_position.shoulderR_joint1=np.pi/4
    # default_position.shoulderL_joint1=np.pi/4
    # default_position.elbowR_joint=np.pi/4
    # default_position.elbowL_joint=np.pi/4
    # default_position.prismatic_z=0.3
    # #pdb.set_trace()
    # # default_position.box_x=0.0
    # # default_position.box_y=0.1
    # # default_position.box_z=0.15

    # #add randomness offset to positions
    # random_offset=PositionView([0]*Np)
    # random_offset.shoulderR_joint1=0.3*(np.random.random()-0.5)
    # random_offset.shoulderL_joint1=0.3*(np.random.random()-0.5)
    # random_offset.shoulderR_joint2=0.3*(np.random.random()-0.5)
    # random_offset.shoulderL_joint2=0.3*(np.random.random()-0.5)  
    # random_offset.elbowR_joint=0.3*(np.random.random()-0.5)
    # random_offset.elbowL_joint=0.3*(np.random.random()-0.5)
    # random_offset.torso_joint1=0.2*(np.random.random()-0.5)
    # random_offset.torso_joint2=0.2*(np.random.random()-0.5)
    # random_offset.torso_joint3=0.6*(np.random.random()-0.5)
    # random_offset.prismatic_z=0.2*(np.random.random()-0.5)


    # # plant.SetPositions(plant_context,
    # #     default_position.__array__()+random_offset.__array__())


def simulate_diagram(diagram, plant, controller_plant, state_logger,
                       simulation_time, target_realtime_rate):

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant,
                                                diagram_context)    
    
    set_home(plant, plant_context)

    #setup the simulator
    simulator_config = SimulatorConfig(
                           target_realtime_rate=target_realtime_rate,
                           publish_every_time_step=True)
    
    simulator = Simulator(diagram,diagram_context)
    ApplySimulatorConfig(simulator, simulator_config)


    print("Initial state variables: ", plant.GetPositionsAndVelocities(plant_context))
   
    state_log = state_logger.FindMutableLog(simulator.get_mutable_context())
    state_log.Clear()
    simulator.Initialize()
    #pdb.set_trace()
    adv_step=0.1
    time=0
    for i in range(int(simulation_time/adv_step)):
        time+=adv_step
        simulator.AdvanceTo(time)
        PrintSimulatorStatistics(simulator)    
        input("Press Enter to continue...")
    return state_log.sample_times(), state_log.data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulation_time", type=float, default=4,
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
        "--target_realtime_rate", type=float, default=1.0,
        help="Target realtime rate. Default 1.0.")
    parser.add_argument(
        "--meshcat", action="store_true",
        help="If set, visualize in meshcat. Use DrakeVisualizer otherwise")        
    args = parser.parse_args()

    if args.meshcat:
        meshcat_server= Meshcat()
        visualizer=meshcat_server
    else:
        visualizer=None

    input("Press Enter to continue...")

    diagram, plant, controller_plant,state_logger,agent_idx = make_environment(
        args.contact_model, args.contact_surface_representation,
        args.time_step,meshcat=visualizer)
    
    time_samples, state_samples = simulate_diagram(
        diagram, plant, controller_plant, state_logger,
        args.simulation_time, 
        args.target_realtime_rate)
    
    print("\nFinal state variables:")
    print(state_samples[:, -1])
