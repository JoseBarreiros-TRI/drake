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
    FindResourceOrThrow,
)
from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewVelocities,
        MakeNamedViewState,
        MakeNamedViewActuation,
        AddShape,
        SetColor
        )

## Env parameters
sim_time_step=0.025
# gym_time_step=0.1
controller_time_step=0.01
# gym_time_limit=5
modes=["IDC","torque"]
control_mode=modes[0]
table_heigth =0.15
box_size=[ 0.2,#0.2+0.1*(np.random.random()-0.5),
        0.2,#0.2+0.1*(np.random.random()-0.5),
         0.2,   #0.2+0.1*(np.random.random()-0.5),
        ]
box_mass=1
box_mu=1.0
contact_model='point'#'hydroelastic_with_fallback'#ContactModel.kHydroelasticWithFallback#kPoint
contact_solver='sap'#ContactSolver.kSap#kTamsi # kTamsi
desired_box_xy=[
    0.+0.8*(np.random.random()-0.5),
    1,+0.5*(np.random.random()-0.5)
    ] 

##

def AddAgent(plant):
    parser = Parser(plant)
    model_file = FindResourceOrThrow("drake/manipulation/models/iiwa_description/iiwa7/iiwa7_with_box_collision.sdf")
    agent = parser.AddModelFromFile(model_file)
    #noodleman = parser.AddModelFromFile(FindResource("models/humanoid_v2_noball.sdf"))
    p_WAgent_fixed = RigidTransform(RollPitchYaw(0, 0, np.pi/2),
                                     np.array([0, 0, 0])) #0.25
    # weld the lower leg of the noodleman to the world frame. 
    # The inverse dynamic controller does not work with floating base
    weld=WeldJoint(
          name="weld_base",
          frame_on_parent_P=plant.world_frame(),
          frame_on_child_C=plant.GetFrameByName("iiwa_link_0", agent), # "waist"
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

def AddTable(plant):
    parser = Parser(plant)
    table = parser.AddModelFromFile(FindResource("models/table.sdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("table", table),
        RigidTransform(RollPitchYaw(0, 0, 0),
                        np.array([0, 1.2, table_heigth]))
                    )
    return table

def AddBox(plant):
    w= box_size[0]
    d= box_size[1]
    h= box_size[2]
    mass= box_mass
    mu= box_mu
    #if contact_model==ContactModel.kHydroelastic or contact_model==ContactModel.kHydroelasticWithFallback:
    if contact_model=='hydroelastic_with_fallback' or contact_model=='hydroelastic':
        parser = Parser(plant)
        box = parser.AddModelFromFile(FindResource("models/box.sdf"))
    else:
        box=AddShape(plant, Box(w,d,h), 
        name="box",mass=mass,mu=mu,
        color=[.4, .4, 1., 0.8])

    return box

def AddTargetPosVisuals(plant,xyz_position,color=[.8, .1, .1, 1.0]):
    parser = Parser(plant)
    marker = parser.AddModelFromFile(FindResource("models/cross.sdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("cross", marker),
        RigidTransform(RollPitchYaw(0, 0, 0),
                        np.array(xyz_position)
                    )
    )

def add_collision_filters(scene_graph, plant):
    filter_manager=scene_graph.collision_filter_manager()
    body_pairs=[
        ["iiwa_link_1","iiwa_link_2"],
        ["iiwa_link_2","iiwa_link_3"],
        ["iiwa_link_3","iiwa_link_4"],
        ["iiwa_link_4","iiwa_link_5"],
        ["iiwa_link_5","iiwa_link_6"],
        ["iiwa_link_6","iiwa_link_7"],
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

def make_environment(meshcat=None, debug=False):

    builder = DiagramBuilder()

    multibody_plant_config = \
        MultibodyPlantConfig(
            time_step=sim_time_step,
            contact_model=contact_model,
            )

    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    #plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)

    agent = AddAgent(plant)
    #AddFloor(plant)
    AddTable(plant)
    box = AddBox(plant)
    target_position=[desired_box_xy[0],desired_box_xy[1],table_heigth]
    AddTargetPosVisuals(plant,target_position)
    plant.Finalize()
    plant.set_name("plant")
    # filter collisison between parent and child of each joint.
    add_collision_filters(scene_graph,plant)

    controller_plant = MultibodyPlant(time_step=controller_time_step)
    AddAgent(controller_plant)

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
        controller_vis = MeshcatVisualizerCpp.AddToBuilder(
            builder, controller_scene_graph, meshcat,
            MeshcatVisualizerParams(prefix="controller"))
        controller_vis.set_name("controller meshcat")
    
    # finalize the plant
    controller_plant.Finalize()
    controller_plant.set_name("controller_plant")
    add_collision_filters(scene_graph,controller_plant)

    Ns = controller_plant.num_multibody_states()
    Nv = controller_plant.num_velocities()
    Na = controller_plant.num_actuators()
    Nj = controller_plant.num_joints()
    Np = controller_plant.num_positions()

    #Make NamedViews
    StateView=MakeNamedViewState(controller_plant, "States")
    PositionView=MakeNamedViewPositions(controller_plant, "Position")
    ActuationView=MakeNamedViewActuation(controller_plant, "Actuation")

    if debug:
        print("\nnumber of position: ",Np,
            ", number of velocities: ",Nv,
            ", number of actuators: ",Na,
            ", number of joints: ",Nj,
            ", number of multibody states: ",Ns,'\n')
        plt.figure()
        plot_graphviz(plant.GetTopologyGraphvizString())
        plt.plot(1)
        plt.show(block=False)
     
        print("\nState view: ", StateView(np.ones(Ns)))
        print("\nActuation view: ", ActuationView(np.ones(Na)))
        print("\nPosition view: ",PositionView(np.ones(Np)))   

    if control_mode=="IDC":
        #Create inverse dynamics controller
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

        #actions are positions sent to IDC
        actions = builder.AddSystem(PassThrough(Na))
        positions_to_state = builder.AddSystem(Multiplexer([Na, Na]))
        builder.Connect(actions.get_output_port(),
                    positions_to_state.get_input_port(0))
        zeros_v = builder.AddSystem(ConstantVectorSource([0] * Na))
        builder.Connect(zeros_v.get_output_port(),
                        positions_to_state.get_input_port(1))
        builder.Connect(positions_to_state.get_output_port(),
                        IDC.get_input_port_desired_state())

    class gate_controller_system(LeafSystem):

        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("control_signal_input", Na)
            self.DeclareVectorOutputPort("gated_control_output", Na, self.CalcControl)
            self.actuation_matrix=controller_plant.MakeActuationMatrix()

        def CalcControl(self, context,output):
            control_signal_input = self.get_input_port(0).Eval(context)
            gated_control_output=control_signal_input.dot(self.actuation_matrix)       
            #print("control_output: ",gated_control_output)  
            #print("control_input: ",control_signal_input)       
            output.set_value(gated_control_output)
    
    if control_mode=="IDC":
        gate_controller=builder.AddSystem(gate_controller_system())
        builder.Connect(IDC.get_output_port(),
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

    nx = plant.num_positions() + plant.num_velocities()
    state_logger = builder.AddSystem(VectorLogSink(nx))
    builder.Connect(plant.get_state_output_port(),
                    state_logger.get_input_port())


    nx = plant.num_positions() + plant.num_velocities()
    state_logger = builder.AddSystem(VectorLogSink(nx))
    builder.Connect(plant.get_state_output_port(),
                    state_logger.get_input_port())


    diagram = builder.Build()

    if debug:
        #visualize plant and diagram
        plt.figure()
        plot_graphviz(controller_plant.GetTopologyGraphvizString())
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.plot(1)
        plt.show(block=False)
        #pdb.set_trace()


    return diagram, plant, controller_plant, state_logger, agent

def set_home(simulator,diagram_context,plant_name="plant"):
    
    diagram = simulator.get_system()
    plant=diagram.GetSubsystemByName(plant_name)
    plant_context = diagram.GetMutableSubsystemContext(plant,
                                                diagram_context)  

    home_positions=[
        ('iiwa_joint_1',0.1*(np.random.random()-0.5)+0.3),
        ('iiwa_joint_2',0.1*(np.random.random()-0.5)+0.3),
        ('iiwa_joint_3',0.1*(np.random.random()-0.5)+0.3),
        ('iiwa_joint_4',0.1*(np.random.random()-0.5)+0.3),
        ('iiwa_joint_5',0.1*(np.random.random()-0.5)+0.3),
        ('iiwa_joint_6',0.1*(np.random.random()-0.5)+0.3),
        ('iiwa_joint_7',0.1*(np.random.random()-0.5)+0.3),

    ]

    #ensure the positions are within the joint limits
    for pair in home_positions:
        joint = plant.GetJointByName(pair[0])
        if joint.type_name()=="revolute":
            joint.set_angle(plant_context,
                        np.clip(pair[1],
                            joint.position_lower_limit(),
                            joint.position_upper_limit()
                            )
                        )
    box=plant.GetBodyByName("box")
    
    box_pose = RigidTransform(
                    RollPitchYaw(0, 0.1, 0),
                    np.array(
                        [
                            0+0.25*(np.random.random()-0.5), 
                            0.75+0.1*(np.random.random()-0.5), 
                            box_size[2]/2+0.005+table_heigth,
                        ])
                    )
    plant.SetFreeBodyPose(plant_context,box,box_pose)


def simulate_diagram(diagram, plant, controller_plant, state_logger,
                       simulation_time, target_realtime_rate):

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant,
                                                diagram_context)    

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


    context = simulator.get_mutable_context()
    context.SetTime(0)
    set_home(simulator, context)

    #pdb.set_trace()
    adv_step=0.3
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
        "--simulation_time", type=float, default=100,
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
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.meshcat:
        meshcat_server= Meshcat()
        visualizer=meshcat_server
    else:
        visualizer=None

    input("Press Enter to continue...")

    diagram, plant, controller_plant,state_logger,agent_idx = make_environment(
        meshcat=visualizer, debug=args.debug)
    
    time_samples, state_samples = simulate_diagram(
        diagram, plant, controller_plant, state_logger,
        args.simulation_time, 
        args.target_realtime_rate)
    
    print("\nFinal state variables:")
    print(state_samples[:, -1])
