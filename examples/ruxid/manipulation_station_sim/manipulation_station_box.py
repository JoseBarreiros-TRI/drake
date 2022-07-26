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
    MeshcatVisualizer,
    FindResourceOrThrow,
)
from pydrake.examples import (
    CreateClutterClearingYcbObjectList, ManipulationStation,
    ManipulationStationHardwareInterface)

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
    1,+0.5*(np.random.random()-0.5),
    0.+0.8*(np.random.random()-0.5),
    ] 

##

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

def make_environment(meshcat=None, debug=False, hardware=False):

    builder = DiagramBuilder()

    target_position=[desired_box_xy[0],desired_box_xy[1],table_heigth]
    
    if hardware:
        camera_ids = []
        station = builder.AddSystem(ManipulationStationHardwareInterface(
            camera_ids,False))
        station.Connect(wait_for_cameras=False,wait_for_wsg=False)     
        controller_plant=station.get_controller_plant()
        plant=None
        #plant=station.get_multibody_plant()
    else:
        station = builder.AddSystem(ManipulationStation(time_step=sim_time_step,contact_model=contact_model,contact_solver=contact_solver))
        station.SetupCitoRlStation()

        # station.AddManipulandFromFile(
        #         "drake/examples/rl_cito_station/models/"
        #         + "061_foam_brick.sdf",
        #         RigidTransform(RotationMatrix.Identity(), [0.8, 0, 0.2]))
                    

        controller_plant=station.get_controller_plant()
        plant=station.get_multibody_plant()

        box = AddBox(plant)
        AddTargetPosVisuals(plant,target_position)
        station.Finalize()

        if meshcat:
            geometry_query_port = station.GetOutputPort("geometry_query")
            meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
                builder=builder,
                query_object_port=geometry_query_port,
                meshcat=meshcat)
            
        # filter collisison between parent and child of each joint.
        #add_collision_filters(scene_graph,plant)


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
        plot_graphviz(controller_plant.GetTopologyGraphvizString())
        plt.plot(1)
        plt.show(block=False)
     
        print("\nState view: ", StateView(np.ones(Ns)))
        print("\nActuation view: ", ActuationView(np.ones(Na)))
        print("\nPosition view: ",PositionView(np.ones(Np)))   
        #pdb.set_trace()


    iiwa_feedforward_torque = builder.AddSystem(PassThrough(Na))
    iiwa_position = builder.AddSystem(PassThrough(Na))
    
    builder.Connect(iiwa_feedforward_torque.get_output_port(),
                    station.GetInputPort("iiwa_feedforward_torque"))
    builder.Connect(iiwa_position.get_output_port(),
                    station.GetInputPort("iiwa_position"))    

    zeros_v = builder.AddSystem(ConstantVectorSource([0]*Na))
    builder.Connect(zeros_v.get_output_port(),
                iiwa_position.get_input_port())

    diagram = builder.Build()

    if debug:
        #visualize plant and diagram
        if not hardware:
            plt.figure()
            plot_graphviz(plant.GetTopologyGraphvizString())
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.plot(1)
        plt.show(block=False)
        #pdb.set_trace()


    return diagram, plant, controller_plant

def set_home(simulator,diagram_context,plant,plant_name="plant"):
    #pdb.set_trace()
    diagram = simulator.get_system()
    #plant=diagram.GetSubsystemByName(plant_name)
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
                            0.75+0.1*(np.random.random()-0.5), 
                            0+0.25*(np.random.random()-0.5), 
                            box_size[2]/2+0.005+table_heigth,
                        ])
                    )
    plant.SetFreeBodyPose(plant_context,box,box_pose)


def simulate_diagram(diagram, plant, controller_plant, 
                       simulation_time, target_realtime_rate, hardware=False):
    #pdb.set_trace()
    diagram_context = diagram.CreateDefaultContext()

    #setup the simulator
    simulator_config = SimulatorConfig(
                           target_realtime_rate=target_realtime_rate,
                           publish_every_time_step=False)
    
    simulator = Simulator(diagram,diagram_context)
    ApplySimulatorConfig(simulator, simulator_config)

    if not hardware:
        plant_context = diagram.GetMutableSubsystemContext(plant,
                                                    diagram_context)    
        print("Initial state variables: ", plant.GetPositionsAndVelocities(plant_context))
   
    simulator.Initialize()

    context = simulator.get_mutable_context()
    context.SetTime(0)

    if hardware:
        simulator.AdvanceTo(1e-6)
    else:
        set_home(simulator, context,plant)

    #pdb.set_trace()
    adv_step=0.3
    time=0
    for i in range(int(simulation_time/adv_step)):
        input("Press Enter to continue...")
        time+=adv_step
        simulator.AdvanceTo(time)
        PrintSimulatorStatistics(simulator)    
        
    return 1

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
    parser.add_argument(
        "--hardware", action="store_true",
        help="If set, visualize in meshcat. Use DrakeVisualizer otherwise")        
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.meshcat:
        meshcat_server= Meshcat()
        visualizer=meshcat_server
    else:
        visualizer=None

    input("Press Enter to continue...")

    diagram, plant, controller_plant = make_environment(
        meshcat=visualizer, debug=args.debug, hardware=args.hardware)
    
    simulate_diagram(
        diagram, plant, controller_plant, 
        args.simulation_time, 
        args.target_realtime_rate,hardware=args.hardware)
    

