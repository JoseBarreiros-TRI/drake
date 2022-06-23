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
from pydrake.multibody.tree import WeldJoint
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
)
from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewVelocities,
        MakeNamedViewState,
        MakeNamedViewActuation)

def AddNoodleman(plant):
    parser = Parser(plant)
    noodleman = parser.AddModelFromFile(FindResource("models/humanoid_v2_noball_noZeroBodies.sdf"))
    #noodleman = parser.AddModelFromFile(FindResource("models/humanoid_v2_noball.sdf"))
    return noodleman

def AddFloor(plant):
    parser = Parser(plant)
    floor = parser.AddModelFromFile(FindResource("models/floor.sdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("floor", floor),
        RigidTransform(RollPitchYaw(0, 0, 0),
                        np.array([0, 0, 0.0]))
                    )
    return floor
    
def make_agent_chair(contact_model, contact_surface_representation,
                     time_step):

    multibody_plant_config = \
        MultibodyPlantConfig(
            time_step=time_step,
            contact_model=contact_model,
            contact_solver="sap", # "tamsi" "sap"
            contact_surface_representation=contact_surface_representation)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
    #plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    contact_model=ContactModel.kPoint
    plant.set_contact_model(contact_model) 

    agent = AddNoodleman(plant)
    AddFloor(plant)
    
    
    plant.set_name("plant")

    p_WAgent_fixed = RigidTransform(RollPitchYaw(0, 0, 0),
                                     np.array([0, 0, 1.05]))
    # weld the lower leg of the noodleman to the world frame. 
    # The inverse dynamic controller does not work with floating base
    weld=WeldJoint(
          name="weld_waist",
          frame_on_parent_P=plant.world_frame(),
          frame_on_child_C=plant.GetFrameByName("waist", agent),
          X_PC=p_WAgent_fixed
        )
    #plant.AddJoint(weld)
    plant.Finalize()


    # #filter self collision of all the bodies of the agent
    # filter_manager=scene_graph.collision_filter_manager()
    # geomIdx=[]
    # for body_idx in plant.GetBodyIndices(agent):
    #     body=plant.get_body(body_idx)
    #     print(body.name())
    #     geomIdx+=plant.GetCollisionGeometriesForBody(body)
    
    # set=mut.GeometrySet(geomIdx)
    
    # filter_manager.Apply(
    # declaration=mut.CollisionFilterDeclaration().ExcludeWithin(
    #             set))

    # filter collisison between parent and child of each joint.
    filter_manager=scene_graph.collision_filter_manager()
    body_pairs=[
        ["head","torso"],
        ["torso","waist"],
        ["waist","hips"],
        ["torso","arm_L"],
        ["arm_L","forearm_L"],
        ["forearm_L","hand_L"],
        ["torso","arm_R"],
        ["arm_R","forearm_R"],
        ["forearm_R","hand_R"],
        ["hips","upper_leg_R"],
        ["upper_leg_R","lower_leg_R"],
        ["lower_leg_R","foot_R"],
        ["foot_R","toes_R"],
        ["hips","upper_leg_L"],
        ["upper_leg_L","lower_leg_L"],
        ["lower_leg_L","foot_L"],
        ["foot_L","toes_L"],
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


           

    #pdb.set_trace()     

                



    print("\nnumber of position: ",plant.num_positions(),
        ", number of velocities: ",plant.num_velocities(),
        ", number of actuators: ",plant.num_actuators(),
        ", number of multibody states: ",plant.num_multibody_states(),'\n')
    
    print('joints:')
    for joint_idx in plant.GetJointIndices(agent):
            print(plant.get_joint(joint_idx).name())
            #pdb.set_trace()

    #Desired state corresponding to a standing up position [tetha0,tetha1,tetha1_dot,tetha2_dot].
    #desired_q=np.zeros(25)
    desired_q=0.0*np.ones(25)

    desired_q[15]=-0.
    #desired_q[19]=0.2
    #desired_q[4]=0.2

    desired_state=desired_q #np.concatenate((desired_q,desired_v))
    print("desired joint state:",desired_state)
    desired_state_source=builder.AddSystem(ConstantVectorSource(desired_state))

    ##Create inverse dynamics controller
    Ns = plant.num_multibody_states()
    Nv = plant.num_velocities()
    Na = plant.num_actuators()
    Nj = plant.num_joints()
    Np = plant.num_positions()
    kp = [100] * Na
    ki = [0] * Na
    kd = [10] * Na

    kp[0]=50
    kd[0]=50  
    kp[1]=50
    kd[1]=50  
    kp[2]=50
    kd[2]=50  
    kp[3]=100
    kd[3]=10  
    kp[4]=50
    kd[4]=5  
    kp[5]=20
    kd[5]=2  
    kp[6]=50
    kd[6]=50    
    kp[7]=50
    kd[7]=50
    kp[8]=50
    kd[8]=50  
    kp[9]=100
    kd[9]=10  
    kp[10]=50
    kd[10]=5  
    kp[11]=20
    kd[11]=2  
 
    kp[12]=100
    kd[12]=10   
    kp[13]=50
    kd[13]=5    
    kp[14]=500
    kd[14]=10

    kp[15]=60 #neck 1
    kd[15]=10
    kp[16]=70 #neck 2
    kd[16]=10
    kp[17]=200 # shoulder L 1
    kd[17]=100
    kp[18]=150 #shoulder L 2
    kd[18]=10
    kp[19]=10
    kd[19]=1
    kp[20]=10
    kd[20]=1
    kp[21]=200 #shoulder R 1
    kd[21]=100
    kp[22]=15 #shoulder R 2
    kd[22]=10
    kp[23]=10
    kd[23]=1
    kp[24]=10
    kd[24]=1
    # Select the joint states (and ignore the floating-base states)
    #pdb.set_trace()
    S=np.zeros((Na*2,Ns))
    j=0
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        print(joint.name())
        print('p_i ',joint.position_start(),', v_i ',joint.velocity_start())

        #pdb.set_trace()
        if joint.num_positions() == 1:            
            S[j, joint.position_start()] = 1
            S[Na+j, joint.velocity_start()] = 1
            j = j+1
        
    #pdb.set_trace()   
    #print(S.shape)
    PID = builder.AddSystem(
        PidController(
            kp=kp, 
            ki=ki, 
            kd=kd,
            state_projection=S,
            output_projection=plant.MakeActuationMatrix()[Nv-Na:,:].T
            )
        )                                       
    
    builder.Connect(plant.get_state_output_port(),
                    PID.get_input_port_estimated_state())


    positions_to_state = builder.AddSystem(Multiplexer([Na, Na]))
    builder.Connect( desired_state_source.get_output_port(),
                    positions_to_state.get_input_port(0))

    zeros_v = builder.AddSystem(ConstantVectorSource([0] * Na))
    builder.Connect(zeros_v.get_output_port(),
                    positions_to_state.get_input_port(1))
    builder.Connect(positions_to_state.get_output_port(),
                    PID.get_input_port_desired_state())

    # option one, using actuation port
    #enable controller
    #builder.Connect(PID.get_output_port(),
    #                 plant.get_actuation_input_port(agent))

    #disable controller
    #constant_zero_torque=builder.AddSystem(ConstantVectorSource(np.zeros(Na)))
    #builder.Connect(constant_zero_torque.get_output_port(),
    #                plant.get_actuation_input_port(agent))   
    # <option one>
    
    #enable controller per joint

    class gate_controller_system(LeafSystem):

        def __init__(self):
            LeafSystem.__init__(self)
            Na = plant.num_actuators()
            self.DeclareVectorInputPort("control_input", Na)
            self.DeclareVectorOutputPort("gated_control_output", Na, self.CalcControl)
            #self.StateView=MakeNamedViewState(plant, "States")


        def CalcControl(self, context,output):
            control_input = self.get_input_port(0).Eval(context)
            control_output=np.zeros(control_input.shape)
            idx=4
            #control_output[idx]=control_input[idx]
            
            #control_output[0:2]=control_input[0:2]
            control_output=control_input

            print("control_output: ",control_output)
            #print("control_input: ",control_input)
            #pdb.set_trace()
            
            output.set_value(control_output)

            


    gate_controller=builder.AddSystem(gate_controller_system())
    builder.Connect(PID.get_output_port(),
                     gate_controller.get_input_port(0))
    builder.Connect(gate_controller.get_output_port(),
                    plant.get_actuation_input_port(agent))       




    # ## option two, using applied_generalized_port()
    # constant_zero_torque=builder.AddSystem(ConstantVectorSource(np.zeros(Na)))
    # builder.Connect(constant_zero_torque.get_output_port(),plant.get_actuation_input_port())

    # helper = builder.AddSystem(Multiplexer([6, Na]))
    # constant_six=builder.AddSystem(ConstantVectorSource(np.zeros(6)))    
    # builder.Connect(constant_six.get_output_port(),
    #                 helper.get_input_port(0))
    
    # # disable controller
    # # constant_twentyfive=builder.AddSystem(ConstantVectorSource(np.zeros(25)))  
    # # builder.Connect(constant_twentyfive.get_output_port(),
    # #                 helper.get_input_port(1))
    # # builder.Connect(helper.get_output_port(),
    # #                 plant.get_applied_generalized_force_input_port())
    # #  

    # # enable controller
    # builder.Connect(PID.get_output_port_control(),
    #                 helper.get_input_port(1))
    # builder.Connect(helper.get_output_port(),
    #                 plant.get_applied_generalized_force_input_port()) 
    # #
    # ## <option two>             


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
    pdb.set_trace()

    return diagram, plant, state_logger, agent

def set_home(plant, context):
    #pdb.set_trace()
    #print(plant)
    waist=plant.GetBodyByName("waist")
    
    Na = plant.num_actuators()
    for i in range(Na):
        joint=plant.get_joint(JointIndex(i))   
        if not "_weld" in joint.name():
            #print(i,' ',joint.name())                                              
            low_joint= joint.position_lower_limit()
            high_joint= joint.position_upper_limit()
            joint.set_angle(context,0.0)
            #pdb.set_trace()

            #joint.set_angle(context,0.1*np.random.random()*(high_joint-low_joint)+0.1*low_joint)

    plant.SetFreeBodyPose(context,waist,RigidTransform([0,0,1.05]))
            

def set_home2(simulator):
    #print('setttt')
    plant = simulator.get_system().GetSubsystemByName("plant")
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)
    set_home(plant, plant_context)

def simulate_diagram(diagram, plant, state_logger,
                     agent_init_position, agent_init_velocity,
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
    #set_home2(simulator)
    
    frame_toesL=plant.GetBodyByName("foot_L").body_frame()
    #frame_toesR=plant.GetBodyByName("foot_R").body_frame()
    frame_world=plant.world_frame()
    p_AQ_lower=[0.09,0.09,-0.01]
    p_AQ_upper=[0.11,0.11,0.01]
    p_BQ=[0,0,0]
    PositionConstraint(plant,frame_world,p_AQ_lower ,p_AQ_upper ,frame_toesL,p_BQ ,plant_context)


    print("Initial state variables: ", plant.GetPositionsAndVelocities(plant_context))
   
    #simulator.get_mutable_context().SetTime(0)
    state_log = state_logger.FindMutableLog(simulator.get_mutable_context())
    state_log.Clear()
    simulator.Initialize()
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
        "--time_step", type=float, default=0.02,
        help="The fixed time step period (in seconds) of discrete updates "
             "for the multibody plant modeled as a discrete system. "
             "If zero, we will use an integrator for a continuous system. "
             "Non-negative. Default 0.001.")
    parser.add_argument(
        "--agent_initial_position", nargs=2, metavar=('tetha1', 'theta2'),
        default=[1.95, -1.87],
        help="Noodleman's initial joint position: tetha1, theta2 (in rad). "
             "Default: 1.95 -1.87. It correspond to a sitting position")
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Target realtime rate. Default 1.0.")
    args = parser.parse_args()

    diagram, humanoid_chair_plant, state_logger,humanoid_idx = make_agent_chair(
        args.contact_model, args.contact_surface_representation,
        args.time_step)
    time_samples, state_samples = simulate_diagram(
        diagram, humanoid_chair_plant, state_logger,
        np.array(args.agent_initial_position),
        np.array([0., 0.]),
        args.simulation_time, args.target_realtime_rate)
    print("\nFinal state variables:")
    print(state_samples[:, -1])
