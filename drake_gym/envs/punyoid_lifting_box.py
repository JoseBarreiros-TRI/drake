import gym
import pdb
import numpy as np
import matplotlib.pyplot as plt
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.math import RollPitchYaw
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ConstantVectorSource,
    ContactVisualizer,
    ContactVisualizerParams,
    DiagramBuilder,
    EventStatus,
    InverseDynamicsController,
    LeafSystem,
    MeshcatVisualizerCpp,
    MeshcatVisualizerParams,
    MultibodyPlant,
    MultibodyPositionToGeometryPose,
    Multiplexer,
    Parser,
    PassThrough,
    RandomGenerator,
    RigidTransform,
    SceneGraph,
    Simulator,
    WeldJoint,
    ContactModel,
)

from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from drake_gym import DrakeGymEnv
from scenarios import AddShape, SetColor, SetTransparency
from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewVelocities,
        MakeNamedViewState,
        MakeNamedViewActuation)
import pydrake.geometry as mut


## Gym parameters
sim_time_step=0.001
gym_time_step=0.005
controller_time_step=0.002
gym_time_limit=5
modes=["IDC","torque"]
control_mode=modes[0]
box_size=[ 0.2+0.1*(np.random.random()-0.5),
            0.2+0.1*(np.random.random()-0.5),
            0.2+0.1*(np.random.random()-0.5) ]
##

def AddAgent(plant):
    parser = Parser(plant)
    agent = parser.AddModelFromFile(FindResource("models/humanoid_torso_v2_noball_noZeroBodies_spring_prismatic.sdf"))
    p_WAgent_fixed = RigidTransform(RollPitchYaw(0, 0, 0),
                                     np.array([0, 0, 0])) #0.25
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
    floor = parser.AddModelFromFile(FindResource("models/floor_v2.sdf"))
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

def make_sim(generator,
                    observations="state",
                    meshcat=None,
                    time_limit=5,debug=False):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)

    #set contact model
    contact_model=ContactModel.kPoint
    plant.set_contact_model(contact_model) 

    #add assets to the plant
    agent = AddAgent(plant)
    AddFloor(plant)
    box = AddBox(plant)
    plant.Finalize()
    plant.set_name("plant")
    # filter collisison between parent and child of each joint.
    add_collision_filters(scene_graph,plant)

    #add assets to the controller plant
    controller_plant = MultibodyPlant(time_step=controller_time_step)
    controller_plant.set_contact_model(contact_model)     
    AddAgent(controller_plant)        
#    SetTransparency(scene_graph, alpha=0.5, source_id=plant.get_source_id())

    if meshcat:
        MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
        ContactVisualizer.AddToBuilder(
            builder, plant, meshcat,
            ContactVisualizerParams(radius=0.005, newtons_per_meter=40.0))

        # Use the controller plant to visualize the set point geometry.
        controller_scene_graph = builder.AddSystem(SceneGraph())
        controller_plant.RegisterAsSourceForSceneGraph(controller_scene_graph)
        SetColor(controller_scene_graph,
                 color=[1.0, 165.0 / 255, 0.0, 1.0],
                 source_id=controller_plant.get_source_id())
        controller_vis = MeshcatVisualizerCpp.AddToBuilder(
            builder, controller_scene_graph, meshcat,
            MeshcatVisualizerParams(prefix="controller"))
        controller_vis.set_name("controller meshcat")

    #finalize the plant
    controller_plant.Finalize()
    controller_plant.set_name("controller_plant")
    add_collision_filters(scene_graph,controller_plant)  

    #extract controller plant information
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

    builder.ExportInput(actions.get_input_port(), "actions")
    builder.ExportOutput(plant.get_state_output_port(), "observations")

    class RewardSystem(LeafSystem):

        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("state", Ns)
            self.DeclareAbstractInputPort("body_poses",AbstractValue.Make([RigidTransform.Identity()]))
            self.DeclareVectorInputPort("actions", Na)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)
            self.StateView=MakeNamedViewState(plant, "States")
            self.PositionView=MakeNamedViewPositions(plant, "Position")
            self.ActuationView=MakeNamedViewActuation(plant, "Actuation") 

        def CalcReward(self, context, output):
            reward=1
       
            # if debug:
            #     print('torso_pose: ',torso_pose)
            #     print('joint_state: ',noodleman_joint_state)
            #     print('act: {a}, j_state: {p}'.format(a=actions,p=noodleman_joint_state))
            #     print('cost: {c}, cost_heigth: {ch}, cost_pos: {cp}, cost_vel: {cv}'.format(c=cost,ch=cost_heigth,cp=cost_pos,cv=cost_vel))
            #     print('rew: {r}\n'.format(r=reward))

            output[0] = reward

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(plant.get_state_output_port(agent), reward.get_input_port(0))
    builder.Connect(plant.get_body_poses_output_port(), reward.get_input_port(1))
    builder.Connect(actions.get_output_port(), reward.get_input_port(2))
    builder.ExportOutput(reward.get_output_port(), "reward")

    diagram = builder.Build()
    simulator = Simulator(diagram)

    # Termination conditions:
    def monitor(context):
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    if debug:
        #visualize plant and diagram
        plt.figure()
        plot_graphviz(controller_plant.GetTopologyGraphvizString())
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.plot(1)
        plt.show(block=False)
        #pdb.set_trace()

    return simulator

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

    Np = plant.num_positions()
    PositionView=MakeNamedViewPositions(plant, "Positions")

    default_position=PositionView([0]*Np)
    default_position.shoulderR_joint1=np.pi/4
    default_position.shoulderL_joint1=np.pi/4
    default_position.elbowR_joint=np.pi/4
    default_position.elbowL_joint=np.pi/4
    default_position.prismatic_z=0.3

    #add randomness offset to positions
    random_offset=PositionView([0]*Np)
    random_offset.shoulderR_joint1=0.3*(np.random.random()-0.5)
    random_offset.shoulderL_joint1=0.3*(np.random.random()-0.5)
    random_offset.shoulderR_joint2=0.3*(np.random.random()-0.5)
    random_offset.shoulderL_joint2=0.3*(np.random.random()-0.5)  
    random_offset.elbowR_joint=0.3*(np.random.random()-0.5)
    random_offset.elbowL_joint=0.3*(np.random.random()-0.5)
    random_offset.torso_joint1=0.2*(np.random.random()-0.5)
    random_offset.torso_joint2=0.2*(np.random.random()-0.5)
    random_offset.torso_joint3=0.6*(np.random.random()-0.5)
    random_offset.prismatic_z=0.2*(np.random.random()-0.5)

def PunyoidBoxLiftingEnv(observations="state", meshcat=None, time_limit=gym_time_limit, debug=False):
    #Make simulation
    simulator = make_sim(RandomGenerator(),
                            observations,
                            meshcat=meshcat,
                            time_limit=time_limit,
                            debug=debug)

    plant = simulator.get_system().GetSubsystemByName("plant")
    
    #Define Action space
    Na=plant.num_actuators()
    low = plant.GetPositionLowerLimits()[:Na]
    high = plant.GetPositionUpperLimits()[:Na]
    # StateView=MakeNamedViewState(plant, "States")
    # PositionView=MakeNamedViewPositions(plant, "Position")
    # ActuationView=MakeNamedViewActuation(plant, "Actuation")
    action_space = gym.spaces.Box(low=np.asarray(low, dtype="float64"), high=np.asarray(high, dtype="float64"),dtype=np.float64)
     
    #Define observation space 
    low = np.concatenate(
        (plant.GetPositionLowerLimits(), plant.GetVelocityLowerLimits()))
    high = np.concatenate(
        (plant.GetPositionUpperLimits(), plant.GetVelocityUpperLimits()))
    observation_space = gym.spaces.Box(low=np.asarray(low, dtype="float64"),
                                       high=np.asarray(high, dtype="float64"),
                                       dtype=np.float64)
    
    env = DrakeGymEnv(simulator=simulator,
                      time_step=gym_time_step,
                      action_space=action_space,
                      observation_space=observation_space,
                      reward="reward",
                      action_port_id="actions",
                      observation_port_id="observations")
    return env

# for standalone testing of the environment
# if __name__ == "__main__":

#     from pydrake.geometry import StartMeshcat
#     from stable_baselines3.common.env_checker import check_env
    
#     # Make a version of the env with meshcat.
#     meshcat = StartMeshcat()
#     gym.envs.register(id="PunyoidBoxLifting-v0",
#                   entry_point="punyoid_lifting_box:PunyoidBoxLiftingEnv")

#     env = gym.make("PunyoidBoxLifting-v0", meshcat=meshcat, observations="state",debug=True)
#     env.simulator.set_target_realtime_rate(1.0)
#     pdb.set_trace()
#     #obs=env.reset()
#     #print(env.observation_space.contains(obs))
    
#     check_env(env)