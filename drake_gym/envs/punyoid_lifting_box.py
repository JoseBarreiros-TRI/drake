import gym
import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R
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
    ContactSolver,
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
gym_time_step=0.01
controller_time_step=0.01
gym_time_limit=5
modes=["IDC","torque"]
control_mode=modes[0]
box_size=[ 0.35,#0.2+0.1*(np.random.random()-0.5),
        0.35,#0.2+0.1*(np.random.random()-0.5),
         0.35,   #0.2+0.1*(np.random.random()-0.5),
        ]
box_mass=5
box_mu=1.0
contact_model=ContactModel.kPoint#kHydroelasticWithFallback#kHydroelastic#kPoint
contact_solver=ContactSolver.kTamsi # kTamsi
desired_box_heigth=0.7 #0.8
##

def AddAgent(plant):
    parser = Parser(plant)
    agent = parser.AddModelFromFile(FindResource("models/humanoid_torso_v2_noball_noZeroBodies_spring_prismatic_v2.sdf"))
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

def AddBox(plant):
    w= box_size[0]
    d= box_size[1]
    h= box_size[2]
    mass= box_mass
    mu= box_mu
    if contact_model==ContactModel.kHydroelastic or contact_model==ContactModel.kHydroelasticWithFallback:
        parser = Parser(plant)
        box = parser.AddModelFromFile(FindResource("models/box.sdf"))
    else:
        box=AddShape(plant, Box(w,d,h), name="box",mass=mass,mu=mu)

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
    plant.set_contact_model(contact_model) 
    plant.set_contact_solver(contact_solver)

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
    controller_plant.set_contact_solver(contact_solver)  
    AddAgent(controller_plant)        
    #SetTransparency(scene_graph, alpha=0.5, source_id=plant.get_source_id())

    if meshcat:
        MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
        ContactVisualizer.AddToBuilder(
            builder, plant, meshcat,
            ContactVisualizerParams(radius=0.005, newtons_per_meter=500.0))

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

    class observation_publisher(LeafSystem):

        def __init__(self):
            LeafSystem.__init__(self)
            Nss = plant.num_multibody_states()
            self.DeclareVectorInputPort("plant_states", Nss)
            self.DeclareAbstractInputPort("body_poses",AbstractValue.Make([RigidTransform.Identity()]))
            self.DeclareVectorOutputPort("observations", Nss+6, self.CalcObs)
            self.box_body_idx=plant.GetBodyByName('box').index()

        def CalcObs(self, context,output):
            plant_state = self.get_input_port(0).Eval(context)
            body_poses=self.get_input_port(1).Eval(context)
   
            box_pose = body_poses[self.box_body_idx].translation()
 
            box_rotation=body_poses[self.box_body_idx].rotation().matrix()
            #pose of the middle point of the farthest edge of the box
            box_L_edge= box_rotation.dot(np.array([box_pose[0]+box_size[0]/2,box_pose[1]+box_size[1]/2,box_pose[2]]))
            box_R_edge= box_rotation.dot(np.array([box_pose[0]-box_size[0]/2,box_pose[1]+box_size[1]/2,box_pose[2]]))
            #pdb.set_trace()
            extension=np.concatenate((box_L_edge,box_R_edge))
            extended_observations=np.concatenate((plant_state,extension))      
            output.set_value(extended_observations)

    obs_pub=builder.AddSystem(observation_publisher())

    builder.Connect(plant.get_state_output_port(),obs_pub.get_input_port(0))
    builder.Connect(plant.get_body_poses_output_port(), obs_pub.get_input_port(1))
    builder.ExportOutput(obs_pub.get_output_port(), "observations")

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
            self.box_body_idx=plant.GetBodyByName('box').index() 
            self.handL_body_idx=plant.GetBodyByName('hand_L').index() 
            self.handR_body_idx=plant.GetBodyByName('hand_R').index() 
            self.torso_body_idx=plant.GetBodyByName('torso').index() 
            self.desired_box_heigth=desired_box_heigth
            #self.Np=plant.num_positions()

        def CalcReward(self, context, output):
            agent_state = self.get_input_port(0).Eval(context)
            body_poses=self.get_input_port(1).Eval(context)
            actions = self.get_input_port(2).Eval(context)
            box_pose = body_poses[self.box_body_idx].translation()
            handL_pose = body_poses[self.handL_body_idx].translation()
            handR_pose = body_poses[self.handR_body_idx].translation()
            torso_pose = body_poses[self.torso_body_idx].translation()

            
            
            box_rotation=body_poses[self.box_body_idx].rotation().matrix()
            box_euler=R.from_dcm(box_rotation).as_euler('zyx', degrees=False)
            #pdb.set_trace()
            #pose of the middle point of the farthest edge of the box
            box_L_edge= box_rotation.dot(np.array([box_pose[0]+box_size[0]/2,box_pose[1]+box_size[1]/2,box_pose[2]]))
            box_R_edge= box_rotation.dot(np.array([box_pose[0]-box_size[0]/2,box_pose[1]+box_size[1]/2,box_pose[2]]))
            

            diff_heigth=self.desired_box_heigth-box_pose[2]

            distance_to_torso=box_pose-torso_pose

            cost_heigth=diff_heigth**2
            #distance to the hands
            diff_Lhand=box_L_edge-handL_pose
            diff_Rhand=box_R_edge-handR_pose
            cost_Lhand=diff_Lhand.dot(diff_Lhand)
            cost_Rhand=diff_Rhand.dot(diff_Rhand)
            cost_torso=distance_to_torso.dot(distance_to_torso)
            cost_rotation=box_euler[1]**2+box_euler[2]**2 

            cost = cost_Lhand + cost_Rhand + 10*cost_heigth + cost_torso + cost_rotation
            reward=1.5-cost
       
            if debug:
                print('box_pose: ',box_pose)
                print("Ledge: ",box_L_edge)
                print("Redge: ",box_R_edge)
                print("torso: ", torso_pose)
                print("Lhand: ",handL_pose)
                print("Rhand: ",handR_pose)
                
                #print('joint_state: ',noodleman_joint_state)
                #print('act: {a}, j_state: {p}'.format(a=actions,p=noodleman_joint_state))
                print('cost: {c}, cost_heigth: {ch}, cost_Lhand: {cl}, cost_Rhand: {cr}, cost_torso: {ct}m cost_rotation: {cro}'.format(c=cost,
                        ch=cost_heigth,
                        cl=cost_Lhand,
                        cr=cost_Rhand,
                        ct= cost_torso,
                        cro=cost_rotation,
                        ))
                print('rew: {r}\n'.format(r=reward))
            #pdb.set_trace()

            output[0] = reward

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(plant.get_state_output_port(agent), reward.get_input_port(0))
    builder.Connect(plant.get_body_poses_output_port(), reward.get_input_port(1))
    builder.Connect(actions.get_output_port(), reward.get_input_port(2))
    builder.ExportOutput(reward.get_output_port(), "reward")

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()

    # Termination conditions:
    def monitor(context,plant=plant):
        #pdb.set_trace()
        plant_context=plant.GetMyContextFromRoot(context)
        box_body_idx=plant.GetBodyByName('box').index() 
        body_poses=plant.get_body_poses_output_port().Eval(plant_context)
        box_pose=body_poses[box_body_idx].translation()
        #print("b_pose: ", box_pose)
        # terminate from time and box out of reach
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")
        elif box_pose[1]>0.9:
            #pdb.set_trace()
            return EventStatus.ReachedTermination(diagram, "box out of reach")
        
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

def set_home(simulator,diagram_context,plant_name="plant"):
    
    diagram = simulator.get_system()
    plant=diagram.GetSubsystemByName(plant_name)
    plant_context = diagram.GetMutableSubsystemContext(plant,
                                                diagram_context)  

    home_positions=[
        ('shoulderR_joint1',0.1*(np.random.random()-0.5)+0.1),
        ('shoulderL_joint1',0.1*(np.random.random()-0.5)+0.1),
        ('shoulderR_joint2',0.1*(np.random.random()-0.5)-0.2),
        ('shoulderL_joint2',0.1*(np.random.random()-0.5)-0.2),
        ('elbowR_joint1',0.2*(np.random.random()-0.5)+0.1),
        ('elbowL_joint1',0.2*(np.random.random()-0.5)+0.1),
        ('torso_joint1',0.2*(np.random.random()-0.5)),
        ('torso_joint2',0.1*(np.random.random()-0.5)),
        ('torso_joint3',0.2*(np.random.random()-0.5)),
        ('prismatic_z',0.2*(np.random.random()-0.5)+0.35),
    ]

    #ensure the positions are within the joint limits
    for pair in home_positions:
        joint = plant.GetJointByName(pair[0])
        if joint.type_name()=="prismatic":
            joint.set_translation(plant_context,
                        np.clip(pair[1],
                            joint.position_lower_limit(),
                            joint.position_upper_limit()
                            )
                        )
        elif joint.type_name()=="revolute":
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
                            0.4+0.15*(np.random.random()-0.5), 
                            box_size[2]/2+0.005,
                        ])
                    )
    plant.SetFreeBodyPose(plant_context,box,box_pose)

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
        (plant.GetPositionLowerLimits(), plant.GetVelocityLowerLimits(),np.array([-np.inf]*6)))
    high = np.concatenate(
        (plant.GetPositionUpperLimits(), plant.GetVelocityUpperLimits(),np.array([np.inf]*6)))
    observation_space = gym.spaces.Box(low=np.asarray(low, dtype="float64"),
                                       high=np.asarray(high, dtype="float64"),
                                       dtype=np.float64)

    env = DrakeGymEnv(simulator=simulator,
                      time_step=gym_time_step,
                      action_space=action_space,
                      observation_space=observation_space,
                      reward="reward",
                      action_port_id="actions",
                      observation_port_id="observations",
                      set_home=set_home)
    return env
