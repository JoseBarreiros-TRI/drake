from re import S
import gym
import pdb
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pydrake.common.value import AbstractValue
from pydrake.math import RigidTransform
from pydrake.math import RollPitchYaw
from pydrake.all import (
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
    AddMultibodyPlant,
    MultibodyPlantConfig,
    FindResourceOrThrow,

)

from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from drake_gym import DrakeGymEnv
from scenarios import AddShape, SetColor, SetTransparency
from utils import (FindResource, MakeNamedViewPositions,
        MakeNamedViewState,
        MakeNamedViewActuation)
import pydrake.geometry as mut


## Gym parameters
sim_time_step=0.01
gym_time_step=0.05
controller_time_step=0.01
gym_time_limit=5
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
    1.0+0.5*(np.random.random()-0.5),
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
    floor = parser.AddModelFromFile(FindResource("models/floor_v3.sdf"))
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
        box = parser.AddModelFromFile(FindResource("models/box_v2.sdf"))
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

def make_sim(generator,
                    observations="state",
                    meshcat=None,
                    time_limit=5,debug=False):

    builder = DiagramBuilder()

    multibody_plant_config = \
        MultibodyPlantConfig(
            time_step=sim_time_step,
            contact_model=contact_model,
            )

    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)

    #add assets to the plant
    agent = AddAgent(plant)
    AddTable(plant)
    box = AddBox(plant)
    target_position=[desired_box_xy[0],desired_box_xy[1],table_heigth+0.01]
    AddTargetPosVisuals(plant,target_position)
    plant.Finalize()
    plant.set_name("plant")
    # filter collisison between parent and child of each joint.
    add_collision_filters(scene_graph,plant)

    #add assets to the controller plant
    controller_plant = MultibodyPlant(time_step=controller_time_step)
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
            self.DeclareVectorOutputPort("observations", Nss+15, self.CalcObs)
            self.box_body_idx=plant.GetBodyByName('box').index()
            self.desired_box_pose=np.array([desired_box_xy[0],desired_box_xy[1],box_size[2]/2+table_heigth])

        def CalcObs(self, context,output):
            plant_state = self.get_input_port(0).Eval(context)
            body_poses=self.get_input_port(1).Eval(context)

            box_pose = body_poses[self.box_body_idx].translation()

            box_rotation=body_poses[self.box_body_idx].rotation().matrix()
            #pose of the middle point of the farthest edge of the box
            box_LF_edge= box_rotation.dot(np.array([box_pose[0]+box_size[0]/2,box_pose[1]+box_size[1]/2,box_pose[2]]))
            box_RF_edge= box_rotation.dot(np.array([box_pose[0]-box_size[0]/2,box_pose[1]+box_size[1]/2,box_pose[2]]))
                        #pose of the middle point of the closest edge of the box
            box_LC_edge= box_rotation.dot(np.array([box_pose[0]+box_size[0]/2,box_pose[1]-box_size[1]/2,box_pose[2]]))
            box_RC_edge= box_rotation.dot(np.array([box_pose[0]-box_size[0]/2,box_pose[1]-box_size[1]/2,box_pose[2]]))
            distance_to_target=self.desired_box_pose-box_pose
            #pdb.set_trace()
            extension=np.concatenate((box_LF_edge,box_RF_edge,box_LC_edge,box_RC_edge,distance_to_target))
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
            self.EE_body_idx=plant.GetBodyByName('iiwa_link_7').index()
            self.desired_box_pose=np.array([desired_box_xy[0],desired_box_xy[1],box_size[2]/2+table_heigth])
            #self.Np=plant.num_positions()

        def CalcReward(self, context, output):
            agent_state = self.get_input_port(0).Eval(context)
            body_poses=self.get_input_port(1).Eval(context)
            actions = self.get_input_port(2).Eval(context)
            box_pose = body_poses[self.box_body_idx].translation()
            EE_pose = body_poses[self.EE_body_idx].translation()

            box_rotation=body_poses[self.box_body_idx].rotation().matrix()
            box_euler=R.from_dcm(box_rotation).as_euler('zyx', degrees=False)
            #pdb.set_trace()

            distance_to_EE=box_pose-EE_pose
            distance_to_target=self.desired_box_pose-box_pose

            cost_EE=distance_to_EE.dot(distance_to_EE)
            cost_to_target=distance_to_target.dot(distance_to_target)

            cost = cost_EE + 10*cost_to_target
            #cost = 10*cost_to_target
            reward=1.2-cost

            if debug:
                print('box_pose: ',box_pose)
                print("EE: ", EE_pose)

                #print('joint_state: ',noodleman_joint_state)
                #print('act: {a}, j_state: {p}'.format(a=actions,p=noodleman_joint_state))
                print('cost: {c}, cost_EE: {ce}, cost_to_target: {ct}'.format(
                        c=cost,
                        ce=cost_EE,
                        ct=cost_to_target,
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
        elif np.linalg.norm(box_pose)>1.4 or box_pose[1]<0.2:
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

def IiwaBoxPushingEnv(observations="state", meshcat=None, time_limit=gym_time_limit, debug=False):

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
        (plant.GetPositionLowerLimits(), plant.GetVelocityLowerLimits(),np.array([-np.inf]*15)))
    high = np.concatenate(
        (plant.GetPositionUpperLimits(), plant.GetVelocityUpperLimits(),np.array([np.inf]*15)))
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
