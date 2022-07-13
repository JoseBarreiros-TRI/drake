import gym, pdb
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
)
from pydrake.common.containers import EqualToDict, namedview, NamedViewBase

from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from drake_gym import DrakeGymEnv
from scenarios import AddShape, SetColor, SetTransparency
from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewVelocities,
        MakeNamedViewState,
        MakeNamedViewActuation)

def AddNoodleman(plant):
    parser = Parser(plant)
    noodleman = parser.AddModelFromFile(FindResource("models/noodleman_w_foot_v1.sdf"))
    return noodleman

def AddFloor(plant):
    parser = Parser(plant)
    floor = parser.AddModelFromFile(FindResource("models/floor.sdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("floor", floor),
        RigidTransform(RollPitchYaw(0, 0, 0),
                        np.array([0, 0, -0.05]))
                    )
    return floor

def make_noodleman_stand_up_sim(generator,
                    observations="state",
                    meshcat=None,
                    time_limit=5,debug=False):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    noodleman = AddNoodleman(plant)
    AddFloor(plant)
    plant.Finalize()
    plant.set_name("plant")
    SetTransparency(scene_graph, alpha=0.5, source_id=plant.get_source_id())
    controller_plant = MultibodyPlant(time_step=0.001)
    AddNoodleman(controller_plant)

    if meshcat:
        MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
        #meshcat.Set2dRenderMode(xmin=-.35, xmax=.35, ymin=-0.1, ymax=0.3)
        # ContactVisualizer.AddToBuilder(
        #     builder, plant, meshcat,
        #     ContactVisualizerParams(radius=0.005, newtons_per_meter=40.0))

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

    controller_plant.Finalize()
    
    if debug:
        print("\nnumber of position: ",plant.num_positions(),
            ", number of velocities: ",plant.num_velocities(),
            ", number of actuators: ",plant.num_actuators(),
            ", number of joints: ",plant.num_joints(),
            ", number of multibody states: ",plant.num_multibody_states(),'\n')
        plt.figure()
        plot_graphviz(plant.GetTopologyGraphvizString())
        plt.plot(1)
        plt.show(block=False)
       
    ##Create inverse dynamics controller
    Ns = controller_plant.num_multibody_states()
    Nv = controller_plant.num_velocities()
    Na = controller_plant.num_actuators()
    kp = [100] * Na
    ki = [0] * Na
    kd = [10] * Na
    # Select the joint states (and ignore the floating-base states)
    S=np.zeros((Na*2,Ns))
    j=0
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        #print(joint.num_positions())
        #pdb.set_trace()
        if joint.num_positions() != 1:
            continue
        S[j, joint.position_start()] = 1
        S[Na+j, joint.velocity_start()] = 1
        j = j+1

    
    controller = builder.AddSystem(
        PidController(
            kp=kp, 
            ki=ki, 
            kd=kd,
            state_projection=S,
            output_projection=plant.MakeActuationMatrix()[Nv-Na:,:].T)
            )
    
    builder.Connect(plant.get_state_output_port(noodleman),
                    controller.get_input_port_estimated_state())

    actions = builder.AddSystem(PassThrough(Na))
    positions_to_state = builder.AddSystem(Multiplexer([Na, Na]))
    builder.Connect(actions.get_output_port(),
                    positions_to_state.get_input_port(0))
    zeros = builder.AddSystem(ConstantVectorSource([0] * Na))
    builder.Connect(zeros.get_output_port(),
                    positions_to_state.get_input_port(1))
    builder.Connect(positions_to_state.get_output_port(),
                    controller.get_input_port_desired_state())
    builder.Connect(controller.get_output_port(),
                    plant.get_actuation_input_port())

    
    if meshcat:
        positions_to_poses = builder.AddSystem(
            MultibodyPositionToGeometryPose(controller_plant))
        builder.Connect(
            positions_to_poses.get_output_port(),
            controller_scene_graph.get_source_pose_port(
                controller_plant.get_source_id()))

    builder.ExportInput(actions.get_input_port(), "actions")
    if observations == "state":
        builder.ExportOutput(plant.get_state_output_port(), "observations")
    else:
        raise ValueError("observations must be one of ['state']")

    # class ObsExtractor(LeafSystem):
    #      def __init__(self):
    #         LeafSystem.__init__(self)
    #         self.DeclareVectorInputPort("noodleman_state", 19)
    #         self.DeclareVectorInputPort("actions", 3)

    class RewardSystem(LeafSystem):

        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("noodleman_state", 19)
            self.DeclareAbstractInputPort("noodleman_poses",AbstractValue.Make([RigidTransform.Identity()]))
            #self.DeclareVectorInputPort("noodleman_poses", 10)
            self.DeclareVectorInputPort("actions", 3)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)
            ##Create inverse dynamics controller
            Ns = controller_plant.num_multibody_states()
            Na = controller_plant.num_actuators()

            # Select the joint states (and ignore the floating-base states)
            S=np.zeros((Na*2,Ns))
            j=0
            for i in range(plant.num_joints()):
                joint = plant.get_joint(JointIndex(i))
                #print(joint.num_positions())
                #pdb.set_trace()
                if joint.num_positions() != 1:
                    continue
                S[j, joint.position_start()] = 1
                S[Na+j, joint.velocity_start()] = 1
                j = j+1
            #self.S=S

        def CalcReward(self, context, output):
            
            torso_body_idx=plant.GetBodyByName('torso').index()
            noodleman_state = self.get_input_port(0).Eval(context)
            torso_pose=self.get_input_port(1).Eval(context)[torso_body_idx].translation()
            actions = self.get_input_port(2).Eval(context)
            
            
            StateView=MakeNamedViewState(plant, "States")
            state=StateView(noodleman_state)
            #ActuationView=MakeNamedViewActuation(plant, "Actuation")
            #actuation=ActuationView(actions)
            

            desired_joint_pose=np.array([0,0,0])
            noodleman_joint_state=np.array([state.ankle_joint_q,state.knee_joint_q,state.hip_joint_q])
            pos_error=desired_joint_pose-noodleman_joint_state
            
            #pdb.set_trace()
            
            cost_heigth=(1.7-torso_pose[2])**2

            cost_pos = 2 * pos_error.dot(pos_error)

            # noodleman velocity
            cost_vel=noodleman_state[2:].dot(noodleman_state[2:])
            #print('cost2: {c}'.format(c=cost_vel))
            cost = 10*cost_heigth+cost_pos + 0.01 * cost_vel

            # Add 20 to make rewards positive (to avoid rewarding simulator
            # crashes).            
            reward= 27 - cost

            if debug:
                print('torso_pose: ',torso_pose)
                print('joint_state: ',noodleman_joint_state)
                print('act: {a}, j_state: {p}'.format(a=actions,p=noodleman_joint_state))
                print('cost: {c}, cost_heigth: {ch}, cost_pos: {cp}, cost_vel: {cv}'.format(c=cost,ch=cost_heigth,cp=cost_pos,cv=cost_vel))
                print('rew: {r}\n'.format(r=reward))

            output[0] = reward

    reward = builder.AddSystem(RewardSystem())
    #pdb.set_trace()


    builder.Connect(plant.get_state_output_port(noodleman), reward.get_input_port(0))
    builder.Connect(plant.get_body_poses_output_port(), reward.get_input_port(1))
    builder.Connect(actions.get_output_port(), reward.get_input_port(2))
    builder.ExportOutput(reward.get_output_port(), "reward")

    # Set random state distributions.
    uniform_random_1 = Variable(name="uniform_random",
                              type=Variable.Type.RANDOM_UNIFORM)
    uniform_random_2 = Variable(name="uniform_random2",
                              type=Variable.Type.RANDOM_UNIFORM)   
    uniform_random_3 = Variable(name="uniform_random3",
                              type=Variable.Type.RANDOM_UNIFORM)                                                            
    hip_joint = plant.GetJointByName("hip_joint")
    knee_joint = plant.GetJointByName("knee_joint")
    ankle_joint = plant.GetJointByName("ankle_joint")

    low_hip= hip_joint.position_lower_limit()
    high_hip= hip_joint.position_upper_limit()
    low_knee= knee_joint.position_lower_limit()
    high_knee= 0.5*knee_joint.position_upper_limit() #to avoid penetration w/ floor at start
    low_ankle= ankle_joint.position_lower_limit()
    high_ankle= ankle_joint.position_upper_limit()

    #print(low_hip,' ',high_hip,' ',low_knee,' ',high_knee)

    knee_joint.set_random_angle_distribution((high_knee-low_knee)*uniform_random_1+low_knee)
    hip_joint.set_random_angle_distribution((high_hip-low_hip)*uniform_random_2+low_hip)
    ankle_joint.set_random_angle_distribution((high_ankle-low_ankle)*uniform_random_3+low_ankle)


    #pdb.set_trace()

    #context = plant.CreateDefaultContext()
    #print(hip_joint.get_angle(context), ' ',knee_joint.get_angle(context))

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
        # plt.figure()
        # plot_graphviz(plant.GetTopologyGraphvizString())
        plt.figure()
        plot_graphviz(controller_plant.GetTopologyGraphvizString())
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.plot(1)
        plt.show(block=False)
    #pdb.set_trace()
    # context = simulator.get_mutable_context()
    # plant_context = plant.GetMyContextFromRoot(context)
    # set_home(plant, plant_context)
    # #if meshcat:
    # context2 = controller_plant.CreateDefaultContext()
    # controller_plant_context = controller_plant.GetMyContextFromRoot(context2)
    # set_home(controller_plant, controller_plant_context)
    return simulator

def set_home(plant, context):
    #pdb.set_trace()
    #print(plant)
    plant.SetFreeBodyPose(context, plant.GetBodyByName("foot"), RigidTransform([0.5, 0.5, 0.5]))

def NoodlemanStandUpEnv(observations="state", meshcat=None, time_limit=5, debug=False):
    simulator = make_noodleman_stand_up_sim(RandomGenerator(),
                                observations,
                                meshcat=meshcat,
                                time_limit=time_limit,debug=debug)

    plant = simulator.get_system().GetSubsystemByName("plant")
    


    low = np.concatenate(
        (plant.GetPositionLowerLimits(), plant.GetVelocityLowerLimits()))
    high = np.concatenate(
        (plant.GetPositionUpperLimits(), plant.GetVelocityUpperLimits()))
    action_space = gym.spaces.Box(low=np.array([-np.pi/2, -np.pi/2, -np.pi/2], dtype="float64"),
                                   high=np.array([np.pi/2, np.pi/2, np.pi/2], dtype="float64"))
    
    #action_space = gym.spaces.Box(low=low[:2], high=high[:2],dtype=np.float64)

    

    if observations == "state":

        observation_space = gym.spaces.Box(low=np.asarray(low, dtype="float64"),
                                           high=np.asarray(high,
                                                           dtype="float64"),dtype=np.float64)

    env = DrakeGymEnv(simulator=simulator,
                      time_step=0.005,
                      action_space=action_space,
                      observation_space=observation_space,
                      reward="reward",
                      action_port_id="actions",
                      observation_port_id="observations")
    #pdb.set_trace()
    return env