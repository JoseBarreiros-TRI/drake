import gym, pdb
import numpy as np
import matplotlib.pyplot as plt

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
)
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from drake_gym import DrakeGymEnv
from scenarios import AddShape, SetColor, SetTransparency
from utils import FindResource


def AddNoodleman(plant):
    parser = Parser(plant)
    noodleman = parser.AddModelFromFile(FindResource("models/noodleman_v1.sdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("lower_leg", noodleman),
        RigidTransform(RollPitchYaw(0, 0, 0),
                        np.array([0, 0.6, 0.35]))
                    )

    return noodleman


def make_noodleman_stand_up_sim(generator,
                    observations="state",
                    meshcat=None,
                    time_limit=5,debug=False):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    noodleman = AddNoodleman(plant)
    plant.Finalize()
    plant.set_name("plant")
    SetTransparency(scene_graph, alpha=0.5, source_id=plant.get_source_id())
    controller_plant = MultibodyPlant(time_step=0.001)
    AddNoodleman(controller_plant)

    if meshcat:
        MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
        #meshcat.Set2dRenderMode(xmin=-.35, xmax=.35, ymin=-0.1, ymax=0.3)
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

    controller_plant.Finalize()
    
    if debug:
        print("\nnumber of position: ",plant.num_positions(),
            ", number of velocities: ",plant.num_velocities(),
            ", number of actuators: ",plant.num_actuators(),
            ", number of multibody states: ",plant.num_multibody_states(),'\n')

    ##Create inverse dynamics controller
    N = controller_plant.num_positions()
    kp = [100] * N
    ki = [1] * N
    kd = [20] * N
    controller = builder.AddSystem(
        InverseDynamicsController(controller_plant, kp, ki, kd, False))
    builder.Connect(plant.get_state_output_port(noodleman),
                    controller.get_input_port_estimated_state())

    actions = builder.AddSystem(PassThrough(N))
    positions_to_state = builder.AddSystem(Multiplexer([N, N]))
    builder.Connect(actions.get_output_port(),
                    positions_to_state.get_input_port(0))
    zeros = builder.AddSystem(ConstantVectorSource([0] * N))
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

    class RewardSystem(LeafSystem):

        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("noodleman_state", 4)
            self.DeclareVectorInputPort("actions", 2)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

        def CalcReward(self, context, output):
            noodleman_state = self.get_input_port(0).Eval(context)
            actions = self.get_input_port(1).Eval(context)
            desired_state=[0,0,0,0]
            pos_error=desired_state[:2]-noodleman_state[:2]

            cost_pos = 2 * pos_error.dot(pos_error)

            # noodleman velocity
            cost_vel=noodleman_state[2:].dot(noodleman_state[2:])
            #print('cost2: {c}'.format(c=cost_vel))
            cost = cost_pos + 0.01 * cost_vel

            # Add 20 to make rewards positive (to avoid rewarding simulator
            # crashes).            
            reward= 17 - cost

            if debug:
                print('act: {a}, pos: {p}'.format(a=actions,p=noodleman_state[:2]))
                print('cost: {c}, cost_pos: {cp}, cost_vel: {cv}'.format(c=cost,cp=cost_pos,cv=cost_vel))
                print('rew: {r}\n'.format(r=reward))

            output[0] = reward

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(plant.get_state_output_port(noodleman), reward.get_input_port(0))
    builder.Connect(actions.get_output_port(), reward.get_input_port(1))
    builder.ExportOutput(reward.get_output_port(), "reward")

    # Set random state distributions.
    uniform_random_1 = Variable(name="uniform_random",
                              type=Variable.Type.RANDOM_UNIFORM)
    uniform_random_2 = Variable(name="uniform_random2",
                              type=Variable.Type.RANDOM_UNIFORM)                              
    hip_joint = plant.GetJointByName("hip_joint")
    knee_joint = plant.GetJointByName("knee_joint")

    low_hip= hip_joint.position_lower_limit()
    high_hip= hip_joint.position_upper_limit()
    low_knee= knee_joint.position_lower_limit()
    high_knee= knee_joint.position_upper_limit()

    #print(low_hip,' ',high_hip,' ',low_knee,' ',high_knee)

    knee_joint.set_random_angle_distribution((high_knee-low_knee)*uniform_random_2+low_knee)
    hip_joint.set_random_angle_distribution((high_hip-low_hip)*uniform_random_1+low_hip)


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
        plt.figure()
        plot_graphviz(plant.GetTopologyGraphvizString())
        plt.figure()
        plot_graphviz(controller_plant.GetTopologyGraphvizString())
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.plot(1)
        plt.show(block=False)

    return simulator


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
    # action_space = gym.spaces.Box(low=np.array([-np.pi/2, -np.pi/2], dtype="float64"),
    #                               high=np.array([np.pi/2, np.pi/2], dtype="float64"))
    action_space = gym.spaces.Box(low=low[:2], high=high[:2],dtype=np.float64)

    

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
    return env