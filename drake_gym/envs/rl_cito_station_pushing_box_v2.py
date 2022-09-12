from functools import partial
from ast import excepthandler
from http.client import ResponseNotReady
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
    MakeRenderEngineVtk,
    LeafSystem,
    RotationMatrix,
    RgbdSensor,
    RenderCameraCore,
    RenderEngineVtkParams,
    MeshcatVisualizer,
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
    Variable,
    CameraInfo,
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    CoulombFriction,
    RoleAssign,
)

from pydrake.systems.primitives import FirstOrderLowPassFilter
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters)

from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from drake_gym import DrakeGymEnv
from utils import (FindResource, MakeNamedViewPositions,
        MakeNamedViewState,
        MakeNamedViewActuation)
import pydrake.geometry as mut

from pydrake.examples import (
    RlCitoStation,
    RlCitoStationHardwareInterface)
from drake.examples.rl_cito_station.differential_ik import DifferentialIK

## Gym parameters
sim_time_step=0.005
gym_time_step=0.05
gym_time_limit=5
table_heigth =0.
box_size=[ 0.075,#0.2+0.1*(np.random.random()-0.5),
        0.05,#0.2+0.1*(np.random.random()-0.5),
         0.05,   #0.2+0.1*(np.random.random()-0.5),
        ]
contact_model='point'#'hydroelastic_with_fallback'#ContactModel.kHydroelasticWithFallback#kPoint
contact_solver='sap'#ContactSolver.kSap#kTamsi # kTamsi

##
optitrack_pose_transform=np.array([[1,0,0],[0,1,0],[0,0,1]])
cross_xyz=np.array([0,0.5,0])

def AddTargetVisual(plant):
    parser = Parser(plant)
    marker = parser.AddModelFromFile(FindResource("models/cross_prismatic_2D.sdf"))
    return marker

def add_collision_filters(scene_graph, plant):
    filter_manager=scene_graph.collision_filter_manager()
    body_pairs=[
        ["iiwa_link_0","iiwa_link_1"],
        ["iiwa_link_1","iiwa_link_2"],
        ["iiwa_link_2","iiwa_link_3"],
        ["iiwa_link_4","iiwa_link_5"],
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
             meshcat=None,
             time_limit=5,
             debug=False,
             hardware=False,
             task="reach",
             mock_hardware=False,
             obs_noise=False,
             monitoring_camera=False,
             add_disturbances=False,
             observation_type=["state"],
             reward_type=["sparse"],
             termination_type=[],
             control_mode=["joint_positions"]):

    assert(task=="reach" or task=="push"),f'_{task}_ task not implemented. valid options are push, reach'
    assert(control_mode=="joint_position" or control_mode=="EE_pose" or control_mode=="EE_delta_pose"),f'_{control_mode}_ control mode not implemented. valid options are joint_positions, EE_pose'

    builder = DiagramBuilder()

    if hardware:
        if mock_hardware:
            station = builder.AddSystem(RlCitoStationHardwareInterface(has_optitrack=True))
        else:
            station = builder.AddSystem(RlCitoStationHardwareInterface(has_optitrack=True, optitrack_frame_transform=optitrack_pose_transform))
        station.Connect(wait_for_optitrack=True)
        controller_plant=station.get_controller_plant()
        plant=None
    else:
        station = builder.AddSystem(RlCitoStation(time_step=sim_time_step,contact_model=contact_model,contact_solver=contact_solver))
        station.SetupCitoRlStation()

        station.AddManipulandFromFile(
                "drake/examples/rl_cito_station/models/"
                + "optitrack_brick_v2.sdf",
                RigidTransform(RotationMatrix.Identity(), [0.6, 0, table_heigth]),
                "box")

        controller_plant=station.get_controller_plant()
        plant=station.get_multibody_plant()
        scene_graph=station.get_scene_graph()

        AddTargetVisual(plant)
        station.Finalize()


        class target_xyz_extractor(LeafSystem):
            def __init__(self,state_view):
                LeafSystem.__init__(self)
                Ns=plant.num_multibody_states()
                self.DeclareVectorInputPort("target_state",Ns)
                self.DeclareVectorOutputPort("target_xyz", 3, self.Extract_xyz)
                self.state_view=state_view

            def Extract_xyz(self, context, output):
                state = self.get_input_port(0).Eval(context)
                s=self.state_view(state)
                #pdb.set_trace()
                xyz=[s.Cross_Slider_x_x,s.Cross_Slider_y_x,table_heigth]
                output.set_value(xyz)

        PlantStateView=MakeNamedViewState(plant, "States")
        target_xyz_ext=builder.AddSystem(target_xyz_extractor(state_view=PlantStateView))
        builder.Connect(station.GetOutputPort("plant_continuous_state"),target_xyz_ext.get_input_port())

        if meshcat:
            geometry_query_port = station.GetOutputPort("geometry_query")
            meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
                builder=builder,
                query_object_port=geometry_query_port,
                meshcat=meshcat)

            # MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
            contact_results_port = station.GetOutputPort("contact_results")
            ContactVisualizer.AddToBuilder(
                builder=builder,
                contact_results_port=contact_results_port,
                query_object_port=geometry_query_port,
                meshcat=meshcat,
                params=ContactVisualizerParams(radius=0.005, newtons_per_meter=50.0),
                )

        # filter collisison between parent and child of each joint.
        add_collision_filters(scene_graph,plant)


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
        plot_graphviz(controller_plant.GetTopologyGraphvizString())
        plt.plot(1)
        plt.show(block=False)

        print("\nState view: ", StateView(np.ones(Ns)))
        print("\nActuation view: ", ActuationView(np.ones(Na)))
        print("\nPosition view: ",PositionView(np.ones(Np)))
        #pdb.set_trace()

    if hardware:
        if control_mode=="joint_positions":
            action_passthrough=builder.AddSystem(PassThrough(Na))
            builder.Connect(action_passthrough.get_output_port(),station.GetInputPort("iiwa_position"))
            builder.ExportInput(action_passthrough.get_input_port(),"actions")
        elif control_mode=="EE_pose":
            # TODO
            pass
    else:
        if control_mode=="joint_positions":
            action_passthrough=builder.AddSystem(PassThrough(Na))
            builder.Connect(action_passthrough.get_output_port(),station.GetInputPort("iiwa_position"))
            #builder.ExportInput(station.GetInputPort("iiwa_position"),"actions")
        elif control_mode=="EE_pose" or control_mode=="EE_delta_pose":
            params = DifferentialInverseKinematicsParameters(controller_plant.num_positions(),
                                                            controller_plant.num_velocities())
            time_step = 0.005
            params.set_timestep(time_step)
            # True velocity limits for the IIWA14 (in rad, rounded down to the first
            # decimal)
            iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
            # Stay within a small fraction of those limits for this teleop demo.
            factor = 1.0
            params.set_joint_velocity_limits((-factor*iiwa14_velocity_limits,
                                            factor*iiwa14_velocity_limits))
            differential_ik = builder.AddSystem(DifferentialIK(
                controller_plant, controller_plant.GetFrameByName("iiwa_link_7"), params, time_step, debug))

            builder.Connect(differential_ik.GetOutputPort("joint_position_desired"),
                            station.GetInputPort("iiwa_position"))
            filter = builder.AddSystem(
                FirstOrderLowPassFilter(time_constant=0.1, size=6))
            builder.Connect(filter.get_output_port(0),
                    differential_ik.GetInputPort("rpy_xyz_desired"))
            differential_ik.set_name("diffIK")
            filter.set_name("diffIK_filter")


        if control_mode=="EE_pose":
            action_passthrough=builder.AddSystem(PassThrough(6))
            builder.Connect(action_passthrough.get_output_port(),filter.get_input_port(0))
            #builder.ExportInput(filter.get_input_port(0),"actions")
            #builder.ExportInput(differential_ik.get_input_port(0),"actions")

        elif control_mode=="EE_delta_pose":

            class EEPoseAdder(LeafSystem):
                def __init__(self, robot,differential_ik):
                    LeafSystem.__init__(self)
                    self.Na= robot.num_actuators()
                    self.robot=robot
                    self.differential_ik=differential_ik
                    self.DeclareVectorInputPort("iiwa_positions_measured",self.Na)
                    self.DeclareVectorInputPort("deltaEE_rpyxyz",6)
                    self.DeclareVectorOutputPort("EE_pose",6,self.CalcOutput)

                def CalcOutput(self, context, output):
                    iiwa_positions_measured = self.get_input_port(0).Eval(context)
                    deltaEE_rpyxyz = self.get_input_port(1).Eval(context)
                    current_EE_pose=self.differential_ik.ForwardKinematics(iiwa_positions_measured)
                    rpy=RollPitchYaw(current_EE_pose.rotation())
                    #pdb.set_trace()
                    current_EE_rpy=np.array([rpy.roll_angle(),
                                    rpy.pitch_angle(),
                                    rpy.yaw_angle()])

                    new_EE_rpy=np.clip(current_EE_rpy+deltaEE_rpyxyz[:3],
                                    -2*np.pi,
                                    2*np.pi)
                    new_EE_xyz=np.clip(current_EE_pose.translation()+deltaEE_rpyxyz[3:],
                                    [0.2,-0.7,0],
                                    [0.8,0.7,0.6])
                    #TODO grap rpy

                    output.set_value(np.concatenate((new_EE_rpy,new_EE_xyz)))

            delta_adder=builder.AddSystem(EEPoseAdder(robot=controller_plant,differential_ik=differential_ik))
            builder.Connect(delta_adder.get_output_port(),filter.get_input_port(0))
            # builder.Connect(delta_adder.get_output_port(),differential_ik.get_input_port(0))
            builder.Connect(station.GetOutputPort("iiwa_position_measured"),delta_adder.get_input_port(0))
            #builder.ExportInput(delta_adder.get_input_port(1),"actions")
            action_passthrough=builder.AddSystem(PassThrough(6))
            builder.Connect(action_passthrough.get_output_port(),delta_adder.get_input_port(1))

        builder.ExportInput(action_passthrough.get_input_port(),"actions")
    class observation_publisher(LeafSystem):

        def __init__(self, robot, obs_type, noise=False):
            LeafSystem.__init__(self)

            self.Na= robot.num_actuators()
            self.DeclareAbstractInputPort("manipuland_pose", AbstractValue.Make(RigidTransform.Identity()))
            self.DeclareVectorInputPort("iiwa_positions_measured",self.Na)
            self.DeclareVectorInputPort("iiwa_velocities_measured",self.Na)
            self.DeclareVectorInputPort("iiwa_torques_measured",self.Na)
            self.DeclareVectorInputPort("iiwa_position_commanded",self.Na)
            self.DeclareVectorInputPort("target_xyz",3)
            if control_mode=="EE_pose" or control_mode=="EE_delta_pose":
                self.DeclareVectorInputPort("EE_rpyxyz",6)

            self.robot = robot
            self.frame_EE = robot.GetFrameByName("iiwa_link_7")
            self.robot_context = robot.CreateDefaultContext()
            self.noise = noise
            self.obs_type = obs_type

            if "buffer_10" in self.obs_type:
                self.num_history = 10
            elif "buffer_20" in self.obs_type:
                self.num_history = 20
            else:
                self.num_history = 1

            self.obs_size = 0
            if "state" in self.obs_type:
                # Obs: measured_positions and velocities
                self.obs_size += 2*self.Na

            if "actions" in self.obs_type:
                # Obs: commanded_positions
                if control_mode=="joint_positions":
                    self.obs_size += self.Na
                elif control_mode=="EE_pose" or control_mode=="EE_delta_pose":
                    self.obs_size += 6

            if "distances" in self.obs_type:
                # Obs: distance_EE_to_target, distance_EE_to_box, distance_EE_to_iiwabase.
                self.obs_size += 3

            if "EE_box_target_xyz" in self.obs_type:
                # Obs: xyz positions of end effector,
                # target, and box in the robot base frame.
                self.obs_size += 9

            if "torques" in self.obs_type:
                # Obs: measuared joint torques
                self.obs_size += self.Na

            self.DeclareVectorOutputPort(
                "observations",
                self.obs_size*self.num_history,
                self.CalcObs
                )

            self.obs_buffer = np.array(
                                self.num_history*[np.zeros(self.obs_size)])

        def CalcObs(self, context, output):
            try:
                box_pose = self.get_input_port(0).Eval(context)
            except:
                box_pose=RigidTransform.Identity()  #this might not be safe

            # This assumes the robot base is centered at the world frame.
            box_xyz = box_pose.translation()
            box_rotation=box_pose.rotation().matrix()

            iiwa_positions_measured = self.get_input_port(1).Eval(context)
            iiwa_velocities_measured = self.get_input_port(2).Eval(context)
            iiwa_torques_measured = self.get_input_port(3).Eval(context)
            iiwa_position_commanded = self.get_input_port(4).Eval(context)
            target_xyz = self.get_input_port(5).Eval(context)
            if control_mode=="EE_pose" or control_mode=="EE_delta_pose":
                actions = self.get_input_port(6).Eval(context)

            #EE pose
            x = self.robot.GetMutablePositionsAndVelocities(
                self.robot_context)
            x[:self.robot.num_positions()] = iiwa_positions_measured
            EE_pose=self.robot.EvalBodyPoseInWorld(
                self.robot_context, self.frame_EE.body())
            EE_xyz=EE_pose.translation()

            #pdb.set_trace()
            observations=np.array([])
            if "state" in self.obs_type:
                observations=np.concatenate((observations,iiwa_positions_measured,iiwa_velocities_measured))

            if "actions" in self.obs_type:
                if control_mode=="joint_positions":
                    observations=np.concatenate((observations,iiwa_position_commanded))
                elif control_mode=="EE_pose":
                    observations=np.concatenate((observations,actions))
                elif control_mode=="EE_delta_pose":
                    observations=np.concatenate((observations,actions))

            if "distances" in self.obs_type:
                distance_EE_to_target=np.linalg.norm(EE_xyz-target_xyz)
                distance_EE_to_box=np.linalg.norm(EE_xyz-box_xyz)
                distance_EE_to_iiwabase=np.linalg.norm(EE_xyz)
                observations=np.concatenate((observations,
                                    [distance_EE_to_target, distance_EE_to_box, distance_EE_to_iiwabase]))

            if "EE_box_target_xyz" in self.obs_type:
                observations=np.concatenate((observations, EE_xyz, box_xyz, target_xyz))

            if "torques" in self.obs_type:
                observations=np.concatenate((observations,iiwa_torques_measured))

            if self.noise:
                observations += np.random.uniform(low=-0.01,
                                                  high=0.01,
                                                  size=observations.shape)

            self.obs_buffer[:-1] = self.obs_buffer[1:]
            self.obs_buffer[-1] = observations

            if debug:
                print(f"obs: {observations}")
                print(f"obs_buffer: {self.obs_buffer.flatten()}")

            output.set_value(self.obs_buffer.flatten())

    obs_pub=builder.AddSystem(
                    observation_publisher(
                        robot=controller_plant,
                        obs_type=observation_type,
                        noise=obs_noise)
                    )
    builder.Connect(station.GetOutputPort("optitrack_manipuland_pose"),obs_pub.get_input_port(0))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),obs_pub.get_input_port(1))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),obs_pub.get_input_port(2))
    builder.Connect(station.GetOutputPort("iiwa_torque_measured"),obs_pub.get_input_port(3))
    builder.Connect(station.GetOutputPort("iiwa_position_commanded"),obs_pub.get_input_port(4))
    if not hardware:
        builder.Connect(target_xyz_ext.get_output_port(),obs_pub.get_input_port(5))
    else:
        target_hardware=builder.AddSystem(ConstantVectorSource(cross_xyz))
        builder.Connect(target_hardware.get_output_port(),obs_pub.get_input_port(5))

    if control_mode=="EE_pose" or control_mode=="EE_delta_pose":
        builder.Connect(action_passthrough.get_output_port(),obs_pub.get_input_port(6))

    builder.ExportOutput(obs_pub.get_output_port(), "observations")

    class RewardSystem(LeafSystem):

        def __init__(self,robot, rew_type):
            LeafSystem.__init__(self)
            self.DeclareAbstractInputPort("manipuland_pose", AbstractValue.Make(RigidTransform.Identity()))
            self.DeclareVectorInputPort("iiwa_positions_measured",Na)
            self.DeclareVectorInputPort("iiwa_velocities_measured",Na)
            self.DeclareVectorInputPort("iiwa_torques_measured",Na)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)
            self.DeclareVectorInputPort("target_xyz",3)

            self.robot=robot
            self.frame_EE = robot.GetFrameByName("iiwa_link_7")
            self.robot_context = robot.CreateDefaultContext()
            self.reward_type=rew_type

        def CalcReward(self, context, output):
            try:
                box_pose = self.get_input_port(0).Eval(context)
            except:
                box_pose=RigidTransform.Identity()  #this might not be safe

            # This assumes the robot base is centered at the world frame.
            box_xyz = box_pose.translation()
            box_rotation=box_pose.rotation().matrix()

            iiwa_positions = self.get_input_port(1).Eval(context)
            iiwa_velocities=self.get_input_port(2).Eval(context)
            iiwa_torques=self.get_input_port(3).Eval(context)
            target_xyz = self.get_input_port(4).Eval(context)

            #EE pose
            x = self.robot.GetMutablePositionsAndVelocities(
                self.robot_context)
            x[:self.robot.num_positions()] = iiwa_positions
            EE_pose=self.robot.EvalBodyPoseInWorld(
                self.robot_context, self.frame_EE.body())
            EE_xyz=EE_pose.translation()

            # Approximation of effort based on positions
            cost_effort = iiwa_positions.dot(iiwa_positions)
            # Approximation of energy based on velocities
            cost_energy = iiwa_velocities.dot(iiwa_velocities)

            # coast to reach the goal
            if task=="reach":
                # Distance EE to box.
                Z_BOX_OFFSET=0.15
                vector_to_goal=EE_xyz-(box_xyz+np.array([0,0,Z_BOX_OFFSET]))
            elif task=="push":
                # Distance box to target.
                vector_to_goal=box_xyz-target_xyz
            else:
                raise ValueError(f"Task {task} not supported.")

            if np.linalg.norm(vector_to_goal)<0.05:
                bonus_success=1
            else:
                bonus_success=0

            # cost of collision with table
            if EE_xyz[2]<0.01:
                cost_collision_w_table=1
            else:
                cost_collision_w_table=0

            reward=0
            if "cost_goal" in self.reward_type:
                cost_goal=vector_to_goal.dot(vector_to_goal)
                reward-= 10*cost_goal
            if "cost_goal_normalized" in self.reward_type:
                MAX_DISTANCE=1.5
                ALPHA=2
                cost_goal_n=np.power(
                                np.linalg.norm(vector_to_goal)/MAX_DISTANCE,
                                ALPHA)
                reward-= 10*cost_goal_n
            if "cost_paper" in self.reward_type:
                MAX_DISTANCE=1.5
                ALPHA=3
                cost_goal_n=np.power(
                                np.linalg.norm(vector_to_goal)/MAX_DISTANCE,
                                ALPHA)
                reward+= 1.0*(1-cost_goal_n)*(1-(context.get_time()/time_limit))
                #pdb.set_trace()
            if "time_decay" in self.reward_type:
                reward-= (context.get_time()/time_limit)
            if "cost_effort" in self.reward_type:
                reward-= cost_effort
            if "bonus_success" in self.reward_type:
                reward+= 10* bonus_success
            if "cost_energy" in self.reward_type:
                reward-= 1e-4*cost_energy
            if "cost_collision" in self.reward_type:
                reward-= 20*cost_collision_w_table

            if debug:
                print(
                    f"EE_xyz: {EE_xyz}, "
                    f"box_xyz: {box_xyz}, "
                    f"target_xyz: {target_xyz}")
                print(
                    f"cost_goal: {cost_goal}, "
                    f"cost_energy: {cost_energy}, "
                    f"cost_effort: {cost_effort}, "
                    f"cost_collision: {cost_collision_w_table}, "
                    f"reward: {reward}")

            #pdb.set_trace()
            output[0] = reward

    reward = builder.AddSystem(
                            RewardSystem(
                            robot= controller_plant,
                            rew_type=reward_type)
                            )

    builder.Connect(station.GetOutputPort("optitrack_manipuland_pose"),reward.get_input_port(0))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),reward.get_input_port(1))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),reward.get_input_port(2))
    builder.Connect(station.GetOutputPort("iiwa_torque_measured"),reward.get_input_port(3))
    if not hardware:
        builder.Connect(target_xyz_ext.get_output_port(),reward.get_input_port(4))
    else:
        builder.Connect(target_hardware.get_output_port(),reward.get_input_port(4))



    builder.ExportOutput(reward.get_output_port(), "reward")

    if not hardware and task=="push":
        # Set random state distributions for target.
        uniform_random_x = Variable(name="uniform_random_x",
                                type=Variable.Type.RANDOM_UNIFORM)
        uniform_random_y = Variable(name="uniform_random_y",
                                type=Variable.Type.RANDOM_UNIFORM)
        #pdb.set_trace()
        target_joint_y = plant.GetJointByName("Cross_Slider_y")
        target_joint_y.set_random_translation_distribution(
            0.5 * (uniform_random_y - 0.5))
        target_joint_x = plant.GetJointByName("Cross_Slider_x")
        target_joint_x.set_random_translation_distribution(
            1+0.5 + (uniform_random_x - 0.5))

    if monitoring_camera:
        # Adds an overhead camera.
        # This is useful for logging videos of rollout evaluation.
        scene_graph.AddRenderer(
            "renderer", MakeRenderEngineVtk(RenderEngineVtkParams()))
        color_camera = ColorRenderCamera(
            RenderCameraCore(
                "renderer",
                CameraInfo(
                    width=640,
                    height=640,
                    fov_y=np.pi/4),
                ClippingRange(0.01, 10.0),
                RigidTransform()
            ), False)
        depth_camera = DepthRenderCamera(color_camera.core(),
                                         DepthRange(0.01, 10.0))
        parent_id = plant.GetBodyFrameIdIfExists(plant.world_body().index())
        X_PB = RigidTransform(RollPitchYaw(np.pi, 0.15, 0),
                              np.array([1.35, 0, 3.7]))
        rgbd_camera = builder.AddSystem(RgbdSensor(parent_id=parent_id,
                                                   X_PB=X_PB,
                                                   color_camera=color_camera,
                                                   depth_camera=depth_camera))
        builder.Connect(station.GetOutputPort("query_object"),
                        rgbd_camera.query_object_input_port())
        builder.ExportOutput(
            rgbd_camera.color_image_output_port(), "color_image")

    diagram = builder.Build()
    simulator = Simulator(diagram)

    if monitoring_camera and control_mode!="EE_delta_pose":
        simulator.set_target_realtime_rate(1.0)

    if debug:
        simulator.set_target_realtime_rate(1)
        #visualize plant and diagram
        if not hardware:
            plt.figure()
            plot_graphviz(plant.GetTopologyGraphvizString())
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.plot(1)
        plt.show(block=False)
        #pdb.set_trace()

    simulator.Initialize()


    # Termination conditions:
    VEL_TOLERANCE=3.0
    vel_low = controller_plant.GetPositionLowerLimits()-VEL_TOLERANCE
    vel_high = controller_plant.GetPositionUpperLimits()+VEL_TOLERANCE

    def monitor(context, state_view=PlantStateView, ter_type=termination_type):
        # terminate from time and box out of reach
        if context.get_time() > time_limit:
            #pdb.set_trace()
            if debug:
                print(f"\nTerminated. Time limit reached at {context.get_time()}")
            return EventStatus.ReachedTermination(diagram, "Episode reached time limit.")

        station_context=diagram.GetMutableSubsystemContext(station,context)
        try:
            box_xyz=station.GetOutputPort("optitrack_manipuland_pose").Eval(station_context).translation()
        except:
            box_xyz=RigidTransform.Identity().translation()

        #EE pose
        controller_plant=station.get_controller_plant()
        robot_context = controller_plant.CreateDefaultContext()
        iiwa_position=station.GetOutputPort("iiwa_position_measured").Eval(station_context)
        x = controller_plant.GetMutablePositionsAndVelocities(robot_context)
        x[:controller_plant.num_positions()] = iiwa_position
        frame_EE_=controller_plant.GetFrameByName("iiwa_link_7")
        EE_pose=controller_plant.EvalBodyPoseInWorld(robot_context, frame_EE_.body())
        EE_xyz=EE_pose.translation()
        if hardware:
            target_xyz=cross_xyz
        else:
            plant=station.get_multibody_plant()
            plant_context = plant.GetMyContextFromRoot(context)
            state = plant.GetOutputPort("continuous_state").Eval(plant_context)
            s=state_view(state)
            target_xyz=np.array([s.Cross_Slider_x_x,s.Cross_Slider_y_x,table_heigth])

        if "box_off_table" in ter_type:
            if box_xyz[0]<0.2 or box_xyz[2]<0.0 or box_xyz[0]>2.2 or np.abs(box_xyz[1])>1.0:
                if debug:
                    print(f"\nTerminated. Box off the table. Box pose: {box_xyz}")
                return EventStatus.ReachedTermination(diagram, "Box off the table.")

        if "success" in ter_type:
            if task=="reach":
                distance_to_goal=box_xyz-EE_xyz
            elif task=="push":
                distance_to_goal=box_xyz-target_xyz

            if np.linalg.norm(distance_to_goal)<0.05:
                if debug:
                    print("\nTerminated. Success. Goal reachead.\n")
                return EventStatus.ReachedTermination(diagram, "Success. Goal reachead.")

        if "collision_w_table" in ter_type:
            if EE_xyz[2]<0.005+table_heigth:
                if debug:
                    print(f"\nTerminated. EE collided with table. EE_xyz: {EE_xyz}")
                return EventStatus.ReachedTermination(diagram, "EE collided with table")

        if "velocity_limits" in ter_type:
            #pdb.set_trace()
            velocities=station.GetOutputPort("iiwa_velocity_estimated").Eval(station_context)
            #print("vel: ",velocities)
            if np.any(velocities<vel_low) or np.any(velocities>vel_high):
                #pdb.set_trace()
                if debug:
                    print(f"\nTerminated. Outside joint velocity limits. vel: {velocities}")
                return EventStatus.ReachedTermination(diagram, "Terminated. Outside joint velocity limits.")


        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    return simulator


def set_home(simulator, diagram_context, set_type=["home"]):

    home_positions = []
    home_velocities = []
    home_body_mass_offset = []
    home_u_friction = []
    home_free_body_pose= []

    if "home" in set_type:
        home_positions = [
            ('iiwa_joint_1',0.0),
            ('iiwa_joint_2',0.0),
            ('iiwa_joint_3',0.0),
            ('iiwa_joint_4',0.0),
            ('iiwa_joint_5',0.0),
            ('iiwa_joint_6',0.0),
            ('iiwa_joint_7',0.0),
        ]
        home_free_body_pose= [
            #rpyxyz
            ('box','base_link',[0,0,0,1,0,0.03])
        ]
    elif "random_positions" in set_type:
        # Randomize the initial position of the joints.
        home_positions = [
            ('iiwa_joint_1', np.random.uniform(low=-1.0, high=1.0)),
            ('iiwa_joint_2', np.random.uniform(low=-0.8, high=0.8)),
            ('iiwa_joint_3', np.random.uniform(low=-1.0, high=1.0)),
            ('iiwa_joint_4', np.random.uniform(low=-1.5, high=1.0)),
            ('iiwa_joint_5', np.random.uniform(low=-1.5, high=1.5)),
            ('iiwa_joint_6', np.random.uniform(low=-1.5, high=1.5)),
            ('iiwa_joint_7', np.random.uniform(low=-2.0, high=2.0)),
        ]
        home_free_body_pose= [
            #rpyxyz
            ('box',
            'base_link',
            [0,
            0,
            np.random.uniform(low=0, high=2*np.pi),
            np.random.uniform(low=0.3, high=1.0),
            np.random.uniform(low=-0.5, high=0.5),
            0.03])
            ]
    elif "random_positions_limited" in set_type:
        # Randomize the initial position of the joints.
        home_positions = [
            ('iiwa_joint_1', np.random.uniform(low=-1.0, high=1.0)),
            ('iiwa_joint_2', np.random.uniform(low=0.0, high=0.8)),
            ('iiwa_joint_3', np.random.uniform(low=-0.2, high=0.2)),
            ('iiwa_joint_4', np.random.uniform(low=-1.5, high=0.0)),
            ('iiwa_joint_5', np.random.uniform(low=-1.0, high=1.0)),
            ('iiwa_joint_6', np.random.uniform(low=-1.0, high=1.0)),
            ('iiwa_joint_7', np.random.uniform(low=-2.0, high=2.0)),
        ]
        home_free_body_pose= [
            #rpyxyz
            ('box',
            'base_link',
            [0,
            0,
            np.random.uniform(low=0, high=2*np.pi),
            np.random.uniform(low=0.3, high=1.0),
            np.random.uniform(low=-0.5, high=0.5),
            0.03])
            ]
    elif "random_positions_diffik" in set_type:
        # Randomize the initial position of the joints.
        home_positions = [
            ('iiwa_joint_1', np.random.uniform(low=-0.5, high=0.5)),
            ('iiwa_joint_2', np.random.uniform(low=0.2, high=0.5)),
            ('iiwa_joint_3', np.random.uniform(low=-0.2, high=0.2)),
            ('iiwa_joint_4', np.random.uniform(low=-2, high=-1.5)),
            ('iiwa_joint_5', np.random.uniform(low=-1.0, high=1.0)),
            ('iiwa_joint_6', np.random.uniform(low=.0, high=0.1)),
            ('iiwa_joint_7', np.random.uniform(low=-.1, high=.1)),
        ]

        #sample a point inside the circular workspace of iiwa
        r = 0.8 * np.sqrt(np.random.random())
        theta = np.random.uniform(low=-np.pi/2, high=np.pi/2)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        home_free_body_pose= [
            #rpyxyz
            ('box',
            'base_link',
            [0,
            0,
            np.random.uniform(low=0, high=2*np.pi),
            x, #np.random.uniform(low=0.3, high=1.0),
            y, #np.random.uniform(low=-0.5, high=0.5),
            0.03])
            ]

    if "random_target_position" in set_type:
        # Randomize the initial position of the joints.
        home_positions = [
            ('Cross_Slider_x', np.random.uniform(low=1.1, high=1.5)),
            ('Cross_Slider_y', np.random.uniform(low=-0.6, high=0.6)),
        ]

    if "random_mass" in set_type:
        # Randomize the mass of a body by adding a mass offset.
        # (instance_name, body_name, value)
        home_body_mass_offset = [
            ('box', 'base_link',
                np.random.uniform(low=-0.05, high=0.05))
        ]

    if "random_friction" in set_type:
        # Randomize the mass the friction of a body.
        # (instance_name, body_name, value)
        home_u_friction = [
            ('table', 'table', np.random.uniform(low=0.1, high=1.0))
        ]

    diagram = simulator.get_system()
    station = diagram.GetSubsystemByName("rl_cito_station")

    # set random states
    plant = station.get_multibody_plant()
    plant_context = diagram.GetMutableSubsystemContext(plant,
                                                       diagram_context)
    #pdb.set_trace()
    scene_graph = station.get_scene_graph()#diagram.GetSubsystemByName("scene_graph")
    scene_graph_context = diagram.GetMutableSubsystemContext(
        scene_graph, diagram_context)

    for pair in home_positions:
        joint = plant.GetJointByName(pair[0])
        if joint.type_name()=="revolute":
            joint.set_angle(plant_context,
                        np.clip(pair[1],
                            joint.position_lower_limit(),
                            joint.position_upper_limit()
                            )
                        )
        if joint.type_name()=="prismatic":
            joint.set_translation(plant_context,
                        np.clip(pair[1],
                            joint.position_lower_limit(),
                            joint.position_upper_limit()
                            )
                        )
    for pair in home_velocities:
        joint = plant.GetJointByName(pair[0])
        if joint.type_name() == "revolute":
            joint.set_angular_rate(plant_context,
                                       np.clip(pair[1],
                                               joint.velocity_lower_limit(),
                                               joint.velocity_upper_limit()
                                               )
                                       )

    for pair in home_body_mass_offset:
        instance = plant.GetModelInstanceByName(pair[0])
        body = plant.GetBodyByName(name=pair[1], model_instance=instance)
        mass = body.get_mass(plant.CreateDefaultContext())
        body.SetMass(plant_context, mass+pair[2])

    for pair in home_u_friction:
        instance = plant.GetModelInstanceByName(pair[0])
        body = plant.GetBodyByName(name=pair[1], model_instance=instance)
        geom_id = plant.GetCollisionGeometriesForBody(body)[0]
        props = scene_graph.model_inspector().GetProximityProperties(
            geom_id)
        props.UpdateProperty("material",
                             "coulomb_friction",
                             CoulombFriction(pair[2], pair[2]))

        scene_graph.AssignRole(context=scene_graph_context,
                               source_id=plant.get_source_id(),
                               geometry_id=geom_id,
                               properties=props,
                               assign=RoleAssign.kReplace
                               )

    for pair in home_free_body_pose:
        instance = plant.GetModelInstanceByName(pair[0])
        body = plant.GetBodyByName(name=pair[1],model_instance=instance)
        rpy=pair[2][:3]
        xyz=pair[2][3:]
        body_pose = RigidTransform(
                RollPitchYaw(rpy[0],rpy[1],rpy[2]),
                np.array(xyz)
                )
        plant.SetFreeBodyPose(plant_context,body,body_pose)

    #pdb.set_trace()
    # init diffIK with updated state

    differential_ik = diagram.GetSubsystemByName("diffIK")
    filter = diagram.GetSubsystemByName("diffIK_filter")
    #differential_ik.Reset()
    station_context=diagram.GetMutableSubsystemContext(station,diagram_context)
    q0 = station.GetOutputPort("iiwa_position_measured").Eval(
        station_context)
    differential_ik.parameters.set_nominal_joint_position(q0)
    diff_ik_context=diagram.GetMutableSubsystemContext(differential_ik, simulator.get_context())#diagram_context)
    differential_ik.SetPositions(diff_ik_context, q0)
    filter_context=diagram.GetMutableSubsystemContext(filter, simulator.get_context())#diagram_co
    EE_init_pose=differential_ik.ForwardKinematics(q0)
    EE_init_rpy=RollPitchYaw(EE_init_pose.rotation())
    EE_rpyxyz=np.concatenate((
        np.array(
            [EE_init_rpy.roll_angle(),
            EE_init_rpy.pitch_angle(),
            EE_init_rpy.yaw_angle()]),
        EE_init_pose.translation()))
    filter.set_initial_output_value(filter_context,EE_rpyxyz)

def RlCitoStationBoxPushingEnv(meshcat=None,
                        time_limit=gym_time_limit,
                        debug=False,
                        hardware=False,
                        task="reach",
                        mock_hardware=False,
                        obs_noise=False,
                        monitoring_camera=False,
                        add_disturbances=False,
                        observation_type=["state"],
                        reward_type=["sparse"],
                        termination_type=["box_off_table","collision_w_table"],
                        reset_type=["home"],
                        control_mode=["joint_positions"]):

    #Make simulation
    simulator = make_sim(RandomGenerator(),
                         meshcat=meshcat,
                         time_limit=time_limit,
                         debug=debug,
                         hardware=hardware,
                         task=task,
                         mock_hardware=mock_hardware,
                         obs_noise=obs_noise,
                         monitoring_camera=monitoring_camera,
                         add_disturbances=add_disturbances,
                         observation_type=observation_type,
                         reward_type=reward_type,
                         termination_type=termination_type,
                         control_mode=control_mode)
    #pdb.set_trace()
    if hardware:
        station = simulator.get_system().GetSubsystemByName("rl_cito_station_hardware_interface")
    else:
        station = simulator.get_system().GetSubsystemByName("rl_cito_station")




    plant=station.get_controller_plant()
    #Define Action space
    Na=plant.num_actuators()
    if control_mode=="joint_positions":
        low_a = plant.GetPositionLowerLimits()
        high_a = plant.GetPositionUpperLimits()
    elif control_mode=="EE_pose":
        #rpyxyz
        low_a=np.array([np.pi-0.005,-0.05,-0.05,0.2,-0.7,0.05]) #0.05 rpy -np.pi 0.2 -0.01
        high_a=np.array([np.pi+0.005,0.05,0.05,0.8,0.7,0.6])
    elif control_mode=="EE_delta_pose":
        #rpyxyz
        factor_a=6
        low_a=factor_a*np.array([-1e-9,-1e-9,-1e-9,-0.025,-0.025,-0.025])
        high_a=factor_a*np.array([1e-9,1e-9,1e-9,0.025,0.025,0.025])

    action_space = gym.spaces.Box(
        low=np.asarray(low_a, dtype="float64"),
        high=np.asarray(high_a, dtype="float64"),
        dtype=np.float64)

    #Define observation space
    low=np.array([])
    high=np.array([])
    POSITION_LIMIT_TOLERANCE = np.full((Na,), 0.2)
    VELOCITY_LIMIT_TOLERANCE = np.full((Na,), 20)

    if "state" in observation_type:
        low = np.concatenate((low,np.concatenate(
            (plant.GetPositionLowerLimits()-POSITION_LIMIT_TOLERANCE,
            plant.GetVelocityLowerLimits()-VELOCITY_LIMIT_TOLERANCE))))
        high = np.concatenate((high,np.concatenate(
            (plant.GetPositionUpperLimits()+POSITION_LIMIT_TOLERANCE,
            plant.GetVelocityUpperLimits()+VELOCITY_LIMIT_TOLERANCE))))

    if "actions" in observation_type:
        if control_mode=="joint_positions":
            low = np.concatenate((low,
                        plant.GetPositionLowerLimits()-POSITION_LIMIT_TOLERANCE))
            high = np.concatenate((high,
                        plant.GetPositionUpperLimits()+POSITION_LIMIT_TOLERANCE))
        elif control_mode=="EE_pose":
            ACTUATION_LIMIT_TOLERANCE=np.array([0.1,0.1,0.1,0.5,0.5,0.5])
            low=np.concatenate((low,
                np.array([-2*np.pi,-2*np.pi,-2*np.pi,0.2,-0.7,0])-ACTUATION_LIMIT_TOLERANCE))
            high=np.concatenate((high,
                np.array([2*np.pi,2*np.pi,2*np.pi,0.8,0.7,0.6])+ACTUATION_LIMIT_TOLERANCE))
        elif control_mode=="EE_delta_pose":
            ACTUATION_LIMIT_TOLERANCE=np.array([0.1]*6)
            low=np.concatenate((low,
                factor_a*np.array([-0.1,-0.1,-0.1,-0.025,-0.025,-0.025])-ACTUATION_LIMIT_TOLERANCE))
            high=np.concatenate((high,
                factor_a*np.array([0.1,0.1,0.1,0.025,0.025,0.025])+ACTUATION_LIMIT_TOLERANCE))

    if "distances" in observation_type:
        low = np.concatenate((low,np.array([-10]*3)))
        high = np.concatenate((high,np.array([10]*3)))

    if "EE_box_target_xyz" in observation_type:
        low = np.concatenate((low,np.array([-10]*9)))
        high = np.concatenate((high,np.array([10]*9)))

    if "torques" in observation_type:
        ACTUATION_LIMIT_TOLERANCE = np.full((Na,), 300)
        low = np.concatenate((low,
                    plant.GetEffortLowerLimits()-ACTUATION_LIMIT_TOLERANCE))
        high = np.concatenate((high,
                    plant.GetEffortUpperLimits()+ACTUATION_LIMIT_TOLERANCE))

    if "buffer_10" in observation_type:
        low = np.tile(low, 10)
        high = np.tile(high, 10)
    elif "buffer_20" in observation_type:
        low = np.tile(low, 20)
        high = np.tile(high, 20)

    #pdb.set_trace()
    observation_space = gym.spaces.Box(low=np.asarray(low, dtype="float64"),
                                       high=np.asarray(high, dtype="float64"),
                                       dtype=np.float64)

    # parse the reset type to set_home()
    reset_handler = partial(set_home, set_type=reset_type)

    env = DrakeGymEnv(simulator=simulator,
                      time_step=gym_time_step,
                      action_space=action_space,
                      observation_space=observation_space,
                      reward="reward",
                      action_port_id="actions",
                      observation_port_id="observations",
                      set_home=reset_handler,
                      hardware=hardware,
                      render_rgb_port_id="color_image" if monitoring_camera else None)


    return env
