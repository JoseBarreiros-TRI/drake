from ast import excepthandler
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
    RotationMatrix,
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

)

from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from drake_gym import DrakeGymEnv
from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewState,
        MakeNamedViewActuation)
import pydrake.geometry as mut

from pydrake.examples import (
    RlCitoStation,
    RlCitoStationHardwareInterface)

## Gym parameters
sim_time_step=0.005
gym_time_step=0.05
gym_time_limit=5
# modes=["IDC","torque"]
# control_mode=modes[0]
table_heigth =0.
box_size=[ 0.075,#0.2+0.1*(np.random.random()-0.5),
        0.05,#0.2+0.1*(np.random.random()-0.5),
         0.05,   #0.2+0.1*(np.random.random()-0.5),
        ]
contact_model='point'#'hydroelastic_with_fallback'#ContactModel.kHydroelasticWithFallback#kPoint
contact_solver='sap'#ContactSolver.kSap#kTamsi # kTamsi

##
optitrack_pose_transform=np.array([[1,0,0],[0,1,0],[0,0,1]])

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
             termination_type=[]):                
    
    assert(task=="reach" or task=="push"),f'_{task}_ task not implemented. valid options are push, reach'
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
                # "drake/drake_gym/models/"
                + "optitrack_brick_v2.sdf",
                RigidTransform(RotationMatrix.Identity(), [0.6, 0, table_heigth]),
                "box")
        
        controller_plant=station.get_controller_plant()
        plant=station.get_multibody_plant()
        scene_graph=station.get_scene_graph()
        AddTargetVisual(plant)
        station.Finalize()

   
        class target_xyz_extractor(LeafSystem):
            def __init__(self):
                LeafSystem.__init__(self)
                Ns=plant.num_multibody_states()
                self.DeclareVectorInputPort("target_state",Ns)
                self.DeclareVectorOutputPort("target_xyz", 3, self.Extract_xyz)
                self.state_view=MakeNamedViewState(plant, "States")
            
            def Extract_xyz(self, context, output):
                state = self.get_input_port(0).Eval(context)
                s=self.state_view(state)
                pdb.set_trace()
                xyz=[s,s,table_heigth]
                output.set_value(xyz)
        
        target_xyz_ext=builder.AddSystem(target_xyz_extractor())
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
        action_passthrough=builder.AddSystem(PassThrough(Na))
        builder.Connect(action_passthrough.get_output_port(),station.GetInputPort("iiwa_position"))
        builder.ExportInput(action_passthrough.get_input_port(),"actions")
    else:
        builder.ExportInput(station.GetInputPort("iiwa_position"),"actions")


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
            if "actions" in self.obs_type:
                # Obs: commanded_positions
                self.obs_size += self.Na

            if "state" in self.obs_type:
                # Obs: commanded_positions and velocities
                self.obs_size += 2*self.Na
            
            if "distances" in self.obs_type:
                # Obs: distance_EE_to target, distance_EE_to_box, distance_EE_to_iiwabase.
                self.obs_size += 3
            
            if "EE_box_target_xyz" in self.obs_type:
                # Obs: xyz position in the robot base frame for EE, target, box.
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
            
            #EE pose
            x = self.robot.GetMutablePositionsAndVelocities(
                self.robot_context)
            x[:self.robot.num_positions()] = iiwa_positions_measured
            EE_pose=self.robot.EvalBodyPoseInWorld(
                self.robot_context, self.frame_EE.body())
            EE_xyz=EE_pose.translation()

            observations=np.array([])
            if "state" in self.obs_type:
                observations=np.concatenate((observations,iiwa_positions_measured,iiwa_velocities_measured))

            if "actions" in self.obs_type:
                observations=np.concatenate((observations,iiwa_position_commanded))

            if "distances" in self.obs_type:
                distance_EE_to_target=np.norm(EE_xyz-target_xyz)
                distance_EE_to_box=np.norm(EE_xyz-box_xyz)
                distance_EE_to_iiwabase=np.norm(EE_xyz)
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
    builder.Connect(target_xyz_ext.get_output_port(),obs_pub.get_input_port(5))

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
            cost_effort = 0.5*iiwa_positions*iiwa_positions
            # Approximation of energy based on velocities
            cost_energy = iiwa_velocities*iiwa_velocities

            # coast to reach the goal
            if task=="reach":
                # Distance EE to box.
                distance_to_goal=EE_xyz-box_xyz
            elif task=="push":
                # Distance box to target.
                distance_to_goal=box_xyz-target_xyz      
            else:
                raise ValueError(f"Task {task} not supported.")

            cost_goal=distance_to_goal.dot(distance_to_goal)

            # cost of collision with table
            if EE_xyz<0.01:
                cost_collision_w_table=2
            else:
                cost_collision_w_table=0

            reward=0
            if "cost_goal" in self.reward_type:
                reward-= cost_goal
            if "cost_effort" in self.reward_type:
                reward-= cost_effort
            if "cost_energy" in self.reward_type:
                reward-= cost_energy                          
            if "cost_collision" in self.reward_type:
                reward-= cost_collision_w_table        

            if debug:
                print(
                    f"EE_xyz: {EE_xyz},"
                    "box_xyz: {box_xyz},"
                    "target_xyz: {target_xyz}")
                print(
                    f"cost_goal: {cost_goal},"
                    "cost_energy: {cost_energy},"
                    "cost_effort: {cost_effort},"
                    "cost_collision: {cost_collision},"
                    "reward: {reward}")

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
    builder.Connect(target_xyz_ext.get_output_port(),reward.get_input_port(4))

    builder.ExportOutput(reward.get_output_port(), "reward")
    
    diagram = builder.Build()
    simulator = Simulator(diagram)
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
        pdb.set_trace()

    simulator.Initialize()

    # Termination conditions:
    def monitor(context, ter_type=termination_type):

        # terminate from time and box out of reach
        if context.get_time() > time_limit:
            if debug:
                print(f"Terminated. Time limit reached at {context.get_time()}")
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

            if np.linalg.norm(distance_to_goal)<0.15:
                if debug:
                    print("\nTerminated. Success. Goal reachead.\n")
                return EventStatus.ReachedTermination(diagram, "Success. Goal reachead.")    

        if "collison_w_table" in ter_type:
            if EE_xyz[2]<0.005+table_heigth:
                if debug:
                    print(f"Terminated. EE collided with table. EE_xyz: {EE_xyz}")
                return EventStatus.ReachedTermination(diagram, "EE collided with table")

        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    return simulator


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
                        termination_type=["out_of_range"],
                        reset_type=["home"]):
    
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
                         termination_type=termination_type)
    #pdb.set_trace()
    if hardware:
        station = simulator.get_system().GetSubsystemByName("rl_cito_station_hardware_interface")
    else:
        station = simulator.get_system().GetSubsystemByName("rl_cito_station")
    plant=station.get_controller_plant()
    
    #Define Action space
    Na=plant.num_actuators()
    low = plant.GetPositionLowerLimits()[:Na]
    high = plant.GetPositionUpperLimits()[:Na]
    #pdb.set_trace()
    # StateView=MakeNamedViewState(plant, "States")
    # PositionView=MakeNamedViewPositions(plant, "Position")
    # ActuationView=MakeNamedViewActuation(plant, "Actuation")
    action_space = gym.spaces.Box(low=np.asarray(low, dtype="float64"), high=np.asarray(high, dtype="float64"),dtype=np.float64)
     
    #Define observation space 
    # low = np.concatenate(
    #     (plant.GetPositionLowerLimits(), plant.GetVelocityLowerLimits(),np.array([-np.inf]*6)))
    # high = np.concatenate(
    #     (plant.GetPositionUpperLimits(), plant.GetVelocityUpperLimits(),np.array([np.inf]*6)))

    low = np.concatenate(
        (np.array([-2*np.pi]*Na), np.array([-100]*Na),np.array([-10]*9)))
    high = np.concatenate(
        (np.array([2*np.pi]*Na), np.array([100]*Na),np.array([10]*9)))

    # low = np.concatenate(
    #     (plant.GetPositionLowerLimits(), plant.GetVelocityLowerLimits()))
    # high = np.concatenate(
    #     (plant.GetPositionUpperLimits(), plant.GetVelocityUpperLimits()))


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
                      set_home=None,
                      hardware=hardware)
    return env
