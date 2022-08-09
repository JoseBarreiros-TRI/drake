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
from scenarios import AddShape, SetColor, SetTransparency
from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewState,
        MakeNamedViewActuation)
import pydrake.geometry as mut

from pydrake.examples import (
    CreateClutterClearingYcbObjectList, ManipulationStation,
    ManipulationStationHardwareInterface)

## Gym parameters
sim_time_step=0.005
gym_time_step=0.05
#controller_time_step=0.01
gym_time_limit=5
modes=["IDC","torque"]
control_mode=modes[0]
table_heigth =0.15
box_size=[ 0.2,#0.2+0.1*(np.random.random()-0.5),
        0.2,#0.2+0.1*(np.random.random()-0.5),
         0.2,   #0.2+0.1*(np.random.random()-0.5),
        ]
# box_mass=1
# box_mu=1.0
contact_model='point'#'hydroelastic_with_fallback'#ContactModel.kHydroelasticWithFallback#kPoint
contact_solver='tamsi'#ContactSolver.kSap#kTamsi # kTamsi
desired_box_xy=[
    1.+0.5*(np.random.random()-0.5),
    0+0.8*(np.random.random()-0.5),
    ] 
##

def AddFloor(plant):
    parser = Parser(plant)
    floor = parser.AddModelFromFile(FindResource("models/floor_v3.sdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("floor", floor),
        RigidTransform(RollPitchYaw(0, 0, 0),
                        np.array([0, 0, 0.0]))
                    )
    return floor

def AddTargetPosVisuals(plant,xyz_position,color=[.8, .1, .1, 1.0]):
    parser = Parser(plant)
    marker = parser.AddModelFromFile(FindResource("models/cross.sdf"))
    plant.WeldFrames(
        plant.world_frame(), plant.GetFrameByName("cross", marker),
        RigidTransform(RollPitchYaw(0, 0, 0),
                        np.array(xyz_position)
                    )
    )


def make_sim(generator,
                    observations="state",
                    meshcat=None,
                    time_limit=5,
                    debug=False,
                    hardware=False):
    
    builder = DiagramBuilder()
 
    target_position=[desired_box_xy[0],desired_box_xy[1],table_heigth]

  
    if hardware:
        camera_ids = []
        station = builder.AddSystem(ManipulationStationHardwareInterface(
            camera_ids,False, True))
        station.Connect(wait_for_cameras=False,wait_for_wsg=False,wait_for_optitrack=False)     
        controller_plant=station.get_controller_plant()
        plant=None   
    else:
        station = builder.AddSystem(ManipulationStation(time_step=sim_time_step,contact_model=contact_model,contact_solver=contact_solver))
        station.SetupCitoRlStation()

        station.AddManipulandFromFile(
                "drake/drake_gym/models/"
                + "optitrack_brick.sdf",
                RigidTransform(RotationMatrix.Identity(), [0.6, 0, 0.15]),
                "box")
        
        controller_plant=station.get_controller_plant()
        plant=station.get_multibody_plant()
        scene_graph=station.get_scene_graph()

        #box = AddBox(plant)
        AddTargetPosVisuals(plant,target_position)
        station.Finalize()

        if meshcat:
            geometry_query_port = station.GetOutputPort("geometry_query")
            meshcat_visualizer = MeshcatVisualizer.AddToBuilder(
                builder=builder,
                query_object_port=geometry_query_port,
                meshcat=meshcat)

            # MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)
            # ContactVisualizer.AddToBuilder(
            #     builder, plant, meshcat,
            #     ContactVisualizerParams(radius=0.005, newtons_per_meter=500.0))

            # # Use the controller plant to visualize the set point geometry.
            # controller_scene_graph = builder.AddSystem(SceneGraph())
            # controller_plant.RegisterAsSourceForSceneGraph(controller_scene_graph)
            # SetColor(controller_scene_graph,
            #         color=[1.0, 165.0 / 255, 0.0, 1.0],
            #         source_id=controller_plant.get_source_id())
            # controller_vis = MeshcatVisualizerCpp.AddToBuilder(
            #     builder, controller_scene_graph, meshcat,
            #     MeshcatVisualizerParams(prefix="controller"))
            # controller_vis.set_name("controller meshcat")

        # plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)
        # # filter collisison between parent and child of each joint.
        # add_collision_filters(scene_graph,plant)


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

    # if meshcat:
    #     positions_to_poses = builder.AddSystem(
    #         MultibodyPositionToGeometryPose(controller_plant))
    #     builder.Connect(
    #         positions_to_poses.get_output_port(),
    #         controller_scene_graph.get_source_pose_port(
    #             controller_plant.get_source_id()))


    # actions = builder.AddSystem(PassThrough(Na))
    # builder.Connect(actions.get_output_port(),
    #                    station.GetInputPort("iiwa_position"))    

    if hardware:
        action_passthrough=builder.AddSystem(PassThrough(Na))
        builder.Connect(action_passthrough.get_output_port(),station.GetInputPort("iiwa_position"))
        builder.ExportInput(action_passthrough.get_input_port(),"actions")
    else:
        builder.ExportInput(station.GetInputPort("iiwa_position"),"actions")


    class observation_publisher(LeafSystem):

        def __init__(self,robot,frame_EE):
            LeafSystem.__init__(self)
            #Nss = plant.num_multibody_states()
            self.DeclareAbstractInputPort("manipuland_pose", AbstractValue.Make(RigidTransform.Identity()))
            self.DeclareVectorInputPort("iiwa_position",Na)
            self.DeclareVectorInputPort("iiwa_velocities",Na)
            self.DeclareVectorInputPort("iiwa_torques",Na)
            #self.DeclareVectorOutputPort("observations", Na*2, self.CalcObs)
            self.DeclareVectorOutputPort("observations", Na*2+6, self.CalcObs)
            self.desired_box_pose=np.array([desired_box_xy[0],desired_box_xy[1],box_size[2]/2+table_heigth])

            self.robot = robot
            self.frame_EE = frame_EE
            self.robot_context = robot.CreateDefaultContext()

        def CalcObs(self, context,output):
            try:
                box_pose = self.get_input_port(0).Eval(context)
            except:
                box_pose=RigidTransform.Identity()  #this might not be safe

            iiwa_position = self.get_input_port(1).Eval(context)
            iiwa_velocities = self.get_input_port(2).Eval(context)
            iiwa_torques = self.get_input_port(3).Eval(context)
   
            box_translation = box_pose.translation()
            box_rotation=box_pose.rotation().matrix()

            #EE pose
            x = self.robot.GetMutablePositionsAndVelocities(
                self.robot_context)
            x[:self.robot.num_positions()] = iiwa_position
            EE_pose=self.robot.EvalBodyPoseInWorld(
                self.robot_context, self.frame_EE.body())
            #pdb.set_trace()

            #pose of the middle point of the farthest edge of the box
            box_LF_edge= box_rotation.dot(np.array([box_translation[0]+box_size[0]/2,box_translation[1]+box_size[1]/2,box_translation[2]]))
            box_RF_edge= box_rotation.dot(np.array([box_translation[0]-box_size[0]/2,box_translation[1]+box_size[1]/2,box_translation[2]]))
                        #pose of the middle point of the closest edge of the box
            box_LC_edge= box_rotation.dot(np.array([box_translation[0]+box_size[0]/2,box_translation[1]-box_size[1]/2,box_translation[2]]))
            box_RC_edge= box_rotation.dot(np.array([box_translation[0]-box_size[0]/2,box_translation[1]-box_size[1]/2,box_translation[2]]))
            distance_box_to_target=self.desired_box_pose-box_translation
            distance_EE_to_box=EE_pose.translation()-box_translation
            #pdb.set_trace()
            observations=np.concatenate((iiwa_position,iiwa_velocities))
            #extension=np.concatenate((box_LF_edge,box_RF_edge,box_LC_edge,box_RC_edge,distance_box_to_target))
            extension=np.concatenate((distance_box_to_target,distance_EE_to_box))
            
            extended_observations=np.concatenate((observations,extension))      
            output.set_value(extended_observations)
            #output.set_value(observations)


    obs_pub=builder.AddSystem(observation_publisher(controller_plant,controller_plant.GetFrameByName("iiwa_link_7")))


    builder.Connect(station.GetOutputPort("optitrack_manipuland_pose"),obs_pub.get_input_port(0))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),obs_pub.get_input_port(1))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),obs_pub.get_input_port(2))
    builder.Connect(station.GetOutputPort("iiwa_torque_measured"),obs_pub.get_input_port(3))

    builder.ExportOutput(obs_pub.get_output_port(), "observations")

    class RewardSystem(LeafSystem):

        def __init__(self,robot,frame_EE):
            LeafSystem.__init__(self)
            self.DeclareAbstractInputPort("manipuland_pose", AbstractValue.Make(RigidTransform.Identity()))
            self.DeclareVectorInputPort("iiwa_position",Na)
            self.DeclareVectorInputPort("iiwa_velocities",Na)
            self.DeclareVectorInputPort("iiwa_torques",Na)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

            self.desired_box_pose=np.array([desired_box_xy[0],desired_box_xy[1],box_size[2]/2+table_heigth])

            self.robot = robot
            self.frame_EE = frame_EE
            self.robot_context = robot.CreateDefaultContext()

        def CalcReward(self, context, output):
            try:
                body_pose = self.get_input_port(0).Eval(context)
            except:
                body_pose=RigidTransform.Identity()  #this might not be safe

            iiwa_position = self.get_input_port(1).Eval(context)
            

            box_translation = body_pose.translation()
            box_rotation=body_pose.rotation().matrix()
            box_euler=R.from_dcm(box_rotation).as_euler('zyx', degrees=False)
            #pdb.set_trace()

            #EE pose
            x = self.robot.GetMutablePositionsAndVelocities(
                self.robot_context)
            x[:self.robot.num_positions()] = iiwa_position
            EE_pose=self.robot.EvalBodyPoseInWorld(
                self.robot_context, self.frame_EE.body())


            distance_box_to_target=self.desired_box_pose-box_translation
            distance_EE_to_box=EE_pose.translation()-box_translation

            cost_EE=distance_EE_to_box.dot(distance_EE_to_box)
            cost_to_target=distance_box_to_target.dot(distance_box_to_target) 

            cost = cost_EE + 10*cost_to_target

            reward=5-cost
       
            if debug:
                print('box_pose: ',box_translation)
                print("EE: ", EE_pose.translation())

                print('cost: {c}, cost_EE: {ce}, cost_to_target: {ct}'.format(
                        c=cost,
                        ce=cost_EE,
                        ct=cost_to_target,
                        ))
                print('rew: {r}\n'.format(r=reward))
            #pdb.set_trace()

            output[0] = reward


    reward = builder.AddSystem(RewardSystem(controller_plant,controller_plant.GetFrameByName("iiwa_link_7")))

    builder.Connect(station.GetOutputPort("optitrack_manipuland_pose"),reward.get_input_port(0))
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),reward.get_input_port(1))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),reward.get_input_port(2))
    builder.Connect(station.GetOutputPort("iiwa_torque_measured"),reward.get_input_port(3))
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
        #pdb.set_trace()

    simulator.Initialize()

    # Termination conditions:
    def monitor(context):

        # terminate from time and box out of reach
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")
        if not hardware:
            #pdb.set_trace()
            station_context=diagram.GetMutableSubsystemContext(station,context)
            box_pose=station.GetOutputPort("optitrack_manipuland_pose").Eval(station_context).translation()
            #print("b_pose: ", box_pose)
            if np.linalg.norm(box_pose)>1.4 or box_pose[0]<0.2 or box_pose[2]<0.1:
                #pdb.set_trace()
                return EventStatus.ReachedTermination(diagram, "box out of reach")
        
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    return simulator


def ManipulationStationBoxPushingEnv(observations="state", meshcat=None, time_limit=gym_time_limit, debug=False,hardware=False):
    
    #Make simulation
    simulator = make_sim(RandomGenerator(),
                            observations,
                            meshcat=meshcat,
                            time_limit=time_limit,
                            debug=debug,
                            hardware=hardware)
    #pdb.set_trace()
    if hardware:
        station = simulator.get_system().GetSubsystemByName("manipulation_station_hardware_interface")
    else:
        station = simulator.get_system().GetSubsystemByName("manipulation_station")
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
        (np.array([-np.pi]*Na), np.array([-100]*Na),np.array([-np.inf]*6)))
    high = np.concatenate(
        (np.array([np.pi]*Na), np.array([100]*Na),np.array([np.inf]*6)))

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
                      set_home=None)
    return env
