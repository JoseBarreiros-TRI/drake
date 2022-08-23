from re import S
import gym
import pdb
import numpy as np

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    EventStatus,
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
    AddMultibodyPlant,
    MultibodyPlantConfig,
    FindResourceOrThrow,
    ColorRenderCamera,
    RenderCameraCore,
    ClippingRange,
    DepthRenderCamera,
    DepthRange,
    CameraInfo,
    RgbdSensor,
    RollPitchYaw,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    
)
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from drake_gym import DrakeGymEnv
from scenarios import SetColor
from utils import (MakeNamedViewPositions, 
                MakeNamedViewState,
                MakeNamedViewActuation)
import pydrake.geometry as mut


## Gym parameters
sim_time_step=0.01
gym_time_step=0.05
controller_time_step=0.01
gym_time_limit=5
drake_contact_models=['point','hydroelastic_with_fallback']
contact_model=drake_contact_models[0]
drake_contact_solvers=['sap','tamsi']
contact_solver=drake_contact_solvers[0]

def AddAgent(plant):
    parser = Parser(plant)
    model_file = FindResourceOrThrow("drake/drake_gym/models/cartpole_BSA.sdf")
    agent = parser.AddModelFromFile(model_file)
    return agent

def make_sim(generator,
            meshcat=None,
            time_limit=5,
            debug=False,
            obs_noise=False,
            monitoring_camera=False):
    
    builder = DiagramBuilder()
    
    multibody_plant_config = \
        MultibodyPlantConfig(
            time_step=sim_time_step,
            contact_model=contact_model,
            )

    plant, scene_graph = AddMultibodyPlant(multibody_plant_config, builder)

    #add assets to the plant
    agent = AddAgent(plant)
    plant.Finalize()
    plant.set_name("plant")

    #add assets to the controller plant
    controller_plant = MultibodyPlant(time_step=controller_time_step)
    AddAgent(controller_plant)        

    if meshcat:
        MeshcatVisualizerCpp.AddToBuilder(builder, scene_graph, meshcat)

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
        print("\nState view: ", StateView(np.ones(Ns)))
        print("\nPosition view: ",PositionView(np.ones(Np)))   
        print("\nActuation view: ", ActuationView(np.ones(Na)))
        
        # visualize the plant
        plt.figure()
        plot_graphviz(plant.GetTopologyGraphvizString())
        plt.plot(1)
        plt.show(block=False)

    #actions are positions sent to plant
    actuation = builder.AddSystem(Multiplexer([1,1]))
    prismatic_actuation_torque = builder.AddSystem(PassThrough(1))
    #zero torque to revolute. it is underactuated
    revolute_actuation_torque = builder.AddSystem(ConstantVectorSource([0]))
    builder.Connect(revolute_actuation_torque.get_output_port(),
                    actuation.get_input_port(1))
    builder.Connect(prismatic_actuation_torque.get_output_port(),
                    actuation.get_input_port(0))
    builder.Connect(actuation.get_output_port(),plant.get_actuation_input_port(agent))

    
    if meshcat:
        positions_to_poses = builder.AddSystem(
            MultibodyPositionToGeometryPose(controller_plant))
        builder.Connect(
            positions_to_poses.get_output_port(),
            controller_scene_graph.get_source_pose_port(
                controller_plant.get_source_id()))

    builder.ExportInput(prismatic_actuation_torque.get_input_port(), "actions")

    class observation_publisher(LeafSystem):
        def __init__(self, noise=False):
            LeafSystem.__init__(self)
            Nss = plant.num_multibody_states()
            self.DeclareVectorInputPort("plant_states", Nss)
            self.DeclareVectorOutputPort("observations", Nss, self.CalcObs)
            self.noise=noise
            
        def CalcObs(self, context,output):
            plant_state = self.get_input_port(0).Eval(context)
            if self.noise:
                plant_state+=np.random.uniform(low=-0.01,high=0.01,shape=Nss)       
            output.set_value(plant_state)

    obs_pub=builder.AddSystem(observation_publisher())

    builder.Connect(plant.get_state_output_port(),obs_pub.get_input_port(0))
    builder.ExportOutput(obs_pub.get_output_port(), "observations")

    class RewardSystem(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            # The state port is not used. 
            # Drake only computes the output of a system that is connected.
            self.DeclareVectorInputPort("state", Ns)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

        def CalcReward(self, context, output):
            reward=1
            output[0] = reward

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(plant.get_state_output_port(agent), reward.get_input_port(0))
    builder.ExportOutput(reward.get_output_port(), "reward")

    if monitoring_camera:
        # add an overhead camera. 
        # This is useful for logging videos of rollout evaluation
        scene_graph.AddRenderer("renderer",MakeRenderEngineVtk(RenderEngineVtkParams()))
        color_camera = ColorRenderCamera(
                RenderCameraCore(
                    "renderer",
                    CameraInfo(
                        width=640,
                        height=480,
                        fov_y=np.pi/4),
                    ClippingRange(0.01, 10.0),
                    RigidTransform()
                ), False)
        depth_camera = DepthRenderCamera(color_camera.core(),
                                        DepthRange(0.01, 10.0))
        parent_id=plant.GetBodyFrameIdIfExists(plant.world_body().index())
        X_PB= RigidTransform(RollPitchYaw(-np.pi/2, 0, 0),
                            np.array([0, -2.5, 0.4]))
        rgbd_camera=builder.AddSystem(RgbdSensor(parent_id=parent_id, X_PB=X_PB,
                                  color_camera=color_camera,
                                  depth_camera=depth_camera))
        builder.Connect(scene_graph.get_query_output_port(),rgbd_camera.query_object_input_port())
        builder.ExportOutput(rgbd_camera.color_image_output_port(),"color_image")
        
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()

    # Episode end conditions:
    def monitor(context,state_view=StateView):
        #pdb.set_trace()
        plant_context=plant.GetMyContextFromRoot(context)
        state=plant.GetOutputPort("continuous_state").Eval(plant_context)
        s=state_view(state)

        # Truncation: the episode duration reaches the time limit
        if context.get_time() > time_limit:
            if debug:
                print("Episode reached time limit.")
            return EventStatus.ReachedTermination(diagram, "time limit")
        
        # Termination: The pole angle exceeded +-0.2 rad
        if abs(s.PolePin_q) > 0.2:
            if debug:
                print("Pole angle exceeded +-0.2 rad.")
            return EventStatus.ReachedTermination(diagram, "pole angle exceeded +-0.2 rad")
        
        # Termination: Cart position exceeded +-2.4 m
        if abs(s.CartSlider_x) > 2.4:
            if debug:
                print("Cart position exceeded +-2.4 m.")
            return EventStatus.ReachedTermination(diagram, "cart position exceeded +-2.4 m")
            
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)

    if debug:
        #visualize the controller plant and diagram
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
        ('CartSlider',np.random.uniform(low=-.1,high=0.1)),
        ('PolePin',np.random.uniform(low=-.15,high=0.15)),
    ]
    
    home_velocities=[
        ('PolePin',np.random.uniform(low=-.1,high=0.1))
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
        if joint.type_name()=="prismatic":
            joint.set_translation(plant_context,
                            np.clip(pair[1],
                            joint.position_lower_limit(),
                            joint.position_upper_limit()
                            )
                        )
    for pair in home_velocities:
        joint = plant.GetJointByName(pair[0])
        if joint.type_name()=="revolute":
            joint.set_angular_rate(plant_context,
                            np.clip(pair[1],
                            joint.velocity_lower_limit(),
                            joint.velocity_upper_limit()
                            )
                        )        

def CartpoleEnv(observations="state", 
                meshcat=None, 
                time_limit=gym_time_limit, 
                debug=False,
                obs_noise=False,
                monitoring_camera=False):
    
    #Make simulation
    simulator = make_sim(RandomGenerator(),
                            meshcat=meshcat,
                            time_limit=time_limit,
                            debug=debug,
                            obs_noise=obs_noise,
                            monitoring_camera=monitoring_camera)
    
    plant = simulator.get_system().GetSubsystemByName("plant")
    
    #Define Action space
    Na=1
    low_a = plant.GetEffortLowerLimits()[:Na]
    high_a = plant.GetEffortUpperLimits()[:Na]
    action_space = gym.spaces.Box(low=np.asarray(low_a, dtype="float64"), 
                                    high=np.asarray(high_a, dtype="float64"),
                                    dtype=np.float64)
     
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
                      observation_port_id="observations",
                      set_home=set_home,
                      render_rgb_port_id="color_image" if monitoring_camera else None)

    # expose parameters that could be useful for learning
    env.time_step=gym_time_step
    env.sim_time_step=sim_time_step
    
    return env
