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
contact_model='point'#'hydroelastic_with_fallback'#ContactModel.kHydroelasticWithFallback#kPoint
contact_solver='sap'#ContactSolver.kSap#kTamsi # kTamsi

def AddAgent(plant):
    parser = Parser(plant)
    model_file = FindResourceOrThrow("drake/drake_gym/models/cartpole_BSA.sdf")
    agent = parser.AddModelFromFile(model_file)
    return agent

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
        plt.figure()
        plot_graphviz(plant.GetTopologyGraphvizString())
        plt.plot(1)
        plt.show(block=False)
     
        print("\nState view: ", StateView(np.ones(Ns)))
        print("\nActuation view: ", ActuationView(np.ones(Na)))
        print("\nPosition view: ",PositionView(np.ones(Np)))   


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

        def __init__(self):
            LeafSystem.__init__(self)
            Nss = plant.num_multibody_states()
            self.DeclareVectorInputPort("plant_states", Nss)
            self.DeclareVectorOutputPort("observations", Nss, self.CalcObs)
            
        def CalcObs(self, context,output):
            plant_state = self.get_input_port(0).Eval(context)
            output.set_value(plant_state)

    obs_pub=builder.AddSystem(observation_publisher())

    builder.Connect(plant.get_state_output_port(),obs_pub.get_input_port(0))
    builder.ExportOutput(obs_pub.get_output_port(), "observations")

    class RewardSystem(LeafSystem):

        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("state", Ns)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)


        def CalcReward(self, context, output):
            reward=1
            output[0] = reward

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(plant.get_state_output_port(agent), reward.get_input_port(0))
    builder.ExportOutput(reward.get_output_port(), "reward")

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

def CartpoleEnv(observations="state", meshcat=None, time_limit=gym_time_limit, debug=False):
    
    #Make simulation
    simulator = make_sim(RandomGenerator(),
                            observations,
                            meshcat=meshcat,
                            time_limit=time_limit,
                            debug=debug)
    plant = simulator.get_system().GetSubsystemByName("plant")
    
    #Define Action space
    #pdb.set_trace()
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
                      set_home=None)

    # expose parameters that could be useful for learning
    env.time_step=gym_time_step
    env.sim_time_step=sim_time_step
    
    return env
