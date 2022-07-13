"""
This is an example for simulating a simplified humanoid (aka. noodleman) through pydrake.
It reads three simple SDFormat files of a hydroelastic humanoid,
a rigid chair, and rigid floor.
It uses an inverse dynamics controller to bring the noodleman from a sitting to standing up position.
"""
import argparse
import numpy as np
import pdb
import cv2
import time
import mediapipe as mp
import queue, threading
from pydrake.common import FindResourceOrThrow
from pydrake.geometry import DrakeVisualizer
from pydrake.math import RigidTransform
from pydrake.math import RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.multibody.plant import ConnectContactResultsToDrakeVisualizer
from pydrake.multibody.plant import MultibodyPlantConfig
from pydrake.systems.analysis import ApplySimulatorConfig
from pydrake.systems.analysis import Simulator
from pydrake.systems.analysis import SimulatorConfig
from pydrake.systems.analysis import PrintSimulatorStatistics
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import VectorLogSink
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.all import (DiagramBuilder,Parser,
                         RigidTransform, Simulator)
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.multibody.tree import WeldJoint, RevoluteJoint, PrismaticJoint
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
import matplotlib.pyplot as plt
import pydrake.geometry as mut
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
    InverseDynamics,
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
    ContactModel,
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
    RandomGenerator,
    PositionConstraint,
    MultibodyForces,
    Box,
    Meshcat,
    ContactSolver,
    LcmSubscriberSystem,
    DrakeLcm,
)
# from pydrake.lcm import DrakeLcm, Subscriber
from drake import lcmt_header, lcmt_quaternion

from utils import (FindResource, MakeNamedViewPositions, 
        MakeNamedViewVelocities,
        MakeNamedViewState,
        MakeNamedViewActuation,
        AddShape,
        SetColor
        )

## Env parameters
sim_time_step=0.025
controller_time_step=0.01
# gym_time_limit=5
teleop_freq=0.3
modes=["IDC","torque"]
control_mode=modes[0]
box_size=[ 0.35,#0.2+0.1*(np.random.random()-0.5),
        0.35,#0.2+0.1*(np.random.random()-0.5),
         0.35,   #0.2+0.1*(np.random.random()-0.5),
        ]
box_mass=2
box_mu=10.0
contact_model=ContactModel.kHydroelasticWithFallback#kPoint
contact_solver=ContactSolver.kSap#kTamsi # kTamsi
desired_box_heigth=0.6 #0.8
camera_index=0
stereo_ZED=True
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
    floor = parser.AddModelFromFile(FindResource("models/floor.sdf"))
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

def make_environment(meshcat=None, 
                   debug = False):

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


    nx = plant.num_positions() + plant.num_velocities()
    state_logger = builder.AddSystem(VectorLogSink(nx))
    builder.Connect(plant.get_state_output_port(),
                    state_logger.get_input_port())

    # bufferless VideoCapture
    class VideoCapture:

        def __init__(self, name):
            self.cap = cv2.VideoCapture(name)
            success=0
            while success == 0:
                success, frame = self.cap.read()
            
            self.im_shape=frame.shape
            self.q = queue.Queue()
            t = threading.Thread(target=self._reader)
            t.daemon = True
            t.start()

        # read frames as soon as they are available, keeping only most recent one
        def _reader(self):
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if not self.q.empty():
                    try:
                        self.q.get_nowait()   # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                if stereo_ZED:
                    #for L stereo
                    imageL=frame[:,:int(self.im_shape[1]/2),:].copy()
                    self.q.put(imageL)
                else:
                    self.q.put(frame)
        def read(self):
            return 1,self.q.get()
        
        def isOpened(self):
            return self.cap.isOpened()

    #mediapipe
    class MediapipeController:
        def __init__(self,camera_port):
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mp_pose = mp.solutions.pose
            # For webcam input:
            #self.cap = cv2.VideoCapture(camera_port)
            self.cap = VideoCapture(camera_port)
            self.pose=self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False,
                model_complexity=2,
            )
            # self.pose=self.mp_pose.Pose(
            #     static_image_mode=True,
            #     model_complexity=2,
            #     enable_segmentation=True,
            #     min_detection_confidence=0.5) 
            #pdb.set_trace()
            time.sleep(1)

        def process_frame(self):
            #pdb.set_trace()
            # with self.pose as pose:
            if self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image)
                
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                cv2.waitKey(1) 
            return results.pose_world_landmarks.landmark

    class MediapipeTeleop(LeafSystem):

        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorOutputPort("actions", Na, self.get_actions)
            self.TeleopManager=MediapipeController(camera_index)
            self.landmark_by_name=self.TeleopManager.mp_pose.PoseLandmark
            self.freq_ms=teleop_freq*1000
            self.last_time_step=0
            self.ActuationView=MakeNamedViewActuation(controller_plant, "Actuation")
            self.actuation_matrix=controller_plant.MakeActuationMatrix()
            ac=ActuationView(np.zeros(Na))
            ac.prismatic_z=0.1
            #pdb.set_trace()
            self.last_actions=ac.__array__()#.dot(self.actuation_matrix.T)   
            #acc=np.zeros(Na)
            #acc[0]=0.4

            #self.last_actions=acc

        def landmark_to_vec(self,landmark_origin, landmark_end):
            v=(np.array([landmark_end.x,landmark_end.y,landmark_end.z])-
                np.array([landmark_origin.x,landmark_origin.y,landmark_origin.z]))
            return v

        def get_elbow_angle(self,results,side, shoulder_tetha,shoulder_psi):
            #pdb.set_trace()
            if side=="RIGHT":
                c=self.landmark_to_vec(results[self.landmark_by_name.RIGHT_SHOULDER],
                            results[self.landmark_by_name.RIGHT_ELBOW])
                d=self.landmark_to_vec(results[self.landmark_by_name.RIGHT_ELBOW],
                            results[self.landmark_by_name.RIGHT_WRIST])
                d_=self.landmark_to_vec(results[self.landmark_by_name.RIGHT_WRIST],
                            results[self.landmark_by_name.RIGHT_ELBOW])
            
            elif side=="LEFT":
                c=self.landmark_to_vec(results[self.landmark_by_name.LEFT_SHOULDER],
                            results[self.landmark_by_name.LEFT_ELBOW])
                d=self.landmark_to_vec(results[self.landmark_by_name.LEFT_ELBOW],
                            results[self.landmark_by_name.LEFT_WRIST])
                d_=self.landmark_to_vec(results[self.landmark_by_name.LEFT_WRIST],
                            results[self.landmark_by_name.LEFT_ELBOW])

            ang=np.arccos(np.dot(c,d_)/(np.linalg.norm(c)*np.linalg.norm(d_)))

            Rz=np.array([
                [np.cos(shoulder_psi), -np.sin(shoulder_psi), 0],
                [np.sin(shoulder_psi), np.cos(shoulder_psi), 0],
                [0, 0, 1],
            ])
            Ry=np.array([
                [np.cos(shoulder_tetha), 0, np.sin(shoulder_tetha)],
                [0,1 , 0],
                [-np.sin(shoulder_tetha), 0, np.cos(shoulder_tetha)],
            ])
            #pdb.set_trace()
            d_=Rz@Ry@d
            psi_elb=np.arccos(d_[2]/np.linalg.norm(d_))
            tetha_elb=np.arccos(d_[0]/(np.linalg.norm(d_)*np.sin(psi_elb)))
            #print("tetha_elb:",tetha_elb)
            #print("psi_elb:",psi_elb)

            j1=np.pi -psi_elb
            j2=tetha_elb - 0.9
            return [j1,j2,np.pi-ang]

        def get_shoulder_angles(self,results,side):

            if side=="RIGHT":
                c=self.landmark_to_vec(results[self.landmark_by_name.RIGHT_SHOULDER],
                        results[self.landmark_by_name.RIGHT_ELBOW])  
                cc=np.array([-c[1],-c[2],-c[0]])
            elif side=="LEFT":    
                c=self.landmark_to_vec(results[self.landmark_by_name.LEFT_SHOULDER],
                        results[self.landmark_by_name.LEFT_ELBOW])  
                cc=np.array([-c[1],-c[2],c[0]])

            # print("r_shoulder: ",v_landmark(results[landmark_by_name.RIGHT_SHOULDER]))
            # print("l_shoulder: ",v_landmark(results[landmark_by_name.LEFT_SHOULDER]))
            # print("r_hip: ",v_landmark(results[landmark_by_name.RIGHT_HIP]))
            # print("r_elbow: ",v_landmark(results[landmark_by_name.RIGHT_ELBOW]))
            # print("l_elbow: ",v_landmark(results[landmark_by_name.LEFT_ELBOW]))

            
            tetha=np.arctan(cc[1]/cc[0])
            psi=np.arccos(cc[2]/np.linalg.norm(cc))
            tetha1=np.arccos(cc[0]/(np.linalg.norm(cc)*np.sin(psi)))
            j1=np.pi-tetha1
            j2=psi-np.pi/2

            # print("tetha:",tetha)
            # print("tetha_1:",tetha1)
            # print("psi:",psi)
            # print("j1:",j1)
            # print("j2:",j2)
            #pdb.set_trace()

            return [j1,j2,tetha1,psi]
        
        def landmarks_to_actions(self, landmarks,context):
            angs_shoulderR=self.get_shoulder_angles(landmarks,"RIGHT")
            angs_elbowR=self.get_elbow_angle(landmarks,"RIGHT",angs_shoulderR[2],angs_shoulderR[3])
            angs_shoulderL=self.get_shoulder_angles(landmarks,"LEFT")
            angs_elbowL=self.get_elbow_angle(landmarks,"LEFT",angs_shoulderL[2],angs_shoulderL[3])

            actions=ActuationView(self.last_actions)
            actions.shoulderR_joint1=angs_shoulderR[0]
            actions.shoulderR_joint2=angs_shoulderR[1]+0.15
            #actions.elbowR_joint1=angs_elbowR[0]
            #actions.elbowR_joint2=angs_elbowR[1]
            actions.elbowR_joint1=angs_elbowR[2]
            actions.elbowR_joint2=0.2*angs_elbowR[2]

            actions.shoulderL_joint1=angs_shoulderL[0]
            actions.shoulderL_joint2=angs_shoulderL[1]+0.2
            #actions.elbowL_joint1=angs_elbowL[0]
            #actions.elbowL_joint2=angs_elbowL[1]
            actions.elbowL_joint1=angs_elbowL[2]
            actions.elbowL_joint2=0.2*angs_elbowL[2]
            #print("shoulder: ",-ang_shoulder)
            #print("elbow: ",np.pi-ang_elbow )
            return actions.__array__()#.dot(self.actuation_matrix.T)

        def get_actions(self, context, output):
            time = context.get_time()
            #print("time: ", time)
            actions=self.last_actions
            if (time*1000)%self.freq_ms == 0:
                self.last_time_step=time+self.freq_ms*0.001
                #print(time, self.last_time_step)
                landmarks=self.TeleopManager.process_frame()
                actions=self.landmarks_to_actions(landmarks,context)
                self.last_actions=actions
            #print(actions)
            
            output.set_value(actions.dot(self.actuation_matrix.T) )

    teleop=builder.AddSystem(MediapipeTeleop())
    builder.Connect(teleop.get_output_port(),
                actions.get_input_port())

    diagram = builder.Build()

    if debug:
        #visualize plant and diagram
        plt.figure()
        plot_graphviz(controller_plant.GetTopologyGraphvizString())
        plt.figure()
        plot_system_graphviz(diagram, max_depth=2)
        plt.plot(1)
        plt.show(block=False)
        #pdb.set_trace()

    return diagram, plant, controller_plant, state_logger, agent

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
        ('prismatic_z',0.1*(np.random.random()-0.5)+0.2),
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
                            0+0.1*(np.random.random()-0.5), 
                            0.3+0.1*(np.random.random()-0.5), 
                            box_size[2]/2+0.005,
                        ])
                    )
    plant.SetFreeBodyPose(plant_context,box,box_pose)

def simulate_diagram(diagram, plant, controller_plant, state_logger,
                       simulation_time, target_realtime_rate):

    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant,
                                                diagram_context)    
    
    
    #setup the simulator
    simulator_config = SimulatorConfig(
                           target_realtime_rate=target_realtime_rate,
                           publish_every_time_step=True)
    


    simulator = Simulator(diagram)
    ApplySimulatorConfig(simulator, simulator_config)



    print("Initial state variables: ", plant.GetPositionsAndVelocities(plant_context))
   
    state_log = state_logger.FindMutableLog(simulator.get_mutable_context())
    state_log.Clear()
    simulator.Initialize()

    context = simulator.get_mutable_context()
    context.SetTime(0)
    set_home(simulator, context)

    #pdb.set_trace()
    adv_step=0.1
    time=0
    for i in range(int(simulation_time/adv_step)):
        time+=adv_step
        simulator.AdvanceTo(time)
        #PrintSimulatorStatistics(simulator)    
        #input("Press Enter to continue...")
    return state_log.sample_times(), state_log.data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulation_time", type=float, default=1000,
        help="Desired duration of the simulation in seconds. "
             "Default 8.0.")
    parser.add_argument(
        "--contact_model", type=str, default="hydroelastic_with_fallback",
        help="Contact model. Options are: 'point', 'hydroelastic', "
             "'hydroelastic_with_fallback'. "
             "Default 'hydroelastic_with_fallback'")
    parser.add_argument(
        "--contact_surface_representation", type=str, default="polygon",
        help="Contact-surface representation for hydroelastics. "
             "Options are: 'triangle' or 'polygon'. Default 'polygon'.")
    parser.add_argument(
        "--time_step", type=float, default=0.001,
        help="The fixed time step period (in seconds) of discrete updates "
             "for the multibody plant modeled as a discrete system. "
             "If zero, we will use an integrator for a continuous system. "
             "Non-negative. Default 0.001.")
    parser.add_argument(
        "--target_realtime_rate", type=float, default=1.0,
        help="Target realtime rate. Default 1.0.")
    parser.add_argument(
        "--meshcat", action="store_true",
        help="If set, visualize in meshcat. Use DrakeVisualizer otherwise")        
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.meshcat:
        meshcat_server= Meshcat()
        visualizer=meshcat_server
    else:
        visualizer=None

    input("Press Enter to continue...")

    diagram, plant, controller_plant,state_logger,agent_idx = make_environment(
        meshcat=visualizer, debug=args.debug)
    
    time_samples, state_samples = simulate_diagram(
        diagram, plant, controller_plant, state_logger,
        args.simulation_time, 
        args.target_realtime_rate)
    
    print("\nFinal state variables:")
    print(state_samples[:, -1])
