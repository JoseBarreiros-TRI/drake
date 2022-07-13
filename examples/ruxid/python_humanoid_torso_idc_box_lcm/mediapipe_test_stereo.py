import cv2
import mediapipe as mp
import pdb
import time
import numpy as np
import queue, threading
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Mapping, Optional, Tuple, Union
import dataclasses

cam_port=-1

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
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
      self.q.put(frame)

  def read(self):
    return 1,self.q.get()
  
  def isOpened(self):
    return self.cap.isOpened()

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)
@dataclasses.dataclass
class DrawingSpec:
  # Color for drawing the annotation. Default to the white color.
  color: Tuple[int, int, int] = WHITE_COLOR
  # Thickness for drawing the annotation. Default to 2 pixels.
  thickness: int = 2
  # Circle radius. Default to 2 pixels.
  circle_radius: int = 2
def _normalize_color(color):
  return tuple(v / 255. for v in color)

class MediapipeController:
  def __init__(self,camera_port):
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_poseL = mp.solutions.pose
    self.mp_poseR = mp.solutions.pose
    # # For webcam input:
    self.cap = VideoCapture(camera_port)
    self.poseL=self.mp_poseL.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=1,
      )
    self.poseR=self.mp_poseR.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=1,
      )
    self.landmark_names=self.mp_poseL.PoseLandmark
    
    success, frame = self.cap.read()
    self.im_shape=frame.shape
    print("Camera shape: ", self.im_shape[0],int(self.im_shape[1]/2),self.im_shape[2])
    
    path=os.path.dirname(os.path.abspath(__file__))
    calibration_dir='/home/josebarreiros/'+path[path.find('drake',path.find('drake')+1):]+'/calibration'
    self.load_camera_params(calibration_dir)
    self.fig=plt.figure()
    self.fig2=plt.figure()
    time.sleep(1)

  def read_camera_parameters(self,camera_id,directory):
    #pdb.set_trace()
    inf = open(directory+'/camera_parameters/'+camera_id + '_intrinsics.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

  def read_rotation_translation(self,camera_id, directory):

    inf = open(directory+'/camera_parameters/' + camera_id+'_rot_trans.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

  def _make_homogeneous_rep_matrix(self,R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

  def get_projection_matrix(self,camera_id,directory):

    #read camera parameters
    cmtx, dist = self.read_camera_parameters(camera_id,directory)
    rvec, tvec = self.read_rotation_translation(camera_id,directory)

    #calculate projection matrix
    P = cmtx @ self._make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P
    
  def load_camera_params(self,calibration_dir):
      self.P0=self.get_projection_matrix('camera0',calibration_dir)
      self.P1=self.get_projection_matrix('camera1',calibration_dir)

  #direct linear transform
  def DLT(self,P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    #print('A: ')
    #print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

  def plot_landmarks(self,landmark_list,
        connections=None,
        landmark_drawing_spec=DrawingSpec(
                       color=RED_COLOR, thickness=5),
        connection_drawing_spec= DrawingSpec(
                       color=BLACK_COLOR, thickness=5),
        elevation = 10,
        azimuth = 10,
        figure = None
        ):
    
    #ax = plt.axes(projection='3d')
    ax= figure.add_subplot(111, projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
      if ((landmark.HasField('visibility') and
          landmark.visibility < _VISIBILITY_THRESHOLD) or
          (landmark.HasField('presence') and
          landmark.presence < _PRESENCE_THRESHOLD)):
        continue
      ax.scatter3D(
          xs=[-landmark.z],
          ys=[landmark.x],
          zs=[-landmark.y],
          color=_normalize_color(landmark_drawing_spec.color[::-1]),
          linewidth=landmark_drawing_spec.thickness)
      plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
    if connections:
      num_landmarks = len(landmark_list.landmark)
      # Draws the connections if the start and end landmarks are both visible.
      for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
          raise ValueError(f'Landmark index is out of range. Invalid connection '
                          f'from landmark #{start_idx} to landmark #{end_idx}.')
        if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
          landmark_pair = [
              plotted_landmarks[start_idx], plotted_landmarks[end_idx]
          ]
          ax.plot3D(
              xs=[landmark_pair[0][0], landmark_pair[1][0]],
              ys=[landmark_pair[0][1], landmark_pair[1][1]],
              zs=[landmark_pair[0][2], landmark_pair[1][2]],
              color=_normalize_color(connection_drawing_spec.color[::-1]),
              linewidth=connection_drawing_spec.thickness)
    plt.pause(0.001)
    #ax.cla()
    #plt.show()

  def visualize_3d(self,kpts3d):
    #pdb.set_trace()
    """Now visualize in 3D"""
    torso = [[0, 1] , [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    fig = self.fig
    ax = fig.add_subplot(111, projection='3d')


    for bodypart, part_color in zip(body, colors):
        for _c in bodypart:
            ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)

        #uncomment these if you want scatter plot of keypoints and their indices.
        # for i in range(12):
        #     #ax.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
        #     #ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])


        # #ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlim3d(-100, 100)
    ax.set_xlabel('x')
    ax.set_ylim3d(-100, 100)
    ax.set_ylabel('y')
    ax.set_zlim3d(-100, 100)
    ax.set_zlabel('z')
    plt.pause(0.01)
    #ax.cla()

  def find_3d_keypoints(self,resultsL,resultsR,imageL,imageR):
    #add here if you need more keypoints
    pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
    #check for keypoints detection
    frame0_keypoints = []
    if resultsL.pose_landmarks:
        for i, landmark in enumerate(resultsL.pose_landmarks.landmark):
            if i not in pose_keypoints: continue #only save keypoints that are indicated in pose_keypoints
            pxl_x = landmark.x * imageL.shape[1]
            pxl_y = landmark.y * imageL.shape[0]
            pxl_x = int(round(pxl_x))
            pxl_y = int(round(pxl_y))
            cv2.circle(imageL,(pxl_x, pxl_y), 3, (0,0,255), -1) #add keypoint detection points into figure
            kpts = [pxl_x, pxl_y]
            frame0_keypoints.append(kpts)
    else:
        #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
        frame0_keypoints = [[-1, -1]]*len(pose_keypoints)

    frame1_keypoints = []
    if resultsR.pose_landmarks:
        for i, landmark in enumerate(resultsR.pose_landmarks.landmark):
            if i not in pose_keypoints: continue
            pxl_x = landmark.x * imageR.shape[1]
            pxl_y = landmark.y * imageR.shape[0]
            pxl_x = int(round(pxl_x))
            pxl_y = int(round(pxl_y))
            cv2.circle(imageR,(pxl_x, pxl_y), 3, (0,0,255), -1)
            kpts = [pxl_x, pxl_y]
            frame1_keypoints.append(kpts)
    else:
        #if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
        frame1_keypoints = [[-1, -1]]*len(pose_keypoints)

    #calculate 3d position
    frame_p3ds = []
    for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
        if uv1[0] == -1 or uv2[0] == -1:
            _p3d = [-1, -1, -1]
        else:
            _p3d = self.DLT(self.P0, self.P1, uv1, uv2) #calculate 3d position of keypoint
        frame_p3ds.append(_p3d)
    
    frame_p3ds = np.array(frame_p3ds).reshape((12, 3))    
    
    return frame_p3ds

  def process_frame(self):
      #pdb.set_trace()
      #with pose:
      if self.cap.isOpened():
          success, frame = self.cap.read()
          imageL=frame[:,:int(self.im_shape[1]/2),:].copy()
          imageR=frame[:,int(self.im_shape[1]/2):,:].copy()

          # To improve performance, optionally mark the image as not writeable to
          # pass by reference.
          imageL.flags.writeable = False
          imageR.flags.writeable = False
          resultsL = self.poseL.process(imageL)
          resultsR = self.poseR.process(imageR)
          
          # # Draw the pose annotation on the image.
          self.mp_drawing.draw_landmarks(
              imageL,
              resultsL.pose_landmarks,
              self.mp_poseL.POSE_CONNECTIONS,
              landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

          self.mp_drawing.draw_landmarks(
              imageR,
              resultsR.pose_landmarks,
              self.mp_poseR.POSE_CONNECTIONS,
              landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())


          # Plot pose world landmarks.
          # self.mp_drawing.plot_landmarks(
          #       resultsL.pose_world_landmarks, 
          #       self.mp_poseL.POSE_CONNECTIONS)

          # self.mp_drawing.plot_landmarks(
          #       resultsR.pose_world_landmarks, 
          #       self.mp_poseR.POSE_CONNECTIONS)

          self.plot_landmarks(
                resultsL.pose_world_landmarks, 
                self.mp_poseL.POSE_CONNECTIONS,figure=self.fig)

          self.plot_landmarks(
                resultsR.pose_world_landmarks, 
                self.mp_poseR.POSE_CONNECTIONS,figure=self.fig2)

          #pdb.set_trace()
          #frame_p3ds=self.find_3d_keypoints(resultsL,resultsR,imageL,imageR)
          #frame_p3ds=self.find_3d_keypoints(resultsR,resultsL,imageR,imageL)
     
          # Flip the image horizontally for a selfie-view display.
          cv2.imshow('MediaPipe Pose L', cv2.flip(imageL, 1))
          cv2.imshow('MediaPipe Pose R', cv2.flip(imageR, 1))
          
          cv2.waitKey(1) 
          pdb.set_trace()
          #print(frame_p3ds)
          #self.visualize_3d(frame_p3ds)

      return 1#results.pose_landmarks.landmark

def landmark_to_vec(landmark_origin, landmark_end):
  v=(np.array([landmark_end.x,landmark_end.y,landmark_end.z])-
          np.array([landmark_origin.x,landmark_origin.y,landmark_origin.z]))
  return v#/np.linalg.norm(v)

def angle_between_vectors(v1,v2):
  return np.arccos( np.dot(v1,v2) / ( np.linalg.norm(v1)*np.linalg.norm(v2) ) )

def elbow_angle(results,landmark_by_name):
  #pdb.set_trace()
  v1=landmark_to_vec(results[landmark_by_name.RIGHT_SHOULDER],
            results[landmark_by_name.RIGHT_ELBOW])
  v2=landmark_to_vec(results[landmark_by_name.RIGHT_WRIST],
              results[landmark_by_name.RIGHT_ELBOW])
  return np.arccos(np.dot(v1,v2))

def v_landmark(landmark):
  return [landmark.x,landmark.y,landmark.z]
  
def shoulder_angle(results,landmark_by_name):
  
  b=landmark_to_vec(results[landmark_by_name.RIGHT_SHOULDER],
            results[landmark_by_name.RIGHT_HIP])
  
  a=landmark_to_vec(results[landmark_by_name.LEFT_SHOULDER],
              results[landmark_by_name.RIGHT_SHOULDER])

  c=landmark_to_vec(results[landmark_by_name.RIGHT_SHOULDER],
              results[landmark_by_name.RIGHT_ELBOW])  
  
  b_=b+(a/np.linalg.norm(a)**2)*(a.dot(b))
  
  #normal to the front of the torso
  ba=np.cross(b,a)
  n1=ba/np.linalg.norm(ba)

  #normal to the side of the torso
  nb_=np.cross(n1,b_)
  n2=nb_/np.linalg.norm(nb_)

  # projection of c vector onto front torso plane
  c_front=c-c.dot(n1)*n1
  alpha=angle_between_vectors(c_front,b_)
  
  # projection of c vector onto side torso plane
  c_side=c-c.dot(n2)*n2
  beta=angle_between_vectors(c_side,b_)


  # print("r_shoulder: ",v_landmark(results[landmark_by_name.RIGHT_SHOULDER]))
  # print("l_shoulder: ",v_landmark(results[landmark_by_name.LEFT_SHOULDER]))
  # print("r_hip: ",v_landmark(results[landmark_by_name.RIGHT_HIP]))
  # print("r_elbow: ",v_landmark(results[landmark_by_name.RIGHT_ELBOW]))
  # print("l_elbow: ",v_landmark(results[landmark_by_name.LEFT_ELBOW]))
  # print("a:",a)
  # print("b:",b)
  # print("c:",c)
  # print("b_:",b_)
  # print("n1:",n1)
  # print("n2:",n2)
  # print("c_front:",c_front)
  # print("c_side:",c_side)
  # print("alpha:",alpha)
  # print("beta:",beta)

  # al=angle_between_vectors(np.array([0,c[1],c[2]]),b)
  # be=angle_between_vectors(np.array([c[0],0,c[2]]),n1)
  # print("al:",al)
  # print("be:",be)
  cc=np.array([c[2],c[1],c[0]])
  tetha=np.arctan(cc[1]/cc[0])
  phi=np.arccos(cc[2]/np.linalg.norm(cc))
  print("tetha:",tetha)
  print("phi:",phi)
  #pdb.set_trace()
  time.sleep(0.01)

  return 1

if __name__ == "__main__":
  TeleopManager=MediapipeController(cam_port)
  landmark_by_name=TeleopManager.landmark_names
  n=10000
  i=0
  while i<n:
    #print("i: ", i,"\n")
    results=TeleopManager.process_frame()
    #elbow_angle(results,landmark_by_name)
    #pdb.set_trace()
    i+=1
