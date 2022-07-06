import cv2
import mediapipe as mp
import pdb
import time
import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose
# pose=mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

class MediapipeController:
  def __init__(self,camera_port):
      self.mp_drawing = mp.solutions.drawing_utils
      self.mp_drawing_styles = mp.solutions.drawing_styles
      self.mp_pose = mp.solutions.pose
      # # For webcam input:
      self.cap = cv2.VideoCapture(camera_port)
      self.pose=self.mp_pose.Pose(
          min_detection_confidence=0.5,
          min_tracking_confidence=0.5)
      self.landmark_names=self.mp_pose.PoseLandmark
      # self.pose=self.mp_pose.Pose(
      #     static_image_mode=True,
      #     model_complexity=2,
      #     enable_segmentation=True,
      #     min_detection_confidence=0.5) 
      time.sleep(1)

      #pdb.set_trace()

  def process_frame(self):
      #pdb.set_trace()
      #with pose:
      if self.cap.isOpened():
          success, image = self.cap.read()
          #image=cv2.flip(image, 1)
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

          # Plot pose world landmarks.
          self.mp_drawing.plot_landmarks(
                results.pose_world_landmarks, 
                self.mp_pose.POSE_CONNECTIONS)
                
          # angle=elbow_angle(results.pose_world_landmarks.landmark,self.landmark_names)       
          # cv2.putText(image,str(np.degrees(angle)),(150, 250),
          #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

          angle=shoulder_angle(results.pose_world_landmarks.landmark,self.landmark_names)       
          # cv2.putText(image,str(np.degrees(angle)),(150, 350),
          #       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

          # Flip the image horizontally for a selfie-view display.
          cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
          cv2.waitKey(1) 
          #pdb.set_trace()
      return results.pose_landmarks.landmark

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
  TeleopManager=MediapipeController(0)
  landmark_by_name=TeleopManager.landmark_names
  n=10000
  i=0
  while i<n:
    #print("i: ", i,"\n")
    results=TeleopManager.process_frame()
    #elbow_angle(results,landmark_by_name)
    #pdb.set_trace()
    i+=1
