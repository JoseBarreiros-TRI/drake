import sys
import cv2
import os
from sys import platform
import pdb

# import openpose as op
from openpose import pyopenpose as op
dir_path = '/home/josebarreiros/openpose/models/'
sys.path.append('usr/local/python')


#Setting OpenPose parameters
def set_params():

        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "-1x368"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
        params["render_threshold"] = 0.05
        # # If GPU version is built, and multiple GPUs are available, set the ID here
        # params["num_gpu_start"] = 0
        # params["disable_blending"] = False
        # Ensure you point to the correct path where models are located
        params["model_folder"] = dir_path # + "/../../../models/"
        return params

def main():


        params = set_params()
        
        opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
        opWrapper.configure(params)
        opWrapper.execute()
        #pdb.set_trace()

        #Constructing OpenPose object allocates GPU memory
        openpose = op.OpenPose(params)

        #Opening OpenCV stream
        stream = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:

                ret,img = stream.read()
                datum = op.Datum()
                datum.cvInputData = img
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))



                # Output keypoints and the image with the human skeleton blended on it
                keypoints= datum.poseKeyPoints
                output_image= datum.cvOutputData 

                # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
                if len(keypoints)>0:
                        print('Human(s) Pose Estimated!')
                        print(keypoints)
                else:
                        print('No humans detected!')


                # Display the stream
                cv2.putText(output_image,'OpenPose using Python-OpenCV',(20,30), font, 1,(255,255,255),1,cv2.LINE_AA)

                cv2.imshow('Human Pose Estimation',output_image)

                key = cv2.waitKey(1)

                if key==ord('q'):
                        break

        stream.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
        main()
