import cv2
import mediapipe as mp
import time 
import threading 
import pygame
import os
import math
import sys


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define custom drawing specifications for landmarks and connections
my_landmark_drawing_specs = mp_drawing.DrawingSpec(color = (255, 0, 0), thickness = 1, circle_radius = 1)
my_connection_specs = mp_drawing.DrawingSpec(color = (0, 0, 0), thickness = 1)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh 

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set the resolution of the webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def rgb2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh, mp_pose.Pose(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
) as pose:
    # Check if the webcam is opened correctly
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
                sys.exit(
                     "Error: Unable to read from webcam. Please verify your webcam settings."
                     )

        image = cv2.flip(rgb2bgr(frame),1)

        image.flags.writeable = False

        pose_results = pose.process(image)
        face_results = face_mesh.process(image)

        image.flags.writeable = True 

        # Draw face mesh results
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
            
                 # Draw face mesh landmarks
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec =  my_landmark_drawing_specs,
                    connection_drawing_spec = my_connection_specs
                )
                
                # Draw face mesh contours
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = my_landmark_drawing_specs,
                    connection_drawing_spec = my_connection_specs
                )

                # Only proceed if pose landmarks are also detected
                if pose_results.pose_landmarks:
                    image.flags.writeable = True

                    def get_landmark_coordinates(landmark, image_width, image_height):
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        return (x, y)
                    #Determine posture given landmarks positions
                    CHIN_CENTER = 152
                    LEFT_SHOULDER = 11
                    RIGHT_SHOULDER = 12
                    LEFT_EAR = 356
                    RIGHT_EAR = 127
                    NOSE = 4

                    left_ear = face_landmarks.landmark[LEFT_EAR]
                    right_ear = face_landmarks.landmark[RIGHT_EAR]
                    nose = face_landmarks.landmark[NOSE]
                    left_shoulder = pose_results.pose_landmarks.landmark[LEFT_SHOULDER]
                    right_shoulder = pose_results.pose_landmarks.landmark[RIGHT_SHOULDER]
                    chin_center = face_landmarks.landmark[CHIN_CENTER]
                    

                    image_height, image_width, _ = image.shape
                    lx, ly = int(left_ear.x * image_width), int(left_ear.y * image_height)
                    rx, ry = int(right_ear.x * image_width), int(right_ear.y * image_height)
                    ex, ey = int(nose.x * image_width), int(nose.y * image_height)
                    sx, sy = int(left_shoulder.x * image_width), int(left_shoulder.y * image_height)
                    dx, dy = int(right_shoulder.x * image_width), int(right_shoulder.y * image_height)
                    cx, cy = int(chin_center.x * image_width), int(chin_center.y * image_height)


                    #Calculate distances between ears and shoulders
                    LEFT_EAR_to_LEFT_SHOULDER = math.dist((lx, ly), (sx, sy))
                    RIGHT_EAR_to_RIGHT_SHOULDER = math.dist((rx, ry), (dx, dy))

                    def midpoint(x1, y1, x2, y2):
                        return ((x1 + x2) // 2, (y1 + y2) // 2)

                    #Calculate distnace between chin and midpoint of shoulders 
                    MIDPOINT_SHOULDERS = midpoint(sx, sy, dx, dy)
                    CHIN_TO_MIDPOINT_SHOULDERS = math.dist((cx, cy), MIDPOINT_SHOULDERS)


                    #Debugging prints
                    print("Left distance:", LEFT_EAR_to_LEFT_SHOULDER)
                    print("Right distance:", RIGHT_EAR_to_RIGHT_SHOULDER)
                    print("Chin to midpoint shoulders:", CHIN_TO_MIDPOINT_SHOULDERS)


                    #Determine head posture based on distances
                    left_posture_color = (0, 255, 0)  # Green
                    right_posture_color = (0, 255, 0)  # Green
                    head_posture_color = (0, 255, 0)  # Green
                    chin_posture_color = (0, 255, 0)  # Green

                    posture_text = "Good Posture"
                    head_posture_margin = 210
                    chin_posture_margin = 100


                    if LEFT_EAR_to_LEFT_SHOULDER < head_posture_margin:
                        left_posture_color = (255, 0, 0)  # Red

                    if RIGHT_EAR_to_RIGHT_SHOULDER < head_posture_margin:
                        right_posture_color = (255, 0, 0) # Red
                        
                    if RIGHT_EAR_to_RIGHT_SHOULDER < head_posture_margin or LEFT_EAR_to_LEFT_SHOULDER < head_posture_margin or CHIN_TO_MIDPOINT_SHOULDERS < chin_posture_margin:
                        head_posture_color = (255, 0, 0)  # Red
                        posture_text = "Bad Posture"


                    if CHIN_TO_MIDPOINT_SHOULDERS < chin_posture_margin:
                        chin_posture_color = (255, 0, 0) # Red
                        posture_text = "Bad Posture"
                        
  

                    cv2.putText(image, posture_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, head_posture_color, 2)
                    cv2.line(image, (lx, ly), (sx, sy), left_posture_color, 2)
                    cv2.line(image, (rx, ry), (dx, dy), right_posture_color, 2)
                    cv2.circle(image, MIDPOINT_SHOULDERS, 1, (0, 255, 0), -1)  # Draw midpoint for visualization
                    cv2.line(image, (cx, cy), MIDPOINT_SHOULDERS, chin_posture_color, 2)

        #Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=pose_results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec = my_landmark_drawing_specs,
                connection_drawing_spec = my_connection_specs
                )

        # Display the image with landmarks and connections
        cv2.imshow("Posture Detector", bgr2rgb(image))
       
        # Exit the loop if 'ESC' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break
    
cap.release()
cv2.destroyAllWindows()
