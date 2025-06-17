import cv2
import mediapipe as mp
import numpy as np
import time 
import os 


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define custom drawing specifications for landmarks and connections
my_landmark_drawing_specs = mp_drawing.DrawingSpec(color = (255, 0, 0), thickness = 1, circle_radius = 1)
my_connection_specs = mp_drawing.DrawingSpec(color = (0, 0, 255), thickness = 1)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh 

# Initialize webcam
cap = cv2.VideoCapture(0)
    # Set the resolution of the webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
            break 

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    #               .get_default_face_mesh_tesselation_style()
                )
                
                # Draw face mesh contours
                mp_drawing.draw_landmarks(
                    image = image,
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = my_landmark_drawing_specs,
                    connection_drawing_spec = my_connection_specs
    #                 .get_default_face_mesh_contours_style()
                )

                #Draw pose landmarks
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=pose_results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec = my_landmark_drawing_specs,
                        connection_drawing_spec = my_connection_specs
                    )
            #Determine posture based on landmarks
            if pose_results.pose_landmarks:
            
                posture = "Good Posture"
            
                # Get the coordinates of landmarks for shoulders and ears

                left_shoulder_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_shoulder_x = int(left_shoulder_landmark.x)
                left_shoulder_y = int(left_shoulder_landmark.y)

                right_shoulder_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_shoulder_x = int(right_shoulder_landmark.x)
                right_shoulder_y = int(right_shoulder_landmark.y)
                
                left_ear_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
                left_ear_x = int(left_ear_landmark.x)
                left_ear_y = int(left_ear_landmark.y)

                right_ear_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
                right_ear_x = int(right_ear_landmark.x)
                right_ear_y = int(right_ear_landmark.y)

                # Calculate the average shoulder position
                average_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
                average_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

               
        # Convert the image color from RGB back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the image with landmarks and connections
        cv2.imshow("Posture Detector", cv2.flip(image, 1))

        # Exit the loop if 'ESC' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break
    
cap.release()
cv2.destroyAllWindows()
