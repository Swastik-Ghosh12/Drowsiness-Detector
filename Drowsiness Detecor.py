#importing all the important libraries

from scipy.spatial import distance as dist
#from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np

import imutils
import time
import dlib
import cv2

#function to calculate Ear Aspect Ratio for each ear
def eye_aspect_ratio(eye):
    A= dist.euclidean(eye[1], eye[5])
    B= dist.euclidean(eye[2], eye[4])
    C= dist.euclidean(eye[0], eye[3])
    EAR= (A+B)/(2.0*C)
    return EAR

#Setting the threshold
EAR_THRESHOLD = 0.3

#Consecutive frames for which the EAR has to be lower than the threshold
EYE_AR_CONSEC_FRAMES = 48

#initialising the counter
COUNTER = 0
ALARM_ON = False

##def sound_alarm(path):
##    

#Initialising the face detector and predictor
detector= dlib.get_frontal_face_detector()
print(type(detector))
predictor= dlib.shape_predictor('C:\\Users\KIIT\AppData\Local\Programs\Python\Python37-32\Lib\site-packages\dlib-19.17.0-py3.7.egg-info\shape_predictor_68_face_landmarks.dat')

#Fetching the co-ordinates
(lStart, lEnd)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

#setting up the camera
#cap=VideoStream(src=0).start()
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
time.sleep(1.0)



while True:
	_,frame = cap.read()
	#frame = rescale_frame(frame, percent=25)
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]

		#calculating the EAR for left eye
		leftEAR = eye_aspect_ratio(leftEye)

		#calulating the EAR for right eye
		rightEAR = eye_aspect_ratio(rightEye)

		#calculating the average EAR
		EAR = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		#Checking if drowsy or not
		if EAR < EAR_THRESHOLD:
                    
                    #Counter starts as soon as EAR falls below threshold
			COUNTER += 1

			#if for 48 consecutive frames eyes are closed then it shows a drowsiness alert
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				if not ALARM_ON:
                                     
					ALARM_ON = True
					#playsound.playsound('D:\bensound-ukulele.mp3')
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                               
                                
				
		else:
			COUNTER = 0
			ALARM_ON = False
		cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #display continous frames making it seem like a video
	cv2.imshow("Frame", frame)
	if cv2.waitKey(0) & OxFF== ord('q'):
            cv2.destroyAllWindows()
	
            
