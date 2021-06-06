import numpy as np
import cv2
import argparse
import imutils
import time
import dlib
#To detect and localize facial landmarks we need dlib package and knowing installation process of this package is important
import playsound
#playsound is imported for playing some small and simple sounds as alarms it can play MP3,WAV formats
from scipy.spatial import distance as dis
#distance is imported for calculation of Eucledian distance between facial landmarks and for finding EAR
from imutils.video import VideoStream
#for getting live videostream from webcam/usb webcam we are using
from imutils import face_utils
#we will use this to extract indexes of certain specific parts like eyes etc, after detecting facial landmarks
from threading import Thread
#We import thread class to play our alarm in another Thread so that it doesnt stop the execution of code
def play_alarm(path):
	playsound.playsound(path) #This function is created for playing the alarm sound
     
ap=argparse.ArgumentParser()
ap.add_argument('-s','--shape_predictor',required=True,help='This is path to facial land mark detector')
ap.add_argument('-a','--alarm',type=str,default='',help='path to alarm audio file')
ap.add_argument('-c','--webcam',type=int,default=0,help='here it is index of webcam on our device to use')
args=vars(ap.parse_args())

EAR_thresh=0.3
#This is threshold EAR value if in a frame EAR value is less than this that frame will be counted
EAR_frames=5
#If consecutively EAR is less than 0.3 for 48 frames then alarm will start ringing
frame_count=0
#This value is initialized for counting frames and if it crosses 48 then alarm rings
alarm_on=False
#This alarm_on is boolean expression if driver is drowsy then boolean value will be converted as True
print("[INFO] loading facial landmark predictor model.....")
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args['shape_predictor'])
#Now we will get the indexes of facial landmarks of left and right eye
(lstart,lend)=face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
#actually FACIAL_LANDMARKS_IDXS this is a dictionary for 'left_eye' key the values (lstart,lend) we get is (42,48) and for below line of code 
#we get (rstart,rend) as (36,42)
(rstart,rend)=face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
print((lstart,lend))
print((rstart,rend))
#one eye gives (36,42) and other eye gives (42,48) for above 2 lines of code
print('[INFO] starting video stream.....')
vs=VideoStream(src=args['webcam']).start()
#This will make webcam stream to start
time.sleep(1)
#In this time cam will warmup and then starts
#we write while:True for looping over each frame of videostream
while True:
     frame=vs.read()
     frame=imutils.resize(frame,width=450)
     #Initially i am not converting it to gray and keeping it as it is
     rectangles=detector(frame,0)
     #This above line detects all the faces in frame
     #Now we will loop through all faces(ingeneral we may have 1 face only which is of driver but incase there are more persons also model can detect)
     for rectangle in rectangles:
          shape1=predictor(frame,rectangle)
          #Here we are determining facial landmarks for face region
          print('shape1 is')
          print(shape1)
          shape2=face_utils.shape_to_np(shape1)
          #In above step we are converting facial land marks (x,y) coordinates to numpy array
          print('shape2 is') #shape2 shape is (68,2) and thing to note is shape2 is a numpy array
          print(shape2.shape)
          print(shape2) # This command prints coordinates of all 68 facial land marks in form of numpy array
          lefteye=shape2[lstart:lend] #left eye contains 6 points (x,y) coordinates of lefteye in form of numpy array 
          print('lefteye is')
          print(lefteye.shape)
          #shape of lefteye numpy array is (6,2) as it contain coordinates of 6 points
          print(lefteye)
          righteye=shape2[rstart:rend] #right eye contains 6 points (x,y) coordinates of righteye in form of numpy array
          print('righteye is')
          print(righteye.shape) #righteye is numpy array of shape (6,2)
          print(righteye)
          A=dis.euclidean(lefteye[1],lefteye[5])
          B=dis.euclidean(lefteye[2],lefteye[4])
          C=dis.euclidean(lefteye[0],lefteye[3])
          lEAR=(A+B)/(2*C)
          print(lEAR)
          D=dis.euclidean(righteye[1],righteye[5])
          E=dis.euclidean(righteye[2],righteye[4])
          F=dis.euclidean(righteye[0],righteye[3])
          rEAR=(D+E)/(2.0*F)
          print(rEAR)
          fEAR=(lEAR+rEAR)/2.0
          #The below 4 lines of code are written for visualizing the eye regions, 1st and 3 rd lines are for visualizing left eye region
          # 2nd and 4 th line of code are visualizing right eye region on the frame
          lefteyehull=cv2.convexHull(lefteye)
          righteyehull=cv2.convexHull(righteye)
          cv2.drawContours(frame,[lefteyehull],-1,(0,255,0),1)
          cv2.drawContours(frame,[righteyehull],-1,(0,255,0),1)
          #In below line of code we are checking wheather fEAR value is less than threshold and if in such case we are increasing frame_count by1
          if fEAR<EAR_thresh:
                 frame_count+=1
                 #In below line of code we are checking if frame_count crosses EAR_frames value(if eyes are closed in more than 'EAR_frames'  consecutive frames)
                 if frame_count>=EAR_frames:
                      #Then we are converting boolean alarm_on to True
                      if not alarm_on:
                           alarm_on=True
                           #In below line we are checking wheather is there any input audio file or not
                           if args['alarm'] !="":
                                T=Thread(target=play_alarm,args=(args['alarm'],)) #The 3 lines of code written below are for starting a seperate thread where alarm can play in that thread
                                #T=Thread(target=playsound.playsound(args['alarm'],)) this sort of approach didnt worked for me in this case
                                #2 threads are not running parallely so instead of this i created function and giving target as function
                                T.deamon=True
                                T.start()
                 #This below line prints "U are drowsy..." that text when person fEAR less than threshold value   
                 cv2.putText(frame,"Looks like U are Drowsy wake up Sid",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)  
          else:
               frame_count=0
               alarm_on=False  
          cv2.putText(frame,"EAR: {:.2f}".format(fEAR),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)                       
     cv2.imshow("frame",frame)
     key=cv2.waitKey(1) & 0xFF
     #The below line of code means that until we press key 'a' process keeps on going and it stops only after pressing that key
     if key == ord("a"):
          sys.exit()
cv2.destroyAllWindows()
vs.stop()  
#1st problem is while making sound its not measuring ear value solved after creating a function and giviing target as that function
#2nd problem is it is not making sound after 8 times this is because audio file alarm is a small file of 5seconds
#If we use some what lenghty audio file then it keeps on ringing  
