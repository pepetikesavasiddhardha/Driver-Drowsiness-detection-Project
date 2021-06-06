 Driver Drowsiness detection Project:
 First we have to import all libraries required for this package.The libraries which we may have to install were scipy,playsound,imutils and dlib.Installaing first 3 libraries is very easy but installing dlib is very long process.I got to know about this process from https://youtu.be/xaDJ5xnc8dc this youtube video
playsound is for playing audio files of format .wav,MP3.Later we will add arguments using argparser one argument is path to shape_predictor file which is helpful in detecting facial landmarks another argument is path for alarm audio file and third argument is for giving input the index of webcam we want to use in our case we took default value zero
Also for doing this project one must have idea on EYE ASPECT RATIO(EAR).EAR in simple terms is length of vertical region divided by length of horizontal region.So from this we can simply understand that EAR is zero if our eye closes,as vertical distance will tend to zero
Now for my case i took EAR threshold value as 0.3 and number of consecutive frames for which EAR can't be less than threshold as 5 for making model more sensitive ,but in general cases we take that value as 48.
Now we will initialize frontal_face_detector for detecting faces and then we will initialize shape_predictor model and then we will find out the indexes of coordinate points of left eye and right eye
Now we will start our webcam and then we will start reading our frame.
Now convert frame width into 450 and then we will detect all faces in frame(ingeneral we will expect one face)  this model can findout wheather a single person is sleeping or not even if there are many persons in frame
Now we will loop through all persons faces avaliable in frame and first we will findout coordinate points of all 68 facial landarks and store this information in form of numpy array
Later we will extract 6 coordinate points of lefteye and 6 points of right eye and store them seperately as 2 numpy arrays for 2 eyes.So shape of lefteye numpy array is (6,2) and this is obviously same as for right eye
Now we will calculate EAR for left eye from information avaliable above.Similarly we will calculate EAR for right eye also and we will name them as lEAR,rEAR respectively
we will take average of these both values and we will call it as fEAR which is final_EAR for the eyes of person in that frame
For visualizing the left eye and right eye regions we used cv2.drawContours function.
Now using some if loops we will create code in such a way that it will count all the consecutive frames where the fEAR is less than threshold EAR value and if this frame count crosses our specific value then it will start running a seperate thread in which code is written for playing the alarm sound.
We are writting in seperate thread, if written in same thread then while playing alarm music it may not check the EAR value(finding EAR value gets halted)to avoid this we do these tasks in seperate Threads
Also a text will print on frame if fEAR value is less than threshold value even for one frame so that he may become alert by seeing this before going into sleep,that text is "Looks like U are Drowsy wake up Sid" 
Finally if keyword "a" is pressed then only the process will get stopped
