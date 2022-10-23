import cv2, time
from datetime import datetime
import pandas

#Step1 triggering the camera
video = cv2.VideoCapture(0) #using VideoCapture method to use main camera and capture screen

first_frame = None   #Creating a first frame
status_list = [None, None]
time_list = []
df = pandas.DataFrame(columns = ["Start", "End"])

while True:
    
    #Step2 reading the first frame of video
    check, frame = video.read()  #creating a video frame #here check is a boolean obj, frame is numpy obj
    status = 0    #Null variable to indicate motionless state

   #coverting frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    gray = cv2.GaussianBlur(gray, (21,21), 0)  #GaussianBlur for further processing (obj, gausian kernel tuple, standard deviation)

    if first_frame is None:  #condition will be true only for first iteration
        first_frame = gray  #assign gray frame to first_frame
        continue   #control will go to start of the loop (while), without executing the lines below 

    delta_frame = cv2.absdiff(first_frame, gray) #to find the difference between 1st motionless frame and 2nd frame that is in motion 
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]  #[1] to access the 2nd item in tuple for binary method
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)  #method to dilate the frame

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #creating a contour obj using various methods

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:   #checking 10000 = 100x100 pixels
            continue
        status = 1   #motion detected
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
    status_list.append(status)
    
    #capturing current time when motion state changes
    if status_list[-1]==1 and status_list[-2]==0:
        time_list.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        time_list.append(datetime.now())

    cv2.imshow("Gray frame", gray)
    cv2.imshow("Delta Image", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Colour Frame", frame)

    key = cv2.waitKey(1)
    #print(gray) 
    #print(delta_frame)
    #print(thresh_frame)

    if key == ord('q'):
        if status == 1:
            time_list.append(datetime.now())
 
        break

print(status_list)
print(time_list)

for i in range(0, len(time_list), 2):
    df = df.append({"Start":time_list[i], "End":time_list[i+1]}, ignore_index = True)


df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows()