import cv2
import numpy as np
import handtrackingmodule as htm
import time
import autopy

wCam,hCam=640,480
frameR=100
smoothening=7
wScr,hScr=autopy.screen.size()
print(wScr,hScr)
cap=cv2.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)
ptime=0
plocX,plocY=0,0
clocX,clocY=0,0
detector=htm.HandDetector(maxHands=2)

while True:
    success,img=cap.read()
    img=detector.findHands()
    lmlist,bbox=detector.findPosition(img)
    print(lmlist)
    if len(lmlist)!=0:
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]
        print (x1,y1,x2,y2)
    fingers=detector.fingersUp()
    print(fingers)
    if fingers[1]==1 and fingers[2]==0:
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hcam-frameR),(255,0,255),2)
        x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
        y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))
        clocX=plocX+(x3-plocX)/smoothening
        clocY=plocY+(y3-plocY)/smoothening
        autopy.mouse.move(wScr-x3,y3)
        cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        plocX,plocY=clocX,clocY
    if fingers[1]==1 and fingers[2]==1:
        length, img, lineinfo=detector.findDistance(8,12,img)
        print(length)
        if length<40:
            cv2.circle(img,(lineinfo[4],lineinfo[5]),15,(255,0,255),cv2.FILLED)
            autopy.mouse.click()
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)
