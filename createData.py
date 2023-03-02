import cv2 as cv
import os
import time


#####################################################

myPath = 'data/images'
cameraNo = 1
cameraBrightness = 180
moduleVal = 5  # SAVE EVERY ITH FRAME TO AVOID REPETITION
minBlur = 500  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
grayImage = True # IMAGES SAVED COLORED OR GRAY
saveData = True   # SAVE DATA FLAG
showImage = True  # IMAGE DISPLAY FLAG
imgWidth = 180
imgHeight = 120


#####################################################

global countFolder
cap = cv.VideoCapture(cameraNo, cv.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10,cameraBrightness)


count = 0
countSave =0

def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists( myPath+ str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + str(countFolder))

if saveData:saveDataFunc()


while True:

    success, img = cap.read()
    img = cv.resize(img,(imgWidth,imgHeight))
    if grayImage:img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    if saveData:
        blur = cv.Laplacian(img, cv.CV_64F).var()
        if count % moduleVal ==0 and blur > minBlur:
            nowTime = time.time()
            cv.imwrite(myPath + str(countFolder) +
                    '/' + str(countSave)+"_"+ str(int(blur))+"_"+str(nowTime)+".png", img)
            countSave+=1
        count += 1

    if showImage:
        cv.imshow("Image", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
