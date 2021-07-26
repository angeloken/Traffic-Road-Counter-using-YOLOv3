import cv2
import numpy as np
import time

cap = cv2.VideoCapture('video/sample_2_Trim.mp4')
#cap = cv2.VideoCapture('Pexels Videos 2103099.mp4')
path_label = 'coco.names'

classes = []
with open(path_label,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

weight_height_target = 640
model_cfg,model_weights = 'yolov3608.cfg','yolov3608.weights'
confThreshold = 0.4
nmsThreshold = 0.2
inccount1 = 0
inccount2 = 0
inccount3 = 0
inccount4 = 0
inccount5 = 0
inccount6 = 0
inccount7 = 0
inccount8 = 0
inccount9 = 0
inccount10 = 0
inccount_reset = 0
start_time = time.time()

net = cv2.dnn.readNetFromDarknet(model_cfg,model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

result = cv2.VideoWriter('result.mp4',cv2.VideoWriter_fourcc(*'XVID'),20,(640,640))

def findObject(outputs,img):
    hT,wT,cT = img.shape
    bbox = []
    classIds = []
    confs = []
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    count8 = 0

    for output in outputs:
        for det in output:
            score = det[5:]
            classId = np.argmax(score)
            conf = score[classId]
            if classId == 2 or classId == 3 or classId == 5 or classId == 7:
                if conf > confThreshold:
                    w,h = int(det[2]*wT), int(det[3]*hT)
                    x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                    xMid = int((x + (x + w)) / 2)
                    yMid = int((y+(y+h))/2)
                    cv2.circle(img, (xMid, yMid), 2, (0, 0, 255), 5)
                    bbox.append([x,y,w,h])
                    confs.append(float(conf))
                    classIds.append(classId)
                    if yMid > 417 and yMid <422 and xMid>100 and xMid<295:
                        if classId== 2:
                            count1 = count1 +1
                        elif classId == 3:
                            count2 = count2 +1
                        elif classId == 5:
                            count3 = count3 +1
                        elif classId == 7:
                            count4 = count4 +1
                    if yMid > 386 and yMid <388 and xMid>340 and xMid<480:
                        if classId == 2:
                            count5 = count5 +1
                        elif classId == 3:
                            count6 = count6 +1
                        elif classId == 5:
                            count7 = count7 +1
                        elif classId == 7:
                            count8 = count8 +1
            else:
                continue
    draw_box = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    for i in draw_box:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.putText(img,f'{classes[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),(cv2.FONT_HERSHEY_SIMPLEX),1,(255,255,0),1)
        cv2.line(img, (100, 420), (295, 420), (0, 0, 255, 2))  # RED Line
        cv2.line(img, (100, 418), (295, 418), (0, 255, 0, 1))  # Green Offset Line
        cv2.line(img, (100, 422), (295, 422), (0, 255, 0, 1))  # Green Offset Line
        cv2.line(img, (340, 387), (480, 387), (255, 255, 255, 2))  # RED Line
        cv2.line(img, (340, 386), (480, 386), (0, 255, 0, 1))  # Green Offset Line
        cv2.line(img, (340, 388), (480, 388), (0, 255, 0, 1))  # Green Offset Line
    return count1,count2,count3,count4,count5,count6,count7,count8

while True:
    _, img = cap.read()
    img = cv2.resize(img,(640,640))
    cv2.imshow('video',img)
    blob = cv2.dnn.blobFromImage(img,1/255,(weight_height_target,weight_height_target),[0,0,0,0],crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputnames = [layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputnames)
    counter1,counter2,counter3,counter4,counter5,counter6,counter7,counter8 = findObject(outputs,img)

    inccount1 = inccount1 + counter1
    inccount2 = inccount2 + counter2
    inccount3 = inccount3 + counter3
    inccount4 = inccount4 + counter4
    inccount5 = inccount5  + (counter1+counter2+counter3+counter4)
    inccount6 = inccount6 + counter5
    inccount7 = inccount7 + counter6
    inccount8 = inccount8 + counter7
    inccount9 = inccount9 + counter8
    inccount10 = inccount10 + (counter5 + counter6 + counter7+ counter8)
    run_time = time.time()
    iccount_reset = int(time.time()-start_time)
    if inccount_reset == 3600:
        inccount1 = 0
        inccount2 = 0
        inccount3 = 0
        inccount4 = 0
        inccount5 = 0
        inccount6 = 0
        inccount7 = 0
        inccount8 = 0
        inccount9 = 0
        inccount10 = 0
        inccount_reset = 0
        start_time = run_time

    cv2.putText(img,f'counting Car : {inccount1}',(25,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
    cv2.putText(img, f'counting Motor : {inccount2}', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(img, f'counting Bus : {inccount3}', (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(img, f'counting Truck : {inccount4}', (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(img, f'Total : {inccount5}', (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.putText(img, f'counting Car2 : {inccount6}', (430, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f'counting Motor2 : {inccount7}', (430, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f'counting Bus2 : {inccount8}', (430, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f'counting Truck2 : {inccount9}', (430, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(img, f'Total : {inccount10}', (430, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    result.write(img)
    cv2.imshow('video',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()

