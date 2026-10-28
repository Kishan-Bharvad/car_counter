import cv2
from ultralytics import YOLO
import cvzone
import numpy
import torch
import math
from sort import *



cap = cv2.VideoCapture("video/4.mp4")                 
model = YOLO("yolov8n.pt")
# classname = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat']

classname = ['person', "bicycle", "car",'motorbike','aeroplane','bus','train','truck','boat',
            'traffic light', "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",'umbrella',
             "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
             'baseball glove', 'skateboard', 'surfboard', "tennis racket", "bottle", 'wine glass', 'cup',
             'fork', "knife", "spoon", "bowl", "banana", 'apple','sandwich', 'orange','broccoli',
             'carrot',"hot dog", 'pizza', 'donut', "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "Hook", "clock", "vase", "scissors",
               'teddy bear', "hair drier", "toothbrush"]

cap.set(3,1280)
cap.set(4,720)
tracker = Sort(max_age=100, min_hits=3, iou_threshold=0.3)

limits=[198,500,700,398]
totalcount=[]
cv2.namedWindow('Image', cv2.WINDOW_FULLSCREEN)
try:
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame,(880,740))
        result = model(frame,stream=True)
        
        detection = np.empty((0,5))
        for i in result:
            boxes = i.boxes
            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                w,h = x2-x1,y2-y1
#                 cvzone.cornerRect(frame,(x1,y1,w,h),l=8,t=3)
                
                conf = math.ceil((box.conf[0]*100))/100
#                 print(conf)
                cls = int(box.cls[0])
                currentclass = classname[cls]
                
                if currentclass =='car' or currentclass=='bicycle' or currentclass=='motorbike' or currentclass=='bus' or currentclass=='truck':
                    cvzone.cornerRect(frame,(x1,y1,w,h),l=8,t=3)
#                     cvzone.putTextRect(frame,f'vehicle {conf}',(max(0,x1),max(35,y1-10)),scale=0.8,thickness=1,offset=8)
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detection = np.vstack((detection,currentArray))
                    
        resulttracker = tracker.update(detection)
        
        cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
        
        for result in resulttracker:
            x1,y1,x2,y2,id = result
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1,y2-y1
            print("print id :",id)
            cvzone.cornerRect(frame,(x1,y1,w,h),l=8,t=3,colorR=(255, 0, 255))
            cvzone.putTextRect(frame,f'{id}',(max(0,x1),max(35,y1-10)),scale=0.8,thickness=1,offset=8)
            
            cx,cy= x1+w//2,y1+h//2
            cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
            
            if limits[0]< cx < limits[2] and limits[1]-100 < cy <limits[1]+100:
                if totalcount.count(id)==0:
                    totalcount.append(id)
            
        cvzone.putTextRect(frame,f' Count: {len(totalcount)}',(50,50))   
        
        print("totalcount ",totalcount)
        
        
        cv2.imshow('Image',frame)
        if cv2.waitKey(25)  & 0xFF == ord('h'):
            break
# except:
#     print('Kuch to gadabad hai daya!!!!!')

finally:
    cap.release()
    cv2.destroyAllWindows()