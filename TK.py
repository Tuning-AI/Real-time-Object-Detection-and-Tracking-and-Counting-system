from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np 
from lib.sort import *
class_names = [ 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
                'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
                'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
tracker = Sort(max_age=20)
track_obj = "person"
track_ind = class_names.index(track_obj)
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("people.mp4")
total = []
while True : 
    _ , img = cap.read()
    # fps counter
    #fps, img = fpsReader.update(img,pos=(50,80),color=(0,0,255),scale=2,thickness=3)
    # make the prediction 
    results = model(img , conf=0.6,device="cpu")
    # creat empty detection array 
    detections = np.empty((0,5))
    for result in results : 
        # unpack results
        bboxs = result.boxes 
        for box in bboxs : 
            # bboxes
            x1  , y1 , x2 , y2 = box.xyxy[0]
            x1  , y1 , x2 , y2 = int(x1)  , int(y1) , int(x2) , int(y2)
            # confidence 
            conf = math.ceil((box.conf[0] * 100 ))
            # class name
            clsn = int(box.cls[0])
            if clsn == track_ind : 
                # creat the x1 y1 x2 y2 conf list for tracking
                cuArr = np.array([x1 , y1 , x2 , y2 , conf])
                # detection list update 
                detections = np.vstack((detections , cuArr))
        # make a tracking step
        result_track = tracker.update(detections)
        # unpack the tracking results
        for result_ in result_track : 
            x1 , y1 , x2 , y2 , id = result_
            # calculate the width and the height
            w,h = x2 - x1 , y2 - y1
            # convert it into int
            x1  , y1 , x2 , y2  , id , w , h = int(x1)  , int(y1) , int(x2) , int(y2) , int(id) , int(w) , int(h)
            cvzone.cornerRect(img , (x1 , y1 , w , h) , l=7)
            cvzone.putTextRect(img , f"{conf} % {class_names[clsn]} ID {id}" , (max(0,x1) , max(20 , y1)),thickness=1 ,colorR=(0,0,255) , scale=0.9 , offset=3)
            if total.count(id) == 0 : 
                total.append(id)  
            cv2.putText(img, f"Total Count : {len(total)}", (50 , 50),cv2.FONT_HERSHEY_PLAIN,2,(255, 0, 0), 3)
        cv2.imshow('Object Detection', img)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()