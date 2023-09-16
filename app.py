from ultralytics import YOLO
import cv2
import cvzone
import math
import streamlit as st
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

with st.sidebar: 
    st.image("icon.png" , width=150)
    select_device = st.selectbox("Select compute Device :  ",
                                            ("CPU", "GPU"))
    track_obj = st.selectbox("select your tracking object : " , class_names)
    track_ind = class_names.index(track_obj)
    type_of_detect = st.radio("Select your Detection type : " , ["Live" , "File"])
    save = st.radio("Do You Want To save The Resuls ? " , ["Yes" , "No"])
    confd = st.slider("Select threshold confidence value : " , min_value=0.1 , max_value=1.0 , value=0.25)
    iou = st.slider("Select Intersection over union (iou) value : " , min_value=0.1 , max_value=1.0 , value=0.5)
tab0 , tab1 = st.tabs(["Home" , "Tracking"])
with tab0 : 
    st.header("About This Project :")
    st.write("""Object detection and tracking and counting in real-time using YOLO V8 and SORT algorithm" is a computer vision project that aims to 
    detect and track objects in real-time video streams using YOLO V8, a state-of-the-art object detection algorithm, and the Simple Online and Realtime
    Tracking (SORT) algorithm, a popular object tracking algorithm. The project involves preprocessing the video stream, applying object detection to 
    detect and classify objects in the scene, and then using SORT algorithm to track the detected objects across subsequent frames in the video. 
    The project also includes the development of a counting mechanism to count the number of detected and tracked objects in the video stream.
    The application of this project can be found in various domains like traffic monitoring, crowd control, and security surveillance.
""")


with tab1 :
    if select_device == "GPU" : 
        DEVICE_NAME = st.selectbox("Select GPU index : " , 
                                     (0, 1 , 2)) 
    elif select_device =="CPU" : 
        DEVICE_NAME = "cpu"
    if type_of_detect == "Live" : 
        source = st.text_input("Enter Your URL here :")
        cap = cv2.VideoCapture(source)
    elif type_of_detect == "File" : 
        file_upload = st.file_uploader("Upload your Video from here : ")
        if file_upload : 
            source = file_upload.name 
            cap = cv2.VideoCapture(source)
    start , _ , sa_ve = st.columns(3)
    with start : 
        st.write("Click to start Tracking ")
        start = st.button("Start")
    if save == "Yes" : 
        with sa_ve : 
            st.write("Double click to save results")
            sa_veb = st.button("save" , key="AZEFGGHYJ")
    if start: 
        frame_window = st.image( [] )
        #fpsReader = cvzone.FPS()
        tracker = Sort(max_age=20)
        # creat the model
        model = YOLO("yolov8n.pt")
        total = []
        if save == "Yes" :
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            fourcc = cv2.VideoWriter_fourcc(*'MP4V') #use any fourcc type to improve quality for the saved video
            out = cv2.VideoWriter(f'results/{source.split(".")[0]}.mp4', fourcc, 10, (w, h)) #Video settings 
        else : 
            pass
        try :
            while True : 
                _ , img = cap.read()
                # fps counter
                #fps, img = fpsReader.update(img,pos=(50,80),color=(0,0,255),scale=2,thickness=3)
                # make the prediction 
                results = model(img , conf=confd , iou=iou,device=DEVICE_NAME)
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
                    try : 
                        out.write(img)
                    except : 
                        pass
                frame  = cv2.cvtColor( img , cv2.COLOR_BGR2RGB)
                frame_window.image(frame)
        except :
            try : 
                cap.release()
            except :
                pass

    try : 
        if sa_veb : 
            cap.release()
            st.success('Done!' , icon="✅")
    except :
        pass
