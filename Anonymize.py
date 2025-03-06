import ultralytics
import cv2
import pandas as pd 
from ultralytics import YOLO
import copy
import easyocr
import torch
import os
import mediapipe as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
import argparse
from argparse import RawTextHelpFormatter
from timeit import default_timer as timer
import numpy as np

#Global Variables


def political(frame):
    reader = easyocr.Reader(['en'],gpu=True)
    results = reader.readtext(frame)
    wordlist = ['trump'] #List of words that should be blurred
    num_to_letter = {
        '0': 'o',
        '1': 'l',
        '2': 'z',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '6': 'b',
        '7': 't',
        '8': 'b',
        '9': 'g'
    }
    translation_table = str.maketrans(num_to_letter)
    
    for t in results:
        bbox,text, score = t  #BBOX [[x1,y1]->top left,[x3,y3]->bottom right]
        text = text.lower()
        text = text.translate(translation_table)
        if text in wordlist:
            y1 = int(bbox[0][1])
            y2 = int(bbox[2][1])+2
            x1 = int(bbox[0][0])
            x2 = int(bbox[2][0])+2
            print(text)
            try:
                frame[y1:y2,x1:x2] = cv2.GaussianBlur(frame[y1:y2,x1:x2],(51,51),0)
                if args.blackboxblur== True:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), thickness=-1)
            except:
                print("ITEM FOUND CANNOT BLUR")
    return frame


def person_tracking(frame):

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(min_detection_confidence =0.0,min_tracking_confidence=0.0) as holistic:
        #Change colour for better detection
        frame_col = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #Detection
        results = holistic.process(frame)
        
        #Revert image changes
        frame = cv2.cvtColor(frame_col,cv2.COLOR_RGB2BGR)

        #Using Blur
        #frame = cv2.GaussianBlur(frame,(51,51),0)
        frame = cv2.GaussianBlur(frame,(51,51),0)
        if args.blackboxblur==True:
            h,w,_ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), thickness=-1)
                        
                    

        

        if args.mediapipe == True:
            #Draw face landmarks
            mp_drawing.draw_landmarks(frame,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION) # USE contour if u just want the face outline
            #Right Hand
            mp_drawing.draw_landmarks(frame,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
            #Left Hand 
            mp_drawing.draw_landmarks(frame,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
            #Pose Detection
            mp_drawing.draw_landmarks(frame,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
        



        return frame




def custom_blur(frame,conf,x1,x2,y1,y2):
    global Number_of_Plates 
    LP_detector = YOLO('./LPlate.pt')
    plates = LP_detector(frame)[0]
    flag = True

    for plate in plates.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id = plate
        if score>0.1:
            if flag:
                Number_of_Plates = Number_of_Plates  + 1
                flag=False
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))
            cv2.putText(frame,f'Plate {conf:.2f}',(int(x1),int(y1)- 10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0), 2)
            ROI = frame[int(y1):int(y2),int(x1):int(x2)]
            frame[int(y1):int(y2),int(x1):int(x2)] = cv2.GaussianBlur(ROI,(51,51),0)
            


    return frame




def ocr_blur(frame):
    global Number_of_Plates
    flag = True
    reader = easyocr.Reader(['en'],gpu=True)
    results = reader.readtext(frame)
    
    for t in results:
        if flag:
            Number_of_Plates = Number_of_Plates + 1
            flag=False
        bbox,text, score = t
        print(bbox[0][0])
        print(bbox[2][0])
        y1 = int(bbox[0][1])
        y2 = int(bbox[2][1])+2
        x1 = int(bbox[0][0])
        x2 = int(bbox[2][0])+2
        frame[y1:y2,x1:x2] = cv2.GaussianBlur(frame[y1:y2,x1:x2],(51,51),0)
        if args.blackboxblur==True:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), thickness=-1)
        
    return frame



#Different Blur Functions

# Basic blur just takes the coordinates of the car puts a blur on half ot it.
def basic_blur(frame,x1,x2,y1,y2):
    blur_frame = copy.deepcopy(frame)
    roi = frame[int(((y1+y2)/2)-10):int(y2-5), int(x1):int(x2)]

    #roi = frame[int(((int(y1)+int(y2))/2)-20):int(y2-(y2-(((int(y1)+int(y2))/2)-20))/2), int(x1+((x2-x1)/4)):int(x1+(((x2-x1)/4)*3))]
    if args.blackboxblur==True:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), thickness=-1)

    blur_image = cv2.GaussianBlur(roi,(51,51),0)
    #blur_frame[int(((int(y1)+int(y2))/2)-20):int(y2-(y2-(((int(y1)+int(y2))/2)-20))/2), int(x1+((x2-x1)/4)):int(x1+(((x2-x1)/4)*3))] = blur_image
    blur_frame[int(((y1+y2)/2)-10):int(y2-5), int(x1):int(x2)] = blur_image
    return blur_frame



#Cascade blur using the haar russian plates
#Detection works but not as effective hence cannot be used.

def cascade_blur(frame,frame_with_plates):
    flag = False
    global Number_of_Plates
    blur_frame = copy.deepcopy(frame)
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    plates = plate_cascade.detectMultiScale(blur_frame,1.2)


    for plate in plates:
        if flag == False:
            frame_with_plates = frame_with_plates  + 1
            Number_of_Plates = Number_of_Plates + 1
            flag = True
        (x,y,w,h) = plate
        detected_plate = blur_frame[y:y+h,x:x+w]
        cv2.rectangle(blur_frame,(x,y),(x+w,y+h),(0,255,0),3)
        blur_frame[y:y+h,x:x+w] = cv2.GaussianBlur(detected_plate,(51,51),0)
        if args.blackboxblur== True:
            cv2.rectangle(blur_frame,(x,y),(x+w,y+h),(0, 0, 0),thickness=-1)
    print("Detections of plates : "+str(frame_with_plates))
    
    return blur_frame,frame_with_plates

#Final Blur
def final_blur(frame,conf,x1,x2,y1,y2,frame_original):
    LP_detector = YOLO('./LPlate.pt')
    plates = LP_detector(frame)[0]
    global Number_of_Plates
    Number_of_Plates = Number_of_Plates + 1
    
    if str(plates.boxes.data.tolist()) == "[]":
        frame_original = basic_blur(frame_original,x1,x2,y1,y2)
        return frame_original[int(y1):int(y2),int(x1):int(x2)]
        
    for plate in plates.boxes.data.tolist():
        print(str(plates.boxes.data.tolist()) + "This is working")
        if plates.boxes.data.tolist() == []:
            print("No detection")
            break
        x1,y1,x2,y2,score,class_id = plate
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))
        cv2.putText(frame,f'Plate {conf:.2f}',(int(x1),int(y1)- 10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0), 2)
        ROI = frame[int(y1):int(y2),int(x1):int(x2)]
        frame[int(y1):int(y2),int(x1):int(x2)] = cv2.GaussianBlur(ROI,(51,51),0)
        if args.blackboxblur== True:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), thickness=-1)



    return frame



def startBlur(input_file,output_file):
    global Number_of_Plates
    global Number_of_Cars
    global Number_of_Person
    print(torch.zeros(1).cuda())       
    check_frame = True
    vehicle_model = YOLO('yolov8n.pt')
    #List of Items in Coco Dataset
    class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    #Get Video
    cap=cv2.VideoCapture(input_file)
    ret = True
    frame_counter = -1
    vehicles = [2,3,5,7]
    frame_with_plates = 0  # for second method
    frame_with_cars = 0 
    # Processed Video
    output_path = output_file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2 .CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path,fourcc,fps,(frame_width,frame_height))
    #Capturing frames one by one for detection
    while ret:
        ret, frame = cap.read()
        frame_counter = frame_counter + 1 # incriment the frame counter]
        flag_car = True


       
        if ret:
            detections = vehicle_model(frame)[0] 
            for detection in detections.boxes.data.tolist():
                x1,y1,x2,y2,conf,cid  = detection
                conf = conf*100

                if int(conf) >20 and cid in vehicles :
                    Number_of_Cars = Number_of_Cars + 1
                    print("Car incrimented : "+str(Number_of_Cars))
                    if flag_car:
                        frame_with_cars = frame_with_cars + 1
                        flag_car = False

                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))
                    cv2.putText(frame,f'Car {conf:.2f}',(int(x1),int(y1)- 10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0), 2)

                    #Processing

                    #Blur
                    if args.blur == "basic_blur":
                        frame = basic_blur(frame,x1,x2,y1,y2) #Uncomment for basic Blur
                    if args.blur =="cascade_blur": 
                        frame[int(y1):int(y2),int(x1):int(x2)],frame_with_plates = cascade_blur(copy.deepcopy(frame[int(y1):int(y2),int(x1):int(x2)]),frame_with_plates) #Uncomment for cascade blur
                    if args.blur =="ocr_blur":
                        frame[int(y1):int(y2),int(x1):int(x2)]=ocr_blur(copy.deepcopy(frame[int(y1):int(y2),int(x1):int(x2)])) #OCR Model
                    if args.blur =="custom_blur":
                        frame[int(y1):int(y2),int(x1):int(x2)]=custom_blur(copy.deepcopy(frame[int(y1):int(y2),int(x1):int(x2)]),conf,x1,x2,y1,y2) #best Model
                    if args.blur == "final_blur":
                        frame[int(y1):int(y2),int(x1):int(x2)]=final_blur(copy.deepcopy(frame[int(y1):int(y2),int(x1):int(x2)]),conf,x1,x2,y1,y2,frame) #best Model + Final Tuning
                    

                if cid == 0 : #Person Detection
                    
                    Number_of_Person=Number_of_Person  + 1
                    #To check if it works
                    frame_print = frame[int(y1):int(y2),int(x1):int(x2)]
                    frame_print=person_tracking(frame_print)
                    frame[int(y1):int(y2),int(x1):int(x2)] = frame_print
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))
                    cv2.putText(frame,f'Person {conf:.2f}',(int(x1),int(y1)- 10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0), 2)

                print(conf)
            if args.ocr == True:
                frame=political(copy.deepcopy(frame)) # To enable detection of political messages.
        output.write(frame)   
        try:
            cv2.imshow('Cars',frame)
            if cv2.waitKey(25) & 0xff == ord('q'):
                break
        except :
            print("Frame Gen Error")
        

    cap.release()
    output.release()
    print("frame counter  : "+ str(frame_counter))
    print("Frames with Cars  : " + str(frame_with_cars))
    print("Old Method Plates " + str(frame_with_plates))




# Cuts video into segements which makes it easier to process.
def cut_video(input,out_dir,seg_dur):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    cap = cv2.VideoCapture(input)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        total_duration = frame_count // fps
    except:
        print("Video corrupted or not found")
        exit()
        

    seg_frame = seg_dur * fps
    success, frame = cap.read()
    count = 0
    segment_index = 0
    segment_frames_written = 0

    out = None

    while success:
        if segment_frames_written == 0:
            out = cv2.VideoWriter(
                os.path.join(out_dir, f'segment_{segment_index}.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'), fps,
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )
        out.write(frame)
        segment_frames_written += 1
        if segment_frames_written == seg_frame:
            out.release()
            segment_index += 1
            segment_frames_written = 0
        success, frame = cap.read()
        count += 1
    if out:
        out.release()

    cap.release()

def combine_videos(segment_dir, output_path):
    segment_files = sorted([os.path.join(segment_dir, f) for f in os.listdir(segment_dir) if f.endswith('.mp4')]) # Sorting the files in order so that combining will be correctly done.
    print(segment_files)
    cap = cv2.VideoCapture(segment_files[0])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path,fourcc,fps,(frame_width,frame_height))
    
    for seg_file in segment_files:
        cap = cv2.VideoCapture(seg_file)
        success, frame = cap.read()
        while success:
            output.write(frame)
            success, frame = cap.read()
        cap.release()


def cut_video_mov(video_path, output_dir, segment_duration):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = VideoFileClip(video_path)
    duration = video.duration
    segment_index = 0

    for start_time in range(0, int(duration), segment_duration):
        end_time = min(start_time + segment_duration, int(duration))
        segment = video.subclip(start_time, end_time)
        segment.write_videofile(os.path.join(output_dir, f'segment_{segment_index}.mp4'), codec="libx264")
        segment_index += 1






if __name__ == "__main__":
    Number_of_Cars=0
    Number_of_Plates=0
    Number_of_Person=0

    parser = argparse.ArgumentParser(description='GDPR based Framework to annonmize Personally indentifiable information',formatter_class=RawTextHelpFormatter)

    #Required Arguments
    parser.add_argument('input',type=str,help='The input file which should be processed')




    #Optional Arguments
    parser.add_argument('--blur',type=str,choices=['basic_blur','cascade_blur','ocr_blur','custom_blur','final_blur'],default='final_blur',
                        help='Specifies which type of method is used to blur out the number plates of vehicles.\n\nbasic_blur : Blurs out approximately half of the vehicle.Fast but a lot of information is lost.\n'
                        'cascade_blur: Uses the cv2 cascade classifier.Works on the haarcascade_russian_plate_number dataset.This method is not effective enough to be used in this project.\n'
                        'ocr_blur : Uses optical character recognition to blur out all letters from the car.Slow and loses a lot of information in the process.\n'
                        'custom_blur : Uses YOLO model which is fined tuned with a number plate dataset.Best model fast and accurate.\n'
                        'final_blur : Uses custom_blur model along with basic_blur to cover most of the cases.This method is best method for blurring.\n'
                        )
    parser.add_argument('--blackboxblur',action='store_true',help='Enables Black Box Blurring instead of Gaussian Blur')
    parser.add_argument('--split',type=int,choices=range(2,120),metavar=' Range [2-120]',default=60,help='Specifies the size of the segments the video is split into.The default value is 60 seconds and this is recommended unless the video is shorter in length.Range is [2-120] seconds')
    parser.add_argument('--ocr',action='store_true',help='Enables detection and bluring of political messages')
    parser.add_argument('--mediapipe',action='store_true',help='Enables mediapipe and shows pose tracking')


    args = parser.parse_args()
    print(args.input)
    print(args.blur)

    #cut_video_mov(args.input,'output_segments',args.split)
    
    cut_video_mov(args.input,'output_segments',args.split)

    #startBlur('./Test1.mp4','Test_Final',) # Only for testing
    
    if not os.path.exists('processed_segments'):
        os.makedirs('processed_segments')
    seg_files = [f for f in os.listdir('output_segments') if f.endswith('.mp4')]
    
    start_time = timer()
    for segment_file in seg_files:
        startBlur(os.path.join('output_segments', segment_file), os.path.join('processed_segments', segment_file))
    end_time = timer()

    
    
    combine_videos('processed_segments', 'final_output.mp4')
    
    print("Number of Cars Detected : "+ str(Number_of_Cars))
    print("Number of Plates Detected : " + str(Number_of_Plates))
    print("Number of People Detected : "+str(Number_of_Person))
    print("Time Taken : "+ str(int(end_time-start_time)))
    

    