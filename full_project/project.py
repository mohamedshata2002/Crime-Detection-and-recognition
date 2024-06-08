import tensorflow as tf
# from Face_information_model.Face_info import face_image
import cv2
import numpy as np
from  full_project.action  import frames_from_video_file
from full_project.lowlight import is_low_light
from full_project.upscaling import diffusion ,upscale_image
import os
import gc
from ultralytics import YOLO 
from django.contrib.auth.models import User    
from django.conf import settings
from myapp.models import Video
from myapp.utils import generate_thumbnail
import torch
import os
from full_project.video import image_enhancement
import torch 
import time

def fight_detction(path):
    model = tf.keras.models.load_model("E:\\workstation\\project\\project\\enhancement_Retinex\\action")
    # video_path = "E:\workstation\project\project\\test7.mp4"
   

    sample_video =frames_from_video_file(path, n_frames =20)
    sample_video = sample_video.reshape((1,20,224,224,3))
    act_score = model.predict(sample_video)
    print(act_score)
    if act_score >=0.5:
        print("fight")
        return "fight"
    else:
        print("No fight")
        return 'No fighting'


def Crime(path,user,Enhancement,plate_detection,low_light):

    # Load the YOLOv8 model
    model = YOLO("E:\\workstation\\prs\\crime_detection\\full_project\\yolo\\best.pt")
    pipeline =diffusion()
    
    
    
    cap = cv2.VideoCapture(path)
    names = model.names
    input_video_name = os.path.basename(path)
    output_video_name = f"{user.username}_{input_video_name.split('.')[0]}_output.mp4"
    output_video_path = os.path.join(settings.MEDIA_ROOT,"result",output_video_name)




    

    # video = Video.objects.create(
    #     user=user,
    #     video_name =output_video_name,
    #     video_path =output_video_path
    # )
    # Generate and save the thumbnail
    
    #video.save()





    os.makedirs(os.path.dirname(output_video_path),exist_ok=True)
    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('E:\workstation\project\project\crime_detection\myapp\saved\output_video.mp4', fourcc, 30.0, (512, 512))
    # out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (512, 512))
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 640))
    # Loop through the video frames
    crimes = []
    names = model.names

    plates = []
    frame_count = 0
    start = time.time()
    print("start time is",start)
    while True:
        
        success, frame = cap.read()
    
        if success :
            frame =cv2.resize(frame,(640,640))
            if frame_count%2==0 or Enhancement:
            
                
                ### low light images 
               
                if low_light:
                    if is_low_light(frame):
                        ### enhancement model
                        frame = image_enhancement(frame ,"E:\\workstation\\project\\project\\crime_detection\\full_project\\weight\\best_psnr_21.66_95000.pth")
                        torch.cuda.empty_cache()
                        gc.collect()
                if Enhancement:
                    print("Enhancement")
                    
                    frame = upscale_image(frame , pipeline=pipeline)
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(source=frame,conf=0.7, persist=True)
                torch.cuda.empty_cache()
                gc.collect()
                if plate_detection:
                    print("plate")
                    from plate_detection.main import plate_detector
                    plate_num = plate_detector(frame)
                    torch.cuda.empty_cache()
                    gc.collect()
                    if plate_num :
                        plates.append(plate_num)
                # Visualize the results on the frame

                annotated_frame = results[0].plot()
                # list of crimes 
                for m in results[0].boxes:
                    cls = m.cls
                    crimes.append(names[int(cls)])
                out.write(annotated_frame)
                
                
            if is_low_light:
                
                # frame,gender = face_image(annotated_frame)

                # Write the annotated frame to the output video

                out.write(annotated_frame)

                
                # Display the annotated frame
                # cv2.imshow("YOLOv8 Tracking", annotated_frame)
                # Break the loop if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord("q"):
            frame_count +=1
            
            
        else:
                # Break the loop if the end of the video is reached
            break

    # Release the video capture object, the VideoWriter object, and close the display window
    # cv2.imwrite('c:\\Users\\moham\\Downloads\\test4.jpg',  annotated_frame)
    end = time.time()
    print("End time is",end)
    execution_time = end -start
    print(f"Execution time for my_function: {execution_time:.6f} seconds")
    cap.release()
    out.release()
    Crimes = set(crimes)
    print("The Crime path of the uploaded video:", output_video_name)
    print("plates is :",plates)
    
    
    return Crimes ,output_video_path,plates
# fight_detction("E:\\workstation\\project\\project\\fire.mp4")
# Crime("E:\\workstation\\project\\project\\fire.mp4")