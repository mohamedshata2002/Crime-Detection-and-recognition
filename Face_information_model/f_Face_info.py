import cv2
import numpy as np
import face_recognition
from Face_information_model.gender_detection import f_my_gender
from Face_information_model.my_face_recognition import f_main

gender_detector =  f_my_gender.Gender_Model()
rec_face = f_main.rec()
#----------------------------------------------

def get_face_info(im):
    # face detection
    
    boxes_face = face_recognition.face_locations(im)
    out = []
    if len(boxes_face)!=0:
        for box_face in boxes_face:
            box_face_fc = box_face
            x0,y1,x1,y0 = box_face
            box_face = np.array([y0,x0,y1,x1])
            face_features = {
                "name":[],
                "gender":[],
                "expression":[],
                "bbx_frontal_face":box_face             
            } 

            face_image = im[x0:x1,y0:y1]

            # -------------------------------------- face_recognition ---------------------------------------
            face_features["name"] = rec_face.recognize_face2(im,[box_face_fc])[0]
            # -------------------------------------- gender_detection ---------------------------------------
            face_features["gender"] = gender_detector.predict_gender(face_image)
            # -------------------------------------- out ---------------------------------------       
            out.append(face_features)
    else:
        face_features = {
            "name":[],
            "gender":[],
            "bbx_frontal_face":[]             
        }
    return out

def bounding_box(out,img):
    genders = []
    names = []
    for data_face in out:
        box = data_face["bbx_frontal_face"]
        
        if len(box) == 0:
            continue
        else:
            x0,y0,x1,y1 = box
            img = cv2.rectangle(img,
                            (x0,y0),
                            (x1,y1),
                            (0,255,0),2);
            thickness = 3
            fontSize = 2
            step = 30
            genders.append(data_face["gender"])
            names.append(data_face["name"])
            try:
                cv2.putText(img, "gender: " +data_face["gender"], (x0, y0-step-10*1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "name: " +data_face["name"], (x0, y0-step-10*10), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
    return img ,genders , names
