'''
load the images that are in the folder database_images
'''
import Face_information_model.config as cfg
import os
from Face_information_model.my_face_recognition import f_main
import cv2
import numpy as np
import traceback


def load_images_to_database():
    list_images = os.listdir(cfg.path_images)
    list_images = [File for File in list_images if File.endswith(('.jpg','.jpeg','JPEG'))]

    name = []
    Feats = []

    for file_name in list_images:
        im = cv2.imread(cfg.path_images+os.sep+file_name)

        # get the features of the rose
        box_face = f_main.rec_face.detect_face(im)
        feat = f_main.rec_face.get_features(im,box_face)
        if len(feat)!=1:
            continue
        else:
            # insert the new characteristics into the data base
            new_name = file_name.split(".")[0]
            if new_name == "":
                continue
            name.append(new_name)
            if len(Feats)==0:
                Feats = np.frombuffer(feat[0], dtype=np.float64)
            else:
                Feats = np.vstack((Feats,np.frombuffer(feat[0], dtype=np.float64)))
    return name, Feats

def insert_new_user(rec_face,name,feat,im):
    try:
        rec_face.db_names.append(name)
        if len(rec_face.db_features)==0:
            rec_face.db_features = np.frombuffer(feat[0], dtype=np.float64)
        else:
            rec_face.db_features = np.vstack((rec_face.db_features,np.frombuffer(feat[0], dtype=np.float64)))
        # save the image
        cv2.imwrite(cfg.path_images+os.sep+name+".jpg", im) 
        return 'ok'
    except Exception as ex:
        error = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        return error