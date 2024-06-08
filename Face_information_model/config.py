# -------------------------------------- emotion_detection ---------------------------------------
# Emotion detection model
# path_model = 'emotion_detection/Modelos/model_dropout.hdf5'
# Model parameters, the image must be converted to a size of 48x48 in gray scale
w,h = 48,48
rgb = False
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

# -------------------------------------- face_recognition ---------------------------------------
# path images folder
path_images = "Face_information_model/images_db"

path_Gendermodel = 'Face_information_model/gender_detection/gender_model_weights.h5'

