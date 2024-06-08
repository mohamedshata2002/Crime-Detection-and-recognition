from deepface import DeepFace
demography = DeepFace.analyze("juan.jpg", actions = [ 'gender', 'emotion'])
# demographics = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])



