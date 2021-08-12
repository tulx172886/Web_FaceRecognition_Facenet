import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from mtcnn.mtcnn import MTCNN
import cv2
from keras.models import load_model
import pickle

model_embedding = load_model("facenet_keras.h5")

#Label encode
out_encoder = LabelEncoder()
labels = np.array(['Bill Gates']*6 + ['Elon Musk']*7 + ['Jack Ma']*7 + ['Jeff Bezos']*7 + ['Mark Zuckerberg']*7)
out_encoder.fit(labels)

filemodel = "ModelSVM.pkl"
with open(filemodel, "rb") as file:
    model = pickle.load(file)




#Read image
img_org = cv2.imread("ti020721-1625219549326676729972.jpg")
img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
pixels = np.asarray(img)

#Detect Face
detector = MTCNN()
results = detector.detect_faces(pixels)
x1, y1, width, height = results[0]['box']
x1, y1 = abs(x1), abs(y1)
x2, y2 = x1 + width, y1+height
face = pixels[y1:y2, x1:x2]

#Resize face image
image = cv2.resize(face, dsize=(160,160))
face_array = np.asarray(image)


#Embedding face
face_pixels = face_array.astype('float32')
mean, std = face_pixels.mean(), face_pixels.std()
face_pixels = (face_pixels - mean) / std
samples = np.expand_dims(face_pixels, axis=0)
samples = model_embedding.predict(samples)



#Predict
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
print(yhat_class)
print(yhat_prob)

#Get name
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_name = out_encoder.inverse_transform(yhat_class)

print("Predict: %s" %(predict_name[0]))
print("Probability: %.3f" %(class_probability))



cv2.rectangle(img_org, (x1,y1), (x2,y2), (0,255,0), 1)
if class_probability>95:
    cv2.putText(img_org, predict_name[0] +" "+ str(round(class_probability,2)), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
    cv2.imshow("Result", img_org)
else:
    cv2.putText(img_org, "Unknown", (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
    cv2.imshow("Result", img_org)

cv2.waitKey(0)








