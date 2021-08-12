import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from mtcnn.mtcnn import MTCNN
import cv2
from keras.models import load_model

model_embedding = load_model("facenet_keras.h5")

#Load các face embedding có kích thước (128, )
data_embedding = np.load("DataEmbedding.npz")
trainX, trainy, testX, testy = data_embedding['arr_0'], data_embedding['arr_1'], data_embedding['arr_2'], data_embedding['arr_3']
print("Dataset with len(trainX)=%d" %(trainX.shape[0]))

#Normalize input vector
in_encoder = Normalizer(norm = "l2")
trainX = in_encoder.transform(trainX)

#Label encode
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)


#Fit  model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #Read image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pixels = np.asarray(img)

    #Detect Face
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if results:
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
        if class_probability > 95:
            predict_name = out_encoder.inverse_transform(yhat_class)
            print("Predict: %s" %(predict_name[0]))
            print("Probability: %.3f" %(class_probability))

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(frame, predict_name[0] +" "+ str(round(class_probability,2)), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

        else:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
            cv2.putText(frame, "Unknown", (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
        cv2.imshow("Result", frame)
        cv2.waitKey(1)

    else:
        continue

cap.release()
cv2.destroyAllWindows()








