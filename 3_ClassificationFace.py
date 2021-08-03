import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt

#Load DataSet Embedding
data = np.load("DataEmbedding.npz")
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print("Dataset: train=%d, test=%d" %(trainX.shape[0], testX.shape[0]))

#Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

#Label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)    #Mã hóa các label từ tên (Bill Gates, Elon Musk) thành các số 0,1,2,..
testy = out_encoder.transform(testy)

#fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

#Predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
#Score
score_train = accuracy_score(trainy, yhat_train)     #Tỉ lệ chính xác giữa nhãn thật và nhãn dự đoán
score_test = accuracy_score(testy, yhat_test)
#Kết quả
print("Accuracy: Train=%.3f, test=%.3f" %(score_train*100, score_test*100))


#Load faces
data_org = np.load("Data.npz")
testX_faces = data_org['arr_2']

#Test model on a random example from test dataset
selection = random.choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
print(random_face_pixels.shape)
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])  #Chuyển đổi ngược từ số ra tên class face

#Prediction for the face
samples = np.expand_dims(random_face_emb, axis=0)    #(1, 128)
print(samples.shape)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)

#Get name
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100   #Xác suất
predict_names = out_encoder.inverse_transform(yhat_class)
print("Predicted; %s (%.3f)" %(predict_names[0], class_probability))
print("Expected: %s" %(random_face_name[0]))

plt.imshow(random_face_pixels)
title = "%s (%.3f)" %(predict_names[0], class_probability)
plt.title(title)
plt.show()



