import numpy as np
from keras.models import load_model

#Load the face dataset
data = np.load("Data.npz")
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print("Data Loaded: ", trainX.shape, trainy.shape, testX.shape, testy.shape)

#Load facenet model
model = load_model("facenet_keras.h5")


#Get the face embedding for one face
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')    #Đưa ma trận về kiểu số thực

    mean, std = face_pixels.mean(), face_pixels.std()      #Chuẩn hóa giá trị pixel trên các kênh
    face_pixels = (face_pixels - mean) / std

    #Biến đổi face thành 1 mẫu
    samples = np.expand_dims(face_pixels, axis=0)

    #Đưa ra dự đoán để tạo embedding
    yhat = model.predict(samples)
    return yhat[0]

y = get_embedding(model, trainX[0])


#Chuyển đổi từng khuôn mặt trong train set về embedding
newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = np.asarray(newTrainX)
print(newTrainX.shape)

#Chuyển đổi từng khuôn mặt trong test set về embedding
newTestX = list()
for face_pixels in testX:
    embedding = get_embedding(model,face_pixels)
    newTestX.append(embedding)
newTestX = np.asarray(newTestX)
print(newTestX.shape)

#Lưu dữ liệu embedding vào file nén
np.savez_compressed("DataEmbedding.npz", newTrainX, trainy, newTestX, testy)


