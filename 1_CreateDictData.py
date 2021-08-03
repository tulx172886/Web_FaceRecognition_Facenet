from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
import numpy as np

#Xác định 1 khuôn mặt từ ảnh
def extract_face(filename, required_size=(160,160)):
    image = Image.open(filename)            #Mở ảnh
    image = image.convert('RGB')            #Chuyển về hệ màu RGB
    pixels = asarray(image)                 #Chuyển về dạng ma trận

    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']       #Xác định khuôn mặt đầu tiên
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]              #Cắt Face sau khi detect

    #Resize pixels về size model
    image = Image.fromarray(face)            #Chuyển từ ma trận về ảnh
    image = image.resize(required_size)      #Resize (160,160)
    face_array = asarray(image)              #Chuyển về ma trận
    return face_array


# folder = "Data/Train/Bill Gates/"
# i = 1
# for filename in os.listdir(folder):
#     path = folder + filename
#     face = extract_face(path)
#     plt.subplot(2,7,i)
#     plt.axis('off')
#     plt.imshow(face)
#     i += 1
# plt.show()

#Load ảnh và xác định khuôn mặc cho tất cả các ảnh vào 1 dictionary
def load_faces(dictionary):     #Dictionary: Bill Gates, Elon Musk....
    faces = list()
    for filename in os.listdir(dictionary):
        path = dictionary + filename
        face = extract_face(path)
        faces.append(face)
    return faces

#Load tập dữ liệu xác định khuôn mặt và nhãn
def load_dataset(dictionary):
    X, y = list(), list()
    for subdir in os.listdir(dictionary):      #subdir: Bill gates, Elon Musk....
        path = dictionary + subdir + "/"      #Data/Train/Bill Gates/

        #Load tất cả face trong subdirectory
        faces = load_faces(path)
        labels = [subdir for i in range(len(faces))]
        print("Load %d examples for class %s" %(len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

trainX, trainy = load_dataset("Data/Train/")
print(trainX.shape, trainy.shape)

testX, testy = load_dataset("Data/Val/")
print(testX.shape, testy.shape)

#Lưu dữ liệu vào 1 file dạng nén
np.savez_compressed("Data.npz", trainX, trainy, testX, testy)
