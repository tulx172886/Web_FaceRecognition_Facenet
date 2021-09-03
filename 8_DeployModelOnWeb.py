from flask import render_template, Flask, request
import cv2
from mtcnn import mtcnn
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

face_detector = mtcnn.MTCNN()
model_embedding = load_model("facenet_keras.h5")

data_embedding = np.load("DataEmbedding.npz")
labels = data_embedding['arr_1']
out_encoder = LabelEncoder()
out_encoder.fit(labels)

#Load model để predict khuôn mặt
with open("ModelSVM.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

#Thư mục lưu các file load lên
app.config["UPLOAD_FOLDER"] = "static"

@app.route("/", methods=["POST", "GET"])
def index():
    list_oldfile = os.listdir('static')
    for file_old in list_oldfile:
        os.remove("static/"+file_old)
    if request.method == "GET":
        return render_template("index.html")
    else:
        #Load ảnh được gửi lên
        image = request.files['FileUpload']
        image_path = 'static/' + image.filename
        image.save(image_path)

        #Tiến hành thao tác nhận diện
        img_org = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
        pixels = np.asarray(img_rgb)
        results = face_detector.detect_faces(img_rgb)
        print(results)
        print("Số khuôn mặt phát hiện trong ảnh: ", len(results))
        if results:
            # colors = [(0,255,0), (0,0,255),(255,0,0),(255,255,0),(0,255,255)]
            colors = np.random.randint(0,255,(len(results), 3))
            print(colors)
            for i in range(len(results)):
                print(i)
                if results[i]['confidence'] > 0.95:
                    x1, y1, width, height = results[i]['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height
                    face = pixels[y1:y2, x1:x2]
                    face = cv2.resize(face, (160,160))
                    face_pixels = face.astype('float32')

                    #Chuẩn hóa các pixels
                    mean , std = face_pixels.mean(), face_pixels.std()
                    face_pixels = (face_pixels - mean) / std
                    samples = np.expand_dims(face_pixels, axis=0)
                    samples = model_embedding.predict(samples)       #face embedding

                    #Predict
                    yhat_class = model.predict(samples)             # Predict label
                    yhat_prob = model.predict_proba(samples)         #Xác suất

                    #Chuyển đổi thành dữ liệu để hiển thị
                    class_index = yhat_class[0]
                    print("Class index: ",class_index)
                    class_probability = yhat_prob[0, class_index] * 100
                    predict_name = out_encoder.inverse_transform(yhat_class)

                    print("Predict: %s" %(predict_name[0]))
                    print("Probability: %.3f" %(class_probability))

                    color = (int(colors[class_index][0]), int(colors[class_index][1]), int(colors[class_index][2]))
                    print(color)
                    cv2.rectangle(img_org, (x1, y1), (x2, y2), color, 2)
                    if class_probability > 95:
                        cv2.putText(img_org, predict_name[0] + " " + str(round(class_probability, 2)), (x1, y1),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                    else:
                        cv2.putText(img_org, "Unknown", (x1, y1),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            cv2.imwrite(image_path, img_org)
            return render_template("index.html", imgname=image.filename,
                                                # facename=predict_name[0],
                                                # probability=class_probability,
                                                success=True)
                    # else:
                    #     cv2.putText(img_org, "Unknown", (x1, y1),
                    #                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    #     cv2.imwrite(image_path, img_org)
                    #     return render_template("index.html", imgname=image.filename,
                    #                                         facename="Unknown",
                    #                                         probability="",
                    #                                         success=True)


        else:
            return render_template("index.html",  imgname=image.filename,
                                                    fail = "Không phát hiện được khuôn mặt trong ảnh")

if __name__ == "__main__":
    app.run()

