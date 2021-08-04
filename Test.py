import mtcnn
import cv2
import dlib

landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img_org = cv2.imread("bill-gates-giau-co-nao-1.jpg")
img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

detector = mtcnn.MTCNN()
results = detector.detect_faces(img)
print(results)
x1, y1, width, height = results[0]['box']
x1, y1 = abs(x1), abs(y1)
x2, y2 = x1+width, y1+height
face = img[y1:y2, x1:x2]
landmark = landmark_detector(img, dlib.rectangle(x1,y1,x2,y2))

for i in range(0,68):
    x = landmark.part(i).x
    y = landmark.part(i).y
    cv2.circle(img_org, (x,y),4,(0,255,0),-1)
cv2.imshow("Result",img_org)
cv2.waitKey(0)

