import cv2
import pickle
import numpy as np
from lbp_class import LocalBinaryPatterns
import cvlib as cv
import os

# Tải mô hình
svm_model_path ='gender_detection_model3.pkl'
if not os.path.exists(svm_model_path):
    raise FileNotFoundError(f"Mô hình SVM không tìm thấy tại {svm_model_path}")

with open(svm_model_path, 'rb') as f:
    model_data = pickle.load(f)

model_svm = model_data['model']
scaler = model_data['scaler']
le = model_data['label_encoder']
feature_length = model_data['feature_length']

# Khởi tạo LBP
desc = LocalBinaryPatterns(24, 8)

# Khởi tạo webcam
video = cv2.VideoCapture(0)
print("Bắt đầu nhận diện giới tính qua webcam. Nhấn 'q' để thoát.")

while True:
    ret, frame = video.read()
    if not ret:
        print("Không thể truy cập webcam!")
        break

    faces, confidences = cv.detect_face(frame)

    for face in faces:
        (startX, startY, endX, endY) = face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        face_crop = frame[startY:endY, startX:endX]

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray_face)

        # Kiểm tra số đặc trưng
        if len(hist) != feature_length:
            raise ValueError(f"Số đặc trưng của hist ({len(hist)}) không khớp với mô hình ({feature_length}).")

        # Chuẩn hóa đặc trưng và dự đoán
        hist = scaler.transform([hist])
        idx = model_svm.predict(hist)[0]
        predict_name = le.inverse_transform([idx])[0]

        # Lấy xác suất và hiển thị
        prob = model_svm.predict_proba(hist)[0]
        confidence = prob[idx] * 100
        label = f"{predict_name} - {confidence:.2f}%"
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Gender Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
