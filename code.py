import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import cv2
from PIL import Image, ImageTk
import numpy as np
import pickle
from lbp_class import LocalBinaryPatterns
import cvlib as cv
import os


#thư viện cho CNN
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model




# Tải mô hình LBP
svm_model_path = 'gender_detection_model3.pkl'
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

# Hàm nhận diện giới tính
def predict_gender(image_path):
    # Đọc và xử lý ảnh
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Mô tả LBP
    hist = desc.describe(gray)
    
    # Kiểm tra số đặc trưng
    if len(hist) != feature_length:
        raise ValueError(f"Số đặc trưng của hist ({len(hist)}) không khớp với mô hình ({feature_length}).")
    
    # Chuẩn hóa đặc trưng và dự đoán
    hist = scaler.transform([hist])
    idx = model_svm.predict(hist)[0]
    predict_name = le.inverse_transform([idx])[0]
    
    # Lấy xác suất dự đoán
    prob = model_svm.predict_proba(hist)[0]
    confidence = prob[idx] * 100
    
    return predict_name, confidence

# Hàm mở ảnh và hiển thị kết quả
import time

def open_image():
    # Chọn file ảnh
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
    # Bắt đầu đo thời gian
    start_time = time.time()
    
    # Hiển thị ảnh trên giao diện
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((300, 300))  # Resize ảnh để hiển thị
    img_tk = ImageTk.PhotoImage(img)
    label_img.config(image=img_tk)
    label_img.image = img_tk
    
    # Nhận diện giới tính
    try:
        gender, confidence = predict_gender(file_path)
        # Kết thúc đo thời gian
        end_time = time.time()
        elapsed_time = end_time - start_time  # Tính thời gian thực hiện
        
        # Hiển thị kết quả và thời gian
        label_result.config(
            text=f"Giới tính: {gender}\nĐộ chính xác: {confidence:.2f}%\nThời gian thực hiện: {elapsed_time:.2f} giây"
        )
    except Exception as e:
        label_result.config(text=f"Lỗi: {e}")


# Hàm mở webcam
def open_webcam():
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

# Hàm mở video
def open_video():
    # Chọn file video
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not file_path:
        return
    
    # Khởi tạo video
    video = cv2.VideoCapture(file_path)
    print("Bắt đầu nhận diện giới tính qua video. Nhấn 'q' để thoát.")

    while True:
        ret, frame = video.read()
        if not ret:
            print("Không thể đọc video!")
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







# using for cnn
# Load model for CNN
model = load_model(r"C:\Users\NGOC LAN\Desktop\demo1\gender_detection.keras")
classes = ['man', 'woman']

# Function to process a frame
def process_frame(frame):
    face, confidence = cv.detect_face(frame)
    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        face_crop = np.copy(frame[startY:endY, startX:endX])
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        label = f"{label}: {conf[idx] * 100:.2f}%"
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

# Function to use webcam
def use_webcam():
    webcam = cv2.VideoCapture(0)
    while webcam.isOpened():
        status, frame = webcam.read()
        if not status:
            break
        frame = process_frame(frame)
        cv2.imshow("Gender Detection - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    webcam.release()
    cv2.destroyAllWindows()

# Function to process video
def process_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not video_path:
        return
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        status, frame = video.read()
        if not status:
            break
        frame = process_frame(frame)
        cv2.imshow("Gender Detection - Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

# Function to select and process an image
def select_image():
    # Mở hộp thoại để người dùng chọn ảnh
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        return

    # Đọc ảnh bằng OpenCV
    image = cv2.imread(image_path)

    # Xử lý nhận diện khuôn mặt và hiển thị nhãn (tích hợp code từ process_frame)
    face, confidence = cv.detect_face(image)  # Phát hiện khuôn mặt
    classes = ['man', 'woman']
    
    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Vẽ khung xanh bao quanh khuôn mặt (tăng độ dày và kích thước khung)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 5)  # Độ dày khung là 5

        # Cắt vùng khuôn mặt
        face_crop = np.copy(image[startY:endY, startX:endX])
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:  # Bỏ qua nếu khuôn mặt quá nhỏ
            continue

        # Tiền xử lý khuôn mặt để dự đoán giới tính
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Dự đoán giới tính
        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        label = f"{label}: {conf[idx] * 100:.2f}%"

        # Hiển thị nhãn giới tính phía trên khung khuôn mặt (tăng cỡ chữ và độ dày chữ)
        Y = startY - 20 if startY - 20 > 20 else startY + 20
        cv2.putText(image, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)  # Cỡ chữ 1.5, độ dày 4

    # Chuyển đổi ảnh từ BGR (OpenCV) sang RGB (Tkinter yêu cầu RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Lấy kích thước gốc của ảnh
    original_height, original_width = image_rgb.shape[:2]

    # Xác định kích thước hiển thị tối đa (300x300)
    max_width, max_height = 300, 300

    # Tính toán tỉ lệ để resize ảnh theo tỉ lệ gốc
    scale = min(max_width / original_width, max_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Resize ảnh theo tỉ lệ
    resized_image = cv2.resize(image_rgb, (new_width, new_height))

    # Chuyển đổi ảnh từ OpenCV (numpy array) sang định dạng PIL Image
    img_pil = Image.fromarray(resized_image)

    # Chuyển đổi ảnh từ PIL Image sang ImageTk để hiển thị trong Tkinter
    img_tk = ImageTk.PhotoImage(img_pil)

    # Hiển thị ảnh trong `label_img_left`
    label_img_left.config(image=img_tk)
    label_img_left.image = img_tk  # Lưu tham chiếu để tránh bị xóa




# Tạo giao diện với Tkinter
root = tk.Tk()
root.title("Nhận dạng giới tính")

# Chia giao diện làm 2 phần: bên trái và bên phải
frame_left = Frame(root, width=300, height=400, bg="#f0f0f0")
frame_left.pack(side="left", fill="both", expand=True)

frame_right = Frame(root, width=300, height=400, bg="#ffffff")
frame_right.pack(side="right", fill="both", expand=True)



# **Bên trái: Nút mở webcam**
#nhãn title
title_left = Label(frame_left, text="using CNN", font=("Arial", 24))
title_left.pack(pady=20)


# Tạo Frame để chứa các nút Webcam và Video (cùng dòng)
button_frame_left = Frame(frame_left, bg="#f0f0f0")
button_frame_left.pack(pady=10)

# Nút Webcam trái
btn_webcam_left = Button(button_frame_left, text="Mở webcam", command=use_webcam, font=("Arial", 14), bg="#2196F3", fg="white")
btn_webcam_left.grid(row=0, column=0, padx=5)

# Nút Video
btn_video_left = Button(button_frame_left, text="Video", command=process_video, font=("Arial", 14), bg="#FF5722", fg="white")
btn_video_left.grid(row=0, column=1, padx=5)

# Nút chọn ảnh
btn_select_left = Button(frame_left, text="Chọn ảnh", command=select_image, font=("Arial", 14), bg="#4CAF50", fg="white")
btn_select_left.pack(pady=10)

# Hiển thị ảnh
label_img_left = Label(frame_left, bg="#f0f0f0")
label_img_left.pack(pady=20)

# Hiển thị kết quả của ảnh
# label_result_left = Label(frame_left, text="Kết quả sẽ hiển thị ở đây", font=("Arial", 14), fg="#333", bg="#fff")
# label_result_left.pack(pady=10)



# **Bên phải: Nút chọn ảnh và hiển thị kết quả**
#nhãn title
# Thêm tiêu đề bên phải
title_right = Label(frame_right, text="using LBP+SVM", font=("Arial", 24), bg="#fff")
title_right.pack(pady=20)

# Tạo Frame để chứa các nút Webcam và Video (cùng dòng)
button_frame = Frame(frame_right, bg="#fff")
button_frame.pack(pady=10)

# Nút Webcam
btn_webcam = Button(button_frame, text="Webcam", command=open_webcam, font=("Arial", 14), bg="#2196F3", fg="white")
btn_webcam.grid(row=0, column=0, padx=5)

# Nút Video
btn_video = Button(button_frame, text="Video", command=open_video, font=("Arial", 14), bg="#FF5722", fg="white")
btn_video.grid(row=0, column=1, padx=5)

# Nút chọn ảnh
btn_select = Button(frame_right, text="Chọn ảnh", command=open_image, font=("Arial", 14), bg="#4CAF50", fg="white")
btn_select.pack(pady=10)

# Hiển thị ảnh
label_img = Label(frame_right, bg="#fff")
label_img.pack(pady=20)

# Hiển thị kết quả của ảnh
label_result = Label(frame_right, text="Kết quả sẽ hiển thị ở đây", font=("Arial", 14), fg="#333", bg="#fff")
label_result.pack(pady=10)


# Chạy giao diện
root.mainloop()
