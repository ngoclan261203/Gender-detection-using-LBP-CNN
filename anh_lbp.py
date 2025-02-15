import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk
import numpy as np
import pickle
from lbp_class import LocalBinaryPatterns

# Tải mô hình
svm_model_path = 'gender_detection_model3.pkl'
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
def open_image():
    # Chọn file ảnh
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
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
        label_result.config(text=f"Giới tính: {gender}\nĐộ chính xác: {confidence:.2f}%")
    except Exception as e:
        label_result.config(text=f"Lỗi: {e}")

# Tạo giao diện với Tkinter
root = tk.Tk()
root.title("Nhận dạng giới tính")

# Nhãn hiển thị ảnh
label_img = Label(root)
label_img.pack()

# Nút chọn ảnh
btn_select = Button(root, text="Chọn ảnh", command=open_image, font=("Arial", 14), bg="#4CAF50", fg="white")
btn_select.pack(pady=10)

# Nhãn hiển thị kết quả
label_result = Label(root, text="Kết quả sẽ hiển thị ở đây", font=("Arial", 14), fg="#333")
label_result.pack(pady=10)

# Chạy giao diện
root.mainloop()
