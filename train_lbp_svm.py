import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from lbp_class import LocalBinaryPatterns
import pickle
from sklearn.metrics import accuracy_score

# Các nhãn phân loại
classes = ['man', 'woman']

# Đường dẫn đến dữ liệu huấn luyện
data_dir = r"C:\Users\NGOC LAN\Desktop\LBP_webcam - Copy - Copy (2)\gender_dataset_face"

# Khởi tạo LBP
desc = LocalBinaryPatterns(24, 8)  # Thay đổi số điểm lân cận và bán kính nếu cần

# Các biến để lưu trữ đặc trưng và nhãn
features = []
labels = []

# Đọc dữ liệu từ các thư mục
for label in classes:
    label_dir = os.path.join(data_dir, label)
    for image_name in os.listdir(label_dir):
        # Đọc ảnh
        image_path = os.path.join(label_dir, image_name)
        image = cv2.imread(image_path)

        # Chuyển ảnh sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mô tả LBP cho ảnh
        hist = desc.describe(gray)

        # Lưu trữ đặc trưng và nhãn
        features.append(hist)
        labels.append(label)

# Chuyển nhãn thành số (0: man, 1: woman)
le = LabelEncoder()
labels = le.fit_transform(labels)

# Chuyển đổi đặc trưng và nhãn thành mảng numpy
features = np.array(features)
labels = np.array(labels)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Khởi tạo SVM với đánh giá tham số
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(svm.SVC(probability=True), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# In ra thông số tốt nhất và độ chính xác
print(f"Best parameters: {grid_search.best_params_}")
model_svm = grid_search.best_estimator_

# Đánh giá mô hình trên tập kiểm tra
y_pred = model_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác trên tập kiểm tra: {accuracy * 100:.2f}%")

# Lưu mô hình vào file
svm_model_path = 'gender_detection_model3.pkl'
model_data = {
    'model': model_svm,
    'scaler': scaler,
    'label_encoder': le,
    'feature_length': features.shape[1],
}
with open(svm_model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"Mô hình đã được lưu tại {svm_model_path}")
