import os
import shutil

# Thư mục nguồn và thư mục đích
folder1 = r"C:\Users\NGOC LAN\Desktop\gop_anh\output_women"  # Đường dẫn đến thư mục 1
folder2 = r"C:\Users\NGOC LAN\Desktop\gop_anh\train\woman"  # Đường dẫn đến thư mục 2
output_folder = r"C:\Users\NGOC LAN\Desktop\gop_anh\gender_dataset_face\woman"  # Đường dẫn thư mục đích

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Hàm gộp và đặt tên lại file
def merge_and_rename_folders(folder1, folder2, output_folder):
    counter = 1  # Bộ đếm để đặt tên file

    # Gộp các file từ folder1
    for file in os.listdir(folder1):
        file_path = os.path.join(folder1, file)
        if os.path.isfile(file_path):  # Chỉ xử lý file (bỏ qua thư mục con nếu có)
            new_name = f'image_{counter}.jpg'  # Đổi tên file (có thể thay .jpg theo định dạng file của bạn)
            shutil.copy(file_path, os.path.join(output_folder, new_name))
            counter += 1

    # Gộp các file từ folder2
    for file in os.listdir(folder2):
        file_path = os.path.join(folder2, file)
        if os.path.isfile(file_path):
            new_name = f'image_{counter}.jpg'
            shutil.copy(file_path, os.path.join(output_folder, new_name))
            counter += 1

    print(f"Gộp file hoàn tất! Tổng cộng {counter-1} file đã được gộp vào {output_folder}.")

# Gọi hàm gộp file
merge_and_rename_folders(folder1, folder2, output_folder)
