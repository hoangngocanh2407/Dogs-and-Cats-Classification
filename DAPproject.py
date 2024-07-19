import cv2
import os
import json
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from skimage import color  # Import color module
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
# Đọc dữ liệu từ file JSON
with open('D:\pychar\DaC.v3i.coco\\train\_annotations.coco.json', 'r') as f:
    data = json.load(f)
img_size = (256, 256)
# Đường dẫn tới thư mục chứa ảnh
image_folder = 'D:\pychar\DaC.v3i.coco\\train'

# Khởi tạo các list để lưu trữ đặc trưng HOG và nhãn
features = []
labels = []

# Duyệt qua các annotations để lấy thông tin bbox và nhãn
for annotation in tqdm(data['annotations']):
    image_id = annotation['image_id']
    image_info = next((item for item in data['images'] if item['id'] == image_id), None)
    if image_info:
        file_name = image_info['file_name']
        img_path = os.path.join(image_folder + '/' + file_name)
        image = Image.open(img_path)
        image = color.rgb2gray(np.array(image))
        if image is not None:
            bbox = annotation['bbox']
            x, y, w, h = bbox
            roi = image[int(y):int(y + h), int(x):int(x + w)]
            resized_roi = cv2.resize(roi, (64, 128))  # Kích thước mẫu cho HOG

            # Tính toán đặc trưng HOG cho mẫu
            hog_features = hog(resized_roi, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            features.append(hog_features)
            labels.append(annotation['category_id'])

# Chuyển đổi features và labels thành numpy arrays
X = np.array(features)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện mô hình SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, 'svm_model.pkl')

# Đánh giá mô hình
accuracy = svm_model.score(X_test, y_test)
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


joblib.dump(scaler, 'scaler.pkl')
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')


def extract_hog_features(image, bbox):
    x_start, y_start, x_end, y_end = bbox
    roi = image[y_start:y_end, x_start:x_end]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, (64, 128))
    hog_features = hog(resized_roi, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    return hog_features

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        clone_image = image.copy()

        bbox = []

        def draw_bbox(event, x, y, flags, param):
            nonlocal clone_image, bbox
            if event == cv2.EVENT_LBUTTONDOWN:
                bbox = [x, y, x, y]  # [x_start, y_start, x_end, y_end]
            elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
                clone_image = image.copy()
                cv2.rectangle(clone_image, (bbox[0], bbox[1]), (x, y), (0, 255, 0), 2)  # Màu xanh
            elif event == cv2.EVENT_LBUTTONUP:
                bbox[2], bbox[3] = x, y
                hog_features = extract_hog_features(image, bbox)
                hog_features = scaler.transform([hog_features])
                label = svm_model.predict(hog_features)
                label_name = "Dog" if label[0] == 2 else "Cat"
                cv2.rectangle(clone_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Màu xanh
                cv2.putText(clone_image, f'Label: {label_name}', (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Màu xanh
                cv2.imshow('Image', clone_image)
                bbox = []

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', draw_bbox)

        while True:
            cv2.imshow('Image', clone_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Object Detection")

# Button để chọn ảnh
btn_select_image = tk.Button(root, text="Select Image", command=select_image)
btn_select_image.pack(padx=10, pady=10)

root.mainloop()
