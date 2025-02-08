# Garbage Classification

## **IDE Setup**

### **Required Software**

- **PyCharm**: [Download here](https://www.jetbrains.com/pycharm/)
- **Python**: [Download here](https://www.python.org/)
  - *Note: This project uses Python 3.12.0, but later versions should work as well.*
  - Check your Python version:
    ```sh
    python --version
    ```

![Python Version](https://github.com/user-attachments/assets/693fa9f6-94a8-4037-b5fd-e84963861fdb)

### **Setting Up Your Environment**

1. **Create a New Project in PyCharm**
   - Open PyCharm and select **"New Project"**.
   - Choose a location and ensure the correct Python interpreter is selected.

2. **Install Required Libraries**
   - Open the terminal and run:
     ```sh
     pip install cvzone
     pip install opencv-python
     pip install numpy
     ```
   - Or install directly within PyCharm:

   ![Installation Screenshot](https://github.com/user-attachments/assets/24d44bec-0c56-4da6-86e4-2699317cb7fa)

---

## **CVZone**

[CVZone GitHub Repository](https://github.com/cvzone/cvzone)

### **Basic Setup**
```python
import cvzone
import cv2
```

### **Example Usage**
```python
img = cv2.imread('path_to_image.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', imgGray)
cv2.waitKey(0)
```

---

## **Implementing YOLO**

### **Loading YOLO in Your Program**
```python
from ultralytics import YOLO
```

### **Adding YOLO Weights**
```python
model = YOLO("../Yolo-Weights/yolov8n.pt")  # Nano
model = YOLO("../Yolo-Weights/yolov8s.pt")  # Small
```

![YOLO Example](https://github.com/user-attachments/assets/a65af0bf-ed07-472b-bd37-aeacf261c181)

### **Expanding Object Detection**
- YOLO detects objects such as:
  - "person", "bicycle", "car", "motorbike", ..., "teddy bear", "hair drier", "toothbrush"
- To detect more objects, YOLO needs additional training with custom datasets.
- Find datasets:
  - [Dataset List](https://en.wikipedia.org/wiki/List_of_datasets_in_computer_vision_and_image_processing)
  - [Kaggle: Garbage Classification](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
  - [Kaggle: Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)

---

## **CVZone & Arduino Integration**

### **Import Required Library**
```python
from cvzone.SerialModule import SerialObject
```

### **Using the Library**
```python
# Uses "pySerial" Package - check serialModule.py (External Libraries -> site-packages -> cvzone)
# Sends Arduino a true/false state. If a face is detected, the light should turn on.
if bboxs:
    arduino.sendData([1])
else:
    arduino.sendData([0])
```

### **Arduino Notes**
- Bounding boxes (**bboxs**) confirm object presence.
- Ensure the **camera is turned off** when running the program.

---

## **Fine-Tuning & Dataset Labeling**
- **Fine-Tuning with Transfer Learning:** [View on Kaggle](https://www.kaggle.com/code/sumn2u/garbage-classification-transfer-learning)
- **Automatic Dataset Labeling:**
  - [Autodistill](https://github.com/autodistill/autodistill)
  - [Roboflow](https://roboflow.com)

---

## **YOLO Example Code**
```python
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)  # Webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", ...]

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
    
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
```

---

## **Additional Resources**
- [Kaggle: Garbage Classification Transfer Learning](https://www.kaggle.com/code/sumn2u/garbage-classification-transfer-learning/notebook#Model-Evaluation)
- [Kaggle: Garbage Classification ResNet](https://www.kaggle.com/code/sumn2u/garbage-classification-resnet/notebook#Model-Base)
- [Deep Waste App](https://github.com/sumn2u/deep-waste-app)
- [Deep Waste REST API](https://github.com/sumn2u/deep-waste-rest-api)
- [Keras Applications](https://keras.io/api/applications/)
- [YOLOv8 Trash Dataset](https://universe.roboflow.com/tap-robotics/yolov8_image_trash-6okhz)

---

