#???

import cv2
import cvzone
import time
from ultralytics import YOLO
from cvzone.SerialModule import SerialObject

WIDTH, HEIGHT = 1280, 720
YOLO_MODEL_PATH = "../Yolo-Weights/yolov8n.pt"
METAL_CLASSES = {"fork", "knife", "spoon", "bowl", "laptop", "mouse", "remote", "keyboard", "cell phone", "clock", "scissors"}

cap = cv2.VideoCapture(1)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

arduino = SerialObject()
model = YOLO(YOLO_MODEL_PATH)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

prev_frame_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detected = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=30, t=5)

            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            label = classNames[cls]

            cvzone.putTextRect(img, f'{label} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
            if label in METAL_CLASSES:
                detected = True

    arduino.sendData([1] if detected else [0])

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
    prev_frame_time = new_frame_time
    print(f'FPS: {fps:.2f}')

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






