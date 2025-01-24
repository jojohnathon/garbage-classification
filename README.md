# garbage-classification

## **IDE**

* **PyCharm**: You can download it from [JetBrains](https://www.jetbrains.com/pycharm/).  
* **Python**: Make sure Python is installed. You can download it from [Python.org](https://www.python.org/).  
  * Note: I am using Python 3.12.0, but later versions should work as well  
    * You can check Python version by doing: Python \-version in Terminal

![][image1]

**Setting Up Your Environment**

* **Create a New Project in PyCharm**:  
  * Open PyCharm and select "New Project".  
  * Choose a location for your project and ensure the correct Python interpreter is selected.  
* **Install Required Libraries**:  
  * Open terminal.  
  * Run the following commands to install CVZone and YOLO dependencies:

| pip install cvzonepip install opencv-pythonpip install numpy |
| :---- |

* Or install directly into Pycharm:   
  * (Scroll down the pictures don't fit)

![][image2]

## **![][image3]**

![][image4]

## [**CVZone**](https://github.com/cvzone/cvzone)

* **Basic Setup**:  
  * Import CVZone in your Python program:

| import cvzoneimport cv2 |
| :---- |

* **Example Usage**:  
  * Use CVZone to perform basic image processing tasks:

| img \= cv2.imread('path\_to\_image.jpg')imgGray \= cv2.cvtColor(img, cv2.COLOR\_BGR2GRAY)cv2.imshow('Gray Image', imgGray)cv2.waitKey(0) |
| :---- |
 

## **Implementing YOLO** 

* **~~Download YOLO Weights and Config~~**~~:~~  
  * ~~Download the YOLO weights and configuration files from the official [YOLO website](https://pjreddie.com/darknet/yolo/).~~  
* [**Pre-Installed Example**](?tab=t.oqqve7ts5fu7)  
* **Load YOLO in Your program**:  
  * Import CVZone in your Python program:

| from ultralytics import YOLO |
| :---- |

  * Adding to YOLO Weights

| model \= YOLO("../Yolo-Weights/yolov8n.pt") //Nanomodel \= YOLO("../Yolo-Weights/yolov8s.pt") //Small //etc (Choose one) |
| :---- |

![][image5]

## **More objects:**

* You may have noticed that there are only several objects that are accounted for”  
  * "person", "bicycle", "car", "motorbike"......."teddy bear", "hair drier", "toothbrush"  
  * To have more objects, YOLO needs to be trained on our need types  
  * Look for [datasets](?tab=t.js0cukk4o030) or make our own ¯\\\_(ツ)\_/¯

## **Cvzone \+ Arduinos:**

* Import necessary library

```from cvzone.SerialModule import SerialObject```

* Using the library

| \#Uses "pySerial" Package \-- check serialModule.py (External Libraries \-\> site-packages \-\> cvzone)\#sents arduino a true or false state, check light on arduino if face is detected light should be on    if bboxs:        arduino.sendData(\[1\])    else:        arduino.sendData(\[0\]) |
| :---- |

* bboxs (bounding boxes)   
  * Check to see if one is present to confirm the true state  
* [Program running on Pycharm](?tab=t.61sjjbkhxvww):  
  * NOTE: FOR THE PROGRAM TO RUN CAMERA NEEDS TO BE TURNED OFF  
* [The program loaded onto Arduino](?tab=t.ovn3vykl5fb1)



# Datasets
[https://en.wikipedia.org/wiki/List\_of\_datasets\_in\_computer\_vision\_and\_image\_processing](https://en.wikipedia.org/wiki/List_of_datasets_in_computer_vision_and_image_processing)

[https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

[https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification)

[Files](https://drive.google.com/drive/folders/1X8rn3YMFMUVygGwvCLpvW29pDGkeoSbQ?usp=drive_link)

fine-tuning:  
[https://www.kaggle.com/code/sumn2u/garbage-classification-transfer-learning](https://www.kaggle.com/code/sumn2u/garbage-classification-transfer-learning)

Automatic Dataset labeling:  
[https://github.com/autodistill/autodistill](https://github.com/autodistill/autodistill)  
[https://roboflow.com](https://roboflow.com)  


# Yolo Example
```
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)


model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone","microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
             ]

prev_frame_time = 0
new_frame_time = 0

while True:
   new_frame_time = time.time()
   success, img = cap.read()
   results = model(img, stream=True)
   for r in results:
       boxes = r.boxes
       for box in boxes:
           # Bounding Box
           x1, y1, x2, y2 = box.xyxy[0]
           x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
           # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
           w, h = x2 - x1, y2 - y1
           cvzone.cornerRect(img, (x1, y1, w, h))
           # Confidence
           conf = math.ceil((box.conf[0] * 100)) / 100
           # Class Name
           cls = int(box.cls[0])

           cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

   fps = 1 / (new_frame_time - prev_frame_time)
   prev_frame_time = new_frame_time
   print(fps)

   cv2.imshow("Image", img)
   cv2.waitKey(1)
```
# Random links

[https://www.kaggle.com/code/sumn2u/garbage-classification-transfer-learning/notebook\#Model-Evaluation](https://www.kaggle.com/code/sumn2u/garbage-classification-transfer-learning/notebook#Model-Evaluation)

[https://www.kaggle.com/code/sumn2u/garbage-classification-resnet/notebook\#Model-Base](https://www.kaggle.com/code/sumn2u/garbage-classification-resnet/notebook#Model-Base):

[https://github.com/sumn2u/deep-waste-app](https://github.com/sumn2u/deep-waste-app)

[https://github.com/sumn2u/deep-waste-rest-api](https://github.com/sumn2u/deep-waste-rest-api)

[https://keras.io/api/applications/](https://keras.io/api/applications/)

[https://universe.roboflow.com/tap-robotics/yolov8\_image\_trash-6okhz](https://universe.roboflow.com/tap-robotics/yolov8_image_trash-6okhz)


$
