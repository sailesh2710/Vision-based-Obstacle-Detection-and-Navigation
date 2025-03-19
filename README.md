# Vision-based-Obstacle-Detection-and-Navigation
This project implements a **real-time obstacle detection and path guidance system** using a mobile-held vision sensor. The system processes video input, detects obstacles, and provides navigation cues based on available space.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/982f46fb-55b2-499a-b8b2-4e46767eab9d" width="50%">
</p>

##  Table of Contents
- [Project Overview](#project-overview)
- [Technical Details](#technical-details)
- [Input & Output Videos](#input--output-videos)

---


## Project Overview
This project implements a **computer vision-based obstacle detection system** for navigation. It processes a pre-recorded video input and detects obstacles in three key directions:
- **Front**
- **Left**
- **Right**

Based on the obstacle placement, the system provides navigation instructions:
- **Move Left** if space is available on the left.
- **Move Right** if space is available on the right.
- **STOP** if no clear path is detected.

---

## Technical Details
### 1. Video Upload & Library Imports
We first import the required libraries and upload the video input using Google Colab:

```python
from google.colab import files

def upload_files():
    uploaded = files.upload()
    for k, v in uploaded.items():
        open(k, 'wb').write(v)
    return list(uploaded.keys())

upload_files()
```

### 2. Object Detection Model

We utilize SSD-Inception v2 (ssd_inception_v2_coco_2017_11_17), a Single Shot Multibox Detector (SSD), for real-time object detection. Faster R-CNN was considered but is computationally slower.
- The model is downloaded in a .tar.gz compressed format.
- The frozen inference graph is loaded as:
```python
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
```
- The system is configured to recognize 10 object classes using the COCO dataset label map:
```python
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')
```

### 3. TensorFlow Detection Graph

A detection graph is constructed to efficiently load the object detection model:
```python
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
```

### 4.  Input & Output Tensors
- Input Tensor: The image input to the model.
- Output Tensors: The detected bounding boxes, confidence scores, and object classes.
```python
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
```

### 5. Image Processing & Region of Interest (ROI)
```python
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    ignore_mask_color = (255,) * img.shape[2] if len(img.shape) > 2 else 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)
```
This function ensures only relevant image regions are processed for object detection.

### 6. Video Processing & Navigation Feedback

We use OpenCV’s VideoWriter to save the output
```python
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width, frame_height))
```
The system then overlays navigation feedback on the processed video:
```python
cv2.putText(frame, 'Move LEFT!', (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
cv2.putText(frame, 'Move RIGHT!', (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
cv2.putText(frame, 'STOPPPPPP!!!', (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
```

### 6. Bounding Box Detection

Extracting bounding box coordinates:
```python
ymin = int((boxes[0][0][0] * frame_height))
xmin = int((boxes[0][0][1] * frame_width))
ymax = int((boxes[0][0][2] * frame_height))
xmax = int((boxes[0][0][3] * frame_width))
```
Detected object boundaries are displayed using:
```python
cv2.line(frame, tuple(left_boundary), tuple(left_boundary_top), (255, 0, 0), 5)
cv2.line(frame, tuple(right_boundary), tuple(right_boundary_top), (255, 0, 0), 5)
```

---

## Input & Output Videos
- Input Video: [Click Here](https://drive.google.com/file/d/1sGsKAW9-5BS317fXfBqWYJa63NB9VFOR/view?usp=sharing)
- Output Video: [Click Here](https://drive.google.com/file/d/1-207slOeWuWauQsgf8CR2zGYsezgdk7J/view?usp=sharing)

---

## Contact

For questions or feedback, feel free to reach out:
- **Email**: [saileshkumar2710@gmail.com](mailto:saileshkumar2710@gmail.com)
- **LinkedIn**: [Sailesh Kumar](https://www.linkedin.com/in/sailesh2710/)
- **GitHub**: [sailesh2710](https://github.com/sailesh2710)
