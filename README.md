# Real-time Object Detection with YOLOv11

## 📌 Project Overview
**YOLOv11-based object detection system** for identifying specific equipment in videos or live webcam feeds. This project demonstrates real-time detection of common office equipment using state-of-the-art computer vision techniques.

## 🎯 Features
- **Real-time Detection**: Process video streams with live object detection
- **YOLOv11 Integration**: Latest YOLO version for optimal speed/accuracy balance
- **Custom Object Tracking**: Monitor and count specific equipment
- **Interactive Controls**: Real-time adjustments during processing
- **Performance Metrics**: FPS counter and detection statistics
- **Multi-source Support**: Works with video files and webcams

## 🛠️ Technical Stack
- **Python 3.7+**
- **OpenCV 4.8+** - Image processing and video handling
- **Ultralytics YOLOv11** - Object detection model
- **NumPy** - Mathematical operations

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/object-detection-yolov11.git
cd object-detection-yolov11
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Quick Start
```bash
python detect.py
```

## 🚀 Usage

### Basic Usage
```python
# Run with default webcam (camera index 0)
python detect.py

# Run with specific video file
python detect.py --video "path/to/video.mp4"

# Run with custom model
python detect.py --model "yolo11s.pt"
```

### Command Line Arguments
```bash
--video      # Video file path (default: webcam)
--model      # YOLO model name (default: yolo11n.pt)
--conf       # Confidence threshold (default: 0.5)
--show       # Display detection window (default: True)
--save       # Save output video (default: False)
```

## 🔧 Configuration

### Target Objects Configuration
Edit `config.py` to customize detection targets:

```python
TARGET_ITEMS = {
    "laptop": ("Laptop", (0, 255, 0)),          # Green - Standard COCO object
    "tv": ("Monitor", (255, 0, 0)),             # Blue - Standard COCO object  
    "keyboard": ("Keyboard", (0, 255, 255)),    # Yellow - Standard COCO object
    "chair": ("Chair", (128, 0, 128)),          # Purple - Standard COCO object
    
    # ⚠️ Note: These require custom training (not in standard COCO dataset)
    "projector": ("Projector", (255, 165, 0)),  # Orange - Custom model needed
    "air conditioner": ("AC", (0, 0, 255)),     # Red - Custom model needed
}
```

### Model Selection
Choose from YOLOv11 variants:
- `yolo11n.pt` - Nano (fastest, lower accuracy)
- `yolo11s.pt` - Small (balanced)
- `yolo11m.pt` - Medium (good balance)
- `yolo11l.pt` - Large (higher accuracy)
- `yolo11x.pt` - Extra Large (highest accuracy)

## 🎮 Controls During Execution

| Key | Function | Description |
|-----|----------|-------------|
| **Q / ESC** | Quit | Exit the application |
| **R** | Reset Counters | Reset all object counters to zero |
| **S** | Save Screenshot | Save current frame as image |
| **+** | Increase Confidence | Raise detection threshold (+5%) |
| **-** | Decrease Confidence | Lower detection threshold (-5%) |
| **Space** | Pause/Resume | Toggle video playback |

## 📊 Detection Results Display

### On-screen Information:
```
┌─────────────────────────────────────┐
│ YOLOv11 OBJECT DETECTION           │
│ FPS: 29.5                          │
│ Frame: 450                         │
│ Confidence: 0.50                   │
│                                    │
│ OBJECT COUNTS (MAX):               │
│ Laptop: 3 (2 now)        ███      │
│ Monitor: 2 (1 now)       ██       │
│ Keyboard: 1 (0 now)      █        │
│ Chair: 4 (3 now)         ████     │
│                                    │
│ Current Frame: 6 objects           │
│ Max Total: 10 objects              │
└─────────────────────────────────────┘
```

### Console Output:
```bash
============================================================
YOLOv11 OBJECT DETECTION SYSTEM
============================================================
[INFO] Loading YOLOv11 model...
[SUCCESS] Model loaded: yolo11n.pt
[INFO] Model can detect 80 different object types
[INFO] Available items in model: 4/6
  Available: laptop, tv, keyboard, chair
  Not available (need custom training): projector, air conditioner
[INFO] Starting detection...
[CONTROLS] Press 'Q' to quit | Press 'R' to reset counters
============================================================
```

## 🏗️ Project Structure

```
object-detection-yolov11/
├── detect.py              # Main detection script
├── config.py             # Configuration file
├── requirements.txt      # Dependencies
├── README.md            # This file
├── assets/              # Sample images and videos
│   ├── sample_video.mp4
│   └── screenshots/
├── models/              # Custom trained models (optional)
│   └── custom_yolo11.pt
└── outputs/             # Generated outputs
    ├── saved_frames/
    └── processed_videos/
```

## 🔍 How It Works

### 1. Video Input
```python
# Opens video source (file or webcam)
cap = cv2.VideoCapture(video_path)
```

### 2. YOLOv11 Inference
```python
# Run detection on each frame
results = model(frame, conf=confidence_threshold)
```

### 3. Bounding Box Processing
```python
# Extract detection information
for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0]  # Coordinates
    confidence = box.conf[0]      # Detection confidence
    class_id = box.cls[0]         # Object class
```

### 4. Visualization
```python
# Draw bounding boxes and labels
cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
cv2.putText(frame, label, (x1, y1-10), font, scale, color, thickness)
```

## 📈 Performance Optimization

### For Better Speed:
1. Use `yolo11n.pt` (nano model)
2. Reduce frame resolution
3. Process every Nth frame
4. Run on GPU (CUDA)

### For Better Accuracy:
1. Use `yolo11l.pt` or `yolo11x.pt`
2. Increase confidence threshold
3. Ensure good lighting conditions
4. Use high-quality video source

## 🎯 Custom Training Guide

### For Objects Not in COCO Dataset:

#### 1. Prepare Dataset
```bash
# Directory structure
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

#### 2. Annotation File Format
```yaml
# dataset.yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: projector
  1: air_conditioner
```

#### 3. Training Command
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # Start with pretrained model
model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='office_equipment'
)
```

## 🚨 Common Issues & Solutions

### Issue 1: Low Detection Accuracy
**Solution**: Lower confidence threshold or use larger model

### Issue 2: Slow Performance  
**Solution**: Use smaller model or reduce input resolution

### Issue 3: Objects Not Detected
**Solution**: Verify object is in COCO dataset or train custom model

### Issue 4: Webcam Not Working
**Solution**: Check camera index (try 0, 1, or 2)

## 📝 Output Examples

### Final Results:
```
============================================================
FINAL DETECTION RESULTS
============================================================
Laptop: 3
Monitor: 2  
Keyboard: 1
Chair: 4
----------------------------------------
TOTAL OBJECTS DETECTED: 10

Processed 450 frames in 15.2 seconds
Average FPS: 29.6
============================================================
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

Distributed under MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv11
- [COCO Dataset](https://cocodataset.org/) for object classes
- [OpenCV](https://opencv.org/) community

## 📧 Contact

Golsa Eslamlou

Project Link: https://github.com/eslamlougolsa-lab/YOLOv11-OBJECT-DETECTION-SYSTEM/edit/main/README.md

---

⭐ **If you find this project useful, please give it a star!** ⭐
