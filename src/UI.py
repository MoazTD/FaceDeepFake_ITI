import sys
import os
import cv2
import numpy as np
import onnxruntime as ort
from urllib.request import urlretrieve
from pathlib import Path
import json
import time
from threading import Thread
import queue

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QTimer, Signal, QThread)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon, QImage, QKeySequence, 
    QLinearGradient, QPainter, QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDial, QGraphicsView, QHBoxLayout,
    QLCDNumber, QLayout, QMainWindow, QMenuBar, QPushButton, QScrollBar, 
    QSizePolicy, QSlider, QSpinBox, QStatusBar, QTabWidget, QVBoxLayout,
    QWidget, QLabel, QFileDialog, QTextEdit, QProgressBar, QGridLayout,
    QFrame, QComboBox, QCheckBox, QGroupBox, QFormLayout, QSplitter)

class VideoProcessor(QThread):
    frame_processed = Signal(np.ndarray)
    progress_updated = Signal(int)
    processing_finished = Signal()
    
    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.output_path = ""
        self.face_detector = None
        self.landmark_detector = None
        self.current_frame = 0
        self.total_frames = 0
        self.processing = False
        
    def set_video_path(self, path):
        self.video_path = path
        
    def set_output_path(self, path):
        self.output_path = path
        
    def initialize_models(self):
        models = {
            "detector": {
                "url": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx",
                "path": "yolov8n-face.onnx"
            },
            "landmark": {
                "url": "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
                "path": "lbfmodel.yaml"
            }
        }
        
        for name, model in models.items():
            if not os.path.exists(model["path"]):
                try:
                    urlretrieve(model["url"], model["path"])
                except:
                    pass
        
        try:
            self.face_detector = ort.InferenceSession("yolov8n-face.onnx", providers=["CPUExecutionProvider"])
            self.landmark_detector = cv2.face.createFacemarkLBF()
            self.landmark_detector.loadModel("lbfmodel.yaml")
        except:
            pass
    
    def run(self):
        if not self.video_path or not self.face_detector:
            return
            
        cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened() and self.processing:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame)
            self.frame_processed.emit(processed_frame)
            self.progress_updated.emit(int((self.current_frame / self.total_frames) * 100))
            self.current_frame += 1
            
        cap.release()
        self.processing_finished.emit()
    
    def process_frame(self, frame):
        if self.face_detector is None:
            return frame
            
        try:
            processed_img, scale, padding = self.prepare_image(frame, 640)
            blob = processed_img.transpose(2,0,1)[None].astype(np.float32)/255.0
            input_name = self.face_detector.get_inputs()[0].name
            detections = self.face_detector.run(None, {input_name: blob})[0][0]
            
            boxes, scores = self.process_detections(detections, scale, padding, 0.5)
            indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4).flatten() if boxes else []
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = [tuple(boxes[i]) for i in indices]
            
            if face_rects and self.landmark_detector:
                _, landmarks = self.landmark_detector.fit(gray, np.array(face_rects, dtype=np.int32))
            else:
                landmarks = []
                
            for i, idx in enumerate(indices):
                x, y, w, h = boxes[idx]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"{scores[idx]:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                if i < len(landmarks):
                    for point in landmarks[i][0]:
                        cv2.circle(frame, tuple(point.astype(int)), 2, (0,0,255), -1)
        except:
            pass
            
        return frame
    
    def prepare_image(self, img, target_size):
        h, w = img.shape[:2]
        scale = min(target_size/w, target_size/h)
        new_w, new_h = int(w*scale), int(h*scale)
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (target_size - new_w)//2, (target_size - new_h)//2
        canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return canvas, scale, (pad_x, pad_y)
    
    def process_detections(self, detections, scale, padding, conf_threshold):
        boxes, scores = [], []
        pad_x, pad_y = padding
        
        for detection in detections.T:
            cx, cy, w, h, conf = detection
            if conf < conf_threshold:
                continue
                
            x1 = int(((cx - w/2) - pad_x) / scale)
            y1 = int(((cy - h/2) - pad_y) / scale)
            x2 = int(x1 + (w / scale))
            y2 = int(y1 + (h / scale))
            
            boxes.append([x1, y1, x2-x1, y2-y1])
            scores.append(conf)
        
        return boxes, scores
    
    def start_processing(self):
        self.processing = True
        self.start()
        
    def stop_processing(self):
        self.processing = False

class DeepFakeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.video_processor = VideoProcessor()
        self.video_processor.initialize_models()
        self.setup_connections()
        self.current_video_path = ""
        self.current_image_path = ""
        self.video_cap = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video_frame)
        self.processing_active = False
        
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1400, 900)
        self.setWindowTitle("Deep Fake Studio Pro")
        self.setStyleSheet("""
                           QMainWindow {
	background-color:#f0f0f0;
}
QCheckBox {
	padding:2px;
}
QCheckBox:hover {
	border:1px solid rgb(255,150,60);
	border-radius:4px;
	padding: 1px;
	background-color:qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(190, 90, 50, 50), stop:1 rgba(250, 130, 40, 50));
}
QCheckBox::indicator:checked {
	border:1px solid rgb(246, 134, 86);
	border-radius:4px;
  	background-color:rgb(246, 134, 86)
}
QCheckBox::indicator:unchecked {
	border-width:1px solid rgb(246, 134, 86);
	border-radius:4px;
  	background-color:rgb(255,255,255);
}
QColorDialog {
	background-color:#f0f0f0;
}
QComboBox {
	color:rgb(81,72,65);
	background: #ffffff;
}
QComboBox:editable {
	selection-color:rgb(81,72,65);
	selection-background-color: #ffffff;
}
QComboBox QAbstractItemView {
	selection-color: #ffffff;
	selection-background-color: rgb(246, 134, 86);
}
QComboBox:!editable:on, QComboBox::drop-down:editable:on {
	color:  #1e1d23;	
}
QDateTimeEdit, QDateEdit, QDoubleSpinBox, QFontComboBox {
	color:rgb(81,72,65);
	background-color: #ffffff;
}

QDialog {
	background-color:#f0f0f0;
}

QLabel,QLineEdit {
	color:rgb(17,17,17);
}
QLineEdit {
	background-color:rgb(255,255,255);
	selection-background-color:rgb(236,116,64);
}
QMenuBar {
	color:rgb(223,219,210);
	background-color:rgb(65,64,59);
}
QMenuBar::item {
	padding-top:4px;
	padding-left:4px;
	padding-right:4px;
	color:rgb(223,219,210);
	background-color:rgb(65,64,59);
}
QMenuBar::item:selected {
	color:rgb(255,255,255);
	padding-top:2px;
	padding-left:2px;
	padding-right:2px;
	border-top-width:2px;
	border-left-width:2px;
	border-right-width:2px;
	border-top-right-radius:4px;
	border-top-left-radius:4px;
	border-style:solid;
	background-color:rgb(65,64,59);
	border-top-color: rgb(47,47,44);
	border-right-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:0, stop:0 rgba(90, 87, 78, 255), stop:1 rgba(47,47,44, 255));
	border-left-color:  qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0 rgba(90, 87, 78, 255), stop:1 rgba(47,47,44, 255));
}
QMenu {
	color:rgb(223,219,210);
	background-color:rgb(65,64,59);
}
QMenu::item {
	color:rgb(223,219,210);
	padding:4px 10px 4px 20px;
}
QMenu::item:selected {
	color:rgb(255,255,255);
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(225, 108, 54, 255), stop:1 rgba(246, 134, 86, 255));
	border-style:solid;
	border-width:3px;
	padding:4px 7px 4px 17px;
	border-bottom-color:qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(175,85,48,255), stop:1 rgba(236,114,67, 255));
	border-top-color:qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(253,156,113,255), stop:1 rgba(205,90,46, 255));
	border-right-color:qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgba(253,156,113,255), stop:1 rgba(205,90,46, 255));
	border-left-color:qlineargradient(spread:pad, x1:1, y1:0.5, x2:0, y2:0.5, stop:0 rgba(253,156,113,255), stop:1 rgba(205,90,46, 255));
}
QPlainTextEdit {
	border: 1px solid transparent;
	color:rgb(17,17,17);
	selection-background-color:rgb(236,116,64);
    background-color: #FFFFFF;
}
QProgressBar {
	text-align: center;
	color: rgb(0, 0, 0);
	border: 1px inset rgb(150,150,150); 
	border-radius: 10px;
	background-color:rgb(221,221,219);
}
QProgressBar::chunk:horizontal {
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(225, 108, 54, 255), stop:1 rgba(246, 134, 86, 255));
	border:1px solid;
	border-radius:8px;
	border-bottom-color:qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(175,85,48,255), stop:1 rgba(236,114,67, 255));
	border-top-color:qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(253,156,113,255), stop:1 rgba(205,90,46, 255));
	border-right-color:qlineargradient(spread:pad, x1:0, y1:0.5, x2:1, y2:0.5, stop:0 rgba(253,156,113,255), stop:1 rgba(205,90,46, 255));
	border-left-color:qlineargradient(spread:pad, x1:1, y1:0.5, x2:0, y2:0.5, stop:0 rgba(253,156,113,255), stop:1 rgba(205,90,46, 255));
}
QPushButton{
	color:rgb(17,17,17);
	border-width: 1px;
	border-radius: 6px;
	border-bottom-color: rgb(150,150,150);
	border-right-color: rgb(165,165,165);
	border-left-color: rgb(165,165,165);
	border-top-color: rgb(180,180,180);
	border-style: solid;
	padding: 4px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(220, 220, 220, 255), stop:1 rgba(255, 255, 255, 255));
}
QPushButton:hover{
	color:rgb(17,17,17);
	border-width: 1px;
	border-radius:6px;
	border-top-color: rgb(255,150,60);
	border-right-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:0, stop:0 rgba(200, 70, 20, 255), stop:1 rgba(255,150,60, 255));
	border-left-color:  qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0 rgba(200, 70, 20, 255), stop:1 rgba(255,150,60, 255));
	border-bottom-color: rgb(200,70,20);
	border-style: solid;
	padding: 2px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(220, 220, 220, 255), stop:1 rgba(255, 255, 255, 255));
}
QPushButton:default{
	color:rgb(17,17,17);
	border-width: 1px;
	border-radius:6px;
	border-top-color: rgb(255,150,60);
	border-right-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:0, stop:0 rgba(200, 70, 20, 255), stop:1 rgba(255,150,60, 255));
	border-left-color:  qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0 rgba(200, 70, 20, 255), stop:1 rgba(255,150,60, 255));
	border-bottom-color: rgb(200,70,20);
	border-style: solid;
	padding: 2px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(220, 220, 220, 255), stop:1 rgba(255, 255, 255, 255));
}
QPushButton:pressed{
	color:rgb(17,17,17);
	border-width: 1px;
	border-radius: 6px;
	border-width: 1px;
	border-top-color: rgba(255,150,60,200);
	border-right-color: qlineargradient(spread:pad, x1:0, y1:1, x2:1, y2:0, stop:0 rgba(200, 70, 20, 255), stop:1 rgba(255,150,60, 200));
	border-left-color:  qlineargradient(spread:pad, x1:1, y1:0, x2:0, y2:0, stop:0 rgba(200, 70, 20, 255), stop:1 rgba(255,150,60, 200));
	border-bottom-color: rgba(200,70,20,200);
	border-style: solid;
	padding: 2px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:0, x2:0.5, y2:1, stop:0 rgba(220, 220, 220, 255), stop:1 rgba(255, 255, 255, 255));
}
QPushButton:disabled{
	color:rgb(174,167,159);
	border-width: 1px;
	border-radius: 6px;
	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(200, 200, 200, 255), stop:1 rgba(230, 230, 230, 255));
}
QRadioButton {
	padding: 1px;
}
QRadioButton::indicator:checked {
	height: 10px;
	width: 10px;
	border-style:solid;
	border-radius:5px;
	border-width: 1px;
	border-color: rgba(246, 134, 86, 255);
	color: #a9b7c6;
	background-color:rgba(246, 134, 86, 255);
}
QRadioButton::indicator:!checked {
	height: 10px;
	width: 10px;
	border-style:solid;
	border-radius:5px;
	border-width: 1px;
	border-color: rgb(246, 134, 86);
	color: #a9b7c6;
	background-color: transparent;
}
QScrollArea {
	color: white;
	background-color:#f0f0f0;
}
QSlider::groove {
	border-style: solid;
	border-width: 1px;
	border-color: rgb(207,207,207);
}
QSlider::groove:horizontal {
	height: 5px;
	background: rgb(246, 134, 86);
}
QSlider::groove:vertical {
	width: 5px;
	background: rgb(246, 134, 86);
}
QSlider::handle:horizontal {
	background: rgb(253,253,253);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(207,207,207);
	width: 12px;
	margin: -5px 0;
	border-radius: 7px;
}
QSlider::handle:vertical {
	background: rgb(253,253,253);
	border-style: solid;
	border-width: 1px;
	border-color: rgb(207,207,207);
	height: 12px;
	margin: 0 -5px;
	border-radius: 7px;
}
QSlider::add-page:horizontal, QSlider::add-page:vertical {
 	background: white;
}
QSlider::sub-page:horizontal, QSlider::sub-page:vertical {
	background: rgb(246, 134, 86);
}
QStatusBar, QSpinBox {
	color:rgb(81,72,65);
}
QSpinBox {
	background-color: #ffffff;
}
QScrollBar:horizontal {
	max-height: 20px;
	border: 1px transparent;
	margin: 0px 20px 0px 20px;
}
QScrollBar::handle:horizontal {
	background: rgb(253,253,253);
	border: 1px solid rgb(207,207,207);
	border-radius: 7px;
	min-width: 25px;
}
QScrollBar::handle:horizontal:hover {
	background: rgb(253,253,253);
	border: 1px solid rgb(255,150,60);
	border-radius: 7px;
	min-width: 25px;
}
QScrollBar::add-line:horizontal {
  	border: 1px solid rgb(207,207,207);
  	border-top-right-radius: 7px;
  	border-top-left-radius: 7px;
  	border-bottom-right-radius: 7px;
  	background: rgb(255, 255, 255);
  	width: 20px;
  	subcontrol-position: right;
  	subcontrol-origin: margin;
}
QScrollBar::add-line:horizontal:hover {
  	border: 1px solid rgb(255,150,60);
  	border-top-right-radius: 7px;
  	border-top-left-radius: 7px;
  	border-bottom-right-radius: 7px;
  	background: rgb(255, 255, 255);
  	width: 20px;
  	subcontrol-position: right;
  	subcontrol-origin: margin;
}
QScrollBar::add-line:horizontal:pressed {
  	border: 1px solid grey;
  	border-top-left-radius: 7px;
  	border-top-right-radius: 7px;
  	border-bottom-right-radius: 7px;
  	background: rgb(231,231,231);
  	width: 20px;
  	subcontrol-position: right;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal {
  	border: 1px solid rgb(207,207,207);
  	border-top-right-radius: 7px;
  	border-top-left-radius: 7px;
  	border-bottom-left-radius: 7px;
  	background: rgb(255, 255, 255);
  	width: 20px;
  	subcontrol-position: left;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal:hover {
  	border: 1px solid rgb(255,150,60);
  	border-top-right-radius: 7px;
  	border-top-left-radius: 7px;
  	border-bottom-left-radius: 7px;
  	background: rgb(255, 255, 255);
  	width: 20px;
  	subcontrol-position: left;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:horizontal:pressed {
  	border: 1px solid grey;
  	border-top-right-radius: 7px;
  	border-top-left-radius: 7px;
  	border-bottom-left-radius: 7px;
  	background: rgb(231,231,231);
  	width: 20px;
  	subcontrol-position: left;
  	subcontrol-origin: margin;
}
QScrollBar::left-arrow:horizontal {
  	border: 1px transparent grey;
  	border-top-left-radius: 3px;
  	border-bottom-left-radius: 3px;
  	width: 6px;
  	height: 6px;
  	background: rgb(230,230,230);
}
QScrollBar::right-arrow:horizontal {
	border: 1px transparent grey;
	border-top-right-radius: 3px;
	border-bottom-right-radius: 3px;
  	width: 6px;
  	height: 6px;
 	background: rgb(230,230,230);
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
 	background: none;
} 
QScrollBar:vertical {
	max-width: 20px;
	border: 1px transparent grey;
	margin: 20px 0px 20px 0px;
}
QScrollBar::add-line:vertical {
	border: 1px solid;
	border-color: rgb(207,207,207);
	border-bottom-right-radius: 7px;
	border-bottom-left-radius: 7px;
	border-top-left-radius: 7px;
	background: rgb(255, 255, 255);
  	height: 20px;
  	subcontrol-position: bottom;
  	subcontrol-origin: margin;
}
QScrollBar::add-line:vertical:hover {
  	border: 1px solid;
  	border-color: rgb(255,150,60);
  	border-bottom-right-radius: 7px;
  	border-bottom-left-radius: 7px;
  	border-top-left-radius: 7px;
  	background: rgb(255, 255, 255);
  	height: 20px;
  	subcontrol-position: bottom;
  	subcontrol-origin: margin;
}
QScrollBar::add-line:vertical:pressed {
  	border: 1px solid grey;
  	border-bottom-left-radius: 7px;
  	border-bottom-right-radius: 7px;
  	border-top-left-radius: 7px;
  	background: rgb(231,231,231);
  	height: 20px;
  	subcontrol-position: bottom;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical {
  	border: 1px solid rgb(207,207,207);
  	border-top-right-radius: 7px;
  	border-top-left-radius: 7px;
  	border-bottom-left-radius: 7px;
  	background: rgb(255, 255, 255);
  	height: 20px;
  	subcontrol-position: top;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical:hover {
  	border: 1px solid rgb(255,150,60);
  	border-top-right-radius: 7px;
  	border-top-left-radius: 7px;
  	border-bottom-left-radius: 7px;
	background: rgb(255, 255, 255);
  	height: 20px;
  	subcontrol-position: top;
  	subcontrol-origin: margin;
}
QScrollBar::sub-line:vertical:pressed {
  	border: 1px solid grey;
  	border-top-left-radius: 7px;
  	border-top-right-radius: 7px;
  	background: rgb(231,231,231);
 	height: 20px;
  	subcontrol-position: top;
  	subcontrol-origin: margin;
}
QScrollBar::handle:vertical {
	background: rgb(253,253,253);
	border: 1px solid rgb(207,207,207);
	border-radius: 7px;
	min-height: 25px;
}
QScrollBar::handle:vertical:hover {
	background: rgb(253,253,253);
	border: 1px solid rgb(255,150,60);
	border-radius: 7px;
	min-height: 25px;
}
QScrollBar::up-arrow:vertical {
	border: 1px transparent grey;
  	border-top-left-radius: 3px;
	border-top-right-radius: 3px;
  	width: 6px;
  	height: 6px;
  	background: rgb(230,230,230);
}
QScrollBar::down-arrow:vertical {
  	border: 1px transparent grey;
  	border-bottom-left-radius: 3px;
  	border-bottom-right-radius: 3px;
  	width: 6px;
  	height: 6px;
  	background: rgb(230,230,230);
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
  	background: none;
}
QTabWidget {
	color:rgb(0,0,0);
	background-color:rgb(247,246,246);
}
QTabWidget::pane {
	border-color: rgb(180,180,180);
	background-color:rgb(247,246,246);
	border-style: solid;
	border-width: 1px;
  	border-radius: 6px;
}
QTabBar::tab {
	padding-left:4px;
	padding-right:4px;
	padding-bottom:2px;
	padding-top:2px;
	color:rgb(81,72,65);
  	background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(221,218,217,255), stop:1 rgba(240,239,238,255));
	border-style: solid;
	border-width: 1px;
  	border-top-right-radius:4px;
	border-top-left-radius:4px;
	border-top-color: rgb(180,180,180);
	border-left-color: rgb(180,180,180);
	border-right-color: rgb(180,180,180);
	border-bottom-color: transparent;
}
QTabBar::tab:selected, QTabBar::tab:last:selected, QTabBar::tab:hover {
  	background-color:rgb(247,246,246);
  	margin-left: 0px;
  	margin-right: 1px;
}
QTabBar::tab:!selected {
	margin-top: 1px;
	margin-right: 1px;
}
QTextEdit {
	border-width: 1px;
	border-style: solid;
	border-color:transparent;
	color:rgb(17,17,17);
	selection-background-color:rgb(236,116,64);
}
QTimeEdit, QToolBox, QToolBox::tab, QToolBox::tab:selected {
	color:rgb(81,72,65);
	background-color: #ffffff;
}
            
        """)
        
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        
        main_layout = QVBoxLayout(self.centralwidget)
        
        self.tabWidget = QTabWidget()
        main_layout.addWidget(self.tabWidget)
        
        self.setup_input_tab()
        self.setup_detection_tab()
        self.setup_landmarks_tab()
        self.setup_alignment_tab()
        self.setup_recognition_tab()
        self.setup_segmentation_tab()
        self.setup_swapping_tab()
        self.setup_export_tab()
        
        self.setup_timeline_controls()
        main_layout.addWidget(self.timeline_widget)
        
        self.statusbar = QStatusBar()
        self.statusbar.setStyleSheet("background-color: #404040; color: #ffffff;")
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")
        
    def setup_input_tab(self):
        self.input_tab = QWidget()
        layout = QVBoxLayout(self.input_tab)
        
        input_group = QGroupBox("Input Files")
        input_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ffffff; }")
        input_layout = QGridLayout(input_group)
        
        self.video_label = QLabel("No video selected")
        self.video_label.setMinimumHeight(40)
        self.video_label.setStyleSheet("QLabel { border: 2px dashed #666; padding: 10px; }")
        
        self.image_label = QLabel("No image selected")
        self.image_label.setMinimumHeight(40)
        self.image_label.setStyleSheet("QLabel { border: 2px dashed #666; padding: 10px; }")
        
        self.select_video_btn = QPushButton("Select Video")
        self.select_image_btn = QPushButton("Select Source Image")
        
        input_layout.addWidget(QLabel("Target Video:"), 0, 0)
        input_layout.addWidget(self.video_label, 0, 1)
        input_layout.addWidget(self.select_video_btn, 0, 2)
        
        input_layout.addWidget(QLabel("Source Image:"), 1, 0)
        input_layout.addWidget(self.image_label, 1, 1)
        input_layout.addWidget(self.select_image_btn, 1, 2)
        
        layout.addWidget(input_group)
        
        preview_group = QGroupBox("Preview")
        preview_layout = QHBoxLayout(preview_group)
        
        self.video_preview = QLabel("Video Preview")
        self.video_preview.setMinimumSize(400, 300)
        self.video_preview.setStyleSheet("QLabel { border: 1px solid #666; background-color: #000; }")
        self.video_preview.setAlignment(Qt.AlignCenter)
        
        self.image_preview = QLabel("Image Preview")
        self.image_preview.setMinimumSize(400, 300)
        self.image_preview.setStyleSheet("QLabel { border: 1px solid #666; background-color: #000; }")
        self.image_preview.setAlignment(Qt.AlignCenter)
        
        preview_layout.addWidget(self.video_preview)
        preview_layout.addWidget(self.image_preview)
        
        layout.addWidget(preview_group)
        
        next_layout = QHBoxLayout()
        next_layout.addStretch()
        self.next_btn_1 = QPushButton("Next: Face Detection")
        next_layout.addWidget(self.next_btn_1)
        layout.addLayout(next_layout)
        
        self.tabWidget.addTab(self.input_tab, "1. Input")
        
    def setup_detection_tab(self):
        self.detection_tab = QWidget()
        layout = QVBoxLayout(self.detection_tab)
        
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.conf_threshold_slider = QSlider(Qt.Horizontal) 
        self.conf_threshold_slider.setRange(10, 90)
        self.conf_threshold_slider.setValue(50)
        self.conf_threshold_label = QLabel("0.50")
        
        self.nms_threshold_slider = QSlider(Qt.Horizontal)
        self.nms_threshold_slider.setRange(10, 90)
        self.nms_threshold_slider.setValue(40)
        self.nms_threshold_label = QLabel("0.40")
        
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.conf_threshold_slider)
        threshold_layout.addWidget(self.conf_threshold_label)
        settings_layout.addRow("Confidence Threshold:", threshold_layout)
        
        nms_layout = QHBoxLayout()
        nms_layout.addWidget(self.nms_threshold_slider)
        nms_layout.addWidget(self.nms_threshold_label)
        settings_layout.addRow("NMS Threshold:", nms_layout)
        
        self.detect_btn = QPushButton("Start Detection")
        settings_layout.addRow(self.detect_btn)
        
        layout.addWidget(settings_group)
        
        self.detection_preview = QLabel("Detection Preview")
        self.detection_preview.setMinimumSize(800, 600)
        self.detection_preview.setStyleSheet("QLabel { border: 1px solid #666; background-color: #000; }")
        self.detection_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.detection_preview)
        
        nav_layout = QHBoxLayout()
        self.prev_btn_2 = QPushButton("Previous: Input")
        self.next_btn_2 = QPushButton("Next: Landmarks")
        nav_layout.addWidget(self.prev_btn_2)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn_2)
        layout.addLayout(nav_layout)
        
        self.tabWidget.addTab(self.detection_tab, "2. Face Detection")
        
    def setup_landmarks_tab(self):
        self.landmarks_tab = QWidget()
        layout = QVBoxLayout(self.landmarks_tab)
        
        settings_group = QGroupBox("Landmark Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.landmark_model_combo = QComboBox()
        self.landmark_model_combo.addItems(["LBF Model", "AAM Model", "ERT Model"])
        settings_layout.addRow("Landmark Model:", self.landmark_model_combo)
        
        self.landmark_points_combo = QComboBox()
        self.landmark_points_combo.addItems(["68 Points", "81 Points", "194 Points"])
        settings_layout.addRow("Landmark Points:", self.landmark_points_combo)
        
        self.detect_landmarks_btn = QPushButton("Detect Landmarks")
        settings_layout.addRow(self.detect_landmarks_btn)
        
        layout.addWidget(settings_group)
        
        self.landmarks_preview = QLabel("Landmarks Preview")
        self.landmarks_preview.setMinimumSize(800, 600)
        self.landmarks_preview.setStyleSheet("QLabel { border: 1px solid #666; background-color: #000; }")
        self.landmarks_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.landmarks_preview)
        
        nav_layout = QHBoxLayout()
        self.prev_btn_3 = QPushButton("Previous: Detection")
        self.next_btn_3 = QPushButton("Next: Alignment")
        nav_layout.addWidget(self.prev_btn_3)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn_3)
        layout.addLayout(nav_layout)
        
        self.tabWidget.addTab(self.landmarks_tab, "3. Landmarks")
        
    def setup_alignment_tab(self):
        self.alignment_tab = QWidget()
        layout = QVBoxLayout(self.alignment_tab)
        
        settings_group = QGroupBox("Alignment Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.alignment_method_combo = QComboBox()
        self.alignment_method_combo.addItems(["Similarity Transform", "Affine Transform", "Perspective Transform"])
        settings_layout.addRow("Alignment Method:", self.alignment_method_combo)
        
        self.face_size_slider = QSlider(Qt.Horizontal)
        self.face_size_slider.setRange(64, 512)
        self.face_size_slider.setValue(256)
        self.face_size_label = QLabel("256")
        
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.face_size_slider)
        size_layout.addWidget(self.face_size_label)
        settings_layout.addRow("Face Size:", size_layout)
        
        self.align_faces_btn = QPushButton("Align Faces")
        settings_layout.addRow(self.align_faces_btn)
        
        layout.addWidget(settings_group)
        
        self.alignment_preview = QLabel("Alignment Preview")
        self.alignment_preview.setMinimumSize(800, 600)
        self.alignment_preview.setStyleSheet("QLabel { border: 1px solid #666; background-color: #000; }")
        self.alignment_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.alignment_preview)
        
        nav_layout = QHBoxLayout()
        self.prev_btn_4 = QPushButton("Previous: Landmarks")
        self.next_btn_4 = QPushButton("Next: Recognition")
        nav_layout.addWidget(self.prev_btn_4)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn_4)
        layout.addLayout(nav_layout)
        
        self.tabWidget.addTab(self.alignment_tab, "4. Alignment")
        
    def setup_recognition_tab(self):
        self.recognition_tab = QWidget()
        layout = QVBoxLayout(self.recognition_tab)
        
        settings_group = QGroupBox("Recognition Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.recognition_model_combo = QComboBox()
        self.recognition_model_combo.addItems(["ArcFace", "CosFace", "FaceNet"])
        settings_layout.addRow("Recognition Model:", self.recognition_model_combo)
        
        self.similarity_threshold_slider = QSlider(Qt.Horizontal)
        self.similarity_threshold_slider.setRange(50, 99)
        self.similarity_threshold_slider.setValue(80)
        self.similarity_threshold_label = QLabel("0.80")
        
        sim_layout = QHBoxLayout()
        sim_layout.addWidget(self.similarity_threshold_slider)
        sim_layout.addWidget(self.similarity_threshold_label)
        settings_layout.addRow("Similarity Threshold:", sim_layout)
        
        self.recognize_faces_btn = QPushButton("Recognize Faces")
        settings_layout.addRow(self.recognize_faces_btn)
        
        layout.addWidget(settings_group)
        
        self.recognition_preview = QLabel("Recognition Preview")
        self.recognition_preview.setMinimumSize(800, 600)
        self.recognition_preview.setStyleSheet("QLabel { border: 1px solid #666; background-color: #000; }")
        self.recognition_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.recognition_preview)
        
        nav_layout = QHBoxLayout()
        self.prev_btn_5 = QPushButton("Previous: Alignment")
        self.next_btn_5 = QPushButton("Next: Segmentation")
        nav_layout.addWidget(self.prev_btn_5)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn_5)
        layout.addLayout(nav_layout)
        
        self.tabWidget.addTab(self.recognition_tab, "5. Recognition")
        
    def setup_segmentation_tab(self):
        self.segmentation_tab = QWidget()
        layout = QVBoxLayout(self.segmentation_tab)
        
        settings_group = QGroupBox("Segmentation Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.segmentation_model_combo = QComboBox()
        self.segmentation_model_combo.addItems(["BiSeNet", "U-Net", "DeepLab"])
        settings_layout.addRow("Segmentation Model:", self.segmentation_model_combo)
        
        self.mask_blur_slider = QSlider(Qt.Horizontal)
        self.mask_blur_slider.setRange(0, 20)
        self.mask_blur_slider.setValue(5)
        self.mask_blur_label = QLabel("5")
        
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(self.mask_blur_slider)
        blur_layout.addWidget(self.mask_blur_label)
        settings_layout.addRow("Mask Blur:", blur_layout)
        
        self.segment_faces_btn = QPushButton("Segment Faces")
        settings_layout.addRow(self.segment_faces_btn)
        
        layout.addWidget(settings_group)
        
        self.segmentation_preview = QLabel("Segmentation Preview")
        self.segmentation_preview.setMinimumSize(800, 600)
        self.segmentation_preview.setStyleSheet("QLabel { border: 1px solid #666; background-color: #000; }")
        self.segmentation_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.segmentation_preview)
        
        nav_layout = QHBoxLayout()
        self.prev_btn_6 = QPushButton("Previous: Recognition")
        self.next_btn_6 = QPushButton("Next: Face Swapping")
        nav_layout.addWidget(self.prev_btn_6)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn_6)
        layout.addLayout(nav_layout)
        
        self.tabWidget.addTab(self.segmentation_tab, "6. Segmentation")
        
    def setup_swapping_tab(self):
        self.swapping_tab = QWidget()
        layout = QVBoxLayout(self.swapping_tab)
        
        settings_group = QGroupBox("Face Swapping Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.swap_model_combo = QComboBox()
        self.swap_model_combo.addItems(["SimSwap", "FaceShifter", "FSGAN"])
        settings_layout.addRow("Swap Model:", self.swap_model_combo)
        
        self.blend_ratio_slider = QSlider(Qt.Horizontal)
        self.blend_ratio_slider.setRange(0, 100)
        self.blend_ratio_slider.setValue(80)
        self.blend_ratio_label = QLabel("0.80")
        
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(self.blend_ratio_slider)
        blend_layout.addWidget(self.blend_ratio_label)
        settings_layout.addRow("Blend Ratio:", blend_layout)
        
        self.color_correction_check = QCheckBox("Color Correction")
        self.color_correction_check.setChecked(True)
        settings_layout.addRow(self.color_correction_check)
        
        self.start_swap_btn = QPushButton("Start Face Swapping")
        settings_layout.addRow(self.start_swap_btn)
        
        layout.addWidget(settings_group)
        
        self.swapping_preview = QLabel("Face Swapping Preview")
        self.swapping_preview.setMinimumSize(800, 600)
        self.swapping_preview.setStyleSheet("QLabel { border: 1px solid #666; background-color: #000; }")
        self.swapping_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.swapping_preview)
        
        nav_layout = QHBoxLayout()
        self.prev_btn_7 = QPushButton("Previous: Segmentation")
        self.next_btn_7 = QPushButton("Next: Export")
        nav_layout.addWidget(self.prev_btn_7)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_btn_7)
        layout.addLayout(nav_layout)
        
        self.tabWidget.addTab(self.swapping_tab, "7. Face Swapping")
        
    def setup_export_tab(self):
        self.export_tab = QWidget()
        layout = QVBoxLayout(self.export_tab)
        
        settings_group = QGroupBox("Export Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(["MP4", "AVI", "MOV", "WebM"])
        settings_layout.addRow("Output Format:", self.output_format_combo)
        
        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 10)
        self.quality_slider.setValue(8)
        self.quality_label = QLabel("8")
        
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(self.quality_label)
        settings_layout.addRow("Quality:", quality_layout)
        
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["24", "25", "30", "60"])
        self.fps_combo.setCurrentText("30")
        settings_layout.addRow("FPS:", self.fps_combo)
        
        self.output_path_label = QLabel("No output path selected")
        self.select_output_btn = QPushButton("Select Output Path")
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_path_label)
        output_layout.addWidget(self.select_output_btn)
        settings_layout.addRow("Output Path:", output_layout)
        
        self.export_btn = QPushButton("Export Video")
        settings_layout.addRow(self.export_btn)
        
        layout.addWidget(settings_group)
        
        self.export_progress = QProgressBar()
        layout.addWidget(self.export_progress)
        
        self.export_preview = QLabel("Export Preview")
        self.export_preview.setMinimumSize(800, 600)
        self.export_preview.setStyleSheet("QLabel { border: 1px solid #666; background-color: #000; }")
        self.export_preview.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.export_preview)
        
        nav_layout = QHBoxLayout()
        self.prev_btn_8 = QPushButton("Previous: Face Swapping")
        nav_layout.addWidget(self.prev_btn_8)
        nav_layout.addStretch()
        layout.addLayout(nav_layout)
        
        self.tabWidget.addTab(self.export_tab, "8. Export")
        
    def setup_timeline_controls(self):
        self.timeline_widget = QWidget()
        self.timeline_widget.setMaximumHeight(150)
        self.timeline_widget.setStyleSheet("QWidget { background-color: #333333; }")
        
        layout = QVBoxLayout(self.timeline_widget)
        
        controls_layout = QHBoxLayout()
        
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setMaximumSize(40, 40)
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setMaximumSize(40, 40)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.frame_label = QLabel("Frame: 0 / 0")
        
        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.frame_slider)
        controls_layout.addWidget(self.time_label)
        controls_layout.addWidget(self.frame_label)
        
        layout.addLayout(controls_layout)
        
        timeline_layout = QHBoxLayout()
        
        self.timeline_scroll = QScrollBar(Qt.Horizontal)
        self.timeline_scroll.setMinimum(0)
        self.timeline_scroll.setMaximum(100)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMaximumWidth(100)
        
        timeline_layout.addWidget(QLabel("Timeline:"))
        timeline_layout.addWidget(self.timeline_scroll)
        timeline_layout.addWidget(QLabel("Zoom:"))
        timeline_layout.addWidget(self.zoom_slider)
        
        layout.addLayout(timeline_layout)
        
    def setup_connections(self):
        self.select_video_btn.clicked.connect(self.select_video)
        self.select_image_btn.clicked.connect(self.select_image)
        self.select_output_btn.clicked.connect(self.select_output_path)
        
        self.next_btn_1.clicked.connect(lambda: self.tabWidget.setCurrentIndex(1))
        self.prev_btn_2.clicked.connect(lambda: self.tabWidget.setCurrentIndex(0))
        self.next_btn_2.clicked.connect(lambda: self.tabWidget.setCurrentIndex(2))
        self.prev_btn_3.clicked.connect(lambda: self.tabWidget.setCurrentIndex(1))
        self.next_btn_3.clicked.connect(lambda: self.tabWidget.setCurrentIndex(3))
        self.prev_btn_4.clicked.connect(lambda: self.tabWidget.setCurrentIndex(2))
        self.next_btn_4.clicked.connect(lambda: self.tabWidget.setCurrentIndex(4))
        self.prev_btn_5.clicked.connect(lambda: self.tabWidget.setCurrentIndex(3))
        self.next_btn_5.clicked.connect(lambda: self.tabWidget.setCurrentIndex(5))
        self.prev_btn_6.clicked.connect(lambda: self.tabWidget.setCurrentIndex(4))
        self.next_btn_6.clicked.connect(lambda: self.tabWidget.setCurrentIndex(6))
        self.prev_btn_7.clicked.connect(lambda: self.tabWidget.setCurrentIndex(5))
        self.next_btn_7.clicked.connect(lambda: self.tabWidget.setCurrentIndex(7))
        self.prev_btn_8.clicked.connect(lambda: self.tabWidget.setCurrentIndex(6))
        
        self.conf_threshold_slider.valueChanged.connect(self.update_conf_threshold)
        self.nms_threshold_slider.valueChanged.connect(self.update_nms_threshold)
        self.face_size_slider.valueChanged.connect(self.update_face_size)
        self.similarity_threshold_slider.valueChanged.connect(self.update_similarity_threshold)
        self.mask_blur_slider.valueChanged.connect(self.update_mask_blur)
        self.blend_ratio_slider.valueChanged.connect(self.update_blend_ratio)
        self.quality_slider.valueChanged.connect(self.update_quality)
        
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_landmarks_btn.clicked.connect(self.detect_landmarks)
        self.align_faces_btn.clicked.connect(self.align_faces)
        self.recognize_faces_btn.clicked.connect(self.recognize_faces)
        self.segment_faces_btn.clicked.connect(self.segment_faces)
        self.start_swap_btn.clicked.connect(self.start_face_swapping)
        self.export_btn.clicked.connect(self.export_video)
        
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.stop_btn.clicked.connect(self.stop_playback)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        
        self.video_processor.frame_processed.connect(self.update_preview)
        self.video_processor.progress_updated.connect(self.update_progress)
        self.video_processor.processing_finished.connect(self.processing_finished)
        
    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )
        if file_path:
            self.current_video_path = file_path
            self.video_label.setText(os.path.basename(file_path))
            self.load_video_preview()
            self.statusbar.showMessage(f"Video loaded: {os.path.basename(file_path)}")
            
    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if file_path:
            self.current_image_path = file_path
            self.image_label.setText(os.path.basename(file_path))
            self.load_image_preview()
            self.statusbar.showMessage(f"Image loaded: {os.path.basename(file_path)}")
            
    def select_output_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Path", "", "Video Files (*.mp4 *.avi *.mov *.webm)"
        )
        if file_path:
            self.output_path_label.setText(os.path.basename(file_path))
            self.video_processor.set_output_path(file_path)
            
    def load_video_preview(self):
        if self.current_video_path:
            self.video_cap = cv2.VideoCapture(self.current_video_path)
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            
            self.frame_slider.setMaximum(self.total_frames - 1)
            self.timeline_scroll.setMaximum(self.total_frames - 1)
            
            ret, frame = self.video_cap.read()
            if ret:
                self.display_frame(frame, self.video_preview)
                self.update_time_labels()
                
    def load_image_preview(self):
        if self.current_image_path:
            image = cv2.imread(self.current_image_path)
            if image is not None:
                self.display_frame(image, self.image_preview)
                
    def display_frame(self, frame, label):
        if frame is not None:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            
    def update_conf_threshold(self, value):
        self.conf_threshold_label.setText(f"{value/100:.2f}")
        
    def update_nms_threshold(self, value):
        self.nms_threshold_label.setText(f"{value/100:.2f}")
        
    def update_face_size(self, value):
        self.face_size_label.setText(str(value))
        
    def update_similarity_threshold(self, value):
        self.similarity_threshold_label.setText(f"{value/100:.2f}")
        
    def update_mask_blur(self, value):
        self.mask_blur_label.setText(str(value))
        
    def update_blend_ratio(self, value):
        self.blend_ratio_label.setText(f"{value/100:.2f}")
        
    def update_quality(self, value):
        self.quality_label.setText(str(value))
        
    def start_detection(self):
        if not self.current_video_path:
            self.statusbar.showMessage("Please select a video first")
            return
            
        self.processing_active = True
        self.detect_btn.setText("Processing...")
        self.detect_btn.setEnabled(False)
        
        self.video_processor.set_video_path(self.current_video_path)
        self.video_processor.start_processing()
        
        self.statusbar.showMessage("Face detection started...")
        
    def detect_landmarks(self):
        self.statusbar.showMessage("Detecting landmarks...")
        self.detect_landmarks_btn.setText("Processing...")
        self.detect_landmarks_btn.setEnabled(False)
        
        if self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                processed_frame = self.video_processor.process_frame(frame)
                self.display_frame(processed_frame, self.landmarks_preview)
                
        self.detect_landmarks_btn.setText("Detect Landmarks")
        self.detect_landmarks_btn.setEnabled(True)
        self.statusbar.showMessage("Landmarks detected")
        
    def align_faces(self):
        self.statusbar.showMessage("Aligning faces...")
        self.align_faces_btn.setText("Processing...")
        self.align_faces_btn.setEnabled(False)
        
        if self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                processed_frame = self.video_processor.process_frame(frame)
                self.display_frame(processed_frame, self.alignment_preview)
                
        self.align_faces_btn.setText("Align Faces")
        self.align_faces_btn.setEnabled(True)
        self.statusbar.showMessage("Faces aligned")
        
    def recognize_faces(self):
        self.statusbar.showMessage("Recognizing faces...")
        self.recognize_faces_btn.setText("Processing...")
        self.recognize_faces_btn.setEnabled(False)
        
        if self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                processed_frame = self.video_processor.process_frame(frame)
                self.display_frame(processed_frame, self.recognition_preview)
                
        self.recognize_faces_btn.setText("Recognize Faces")
        self.recognize_faces_btn.setEnabled(True)
        self.statusbar.showMessage("Face recognition completed")
        
    def segment_faces(self):
        self.statusbar.showMessage("Segmenting faces...")
        self.segment_faces_btn.setText("Processing...")
        self.segment_faces_btn.setEnabled(False)
        
        if self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                processed_frame = self.video_processor.process_frame(frame)
                self.display_frame(processed_frame, self.segmentation_preview)
                
        self.segment_faces_btn.setText("Segment Faces")
        self.segment_faces_btn.setEnabled(True)
        self.statusbar.showMessage("Face segmentation completed")
        
    def start_face_swapping(self):
        self.statusbar.showMessage("Starting face swapping...")
        self.start_swap_btn.setText("Processing...")
        self.start_swap_btn.setEnabled(False)
        
        if self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                processed_frame = self.video_processor.process_frame(frame)
                self.display_frame(processed_frame, self.swapping_preview)
                
        self.start_swap_btn.setText("Start Face Swapping")
        self.start_swap_btn.setEnabled(True)
        self.statusbar.showMessage("Face swapping completed")
        
    def export_video(self):
        self.statusbar.showMessage("Exporting video...")
        self.export_btn.setText("Exporting...")
        self.export_btn.setEnabled(False)
        
        self.export_progress.setValue(0)
        for i in range(101):
            self.export_progress.setValue(i)
            QApplication.processEvents()
            time.sleep(0.01)
            
        self.export_btn.setText("Export Video")
        self.export_btn.setEnabled(True)
        self.statusbar.showMessage("Video exported successfully")
        
    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_pause_btn.setText("▶")
        else:
            if self.video_cap and self.video_cap.isOpened():
                self.timer.start(int(1000 / self.fps))
                self.play_pause_btn.setText("⏸")
                
    def stop_playback(self):
        self.timer.stop()
        self.play_pause_btn.setText("▶")
        self.current_frame_idx = 0
        self.frame_slider.setValue(0)
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
    def seek_frame(self, frame_idx):
        if self.video_cap and self.video_cap.isOpened():
            self.current_frame_idx = frame_idx
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_cap.read()
            if ret:
                self.display_frame(frame, self.video_preview)
                self.update_time_labels()
                
    def update_video_frame(self):
        if self.video_cap and self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                self.current_frame_idx += 1
                self.frame_slider.setValue(self.current_frame_idx)
                self.display_frame(frame, self.video_preview)
                self.update_time_labels()
            else:
                self.stop_playback()
                
    def update_time_labels(self):
        if self.fps > 0:
            current_seconds = self.current_frame_idx / self.fps
            total_seconds = self.total_frames / self.fps
            
            current_time = f"{int(current_seconds//60):02d}:{int(current_seconds%60):02d}"
            total_time = f"{int(total_seconds//60):02d}:{int(total_seconds%60):02d}"
            
            self.time_label.setText(f"{current_time} / {total_time}")
            self.frame_label.setText(f"Frame: {self.current_frame_idx} / {self.total_frames}")
            
    def update_preview(self, frame):
        current_tab = self.tabWidget.currentIndex()
        if current_tab == 1:
            self.display_frame(frame, self.detection_preview)
        elif current_tab == 2:
            self.display_frame(frame, self.landmarks_preview)
        elif current_tab == 3:
            self.display_frame(frame, self.alignment_preview)
        elif current_tab == 4:
            self.display_frame(frame, self.recognition_preview)
        elif current_tab == 5:
            self.display_frame(frame, self.segmentation_preview)
        elif current_tab == 6:
            self.display_frame(frame, self.swapping_preview)
        elif current_tab == 7:
            self.display_frame(frame, self.export_preview)
            
    def update_progress(self, value):
        self.export_progress.setValue(value)
        
    def processing_finished(self):
        self.processing_active = False
        self.detect_btn.setText("Start Detection")
        self.detect_btn.setEnabled(True)
        self.statusbar.showMessage("Processing completed")
        
    def closeEvent(self, event):
        if self.video_cap:
            self.video_cap.release()
        if self.processing_active:
            self.video_processor.stop_processing()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DeepFakeMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()