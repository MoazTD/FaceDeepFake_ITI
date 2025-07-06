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
from skimage import exposure
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

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

# ===== ADVANCED FACE SWAP CONFIGURATION =====
CONF_THRES = 0.45
NMS_IOU = 0.35
MODEL_SIZE = 640
SEAMLESS_CLONE_FLAG = cv2.NORMAL_CLONE
FEATHER_AMOUNT = 15
BLEND_RATIO = 0.85
FACE_PADDING = 0.25
MASK_BLUR_KERNEL = 21
HISTOGRAM_MATCH_STRENGTH = 0.7
SHARPEN_STRENGTH = 0.3
NOISE_REDUCTION = True
CONTRAST_ENHANCEMENT = True
GAMMA_CORRECTION = 1.1
MULTI_SCALE_LEVELS = 4
EDGE_PRESERVATION = True
DETAIL_ENHANCEMENT = True

# ===== ADVANCED FACE SWAP FUNCTIONS =====
def enhance_image_quality(image):
    enhanced = image.copy()
    if NOISE_REDUCTION:
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    if CONTRAST_ENHANCEMENT:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = np.power(enhanced / 255.0, 1.0 / GAMMA_CORRECTION)
    enhanced = (enhanced * 255).astype(np.uint8)
    return enhanced

def sharpen_image(image, strength=SHARPEN_STRENGTH):
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
    return sharpened

def reduce_noise(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised

def prepare_image_adv(img, target_size):
    h, w = img.shape[:2]
    scale = min(target_size/w, target_size/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (target_size - new_w)//2, (target_size - new_h)//2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
    return canvas, scale, (pad_x, pad_y)

def process_detections_adv(detections, scale, padding, conf_threshold):
    boxes, scores = [], []
    pad_x, pad_y = padding
    for detection in detections.T:
        if len(detection) >= 5:
            cx, cy, w, h, conf = detection[:5]
            if conf < conf_threshold:
                continue
            x1 = int(((cx - w/2) - pad_x) / scale)
            y1 = int(((cy - h/2) - pad_y) / scale)
            x2 = int(x1 + (w / scale))
            y2 = int(y1 + (h / scale))
            boxes.append([x1, y1, x2-x1, y2-y1])
            scores.append(float(conf))
    return boxes, scores

def detect_faces_adv(image, detector, detector_input_name, conf_threshold, nms_threshold):
    processed_img, scale, padding = prepare_image_adv(image, MODEL_SIZE)
    blob = processed_img.transpose(2,0,1)[None].astype(np.float32)/255.0
    detections = detector.run(None, {detector_input_name: blob})[0][0]
    boxes, scores = process_detections_adv(detections, scale, padding, conf_threshold)
    if len(boxes) == 0:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []
    return [tuple(boxes[i]) for i in indices.flatten()]

def extract_face_region_advanced(image, face_box, padding_ratio=FACE_PADDING):
    x, y, w, h = face_box
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)
    face_region = image[y1:y2, x1:x2]
    if face_region.size > 0:
        face_region = enhance_image_quality(face_region)
    return face_region, (x1, y1, x2-x1, y2-y1)

def create_advanced_mask(face_region_shape, feather_amount=MASK_BLUR_KERNEL):
    h, w = face_region_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center_x, center_y = w // 2, h // 2
    inner_axes = (int(w * 0.35), int(h * 0.35))
    cv2.ellipse(mask, (center_x, center_y), inner_axes, 0, 0, 360, 255, -1)
    outer_axes = (int(w * 0.45), int(h * 0.45))
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(outer_mask, (center_x, center_y), outer_axes, 0, 0, 360, 255, -1)
    gradient_mask = outer_mask - mask
    if feather_amount > 0:
        gradient_mask = cv2.GaussianBlur(gradient_mask, (feather_amount, feather_amount), 0)
    final_mask = mask + gradient_mask
    if feather_amount > 0:
        final_mask = cv2.GaussianBlur(final_mask, (feather_amount, feather_amount), 0)
    return final_mask

def advanced_color_matching(source_region, target_region):
    if source_region.shape[:2] != target_region.shape[:2]:
        target_region = cv2.resize(target_region, (source_region.shape[1], source_region.shape[0]))
    source_lab = cv2.cvtColor(source_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    for i in range(3):
        source_mean = np.mean(source_lab[:, :, i])
        source_std = np.std(source_lab[:, :, i])
        target_mean = np.mean(target_lab[:, :, i])
        target_std = np.std(target_lab[:, :, i])
        if source_std > 0:
            source_lab[:, :, i] = ((source_lab[:, :, i] - source_mean) * (target_std / source_std) + target_mean)
    source_lab = np.clip(source_lab, 0, 255)
    lab_matched = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    hist_matched = source_region.copy()
    for i in range(3):
        hist_matched[:, :, i] = exposure.match_histograms(
            source_region[:, :, i], target_region[:, :, i]
        )
    final_result = cv2.addWeighted(
        lab_matched, 1 - HISTOGRAM_MATCH_STRENGTH, 
        hist_matched, HISTOGRAM_MATCH_STRENGTH, 0
    )
    return final_result

def multi_scale_blending(source, target, mask):
    if source.shape[:2] != target.shape[:2] or source.shape[:2] != mask.shape[:2]:
        h, w = target.shape[:2]
        source = cv2.resize(source, (w, h))
        mask = cv2.resize(mask, (w, h))
    levels = min(MULTI_SCALE_LEVELS, 3)
    source_pyramid = [source.copy()]
    target_pyramid = [target.copy()]
    mask_pyramid = [mask.copy()]
    for i in range(levels):
        source_pyramid.append(cv2.pyrDown(source_pyramid[-1]))
        target_pyramid.append(cv2.pyrDown(target_pyramid[-1]))
        mask_pyramid.append(cv2.pyrDown(mask_pyramid[-1]))
    source_laplacian = [source_pyramid[-1]]
    target_laplacian = [target_pyramid[-1]]
    for i in range(levels - 1, 0, -1):
        size = (source_pyramid[i-1].shape[1], source_pyramid[i-1].shape[0])
        source_upscaled = cv2.pyrUp(source_pyramid[i])
        target_upscaled = cv2.pyrUp(target_pyramid[i])
        if source_upscaled.shape[:2] != source_pyramid[i-1].shape[:2]:
            source_upscaled = cv2.resize(source_upscaled, size)
        if target_upscaled.shape[:2] != target_pyramid[i-1].shape[:2]:
            target_upscaled = cv2.resize(target_upscaled, size)
        source_laplacian.append(source_pyramid[i-1] - source_upscaled)
        target_laplacian.append(target_pyramid[i-1] - target_upscaled)
    blended_pyramid = []
    for i in range(levels):
        if len(mask_pyramid[i].shape) == 2:
            mask_norm = mask_pyramid[i].astype(np.float32) / 255.0
            mask_3d = np.stack([mask_norm, mask_norm, mask_norm], axis=2)
        else:
            mask_3d = mask_pyramid[i].astype(np.float32) / 255.0
        blended = (source_laplacian[i].astype(np.float32) * mask_3d + 
                  target_laplacian[i].astype(np.float32) * (1 - mask_3d))
        blended_pyramid.append(np.clip(blended, 0, 255).astype(np.uint8))
    result = blended_pyramid[0]
    for i in range(1, levels):
        size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
        result = cv2.pyrUp(result)
        if result.shape[:2] != blended_pyramid[i].shape[:2]:
            result = cv2.resize(result, size)
        result = result + blended_pyramid[i]
    return np.clip(result, 0, 255).astype(np.uint8)

def poisson_blend_advanced(source, target, mask, target_region_box):
    x, y, w, h = target_region_box
    x = max(0, min(x, target.shape[1] - 1))
    y = max(0, min(y, target.shape[0] - 1))
    w = min(w, target.shape[1] - x)
    h = min(h, target.shape[0] - y)
    if w <= 0 or h <= 0:
        return target
    source_resized = cv2.resize(source, (w, h))
    mask_resized = cv2.resize(mask, (w, h))
    result = target.copy()
    target_region = target[y:y+h, x:x+w]
    try:
        moments = cv2.moments(mask_resized)
        if moments['m00'] > 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            center_x = max(1, min(center_x, w - 1))
            center_y = max(1, min(center_y, h - 1))
            cloned_region = cv2.seamlessClone(
                source_resized, target_region, mask_resized, 
                (center_x, center_y), cv2.NORMAL_CLONE
            )
            result[y:y+h, x:x+w] = cloned_region
            return result
    except Exception as e:
        print(f"  - Seamless cloning failed, using multi-scale blending: {e}")
    try:
        blended_region = multi_scale_blending(source_resized, target_region, mask_resized)
        result[y:y+h, x:x+w] = blended_region
        return result
    except Exception as e:
        print(f"  - Multi-scale blending failed, using alpha blending: {e}")
    if len(mask_resized.shape) == 2:
        mask_normalized = mask_resized.astype(np.float32) / 255.0
        mask_3d = np.stack([mask_normalized, mask_normalized, mask_normalized], axis=2)
    else:
        mask_3d = mask_resized.astype(np.float32) / 255.0
    blended_region = (source_resized.astype(np.float32) * mask_3d + 
                     target_region.astype(np.float32) * (1 - mask_3d))
    result[y:y+h, x:x+w] = np.clip(blended_region, 0, 255).astype(np.uint8)
    return result

def advanced_face_swap_engine(source_img, target_img, detector, detector_input_name):
    source_img = enhance_image_quality(source_img)
    target_img = enhance_image_quality(target_img)
    source_faces = detect_faces_adv(source_img, detector, detector_input_name, CONF_THRES, NMS_IOU)
    target_faces = detect_faces_adv(target_img, detector, detector_input_name, CONF_THRES, NMS_IOU)
    if not source_faces or not target_faces:
        print("❌ Error: Could not detect faces in both images")
        return None
    source_face = max(source_faces, key=lambda x: x[2] * x[3])
    target_face = max(target_faces, key=lambda x: x[2] * x[3])
    source_face_region, _ = extract_face_region_advanced(source_img, source_face)
    target_face_region, target_region_box = extract_face_region_advanced(target_img, target_face)
    if source_face_region.size == 0 or target_face_region.size == 0:
        print("❌ Error: Could not extract valid face regions")
        return None
    color_matched_face = advanced_color_matching(source_face_region, target_face_region)
    mask = create_advanced_mask(color_matched_face.shape)
    result = poisson_blend_advanced(color_matched_face, target_img, mask, target_region_box)
    if DETAIL_ENHANCEMENT:
        result = sharpen_image(result, SHARPEN_STRENGTH)
    if NOISE_REDUCTION:
        result = reduce_noise(result)
    return result

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
        
########################################################################################################################
########################################################################################################################
########################################################################################################################
        
    def initialize_models(self):
        """Initialize AI models for face detection and processing, using GPU if available."""
        import torch
        models = {
            "detector": {
                "url": "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx",
                "path": "yolov8n-face.onnx"
            },
            "landmark": {
                "url": "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml",
                "path": "lbfmodel.yaml"
            },
            "swapper": {
                "url": "https://huggingface.co/MoazTD/FaceDeepFake_ITI/resolve/main/models/face_swapping/inswapper_128.fp16.onnx?download=true",  # No download URL, user must provide
                ##############################################################################################
                ##############################################################################################
                ##############################################################################################
                "path": r"inswapper_128.fp16.onnx"
                ##############################################################################################
                ##############################################################################################
                ##############################################################################################
            }
        }
        # Download models if they don't exist (except swapper)
        for name, model in models.items():
            if name != "swapper" and not os.path.exists(model["path"]):
                try:
                    print(f"Downloading {name} model...")
                    urlretrieve(model["url"], model["path"])
                    print(f"{name} model downloaded successfully")
                except Exception as e:
                    print(f"Failed to download {name} model: {e}")
        # Set ONNX providers for GPU if available
        providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        print(f"ONNX providers: {providers}")
        # Initialize models
        try:
            if os.path.exists(models["detector"]["path"]):
                self.face_detector = ort.InferenceSession(models["detector"]["path"], providers=providers)
                print("Face detector loaded successfully")
            else:
                print("Face detector model not found")
            # Initialize landmark detector with fallback
            try:
                self.landmark_detector = cv2.face.createFacemarkLBF()
                if os.path.exists(models["landmark"]["path"]):
                    self.landmark_detector.loadModel(models["landmark"]["path"])
                    print("Landmark detector loaded successfully")
                else:
                    print("Landmark model not found")
                    self.landmark_detector = None
            except AttributeError:
                print("OpenCV face module not available - landmark detection disabled")
                self.landmark_detector = None
            # Initialize face swapper with user-specified absolute path only
            face_swap_path = models["swapper"]["path"]
            if os.path.exists(face_swap_path):
                try:
                    self.face_swapper = ort.InferenceSession(face_swap_path, providers=providers)
                    print(f"Face swapper loaded from {face_swap_path}")
                except Exception as e:
                    print(f"Failed to load face swapper: {e}")
                    self.face_swapper = None
            else:
                print(f"Face swapper model not found at {face_swap_path}")
                self.face_swapper = None
        except Exception as e:
            print(f"Error initializing models: {e}")
########################################################################################################################
########################################################################################################################
########################################################################################################################

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
                
            # For each detected face, run face swapper if available
            for i, idx in enumerate(indices):
                x, y, w, h = boxes[idx]
                face_img = frame[y:y+h, x:x+w]
                if self.face_swapper is not None and face_img.size > 0:
                    # Use the same preprocessing as swap_faces
                    def preprocess(img):
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img_rgb, (128, 128))
                        img_float = img_resized.astype(np.float32) / 255.0
                        img_transposed = np.transpose(img_float, (2, 0, 1))
                        return img_transposed[None]
                    # For demo, swap face with itself (or you can use a source_img from UI)
                    input_names = [i.name for i in self.face_swapper.get_inputs()]
                    input_feed = {input_names[0]: preprocess(face_img), input_names[1]: preprocess(face_img)}
                    try:
                        output = self.face_swapper.run(None, input_feed)
                        swapped = output[0]
                        if swapped.shape[-1] == 3:
                            swapped = np.clip(swapped[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                            swapped = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)
                            # Replace face region with swapped face
                            swapped_resized = cv2.resize(swapped, (w, h))
                            frame[y:y+h, x:x+w] = swapped_resized
                    except Exception as e:
                        print(f"Face swap error in process_frame: {e}")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"{scores[idx]:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                if i < len(landmarks):
                    for point in landmarks[i][0]:
                        cv2.circle(frame, tuple(point.astype(int)), 2, (0,0,255), -1)
        except Exception as e:
            print(f"Error in process_frame: {e}")
            
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

        self.target_image_label = QLabel("No target image selected")
        self.target_image_label.setMinimumHeight(40)
        self.target_image_label.setStyleSheet("QLabel { border: 2px dashed #666; padding: 10px; }")

        self.image_label = QLabel("No source image/video selected")
        self.image_label.setMinimumHeight(40)
        self.image_label.setStyleSheet("QLabel { border: 2px dashed #666; padding: 10px; }")

        self.select_video_btn = QPushButton("Select Video")
        self.select_target_image_btn = QPushButton("Select Target Image")
        self.select_image_btn = QPushButton("Select Source Image")
        self.select_source_video_btn = QPushButton("Select Source Video")

        input_layout.addWidget(QLabel("Target Video:"), 0, 0)
        input_layout.addWidget(self.video_label, 0, 1)
        input_layout.addWidget(self.select_video_btn, 0, 2)

        input_layout.addWidget(QLabel("Target Image:"), 1, 0)
        input_layout.addWidget(self.target_image_label, 1, 1)
        input_layout.addWidget(self.select_target_image_btn, 1, 2)

        input_layout.addWidget(QLabel("Source Image:"), 2, 0)
        input_layout.addWidget(self.image_label, 2, 1)
        input_layout.addWidget(self.select_image_btn, 2, 2)
        input_layout.addWidget(self.select_source_video_btn, 2, 3)

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
        self.face_size_slider.setValue(128)
        self.face_size_label = QLabel("128")
        
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
        self.swap_model_combo.addItems(["inswapper_128", "FaceShifter", "FSGAN"])
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
        self.select_target_image_btn.clicked.connect(self.select_target_image)
        self.select_image_btn.clicked.connect(self.select_image)
        self.select_source_video_btn.clicked.connect(self.select_source_video)
        
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
            
    def select_target_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if file_path:
            self.current_target_image_path = file_path
            self.target_image_label.setText(os.path.basename(file_path))
            self.load_target_image_preview()
            self.statusbar.showMessage(f"Target image loaded: {os.path.basename(file_path)}")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if file_path:
            self.current_image_path = file_path
            self.image_label.setText(os.path.basename(file_path))
            self.load_image_preview()
            self.statusbar.showMessage(f"Image loaded: {os.path.basename(file_path)}")
            
    def select_source_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Source Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)"
        )
        if file_path:
            self.current_source_video_path = file_path
            self.image_label.setText(os.path.basename(file_path))
            self.load_source_video_preview()
            self.statusbar.showMessage(f"Source video loaded: {os.path.basename(file_path)}")

    def load_source_video_preview(self):
        if hasattr(self, 'current_source_video_path') and self.current_source_video_path:
            self.source_video_cap = cv2.VideoCapture(self.current_source_video_path)
            ret, frame = self.source_video_cap.read()
            if ret:
                self.display_frame(frame, self.image_preview)

    def select_frame(self, frame_idx=None, from_source=False):
        cap = self.source_video_cap if from_source and hasattr(self, 'source_video_cap') else self.video_cap
        if cap and cap.isOpened():
            if frame_idx is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                if from_source:
                    self.display_frame(frame, self.image_preview)
                else:
                    self.display_frame(frame, self.video_preview)
            return frame if ret else None
        return None

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
                
    def load_target_image_preview(self):
        if hasattr(self, 'current_target_image_path') and self.current_target_image_path:
            image = cv2.imread(self.current_target_image_path)
            if image is not None:
                self.display_frame(image, self.video_preview)

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

        # Prefer image-to-image swap if both images are selected
        if hasattr(self, 'current_target_image_path') and self.current_target_image_path and \
           hasattr(self, 'current_image_path') and self.current_image_path:
            target_frame = cv2.imread(self.current_target_image_path)
            source_img = cv2.imread(self.current_image_path)
        else:
            # Fallback to video frame and source image/video
            if self.video_cap and self.video_cap.isOpened():
                target_frame = self.select_frame(self.current_frame_idx)
            else:
                target_frame = None
            if hasattr(self, 'current_image_path') and self.current_image_path:
                source_img = cv2.imread(self.current_image_path)
            elif hasattr(self, 'source_video_cap') and self.source_video_cap.isOpened():
                source_img = self.select_frame(0, from_source=True)
            else:
                source_img = None

        if target_frame is not None and source_img is not None and hasattr(self.video_processor, 'face_swapper'):
            swapped_frame = self.swap_faces(target_frame, source_img)
            self.display_frame(swapped_frame, self.swapping_preview)
        else:
            self.statusbar.showMessage("Face swapping failed: missing model or input.")

        self.start_swap_btn.setText("Start Face Swapping")
        self.start_swap_btn.setEnabled(True)
        self.statusbar.showMessage("Face swapping completed")
        
    def swap_faces(self, target_frame, source_img):
        """
        Swap faces between target_frame and source_img using the advanced face swap engine for images.
        Falls back to ONNX face swapper for video or if advanced swap fails.
        """
        # Use advanced face swap for image-to-image
        try:
            detector = self.video_processor.face_detector
            detector_input_name = detector.get_inputs()[0].name if detector else None
            if detector and detector_input_name:
                result = advanced_face_swap_engine(source_img, target_frame, detector, detector_input_name)
                if result is not None:
                    return result
        except Exception as e:
            print(f"Advanced face swap error: {e}")
        # fallback to ONNX swapper
        swapper = self.video_processor.face_swapper
        if swapper is None:
            print("Face swapper model is not loaded.")
            return target_frame
        try:
            def preprocess(img):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (128, 128))
                img_float = img_resized.astype(np.float32) / 255.0
                img_transposed = np.transpose(img_float, (2, 0, 1))
                return img_transposed[None]
            target_input = preprocess(target_frame)
            source_input = preprocess(source_img)
            input_names = [i.name for i in swapper.get_inputs()]
            input_feed = {}
            if len(input_names) == 2:
                input_feed[input_names[0]] = target_input
                input_feed[input_names[1]] = source_input
            else:
                input_feed = {input_names[0]: target_input, input_names[1]: source_input}
            output = swapper.run(None, input_feed)
            swapped = output[0]
            if swapped.shape[-1] == 3:
                swapped = np.clip(swapped[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                swapped = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)
            else:
                swapped = target_frame
            return swapped
        except Exception as e:
            print(f"Face swap error: {e}")
            return target_frame

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

    def select_output_path(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Path", "", "Video Files (*.mp4 *.avi *.mov *.webm)")
        if file_path:
            self.output_path_label.setText(os.path.basename(file_path))
            self.video_processor.set_output_path(file_path)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DeepFakeMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()