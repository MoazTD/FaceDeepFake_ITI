import cv2
import numpy as np
from scipy.spatial.distance import euclidean
from PySide6.QtCore import QTimer, QObject, Qt, QThread, QMutex, QWaitCondition
from PySide6.QtWidgets import *
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Signal
import threading
import time
import json
import os
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
from dataclasses import dataclass
import queue
import sys

class FaceAligner:

    def __init__(self):
        
        
        self.FACIAL_LANDMARKS_68_IDXS = {
            "mouth": (48, 68),
            "inner_mouth": (60, 68),
            "right_eyebrow": (17, 22),
            "left_eyebrow": (22, 27),
            "right_eye": (36, 42),
            "left_eye": (42, 48),
            "nose": (27, 36),
            "jaw": (0, 17)
        }

        
        self.FACE_TEMPLATE = np.array([
            [0.31556875000000000, 0.46031746031746035],  
            [0.68262291666666670, 0.46031746031746035],  
            [0.50026249999999990, 0.64050264550264551],  
            [0.34947187500000004, 0.84645502645502645],  
            [0.65343645833333330, 0.84645502645502645]   
        ])

        self.quality_settings = {
            'low': {'resolution': 128, 'interpolation': cv2.INTER_LINEAR},
            'medium': {'resolution': 256, 'interpolation': cv2.INTER_CUBIC},
            'high': {'resolution': 512, 'interpolation': cv2.INTER_LANCZOS4}
        }
        self.current_quality = 'medium'

    def set_quality(self, quality: str):
        if quality in self.quality_settings:
            self.current_quality = quality

    def align_face(self, image: np.ndarray, landmarks: np.ndarray,
                     output_width: int = None, output_height: int = None) -> Tuple[np.ndarray, np.ndarray]:
        try:
            
            if output_width is None or output_height is None:
                resolution = self.quality_settings[self.current_quality]['resolution']
                output_width = output_height = resolution

            interpolation = self.quality_settings[self.current_quality]['interpolation']

            
            if landmarks.shape[0] >= 5:
                face_landmarks = self._extract_5_point_landmarks(landmarks)
            else:
                face_landmarks = landmarks[:5] if len(landmarks) >= 5 else landmarks

            
            transform_matrix = self._get_transform_matrix(
                face_landmarks, output_width, output_height
            )

            
            aligned_face = cv2.warpAffine(
                image, transform_matrix, (output_width, output_height),
                flags=interpolation, borderMode=cv2.BORDER_REFLECT_101
            )

            return aligned_face, transform_matrix

        except Exception as e:
            print(f"Error in face alignment: {e}")
            
            h, w = image.shape[:2]
            if h > 0 and w > 0:
                fallback = cv2.resize(image, (output_width, output_height))
                identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                return fallback, identity_matrix
            else:
                blank = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
                return blank, identity_matrix

    def _extract_5_point_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        if landmarks.shape[0] == 68:
            
            left_eye = np.mean(landmarks[36:42], axis=0)
            right_eye = np.mean(landmarks[42:48], axis=0)
            nose = landmarks[30]
            left_mouth = landmarks[48]
            right_mouth = landmarks[54]
            return np.array([left_eye, right_eye, nose, left_mouth, right_mouth])
        elif landmarks.shape[0] == 5:
            return landmarks
        else:
            return landmarks[:5] if len(landmarks) >= 5 else landmarks

    def _get_transform_matrix(self, landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
        template = self.FACE_TEMPLATE.copy()
        template[:, 0] *= width
        template[:, 1] *= height

        num_points = min(len(landmarks), len(template))
        if num_points >= 3:
            src_points = landmarks[:num_points].astype(np.float32)
            dst_points = template[:num_points].astype(np.float32)

            transform_matrix = cv2.getAffineTransform(src_points[:3], dst_points[:3])
        elif num_points >= 2:
            src_points = landmarks[:num_points].astype(np.float32)
            dst_points = template[:num_points].astype(np.float32)
            transform_matrix = cv2.estimateAffinePartial2D(
                src_points, dst_points, method=cv2.LMEDS
            )[0]

            if transform_matrix is None:
                transform_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        else:
            transform_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        return transform_matrix



class PerformanceMonitor:

    def __init__(self):
        self.timers = {}
        self.stats = deque(maxlen=100)  

    def start_timer(self, name: str):
        self.timers[name] = time.time()

    def end_timer(self, name: str) -> float:
        if name in self.timers:
            duration = time.time() - self.timers[name]
            self.stats.append(duration)
            del self.timers[name]
            return duration
        return 0.0

    def get_average_time(self) -> float:
        return sum(self.stats) / len(self.stats) if self.stats else 0.0

    def get_fps(self) -> float:
        avg_time = self.get_average_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0



class PresetManager:

    def __init__(self, preset_dir: str = "presets"):
        self.preset_dir = preset_dir
        os.makedirs(preset_dir, exist_ok=True)

        
        self.default_presets = {
            "Natural": {
                'brightness': 105, 'contrast': 110, 'saturation': 105,
                'gamma': 95, 'hue_shift': 0, 'vibrance': 10,
                'temperature': 5, 'sharpen': 20
            },
            "Vibrant": {
                'brightness': 110, 'contrast': 120, 'saturation': 130,
                'gamma': 90, 'hue_shift': 0, 'vibrance': 30,
                'temperature': -10, 'sharpen': 40
            }
        }

    def save_preset(self, name: str, adjustments: Dict[str, Any]) -> bool:
        try:
            preset_path = os.path.join(self.preset_dir, f"{name}.json")
            with open(preset_path, 'w') as f:
                json.dump(adjustments, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving preset: {e}")
            return False

    def load_preset(self, name: str) -> Optional[Dict[str, Any]]:
        try:
            if name in self.default_presets:
                return self.default_presets[name].copy()

            preset_path = os.path.join(self.preset_dir, f"{name}.json")
            if os.path.exists(preset_path):
                with open(preset_path, 'r') as f:
                    return json.load(f)

            return None
        except Exception as e:
            print(f"Error loading preset: {e}")
            return None

    def get_preset_names(self) -> List[str]:
        presets = list(self.default_presets.keys())

        try:
            for file in os.listdir(self.preset_dir):
                if file.endswith('.json'):
                    preset_name = file[:-5]
                    if preset_name not in presets:
                        presets.append(preset_name)
        except Exception as e:
            print(f"Error reading preset directory: {e}")

        return sorted(presets)



class AdvancedOpenCVControls(QWidget):

    adjustment_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.preset_manager = PresetManager()
        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        
        self.enabled_cb = QCheckBox("Enable OpenCV Adjustments")
        self.enabled_cb.setChecked(True)
        layout.addWidget(self.enabled_cb)

        
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["low", "medium", "high"])
        self.quality_combo.setCurrentText("medium")
        quality_layout.addWidget(self.quality_combo)
        layout.addLayout(quality_layout)

        
        self.brightness_slider = self.create_slider("Brightness", 0, 200, 100)
        self.contrast_slider = self.create_slider("Contrast", 0, 200, 100)
        self.saturation_slider = self.create_slider("Saturation", 0, 200, 100)
        self.gamma_slider = self.create_slider("Gamma", 50, 150, 100)
        self.hue_slider = self.create_slider("Hue Shift", -50, 50, 0)
        self.vibrance_slider = self.create_slider("Vibrance", 0, 100, 0)
        self.temperature_slider = self.create_slider("Temperature", -50, 50, 0)
        self.sharpen_slider = self.create_slider("Sharpen", 0, 100, 0)

        layout.addWidget(self.brightness_slider)
        layout.addWidget(self.contrast_slider)
        layout.addWidget(self.saturation_slider)
        layout.addWidget(self.gamma_slider)
        layout.addWidget(self.hue_slider)
        layout.addWidget(self.vibrance_slider)
        layout.addWidget(self.temperature_slider)
        layout.addWidget(self.sharpen_slider)

        
        preset_layout = QHBoxLayout()
        self.preset_btn = QPushButton("Load Preset")
        self.preset_btn.setMenu(self._create_preset_menu())
        preset_layout.addWidget(self.preset_btn)
        layout.addLayout(preset_layout)

        
        save_layout = QHBoxLayout()
        self.preset_name_input = QLineEdit()
        self.preset_name_input.setPlaceholderText("Enter preset name...")
        self.save_preset_btn = QPushButton("Save Preset")
        save_layout.addWidget(self.preset_name_input)
        save_layout.addWidget(self.save_preset_btn)
        layout.addLayout(save_layout)

        
        self.reset_btn = QPushButton("Reset All")
        layout.addWidget(self.reset_btn)

    def create_slider(self, name: str, min_val: int, max_val: int, default: int) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)  # Reduce margins

        
        label_layout = QHBoxLayout()
        label = QLabel(name)
        value_label = QLabel(str(default))
        value_label.setMinimumWidth(40)
        label_layout.addWidget(label)
        label_layout.addStretch()
        label_layout.addWidget(value_label)
        layout.addLayout(label_layout)

        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        layout.addWidget(slider)

        
        widget.slider = slider
        widget.label = value_label

        return widget

    def connect_signals(self):
        controls = [
            self.enabled_cb, self.quality_combo,
            self.brightness_slider.slider, self.contrast_slider.slider,
            self.saturation_slider.slider, self.gamma_slider.slider,
            self.hue_slider.slider, self.vibrance_slider.slider,
            self.temperature_slider.slider, self.sharpen_slider.slider
        ]

        for control in controls:
            if hasattr(control, 'valueChanged'):
                control.valueChanged.connect(self._emit_adjustment_changed)
            elif hasattr(control, 'toggled'):
                control.toggled.connect(self._emit_adjustment_changed)
            elif hasattr(control, 'currentTextChanged'):
                control.currentTextChanged.connect(self._emit_adjustment_changed)

        self.save_preset_btn.clicked.connect(self.save_preset)
        self.reset_btn.clicked.connect(self.reset_all)

    def _create_preset_menu(self) -> QMenu:
        menu = QMenu()

        preset_names = self.preset_manager.get_preset_names()
        for name in preset_names:
            action = menu.addAction(name)
            action.triggered.connect(lambda checked, n=name: self._load_preset_by_name(n))

        if preset_names:
            menu.addSeparator()

        refresh_action = menu.addAction("Refresh List")
        refresh_action.triggered.connect(self._refresh_preset_menu)

        return menu

    def _load_preset_by_name(self, name: str):
        adjustments = self.preset_manager.load_preset(name)
        if adjustments:
            self._apply_preset_values(adjustments)
            self._emit_adjustment_changed()

    def _apply_preset_values(self, adjustments: Dict[str, Any]):
        mapping = {
            'brightness': self.brightness_slider.slider,
            'contrast': self.contrast_slider.slider,
            'saturation': self.saturation_slider.slider,
            'gamma': self.gamma_slider.slider,
            'hue_shift': self.hue_slider.slider,
            'vibrance': self.vibrance_slider.slider,
            'temperature': self.temperature_slider.slider,
            'sharpen': self.sharpen_slider.slider
        }

        for key, slider in mapping.items():
            if key in adjustments:
                slider.setValue(adjustments[key])

    def _refresh_preset_menu(self):
        self.preset_btn.setMenu(self._create_preset_menu())

    def save_preset(self):
        name = self.preset_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Warning", "Please enter a preset name.")
            return

        adjustments = self.get_current_adjustments()
        if self.preset_manager.save_preset(name, adjustments):
            QMessageBox.information(self, "Success", f"Preset '{name}' saved successfully.")
            self.preset_name_input.clear()
            self._refresh_preset_menu()
        else:
            QMessageBox.critical(self, "Error", "Failed to save preset.")

    def reset_all(self):
        self.brightness_slider.slider.setValue(100)
        self.contrast_slider.slider.setValue(100)
        self.saturation_slider.slider.setValue(100)
        self.gamma_slider.slider.setValue(100)
        self.hue_slider.slider.setValue(0)
        self.vibrance_slider.slider.setValue(0)
        self.temperature_slider.slider.setValue(0)
        self.sharpen_slider.slider.setValue(0)
        self.enabled_cb.setChecked(True)
        self.quality_combo.setCurrentText("medium")

    def get_current_adjustments(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled_cb.isChecked(),
            'quality': self.quality_combo.currentText(),
            'brightness': self.brightness_slider.slider.value(),
            'contrast': self.contrast_slider.slider.value(),
            'saturation': self.saturation_slider.slider.value(),
            'gamma': self.gamma_slider.slider.value(),
            'hue_shift': self.hue_slider.slider.value(),
            'vibrance': self.vibrance_slider.slider.value(),
            'temperature': self.temperature_slider.slider.value(),
            'sharpen': self.sharpen_slider.slider.value()
        }

    def _emit_adjustment_changed(self):
        self.adjustment_changed.emit(self.get_current_adjustments())



class EnhancedFaceDetector:

    def __init__(self):
        self.detectors = {}
        self.current_detector = "opencv"
        self.init_detectors()

    def init_detectors(self):
        try:
            
            self.detectors["opencv"] = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            print(f"Error loading OpenCV detector: {e}")

        try:
            
            pb_file = 'opencv_face_detector_uint8.pb'
            pbtxt_file = 'opencv_face_detector.pbtxt'
            if os.path.exists(pb_file) and os.path.exists(pbtxt_file):
                net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
                self.detectors["dnn"] = net
            else:
                print(f"DNN model files not found. Please download '{pb_file}' and '{pbtxt_file}'.")
        except Exception as e:
            print(f"DNN detector not available: {e}")

    def detect_faces(self, image: np.ndarray) -> List[Any]:
        if self.current_detector == "opencv" and "opencv" in self.detectors:
            return self._detect_opencv(image)
        elif self.current_detector == "dnn" and "dnn" in self.detectors:
            return self._detect_dnn(image)
        else:
            if "opencv" not in self.detectors:
                print("No face detectors available.")
            else:
                 print("Defaulting to OpenCV detector.")
                 self.current_detector = "opencv"
                 return self._detect_opencv(image)
            return []

    def _detect_opencv(self, image: np.ndarray) -> List[Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detectors["opencv"].detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        results = []
        for (x, y, w, h) in faces:
            
            face_dict = {
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.8,  
                'kps': self._estimate_landmarks(x, y, w, h)
            }
            results.append(type('Face', (), face_dict)())

        return results

    def _detect_dnn(self, image: np.ndarray) -> List[Any]:
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])

        net = self.detectors["dnn"]
        net.setInput(blob)
        detections = net.forward()

        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                face_dict = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'kps': self._estimate_landmarks(x1, y1, x2-x1, y2-y1)
                }
                results.append(type('Face', (), face_dict)())

        return results

    def _estimate_landmarks(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        
        landmarks = np.array([
            [x + w * 0.3, y + h * 0.4],  
            [x + w * 0.7, y + h * 0.4],  
            [x + w * 0.5, y + h * 0.6],  
            [x + w * 0.35, y + h * 0.8], 
            [x + w * 0.65, y + h * 0.8]  
        ])
        return landmarks

    def set_detector(self, detector_name: str):
        if detector_name in self.detectors:
            self.current_detector = detector_name



class AdvancedImageProcessor:

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def adjust_temperature(image: np.ndarray, temp: float) -> np.ndarray:
        result = image.astype(np.float32)

        if temp > 0:  
            result[:, :, 2] *= (1 + temp / 100)  
            result[:, :, 0] *= (1 - temp / 200)  
        else:  
            result[:, :, 0] *= (1 + abs(temp) / 100)  
            result[:, :, 2] *= (1 - abs(temp) / 200)  

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def adjust_vibrance(image: np.ndarray, vibrance: float) -> np.ndarray:
        if vibrance == 0:
            return image

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        saturation = hsv[:, :, 1] / 255.0
        vibrance_factor = 1 + (vibrance / 100) * (1 - saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * vibrance_factor, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_hue(image: np.ndarray, hue_shift: float) -> np.ndarray:
        if hue_shift == 0:
            return image

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def sharpen_image(image: np.ndarray, strength: float) -> np.ndarray:
        if strength == 0:
            return image
        
        strength_val = strength / 50.0
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.0 + strength_val, gaussian, -strength_val, 0)

        return np.clip(unsharp_mask, 0, 255).astype(np.uint8)



def create_smooth_mask(shape: Tuple[int, ...], center: Tuple[float, float],
                       size: Tuple[float, float], feather: int = 20) -> np.ndarray:
    try:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        center_x, center_y = center
        radius_x, radius_y = size[0] / 2, size[1] / 2

        y, x = np.ogrid[:h, :w]
        distance = np.sqrt(((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2)
        mask = np.clip(1.0 - distance, 0.0, 1.0)

        if feather > 0:
            mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), feather / 3)

        return mask

    except Exception as e:
        print(f"Error creating mask: {e}")
        return np.ones((shape[0], shape[1]), dtype=np.float32)


def apply_enhanced_opencv_adjustments(image: np.ndarray, adjustments: Dict[str, Any]) -> np.ndarray:
    try:
        result = image.copy()
        processor = AdvancedImageProcessor()

        
        brightness = adjustments.get('brightness', 100) / 100.0
        contrast = adjustments.get('contrast', 100) / 100.0
        saturation = adjustments.get('saturation', 100) / 100.0
        gamma = adjustments.get('gamma', 100) / 100.0
        hue_shift = adjustments.get('hue_shift', 0)
        vibrance = adjustments.get('vibrance', 0)
        temperature = adjustments.get('temperature', 0)
        sharpen = adjustments.get('sharpen', 0)

        
        if brightness != 1.0 or contrast != 1.0:
            beta = (brightness - 1.0) * 100
            result = cv2.convertScaleAbs(result, alpha=contrast, beta=beta)
        if gamma != 1.0:
            result = processor.adjust_gamma(result, gamma)
        if temperature != 0:
            result = processor.adjust_temperature(result, temperature)
        if hue_shift != 0:
            result = processor.adjust_hue(result, hue_shift)
        if saturation != 1.0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if vibrance != 0:
            result = processor.adjust_vibrance(result, vibrance)
        if sharpen != 0:
            result = processor.sharpen_image(result, sharpen)

        return result

    except Exception as e:
        print(f"Error applying enhanced adjustments: {e}")
        return image


def apply_opencv_to_aligned_face(image: np.ndarray, face: Any,
                                 adjustments: Dict[str, Any], aligner: FaceAligner) -> np.ndarray:
    try:
        
        if hasattr(face, 'bbox'):
            bbox = face.bbox
        elif hasattr(face, 'box'):
            bbox = face.box
        else:
            return image

        x1, y1, x2, y2 = map(int, bbox)

        
        img_h, img_w = image.shape[:2]
        padding = 30
        face_x1 = max(0, x1 - padding)
        face_y1 = max(0, y1 - padding)
        face_x2 = min(img_w, x2 + padding)
        face_y2 = min(img_h, y2 + padding)
        
        if x2 <= x1 or y2 <= y1 or face_x2 <= face_x1 or face_y2 <= face_y1:
            return image

        
        face_crop = image[face_y1:face_y2, face_x1:face_x2].copy()
        if face_crop.size == 0:
            return image

        
        if hasattr(face, 'kps'):
            landmarks = face.kps
        else: 
            landmarks = np.array([
                [x1, y1], [x2, y1], [(x1+x2)/2, (y1+y2)/2],
                [x1, y2], [x2, y2]
            ])

        
        adjusted_landmarks = landmarks.copy()
        adjusted_landmarks[:, 0] -= face_x1
        adjusted_landmarks[:, 1] -= face_y1

        
        aligned_face, transform_matrix = aligner.align_face(face_crop, adjusted_landmarks)

        
        adjusted_aligned = apply_enhanced_opencv_adjustments(aligned_face, adjustments)

        
        inv_transform = cv2.invertAffineTransform(transform_matrix)
        crop_h, crop_w = face_crop.shape[:2]
        restored_face = cv2.warpAffine(
            adjusted_aligned, inv_transform, (crop_w, crop_h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101
        )

        
        face_center = ((x1 + x2) / 2 - face_x1, (y1 + y2) / 2 - face_y1)
        face_size = (x2 - x1, y2 - y1)
        mask = create_smooth_mask(face_crop.shape, face_center, face_size, feather=20)
        mask_3d = np.stack([mask, mask, mask], axis=2)

        
        blended_crop = (restored_face * mask_3d + face_crop * (1 - mask_3d)).astype(np.uint8)

        
        result = image.copy()
        result[face_y1:face_y2, face_x1:face_x2] = blended_crop

        return result

    except Exception as e:
        print(f"Error applying OpenCV to face: {e}")
        return image



class FaceadjustmentsDialog(QDialog):
    
    
    image_processed = Signal(np.ndarray)  
    processing_stats = Signal(dict)  

    def __init__(self, parent=None, initial_image: np.ndarray = None, target_face_embedding: np.ndarray = None):

        super().__init__(parent)
        self.original_image = initial_image
        self.processed_image = None
        self.current_faces = []
        self.target_face_embedding = target_face_embedding
        self.selected_face_index = -1

        
        self.setWindowTitle("Advanced Face Manual Adjustment System")
        self.setModal(True)        
        self.setup_ui()
        self.setup_processing()
        self.setup_connections()
        
        # Make window smaller and resizable
        self.setMinimumSize(900, 500)  # Set minimum size
        self.resize(1000, 600)  # Set initial size (smaller height)
        self.setSizeGripEnabled(True)  # Enable resize grip
        
        if self.original_image is not None:
            self.set_image(self.original_image)
            
    def set_video(self, video_path: str):
        if not os.path.exists(video_path):
            QMessageBox.critical(self, "Error", f"Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                
                self.set_image(frame)
            else:
                QMessageBox.warning(self, "Warning", "Could not read the first frame from the video.")
        else:
            QMessageBox.critical(self, "Error", "Could not open the video file.")


    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)  # Reduce spacing
        main_layout.setContentsMargins(10, 10, 10, 10)  # Reduce margins

        
        image_area = QWidget()
        image_layout = QVBoxLayout(image_area)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        
        self.image_label = QLabel("Load an image to begin...")
        self.image_label.setMinimumSize(480, 360)  # Smaller minimum size
        self.image_label.setStyleSheet("border: 1px solid black; background-color: #f0f0f0;")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        image_layout.addWidget(self.image_label)

        
        image_controls = QHBoxLayout()
        image_controls.setSpacing(5)
        self.load_btn = QPushButton("Load Image")
        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.setEnabled(False)
        image_controls.addWidget(self.load_btn)
        image_controls.addWidget(self.reset_btn)
        image_controls.addStretch()
        image_layout.addLayout(image_controls)

        main_layout.addWidget(image_area, 2)

        
        control_panel = self.create_control_panel()
        control_panel.setMaximumWidth(350)  # Limit control panel width
        main_layout.addWidget(control_panel, 1)

        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.addStretch()
        
        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setEnabled(False)
        
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.ok_btn)
        
        
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(5, 5, 5, 5)
        dialog_layout.addWidget(main_widget)
        dialog_layout.addLayout(button_layout)

    def create_control_panel(self) -> QWidget:
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(5)  # Reduce spacing between groups
        control_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # Collapsible Detection Group
        detection_group = QGroupBox("Face Detection")
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.setSpacing(3)  # Tighter spacing

        detector_layout = QHBoxLayout()
        detector_layout.addWidget(QLabel("Detector:"))
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["opencv", "dnn"])
        detector_layout.addWidget(self.detector_combo)
        detection_layout.addLayout(detector_layout)

        checkbox_layout = QHBoxLayout()
        self.show_landmarks_cb = QCheckBox("Landmarks")
        self.show_bbox_cb = QCheckBox("Bounding Boxes")
        checkbox_layout.addWidget(self.show_landmarks_cb)
        checkbox_layout.addWidget(self.show_bbox_cb)
        detection_layout.addLayout(checkbox_layout)

        control_layout.addWidget(detection_group)

        # Face Selection Group (compact)
        self.face_selection_group = QGroupBox("Select Face")
        self.face_selection_layout = QVBoxLayout(self.face_selection_group)
        self.face_selection_layout.setSpacing(3)
        self.face_buttons_layout = QHBoxLayout()
        self.face_selection_layout.addLayout(self.face_buttons_layout)
        control_layout.addWidget(self.face_selection_group)

        # Enhancement Group (scrollable if needed)
        enhancement_group = QGroupBox("Face Enhancement")
        enhancement_layout = QVBoxLayout(enhancement_group)
        enhancement_layout.setSpacing(3)

        # Create a scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(300)  # Limit height
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.opencv_controls = AdvancedOpenCVControls()
        scroll.setWidget(self.opencv_controls)
        
        enhancement_layout.addWidget(scroll)
        control_layout.addWidget(enhancement_group)

        # Compact Stats Group
        stats_group = QGroupBox("Stats")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(2)

        self.faces_detected_label = QLabel("Faces: 0")
        self.processing_time_label = QLabel("Time: N/A")
        self.image_size_label = QLabel("Size: N/A")

        # Use smaller font for stats
        small_font = self.faces_detected_label.font()
        small_font.setPointSize(8)
        
        for label in [self.faces_detected_label, self.processing_time_label, self.image_size_label]:
            label.setFont(small_font)
            stats_layout.addWidget(label)

        control_layout.addWidget(stats_group)
        control_layout.addStretch()
        
        return control_panel

    def setup_processing(self):
        self.face_detector = EnhancedFaceDetector()
        self.face_aligner = FaceAligner()
        self.performance_monitor = PerformanceMonitor()

    def setup_connections(self):
        
        self.load_btn.clicked.connect(self.load_image)
        self.reset_btn.clicked.connect(self.reset_to_original)
        
        
        self.detector_combo.currentTextChanged.connect(self.change_detector)
        self.opencv_controls.adjustment_changed.connect(self.run_full_processing)
        self.show_landmarks_cb.stateChanged.connect(self.run_full_processing)
        self.show_bbox_cb.stateChanged.connect(self.run_full_processing)
        
        
        self.apply_btn.clicked.connect(self.apply_changes)
        self.cancel_btn.clicked.connect(self.reject)
        self.ok_btn.clicked.connect(self.accept_changes)

    def set_image(self, image: np.ndarray):
        self.original_image = image.copy() if image is not None else None
        self.processed_image = None
        self.current_faces = []
        
        if self.original_image is not None:
            self.reset_btn.setEnabled(True)
            self.apply_btn.setEnabled(True)
            self.ok_btn.setEnabled(True)
            self.update_image_stats()
            self.run_full_processing()
        else:
            self.reset_btn.setEnabled(False)
            self.apply_btn.setEnabled(False)
            self.ok_btn.setEnabled(False)
            self.image_label.setText("Load an image to begin...")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        )
        
        if file_path:
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.critical(self, "Error", f"Failed to load image from {file_path}")
                return
            
            self.set_image(image)

    def reset_to_original(self):
        if self.original_image is not None:
            self.run_full_processing()

    def change_detector(self, detector_name: str):
        self.face_detector.set_detector(detector_name)
        if self.original_image is not None:
            self.run_full_processing()

    def run_full_processing(self, _=None):
        if self.original_image is None:
            return

        self.performance_monitor.start_timer("process")

        adjustments = self.opencv_controls.get_current_adjustments()
        self.face_aligner.set_quality(adjustments.get('quality', 'medium'))

        processing_image = self.original_image.copy()

        if (adjustments.get('enabled', False) or 
            self.show_bbox_cb.isChecked() or 
            self.show_landmarks_cb.isChecked()):
            self.current_faces = self.face_detector.detect_faces(self.original_image)
        else:
            self.current_faces = []

        face_to_process = None
        if self.target_face_embedding is not None and self.current_faces:
            for face in self.current_faces:
                if hasattr(face, 'normed_embedding') and np.dot(self.target_face_embedding, face.normed_embedding) > 0.5:
                    face_to_process = face
                    break
        elif self.current_faces and self.selected_face_index != -1:
            face_to_process = self.current_faces[self.selected_face_index]
        elif self.current_faces:
            face_to_process = self.current_faces[0] 

        if adjustments.get('enabled', False) and face_to_process:
            processing_image = apply_opencv_to_aligned_face(
                processing_image, face_to_process, adjustments, self.face_aligner
            )

        display_image = self.draw_overlays(processing_image, self.current_faces)

        self.processed_image = display_image
        self.display_frame(self.processed_image)
        
        duration = self.performance_monitor.end_timer("process")
        self.update_processing_stats(duration)

        stats = {
            'faces_detected': len(self.current_faces),
            'processing_time': duration,
            'image_shape': self.original_image.shape if self.original_image is not None else None
        }
        self.processing_stats.emit(stats)
        
    def draw_overlays(self, frame: np.ndarray, faces: List[Any]) -> np.ndarray:
        for face in faces:
            if self.show_bbox_cb.isChecked() and hasattr(face, 'bbox'):
                bbox = face.bbox
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                
                
                if hasattr(face, 'confidence'):
                    conf_text = f"{face.confidence:.2f}"
                    cv2.putText(frame, conf_text, (int(bbox[0]), int(bbox[1]) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if self.show_landmarks_cb.isChecked() and hasattr(face, 'kps'):
                for i, point in enumerate(face.kps):
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)
                    
                    cv2.putText(frame, str(i), (int(point[0]) + 3, int(point[1]) + 3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        return frame

    def display_frame(self, frame: np.ndarray):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Error displaying frame: {e}")
            self.image_label.setText(f"Error displaying image: {str(e)}")

    def update_image_stats(self):
        if self.original_image is not None:
            h, w, c = self.original_image.shape
            self.image_size_label.setText(f"Size: {w}x{h}")
        else:
            self.image_size_label.setText("Size: N/A")

    def update_processing_stats(self, processing_time: float):
        self.faces_detected_label.setText(f"Faces: {len(self.current_faces)}")
        self.processing_time_label.setText(f"Time: {processing_time*1000:.1f}ms")

    def apply_changes(self):
        if self.processed_image is not None:
            self.image_processed.emit(self.processed_image.copy())
            QMessageBox.information(self, "Applied", "Changes have been applied successfully!")

    def accept_changes(self):
        if self.processed_image is not None:
            self.image_processed.emit(self.processed_image.copy())
        self.accept()

    def get_processed_image(self) -> Optional[np.ndarray]:
        return self.processed_image.copy() if self.processed_image is not None else None

    def get_original_image(self) -> Optional[np.ndarray]:
        return self.original_image.copy() if self.original_image is not None else None

    def get_detected_faces(self) -> List[Any]:
        return self.current_faces.copy()

    def closeEvent(self, event):
        
        self.original_image = None
        self.processed_image = None
        self.current_faces = []
        event.accept()

    def resizeEvent(self, event):
        """Handle window resize events to maintain proper image scaling"""
        super().resizeEvent(event)
        if hasattr(self, 'processed_image') and self.processed_image is not None:
            self.display_frame(self.processed_image)



def test_face_alignment_dialog():
    import sys
    
    app = QApplication(sys.argv)
    
    
    dialog = FaceadjustmentsDialog()
    result = dialog.exec()
    
    
    def on_image_processed(image):
        print(f"Image processed! Shape: {image.shape}")
    
    def on_processing_stats(stats):
        print(f"Processing stats: {stats}")
    
    dialog.image_processed.connect(on_image_processed)
    dialog.processing_stats.connect(on_processing_stats)
    
    
    result = dialog.exec_()
    
    if result == QDialog.Accepted:
        print("Dialog accepted")
        processed_image = dialog.get_processed_image()
        if processed_image is not None:
            print(f"Final processed image shape: {processed_image.shape}")
    else:
        print("Dialog cancelled")
    
    sys.exit()


if __name__ == "__main__":
    test_face_alignment_dialog()