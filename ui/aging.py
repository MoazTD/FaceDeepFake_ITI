
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import traceback
from datetime import datetime
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
                               QPushButton, QMessageBox, QSpinBox, QGroupBox, QDialogButtonBox,
                               QProgressBar, QRadioButton)

from core.apply_age import apply_ageing


DEBUG_MODE = True


def find_model_path(model_name: str) -> Path | None:
    search_paths = [
        Path(__file__).parent / "Models" / model_name,
        Path.cwd() / "Models" / model_name,
        Path(__file__).parent.parent / "Models" / model_name,
        Path(__file__).parent / "models" / model_name,
        Path.cwd() / "models" / model_name,
        Path(__file__).parent.parent / "models" / model_name,
    ]
    for path in search_paths:
        if path.is_file():
            return path
    return None


def detect_faces_opencv(detector, frame):
    try:
        faces = detector.detect_faces(frame)
        return faces if faces else []
    except Exception as e:
        print(f"[ERROR] OpenCV face detection failed: {e}")
        return []


class AgingWorker(QThread):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
    
    def run(self):
        try:
            face_crop = self.settings['face_crop']
            target_age = self.settings['target_age']
            
            self.progress.emit(f"Applying age adjustment: {target_age} years...")
            
            if face_crop is None or face_crop.size == 0:
                self.error.emit("Invalid face crop provided")
                return
            
            
            aged_face = apply_ageing(face_crop, target_age)

            if aged_face is not None:
                self.progress.emit("Aging completed successfully!")
                self.finished.emit(aged_face)
            else:
                self.error.emit("Failed to process face - no result returned")
                
        except Exception as e:
            import traceback
            error_msg = f"Aging process error: {str(e)}"
            traceback.print_exc()
            self.error.emit(error_msg)

class FaceAgingWindow(QDialog): 
    log_message = Signal(str)
    settings_accepted = Signal(dict) 

    def __init__(self, processor, initial_frame, stylegan_session, parent=None):
        super().__init__(parent) 
        self.setWindowTitle("Advanced Face Aging")
        self.setMinimumSize(1200, 800)

        self.processor = processor
        self.original_frame = initial_frame.copy()
        self.stylegan_session = stylegan_session 
        self.face_bbox = None
        self.worker = None
        self.current_aged_result = None

        self.setup_ui()
        self.set_initial_frame()
    
    def setup_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout(self) 
        
        
        preview_layout = QHBoxLayout()
        self.original_label = QLabel("Original")
        self.processed_label = QLabel("Aged Preview")
        for label in [self.original_label, self.processed_label]:
            label.setFrameShape(QLabel.Shape.Box)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(450, 350)
            label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0;")
        preview_layout.addWidget(self.original_label)
        preview_layout.addWidget(self.processed_label)
        layout.addLayout(preview_layout)

        
        controls_layout = QHBoxLayout()
        
        
        detector_group = QGroupBox("1. Face Detection Method")
        detector_layout = QVBoxLayout(detector_group)
        self.detector_opencv_radio = QRadioButton("OpenCV (Fast, Basic)")
        self.detector_opencv_radio.setChecked(True)
        detector_layout.addWidget(self.detector_opencv_radio)
        controls_layout.addWidget(detector_group)

        
        aging_group = QGroupBox("2. Aging Method")
        aging_layout = QVBoxLayout(aging_group)
        self.aging_styleganex_radio = QRadioButton("StyleGANex (Direct)")
        self.aging_styleganex_radio.setChecked(True)
        aging_layout.addWidget(self.aging_styleganex_radio)
        controls_layout.addWidget(aging_group)
        
        layout.addLayout(controls_layout)

        
        age_group = QGroupBox("3. Target Age Adjustment")
        age_layout = QHBoxLayout(age_group)
        
        
        age_control_layout = QVBoxLayout()
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Younger"))
        self.age_slider = QSlider(Qt.Orientation.Horizontal)
        self.age_slider.setRange(-100, 100)
        self.age_slider.setValue(0)
        slider_layout.addWidget(self.age_slider)
        slider_layout.addWidget(QLabel("Older"))
        age_control_layout.addLayout(slider_layout)
        
        spinbox_layout = QHBoxLayout()
        spinbox_layout.addWidget(QLabel("Age Adjustment:"))
        self.age_spinbox = QSpinBox()
        self.age_spinbox.setRange(-100, 100)
        self.age_spinbox.setValue(0)
        self.age_spinbox.setSuffix(" years")
        spinbox_layout.addWidget(self.age_spinbox)
        spinbox_layout.addStretch()
        age_control_layout.addLayout(spinbox_layout)
        
        age_layout.addLayout(age_control_layout)
        layout.addWidget(age_group)
        
        
        self.age_slider.valueChanged.connect(self.age_spinbox.setValue)
        self.age_spinbox.valueChanged.connect(self.age_slider.setValue)
        
        

        
        self.detector_opencv_radio.toggled.connect(self.update_controls_state)
        self.aging_styleganex_radio.toggled.connect(self.update_controls_state)

        
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.status_label)

        
        button_layout = QHBoxLayout()
        self.preview_button = QPushButton("Update Preview")
        self.preview_button.clicked.connect(self.process_preview)
        button_layout.addWidget(self.preview_button)
        button_layout.addStretch() 
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        button_layout.addWidget(self.button_box)
        
        layout.addLayout(button_layout)

    def update_controls_state(self):
        stylegan_available = self.stylegan_session is not None
        
        if not stylegan_available:
            self.aging_styleganex_radio.setEnabled(False)
            self.aging_styleganex_radio.setText("StyleGANex (Model Not Found)")
        else:
            self.aging_styleganex_radio.setEnabled(True)
            self.aging_styleganex_radio.setText("StyleGANex (Direct)")

        self.update_status()

    def update_status(self):
        detector = "OpenCV"
        aging = "StyleGANex"
        self.status_label.setText(f"Detection: {detector} | Aging: {aging}")

    def set_initial_frame(self):
        self.display_image(self.original_frame, self.original_label)
        self.display_image(self.original_frame, self.processed_label)

    def get_face_bbox(self):
        self.log_message.emit("Detecting face using OpenCV...")

        try:
            faces = detect_faces_opencv(self.processor.detector, self.original_frame)

            if not faces:
                self.log_message.emit("⚠️ No face detected.")
                return None
            
            self.log_message.emit("✅ Face detected successfully.")
            
            
            main_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            bbox = main_face.bbox.astype(int)
            
            
            h, w = self.original_frame.shape[:2]
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                self.log_message.emit("⚠️ Invalid face bounding box.")
                return None
                
            return [x1, y1, x2, y2]
            
        except Exception as e:
            self.log_message.emit(f"⚠️ Face detection error: {str(e)}")
            return None

    def process_preview(self):
        try:
            if self.worker and self.worker.isRunning():
                self.log_message.emit("⏳ Processing already in progress...")
                return
            
            settings = self.get_settings()
            settings['face_crop'] = self.original_frame  

            
            self.log_message.emit(f"⏳ Processing...")
            self.progress_bar.setVisible(True)
            self.status_label.setText("Processing...")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.preview_button.setEnabled(False)
            self.button_box.setEnabled(False) 
            
            self.worker = AgingWorker(settings)
            self.worker.finished.connect(self.on_aging_finished)
            self.worker.error.connect(self.on_aging_error)
            self.worker.progress.connect(self.log_message.emit)
            self.worker.start()
            
        except Exception as e:
            self.log_message.emit(f"❌ Preview processing error: {str(e)}")
            self.progress_bar.setVisible(False)
            self.preview_button.setEnabled(True)
            self.button_box.setEnabled(True)

    def on_aging_finished(self, aged_face):
        try:
            self.log_message.emit("✅ Aging preview updated successfully!")
            self.progress_bar.setVisible(False)
            self.status_label.setText("✅ Preview updated")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.preview_button.setEnabled(True)
            self.button_box.setEnabled(True)
            
            
            self.current_aged_result = aged_face.copy()
            self.display_image(aged_face, self.processed_label)
            
        except Exception as e:
            self.log_message.emit(f"❌ Error updating preview: {str(e)}")
            self.progress_bar.setVisible(False)
            self.preview_button.setEnabled(True)
            self.button_box.setEnabled(True)
        
        self.worker = None

    def on_aging_error(self, error):
        self.log_message.emit(f"❌ Aging failed: {error}")
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Processing failed")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.preview_button.setEnabled(True)
        self.button_box.setEnabled(True)
        self.worker = None

    def display_image(self, cv_img, label):
        try:
            if cv_img is None or cv_img.size == 0:
                return
                
            h, w, ch = cv_img.shape
            bytes_per_line = ch * w
            q_img = QImage(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB).data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(pixmap)
        except Exception as e:
            print(f"[ERROR] Display image failed: {e}")

    def get_settings(self):
        return {
            'detector_type': 'opencv',
            'aging_method': 'styleganex',
            'target_age': self.age_slider.value(),
            'face_bbox': self.face_bbox,
            'aged_result': self.current_aged_result
        }

    

    def accept(self):
        settings = self.get_settings()
        self.settings_accepted.emit(settings)
        super().accept()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.log_message.emit("Stopping background process...")
            self.worker.terminate()
            self.worker.wait(3000)
        event.accept()


def apply_aging_to_video(processor, aging_session, input_video_path, output_video_path, settings, progress_callback=None):
    try:
        if progress_callback:
            progress_callback("Starting video aging process...")
        
        cap = cv2.VideoCapture(str(input_video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Cannot create output video: {output_video_path}")

        frame_count = 0
        
        target_age = settings['target_age']
        
        if progress_callback:
            progress_callback(f"Processing {total_frames} frames with age adjustment: {target_age} years")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if progress_callback and frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                progress_callback(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")


            processed_frame = apply_ageing(frame, target_age)
            
            if processed_frame is not None:
                out.write(processed_frame)
            else:
                out.write(frame) 

        cap.release()
        out.release()
        
        if progress_callback:
            progress_callback(f"✅ Video aging completed! Output saved to {output_video_path}")

        return {
            'success': True,
            'output_path': output_video_path
        }
        
    except Exception as e:
        print(f"[ERROR] Video aging failed: {str(e)}")
        traceback.print_exc()
        
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        
        return {
            'success': False,
            'error': str(e)
        }
