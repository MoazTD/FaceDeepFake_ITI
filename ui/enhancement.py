import os
import cv2
import numpy as np
from pathlib import Path
import datetime
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QGroupBox, QGridLayout, QCheckBox, QComboBox,
    QSpinBox, QMessageBox, QProgressBar, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

try:
    from gfpgan.utils import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    try:
        from gfpgan import GFPGANer  
        GFPGAN_AVAILABLE = True
    except ImportError:
        GFPGAN_AVAILABLE = False


class EnhancementWorker(QThread):
    
    finished = Signal(np.ndarray)  
    error = Signal(str)
    progress = Signal(int)
    
    def __init__(self, image, enhancer, settings):
        super().__init__()
        self.image = image
        self.enhancer = enhancer
        self.settings = settings
        
    def run(self):
        try:
            self.progress.emit(10)
            
            
            only_center_face = self.settings.get('only_center_face', False)
            has_aligned = self.settings.get('has_aligned', False)
            paste_back = self.settings.get('paste_back', True)
            
            self.progress.emit(30)
            
            
            _, _, enhanced_img = self.enhancer.enhance(
                self.image, 
                has_aligned=has_aligned, 
                only_center_face=only_center_face, 
                paste_back=paste_back
            )
            
            self.progress.emit(80)
            
            if enhanced_img is not None:
                
                strength = self.settings.get('strength', 1.0)
                if strength < 1.0:
                    enhanced_img = cv2.addWeighted(
                        self.image, 1.0 - strength, 
                        enhanced_img, strength, 0
                    )
                
                self.progress.emit(100)
                self.finished.emit(enhanced_img)
            else:
                self.error.emit("Enhancement failed - no output received")
                
        except Exception as e:
            self.error.emit(f"Enhancement error: {str(e)}")


class FaceEnhancementDialog(QDialog):
    
    def __init__(self, processor=None, initial_frame=None, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.original_frame = initial_frame
        self.enhanced_frame = None
        self.enhancer = None
        self.enhancement_worker = None
        self.settings = {}
        
        self.setWindowTitle("Face Enhancement")
        self.setFixedSize(1000, 700)
        self.setModal(True)
        
        self.setup_ui()
        self.setup_enhancer()
        
        if self.original_frame is not None:
            self.display_original_image()
    
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        
        title_label = QLabel("Face Enhancement")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(16)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        
        self.setup_image_comparison(main_layout)
        
        
        self.setup_enhancement_controls(main_layout)
        
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        
        self.setup_action_buttons(main_layout)
    
    def setup_image_comparison(self, main_layout):
        comparison_group = QGroupBox("Enhancement Preview")
        comparison_layout = QGridLayout(comparison_group)
        
        
        self.before_label = QLabel("Original")
        self.before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.before_image = QLabel()
        self.before_image.setFixedSize(400, 300)
        
        self.before_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.before_image.setText("No Image")
        
        
        self.after_label = QLabel("Enhanced")
        self.after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.after_image = QLabel()
        self.after_image.setFixedSize(400, 300)
        
        self.after_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.after_image.setText("Click 'Preview Enhancement'")
        
        
        comparison_layout.addWidget(self.before_label, 0, 0)
        comparison_layout.addWidget(self.after_label, 0, 1)
        comparison_layout.addWidget(self.before_image, 1, 0)
        comparison_layout.addWidget(self.after_image, 1, 1)
        
        main_layout.addWidget(comparison_group)
    
    def setup_enhancement_controls(self, main_layout):
        controls_group = QGroupBox("Enhancement Settings")
        controls_layout = QVBoxLayout(controls_group)
        
        
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Enhancement Strength:"))
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(80)
        self.strength_spinbox = QSpinBox()
        self.strength_spinbox.setRange(0, 100)
        self.strength_spinbox.setValue(80)
        self.strength_spinbox.setSuffix("%")
        
        
        self.strength_slider.valueChanged.connect(self.strength_spinbox.setValue)
        self.strength_spinbox.valueChanged.connect(self.strength_slider.setValue)
        self.strength_slider.valueChanged.connect(self.on_settings_changed)
        
        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_spinbox)
        controls_layout.addLayout(strength_layout)
        
        
        options_layout = QHBoxLayout()
        
        self.only_center_face_cb = QCheckBox("Enhance Only Center Face")
        self.only_center_face_cb.setChecked(False)
        self.only_center_face_cb.toggled.connect(self.on_settings_changed)
        
        self.paste_back_cb = QCheckBox("Paste Enhanced Face Back")
        self.paste_back_cb.setChecked(True)
        self.paste_back_cb.toggled.connect(self.on_settings_changed)
        
        options_layout.addWidget(self.only_center_face_cb)
        options_layout.addWidget(self.paste_back_cb)
        controls_layout.addLayout(options_layout)
        
        
        advanced_layout = QHBoxLayout()
        
        advanced_layout.addWidget(QLabel("Architecture:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["clean", "original", "RestoreFormer"])
        self.arch_combo.setCurrentText("clean")
        self.arch_combo.currentTextChanged.connect(self.on_settings_changed)
        advanced_layout.addWidget(self.arch_combo)
        
        advanced_layout.addWidget(QLabel("Channel Multiplier:"))
        self.channel_spinbox = QSpinBox()
        self.channel_spinbox.setRange(1, 4)
        self.channel_spinbox.setValue(2)
        self.channel_spinbox.valueChanged.connect(self.on_settings_changed)
        advanced_layout.addWidget(self.channel_spinbox)
        
        controls_layout.addLayout(advanced_layout)
        
        
        self.preview_btn = QPushButton("Preview Enhancement")
        self.preview_btn.clicked.connect(self.preview_enhancement)
        
        controls_layout.addWidget(self.preview_btn)
        
        main_layout.addWidget(controls_group)
    
    def setup_action_buttons(self, main_layout):
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply to Video")
        self.apply_btn.clicked.connect(self.accept)
        self.apply_btn.setEnabled(False)
        
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        
        button_layout.addStretch()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
    
    def setup_enhancer(self):
        if not GFPGAN_AVAILABLE:
            QMessageBox.critical(self, "Error", "GFPGAN library not found!\nPlease install it with: pip install gfpgan")
            return
        
        try:
            
            model_paths = [
                "models/GFPGANv1.4.pth",
                "Models/GFPGANv1.4.pth",
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                model_path = model_paths[-1]  
            
            self.enhancer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize GFPGAN enhancer:\n{str(e)}")
            self.enhancer = None
    
    def set_image(self, image):
        self.original_frame = image.copy() if image is not None else None
        self.enhanced_frame = None
        self.display_original_image()
        self.after_image.setText("Click 'Preview Enhancement'")
        self.apply_btn.setEnabled(False)
    
    def display_original_image(self):
        if self.original_frame is None:
            return
        
        
        rgb_image = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        
        scaled_pixmap = pixmap.scaled(
            self.before_image.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.before_image.setPixmap(scaled_pixmap)
    
    def display_enhanced_image(self, enhanced_image):
        if enhanced_image is None:
            return
        
        self.enhanced_frame = enhanced_image.copy()
        
        
        rgb_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        
        scaled_pixmap = pixmap.scaled(
            self.after_image.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.after_image.setPixmap(scaled_pixmap)
        self.apply_btn.setEnabled(True)
    
    def on_settings_changed(self):
        self.apply_btn.setEnabled(False)
        if self.enhanced_frame is not None:
            self.after_image.setText("Settings changed\nClick 'Preview Enhancement'")
            self.enhanced_frame = None
    
    def preview_enhancement(self):
        if self.original_frame is None:
            QMessageBox.warning(self, "Warning", "No image to enhance!")
            return
        
        if self.enhancer is None:
            QMessageBox.critical(self, "Error", "Enhancement model not available!")
            return
        
        
        self.settings = {
            'strength': self.strength_slider.value() / 100.0,
            'only_center_face': self.only_center_face_cb.isChecked(),
            'has_aligned': False,
            'paste_back': self.paste_back_cb.isChecked(),
            'arch': self.arch_combo.currentText(),
            'channel_multiplier': self.channel_spinbox.value()
        }
        
        
        self.preview_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.enhancement_worker = EnhancementWorker(
            self.original_frame, 
            self.enhancer, 
            self.settings
        )
        
        self.enhancement_worker.finished.connect(self.on_enhancement_finished)
        self.enhancement_worker.error.connect(self.on_enhancement_error)
        self.enhancement_worker.progress.connect(self.progress_bar.setValue)
        
        self.enhancement_worker.start()
    
    def on_enhancement_finished(self, enhanced_image):
        self.display_enhanced_image(enhanced_image)
        self.preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if self.enhancement_worker:
            self.enhancement_worker.quit()
            self.enhancement_worker.wait()
            self.enhancement_worker = None
    
    def on_enhancement_error(self, error_message):
        QMessageBox.critical(self, "Enhancement Error", error_message)
        self.preview_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if self.enhancement_worker:
            self.enhancement_worker.quit()
            self.enhancement_worker.wait()
            self.enhancement_worker = None
    
    def get_settings(self):
        return self.settings.copy() if self.settings else {}
    
    def closeEvent(self, event):
        if self.enhancement_worker and self.enhancement_worker.isRunning():
            self.enhancement_worker.quit()
            self.enhancement_worker.wait()
        event.accept()





def open_enhancement_dialog(self):
    if not self.cached_swapped_video and not self.target_video_path:
        QMessageBox.warning(self, "Warning", "Please load a video first!")
        return
    
    if not self.processor:
        QMessageBox.critical(self, "Error", "Face processor is not available.")
        return

    try:
        
        current_frame = self.get_current_preview_frame()
        if current_frame is None:
            QMessageBox.warning(self, "Frame Error", "Could not capture a frame from the video for preview.")
            return

        
        dialog = FaceEnhancementDialog(
            processor=self.processor, 
            initial_frame=current_frame,
            parent=self
        )
        
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            if settings:
                self.update_progress("Applying enhancement to the entire video...")
                
                
                input_video_path = self.cached_swapped_video or self.target_video_path
                
                
                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_video_path = str(output_dir / f"enhanced_{timestamp}.mp4")

                
                self.apply_enhancement_to_video(input_video_path, output_video_path, settings)
            else:
                self.update_progress("No enhancement settings to apply.")
        else:
            self.update_progress("Enhancement cancelled by user.")

    except Exception as e:
        QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        
def apply_enhancement_to_video(self, input_path, output_path, settings):
    if not GFPGAN_AVAILABLE:
        self.update_progress("❌ Error: GFPGAN library not found.")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        self.update_progress(f"❌ Error: Could not open video file {input_path}")
        return

    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    
    try:
        model_paths = [
            "models/GFPGANv1.4.pth",
            "Models/GFPGANv1.4.pth",
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            model_path = model_paths[-1]  
        
        enhancer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch=settings.get('arch', 'clean'),
            channel_multiplier=settings.get('channel_multiplier', 2),
            bg_upsampler=None
        )
    except Exception as e:
        self.update_progress(f"❌ Failed to initialize GFPGAN enhancer: {e}")
        cap.release()
        writer.release()
        return

    
    detector = self.processor.detector if self.processor else None
    strength = settings.get('strength', 0.8)
    only_center_face = settings.get('only_center_face', False)
    paste_back = settings.get('paste_back', True)
    
    self.update_progress(f"Enhancing {frame_count} frames with strength {strength:.2f}...")

    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret: 
            break

        processed_frame = frame.copy()

        try:
            
            _, _, enhanced_frame = enhancer.enhance(
                frame, 
                has_aligned=False, 
                only_center_face=only_center_face, 
                paste_back=paste_back
            )
            
            if enhanced_frame is not None:
                
                if strength < 1.0:
                    processed_frame = cv2.addWeighted(
                        frame, 1.0 - strength, 
                        enhanced_frame, strength, 0
                    )
                else:
                    processed_frame = enhanced_frame
                    
        except Exception as e:
            
            print(f"Could not enhance frame {i}: {e}")

        writer.write(processed_frame)

        
        if hasattr(self.ui, 'overall_progress') and i % 10 == 0:
            progress_percent = int(((i + 1) / frame_count) * 100)
            self.ui.overall_progress.setValue(progress_percent)

    cap.release()
    writer.release()
    
    
    self.update_progress(f"✅ Video enhancement complete! Saved to {output_path}")
    QMessageBox.information(self, "Success", f"Enhanced video saved successfully to:\n{output_path}")
    
    
    self.cached_swapped_video = output_path
    if hasattr(self, 'load_swapped_video_preview'):
        self.load_swapped_video_preview()
