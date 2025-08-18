import os
import time
import threading
import numpy as np
import cv2
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QCheckBox, QGroupBox, QFileDialog, QTextEdit,
    QSplitter, QSizePolicy, QScrollArea, QFrame, QRadioButton, QLineEdit,
    QDialog, QProgressBar, QGridLayout, QButtonGroup, QMessageBox, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize, QTimer, QRect
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QFont, QPalette


import ui
from .controller3D import MeshController

from PIL import Image, ImageDraw
from PySide6.QtCore import QThreadPool, QRunnable, Signal, Slot
from ui.themes import apply_theme

class ScalableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)
        self._pixmap = None

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap is None or self._pixmap.isNull():
            # Clear the display when pixmap is None or null
            super().setPixmap(QPixmap())
            return

        
        scaled = self._pixmap.scaled(
            self.width(),
            self.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        super().setPixmap(scaled)

    def resizeEvent(self, event):
        self._update_pixmap()
        super().resizeEvent(event)

    def clear_display(self):
        """Clear the display completely"""
        self._pixmap = None
        super().setPixmap(QPixmap())
        self.update()


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()

class WorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)
    result = Signal(object)
    



class FaceSwapUI(QMainWindow):

    update_video_frame_signal = Signal(np.ndarray)


    def __init__(self, target_video_path=None, cached_swapped_video=None, theme_name="dark"):

        super().__init__()

        
        apply_theme(self, theme_name)


        self.setWindowTitle("PixelSwap")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 700)
        self.thread_pool = QThreadPool()
        self.source_face_data = None
        self.target_face_data = None
        self.show_3d_overlay = False
        self.update_video_frame_signal.connect(self.update_video_frame)

        if cached_swapped_video:
            self.target_path = cached_swapped_video
        else:
            self.target_path = target_video_path


        
        self.playback_cap = None
        self.is_playing = False
        self.video_fps = 30
        self.video_frame_count = 0
        self.current_frame_num = 0
        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._video_playback_frame)

        
        self._create_ui()
        
        
        self.controller = MeshController(progress_callback=self.update_log)
        self._initialize_models()

        if self.target_path:
            self.set_target_video(self.target_path)
            # Show clear button if video is loaded
            self.target_clear_button.setVisible(True)

    def set_target_video(self, path):
        self.stop_playback()
        self.target_path = path
        try:
            
            if self.playback_cap:
                self.playback_cap.release()

            self.playback_cap = cv2.VideoCapture(path)
            if not self.playback_cap.isOpened():
                raise IOError("Cannot open video file.")

            self.video_frame_count = int(self.playback_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.video_frame_count <= 0:
                raise ValueError("Video has no readable frames")

            self.video_fps = self.playback_cap.get(cv2.CAP_PROP_FPS)
            self.video_slider.setRange(0, self.video_frame_count - 1)

            
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.playback_cap.read()

            if not success or frame is None or frame.size == 0:
                raise ValueError("Failed to read first video frame")

            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._update_thumbnail_from_image(
                self.target_canvas,
                Image.fromarray(rgb_frame)
            )

            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.video_slider.setEnabled(True)
            self.current_frame_num = 0

            self.update_log(f"✅ Target video loaded: {Path(path).name}")
            
            # Enable processing buttons if models are initialized
            if hasattr(self, 'controller') and self.controller.mesh_generator is not None:
                self.generate_overlay_button.setEnabled(True)
                self.generate_maps_button.setEnabled(True)
                self.generate_uv_button.setEnabled(True)

        except Exception as e:
            self.update_log(f"❌ Target video error: {e}")
            
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.video_slider.setEnabled(False)
            if self.playback_cap:
                self.playback_cap.release()
                self.playback_cap = None
                
            # Disable processing buttons on error
            self.generate_overlay_button.setEnabled(False)
            self.generate_maps_button.setEnabled(False)
            self.generate_uv_button.setEnabled(False)


    def update_video_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._update_video_frame(Image.fromarray(rgb_frame))



    def _create_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Create splitter for source and target sections
        content_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(content_splitter, 1)

        # Source Image Section
        source_widget = QWidget()
        source_layout = QVBoxLayout(source_widget)

        source_title = QLabel("Source Image")
        source_title.setAlignment(Qt.AlignCenter)
        font = source_title.font()
        font.setPointSize(14)
        font.setBold(True)
        source_title.setFont(font)
        source_layout.addWidget(source_title)

        # Source image canvas with clear button
        source_canvas_container = QWidget()
        source_canvas_layout = QVBoxLayout(source_canvas_container)
        source_canvas_layout.setContentsMargins(0, 0, 0, 0)
        source_canvas_layout.setSpacing(0)

        # Clear button for source
        self.source_clear_button = QPushButton("✕")
        self.source_clear_button.setFixedSize(30, 30)
        self.source_clear_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc3333;
            }
        """)
        self.source_clear_button.clicked.connect(self.clear_source)
        self.source_clear_button.setVisible(False)
        
        # Button container with right alignment
        source_button_container = QWidget()
        source_button_layout = QHBoxLayout(source_button_container)
        source_button_layout.setContentsMargins(5, 5, 5, 0)
        source_button_layout.addStretch()
        source_button_layout.addWidget(self.source_clear_button)
        source_canvas_layout.addWidget(source_button_container)

        # Source image canvas
        self.source_canvas = ScalableLabel()
        self.source_canvas.setAlignment(Qt.AlignCenter)
        self.source_canvas.setMinimumSize(400, 300)
        self.source_canvas.setFrameStyle(QFrame.StyledPanel)
        self.source_canvas.mousePressEvent = self.select_source
        source_canvas_layout.addWidget(self.source_canvas, 1)
        
        source_layout.addWidget(source_canvas_container, 1)

        # Source image selection button
        source_button = QPushButton("Select Source Image")
        source_button.clicked.connect(self.select_source)
        source_layout.addWidget(source_button)

        # Source processing buttons
        source_buttons_layout = QHBoxLayout()
        
        self.source_overlay_button = QPushButton("Generate 3D Overlay")
        self.source_overlay_button.clicked.connect(self.generate_source_overlay)
        self.source_overlay_button.setMinimumHeight(40)
        self.source_overlay_button.setEnabled(False)
        source_buttons_layout.addWidget(self.source_overlay_button)

        self.source_maps_button = QPushButton("Generate Maps")
        self.source_maps_button.clicked.connect(self.generate_source_maps)
        self.source_maps_button.setMinimumHeight(40)
        self.source_maps_button.setEnabled(False)
        source_buttons_layout.addWidget(self.source_maps_button)

        self.source_uv_button = QPushButton("Generate UV Texture")
        self.source_uv_button.clicked.connect(self.generate_source_uv_texture)
        self.source_uv_button.setMinimumHeight(40)
        self.source_uv_button.setEnabled(False)
        source_buttons_layout.addWidget(self.source_uv_button)

        self.source_obj_button = QPushButton("Export OBJ File")
        self.source_obj_button.clicked.connect(self.export_source_obj)
        self.source_obj_button.setMinimumHeight(40)
        self.source_obj_button.setEnabled(False)
        source_buttons_layout.addWidget(self.source_obj_button)

        source_layout.addLayout(source_buttons_layout)
        content_splitter.addWidget(source_widget)

        # Target Video Section
        target_widget = QWidget()
        target_layout = QVBoxLayout(target_widget)

        target_title = QLabel("Target Video")
        target_title.setAlignment(Qt.AlignCenter)
        font = target_title.font()
        font.setPointSize(14)
        font.setBold(True)
        target_title.setFont(font)
        target_layout.addWidget(target_title)

        # Target video canvas with clear button
        target_canvas_container = QWidget()
        target_canvas_layout = QVBoxLayout(target_canvas_container)
        target_canvas_layout.setContentsMargins(0, 0, 0, 0)
        target_canvas_layout.setSpacing(0)

        # Clear button for target
        self.target_clear_button = QPushButton("✕")
        self.target_clear_button.setFixedSize(30, 30)
        self.target_clear_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc3333;
            }
        """)
        self.target_clear_button.clicked.connect(self.clear_target)
        self.target_clear_button.setVisible(False)
        
        # Button container with right alignment
        target_button_container = QWidget()
        target_button_layout = QHBoxLayout(target_button_container)
        target_button_layout.setContentsMargins(5, 5, 5, 0)
        target_button_layout.addStretch()
        target_button_layout.addWidget(self.target_clear_button)
        target_canvas_layout.addWidget(target_button_container)

        # Target video canvas
        self.target_canvas = ScalableLabel()
        self.target_canvas.setAlignment(Qt.AlignCenter)
        self.target_canvas.setMinimumSize(400, 300)
        self.target_canvas.setFrameStyle(QFrame.StyledPanel)
        self.target_canvas.mousePressEvent = self.select_target
        target_canvas_layout.addWidget(self.target_canvas, 1)
        
        target_layout.addWidget(target_canvas_container, 1)

        # Video playback controls
        video_controls = QWidget()
        video_layout = QHBoxLayout(video_controls)

        self.play_button = QPushButton("▶")
        self.play_button.clicked.connect(self.toggle_play_pause)
        self.play_button.setEnabled(False)
        self.play_button.setFixedSize(40, 30)
        video_layout.addWidget(self.play_button)

        self.stop_button = QPushButton("■")
        self.stop_button.clicked.connect(self.stop_playback)
        self.stop_button.setEnabled(False)
        self.stop_button.setFixedSize(40, 30)
        video_layout.addWidget(self.stop_button)

        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.valueChanged.connect(self.seek_video)
        video_layout.addWidget(self.video_slider, 1)

        target_layout.addWidget(video_controls)

        # Video selection button
        target_button = QPushButton("Select Target Video")
        target_button.clicked.connect(self.select_target)
        target_layout.addWidget(target_button)

        # Video processing buttons
        video_buttons_layout = QHBoxLayout()
        
        self.generate_overlay_button = QPushButton("Generate 3D Video Overlay")
        self.generate_overlay_button.clicked.connect(self.generate_video_overlay)
        self.generate_overlay_button.setMinimumHeight(40)
        self.generate_overlay_button.setEnabled(False)
        video_buttons_layout.addWidget(self.generate_overlay_button)

        self.generate_maps_button = QPushButton("Generate Depth Maps & Texture")
        self.generate_maps_button.clicked.connect(self.generate_video_maps)
        self.generate_maps_button.setMinimumHeight(40)
        self.generate_maps_button.setEnabled(False)
        video_buttons_layout.addWidget(self.generate_maps_button)

        self.generate_uv_button = QPushButton("Generate UV Textures")
        self.generate_uv_button.clicked.connect(self.generate_uv_texture_only)
        self.generate_uv_button.setMinimumHeight(40)
        self.generate_uv_button.setEnabled(False)
        video_buttons_layout.addWidget(self.generate_uv_button)

        target_layout.addLayout(video_buttons_layout)
        content_splitter.addWidget(target_widget)

        # Output path selection section
        output_section = QWidget()
        output_layout = QHBoxLayout(output_section)
        self.output_path_button = QPushButton("Select Output Folder")
        self.output_path_button.clicked.connect(self.select_output)
        output_layout.addWidget(self.output_path_button)

        self.output_path_label = QLabel("No output path selected")
        self.output_path_label.setWordWrap(True)
        output_layout.addWidget(self.output_path_label, 1)
        main_layout.addWidget(output_section)

        
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        main_layout.addWidget(log_group)


    def _initialize_models(self):
        self.update_log("🚀 Initializing models...")

        worker = Worker(self._initialize_models_threaded)
        worker.signals.result.connect(self._models_initialized)
        worker.signals.error.connect(self.update_log)
        self.thread_pool.start(worker)


    def _initialize_models_threaded(self):

        try:
            success = self.controller.initialize_models()
            return success
        except Exception as e:
            self.update_log(f"❌ Model initialization error: {e}")
            return False


    def _models_initialized(self, success):
        if success:
            self.update_log("✅ Models initialized successfully.")
            # Enable target video buttons
            self.generate_overlay_button.setEnabled(True)
            self.generate_maps_button.setEnabled(True)
            self.generate_uv_button.setEnabled(True)
            # Enable source image buttons
            self.source_overlay_button.setEnabled(True)
            self.source_maps_button.setEnabled(True)
            self.source_uv_button.setEnabled(True)
            self.source_obj_button.setEnabled(True)
        else:
            self.update_log("❌ Failed to initialize models.")


    def update_log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )



    def clear_source(self):
        """Clear the source image"""
        self.source_canvas.clear_display()
        if hasattr(self, 'source_path'):
            delattr(self, 'source_path')
        
        # Hide clear button
        self.source_clear_button.setVisible(False)
        
        # Disable processing buttons
        self.source_overlay_button.setEnabled(False)
        self.source_maps_button.setEnabled(False)
        self.source_uv_button.setEnabled(False)
        self.source_obj_button.setEnabled(False)
        
        self.update_log("🗑️ Source image cleared")

    def clear_target(self):
        """Clear the target video"""
        self.update_log("🔍 Clear target button clicked - starting clear process...")
        try:
            # Stop playback first
            self.stop_playback()
            
            # Force stop any ongoing playback
            self.is_playing = False
            self.playback_timer.stop()
            
            # Clear the canvas completely
            self.target_canvas.clear_display()
            
            # Clear the target path
            if hasattr(self, 'target_path'):
                old_path = self.target_path
                delattr(self, 'target_path')
                self.update_log(f"🗑️ Cleared target path: {old_path}")
            else:
                self.update_log("🗑️ No target path to clear")
            
            # Release video capture
            if self.playback_cap:
                self.playback_cap.release()
                self.playback_cap = None
                self.update_log("🗑️ Released video capture")
            else:
                self.update_log("🗑️ No video capture to release")
            
            # Reset video frame count and current frame
            self.video_frame_count = 0
            self.current_frame_num = 0
            
            # Reset video slider
            self.video_slider.setRange(0, 0)
            self.video_slider.setValue(0)
            
            # Hide clear button
            self.target_clear_button.setVisible(False)
            
            # Disable video controls
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.video_slider.setEnabled(False)
            
            # Disable processing buttons
            self.generate_overlay_button.setEnabled(False)
            self.generate_maps_button.setEnabled(False)
            self.generate_uv_button.setEnabled(False)
            
            self.update_log("🗑️ Target video cleared successfully")
            
        except Exception as e:
            self.update_log(f"❌ Error clearing target video: {e}")

    def select_source(self, event=None):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Source Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return

        self.source_path = path
        try:
            # Load and display the image
            self._update_thumbnail(self.source_canvas, path, 'image')
            
            # Show clear button
            self.source_clear_button.setVisible(True)
            
            self.update_log(f"✅ Source image loaded: {Path(path).name}")
            
            # Enable processing buttons if models are initialized
            if hasattr(self, 'controller') and self.controller.mesh_generator is not None:
                self.source_overlay_button.setEnabled(True)
                self.source_maps_button.setEnabled(True)
                self.source_uv_button.setEnabled(True)
                self.source_obj_button.setEnabled(True)

        except Exception as e:
            self.update_log(f"❌ Source image error: {e}")
            
            # Disable processing buttons on error
            self.source_overlay_button.setEnabled(False)
            self.source_maps_button.setEnabled(False)
            self.source_uv_button.setEnabled(False)
            self.source_obj_button.setEnabled(False)

    def select_target(self, event=None):
        self.stop_playback()
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Video", "",
            "Video Files (*.mp4 *.mov *.avi)"
        )
        if not path:
            return

        self.target_path = path
        try:
            
            if self.playback_cap:
                self.playback_cap.release()

            self.playback_cap = cv2.VideoCapture(path)
            if not self.playback_cap.isOpened():
                raise IOError("Cannot open video file.")

            # Show clear button
            self.target_clear_button.setVisible(True)
            
            self.video_frame_count = int(self.playback_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.video_frame_count <= 0:
                raise ValueError("Video has no readable frames")

            self.video_fps = self.playback_cap.get(cv2.CAP_PROP_FPS)
            self.video_slider.setRange(0, self.video_frame_count - 1)

            
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.playback_cap.read()

            if not success or frame is None or frame.size == 0:
                raise ValueError("Failed to read first video frame")

            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._update_thumbnail_from_image(
                self.target_canvas,
                Image.fromarray(rgb_frame)
            )

            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.video_slider.setEnabled(True)
            self.current_frame_num = 0

            self.update_log(f"✅ Target video loaded: {Path(path).name}")
            
            # Enable processing buttons if models are initialized
            if hasattr(self, 'controller') and self.controller.mesh_generator is not None:
                self.generate_overlay_button.setEnabled(True)
                self.generate_maps_button.setEnabled(True)
                self.generate_uv_button.setEnabled(True)

        except Exception as e:
            self.update_log(f"❌ Target video error: {e}")
            
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.video_slider.setEnabled(False)
            if self.playback_cap:
                self.playback_cap.release()
                self.playback_cap = None
                
            # Disable processing buttons on error
            self.generate_overlay_button.setEnabled(False)
            self.generate_maps_button.setEnabled(False)
            self.generate_uv_button.setEnabled(False)

    def select_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if path:
            self.output_path_label.setText(Path(path).name)
            self.output_path = path

    
    def toggle_play_pause(self):
        if self.is_playing:
            self.is_playing = False
            self.playback_timer.stop()
            self.play_button.setText("▶")
        else:
            if not self.playback_cap:
                return
            self.is_playing = True
            self.play_button.setText("❚❚")
            interval = int(1000 / self.video_fps)
            self.playback_timer.start(interval)

    def stop_playback(self):
        self.is_playing = False
        self.playback_timer.stop()
        self.play_button.setText("▶")
        if self.playback_cap:
            self.current_frame_num = 0
            self.video_slider.setValue(0)
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.playback_cap.read()
            if success:
                self._update_video_frame(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def _video_playback_frame(self):
        if self.playback_cap and self.playback_cap.isOpened():
            success, frame = self.playback_cap.read()
            if success and frame is not None and frame.size > 0:
                self.current_frame_num += 1
                self.video_slider.setValue(self.current_frame_num)

                
                self.update_video_frame_signal.emit(frame)
            else:
                self.stop_playback()


    def seek_video(self, value):
        if self.playback_cap and not self.is_playing:
            self.current_frame_num = value
            self.playback_cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            success, frame = self.playback_cap.read()
            if success:
                self._update_video_frame(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def generate_video_overlay(self):
        """Generate 3D video overlay with face mesh"""
        if not self.target_path:
            QMessageBox.critical(self, "Input Error", "Please select a target video first.")
            return

        if not hasattr(self, 'output_path'):
            QMessageBox.critical(self, "Output Error", "Please select an output folder.")
            return

        self.generate_overlay_button.setEnabled(False)
        self.generate_overlay_button.setText("Generating Overlay...")

        self.update_log(f"Starting 3D overlay video generation...")

        def worker_function():
            try:
                output_video = self.controller.generate_video_overlay(
                    video_path=self.target_path,
                    output_dir=self.output_path
                )
                self.update_log(f"✅ 3D overlay video complete! Saved to: {output_video}")
                QTimer.singleShot(100, lambda: QMessageBox.information(
                    self, "Success", f"3D overlay video complete!\nSaved to: {Path(output_video).name}"
                ))
            except Exception as e:
                self.update_log(f"❌ 3D overlay generation failed: {e}")
                QTimer.singleShot(100, lambda: QMessageBox.critical(
                    self, "Processing Error", f"3D overlay generation failed:\n{e}"
                ))
            finally:
                QTimer.singleShot(100, self._reset_overlay_button)

        worker = Worker(worker_function)
        worker.signals.error.connect(self.update_log)
        self.thread_pool.start(worker)

    def generate_video_maps(self):
        """Generate depth maps, PNCC maps, texture, and UV texture from first frame"""
        if not self.target_path:
            QMessageBox.critical(self, "Input Error", "Please select a target video first.")
            return

        if not hasattr(self, 'output_path'):
            QMessageBox.critical(self, "Output Error", "Please select an output folder.")
            return

        self.generate_maps_button.setEnabled(False)
        self.generate_maps_button.setText("Generating Maps...")

        self.update_log(f"Starting depth maps, texture, and UV texture generation...")

        def worker_function():
            try:
                # Generate maps for the video
                maps_result = self.controller.generate_video_maps(
                    video_path=self.target_path,
                    output_dir=self.output_path
                )
                
                # Generate texture from first frame
                first_frame_path = self._extract_first_frame()
                if first_frame_path:
                    tex_path, depth_path, pncc_path = self.controller.generate_3d_maps(first_frame_path)
                    self.update_log(f"✅ Texture from first frame: {Path(tex_path).name}")
                
                self.update_log(f"✅ Maps generation complete!")
                self.update_log(f"   Depth maps: {len(maps_result.get('depth', []))} files")
                self.update_log(f"   PNCC maps: {len(maps_result.get('pncc', []))} files")
                self.update_log(f"   Texture maps: {len(maps_result.get('texture', []))} files")
                
                QTimer.singleShot(100, lambda: QMessageBox.information(
                    self, "Success", f"Maps generation complete!\nCheck the output folder for results."
                ))
            except Exception as e:
                self.update_log(f"❌ Maps generation failed: {e}")
                QTimer.singleShot(100, lambda: QMessageBox.critical(
                    self, "Processing Error", f"Maps generation failed:\n{e}"
                ))
            finally:
                QTimer.singleShot(100, self._reset_maps_button)

        worker = Worker(worker_function)
        worker.signals.error.connect(self.update_log)
        self.thread_pool.start(worker)

    def generate_source_overlay(self):
        """Generate 3D overlay for source image"""
        if not hasattr(self, 'source_path'):
            QMessageBox.critical(self, "Input Error", "Please select a source image first.")
            return

        if not hasattr(self, 'output_path'):
            QMessageBox.critical(self, "Output Error", "Please select an output folder.")
            return

        self.source_overlay_button.setEnabled(False)
        self.source_overlay_button.setText("Generating Overlay...")

        self.update_log(f"Starting 3D overlay generation for source image...")

        def worker_function():
            try:
                # Generate 3D overlay for source image
                overlay_path = self.controller.generate_3d_mesh(self.source_path)
                
                # Copy to output directory if specified
                if hasattr(self, 'output_path'):
                    import shutil
                    output_file = Path(self.output_path) / Path(overlay_path).name
                    shutil.copy2(overlay_path, output_file)
                    overlay_path = str(output_file)
                
                self.update_log(f"✅ 3D overlay generated successfully!")
                self.update_log(f"   Overlay: {Path(overlay_path).name}")
                
                QTimer.singleShot(100, lambda: QMessageBox.information(
                    self, "Success", f"3D overlay generated successfully!\nSaved to: {Path(overlay_path).name}"
                ))
                    
            except Exception as e:
                self.update_log(f"❌ 3D overlay generation failed: {e}")
                QTimer.singleShot(100, lambda: QMessageBox.critical(
                    self, "Processing Error", f"3D overlay generation failed:\n{e}"
                ))
            finally:
                QTimer.singleShot(100, self._reset_source_overlay_button)

        worker = Worker(worker_function)
        worker.signals.error.connect(self.update_log)
        self.thread_pool.start(worker)

    def generate_source_maps(self):
        """Generate maps for source image"""
        if not hasattr(self, 'source_path'):
            QMessageBox.critical(self, "Input Error", "Please select a source image first.")
            return

        if not hasattr(self, 'output_path'):
            QMessageBox.critical(self, "Output Error", "Please select an output folder.")
            return

        self.source_maps_button.setEnabled(False)
        self.source_maps_button.setText("Generating Maps...")

        self.update_log(f"Starting maps generation for source image...")

        def worker_function():
            try:
                # Generate maps for source image
                tex_path, depth_path, pncc_path = self.controller.generate_3d_maps(self.source_path)
                
                # Copy to output directory if specified
                if hasattr(self, 'output_path'):
                    import shutil
                    output_dir = Path(self.output_path)
                    for path in [tex_path, depth_path, pncc_path]:
                        output_file = output_dir / Path(path).name
                        shutil.copy2(path, output_file)
                
                self.update_log(f"✅ Maps generation complete!")
                self.update_log(f"   Texture: {Path(tex_path).name}")
                self.update_log(f"   Depth: {Path(depth_path).name}")
                self.update_log(f"   PNCC: {Path(pncc_path).name}")
                
                QTimer.singleShot(100, lambda: QMessageBox.information(
                    self, "Success", f"Maps generation complete!\nCheck the output folder for results."
                ))
                    
            except Exception as e:
                self.update_log(f"❌ Maps generation failed: {e}")
                QTimer.singleShot(100, lambda: QMessageBox.critical(
                    self, "Processing Error", f"Maps generation failed:\n{e}"
                ))
            finally:
                QTimer.singleShot(100, self._reset_source_maps_button)

        worker = Worker(worker_function)
        worker.signals.error.connect(self.update_log)
        self.thread_pool.start(worker)

    def generate_source_uv_texture(self):
        """Generate UV texture for source image"""
        if not hasattr(self, 'source_path'):
            QMessageBox.critical(self, "Input Error", "Please select a source image first.")
            return

        if not hasattr(self, 'output_path'):
            QMessageBox.critical(self, "Output Error", "Please select an output folder.")
            return

        self.source_uv_button.setEnabled(False)
        self.source_uv_button.setText("Generating UV Texture...")

        self.update_log(f"Starting UV texture generation for source image...")

        def worker_function():
            try:
                # Generate UV texture for source image
                uv_tex_path = self.controller.generate_uv_texture(self.source_path, self.output_path)
                
                self.update_log(f"✅ UV texture generated successfully!")
                self.update_log(f"   UV texture: {Path(uv_tex_path).name}")
                
                QTimer.singleShot(100, lambda: QMessageBox.information(
                    self, "Success", f"UV texture generated successfully!\nSaved to: {Path(uv_tex_path).name}"
                ))
                    
            except Exception as e:
                self.update_log(f"❌ UV texture generation failed: {e}")
                QTimer.singleShot(100, lambda: QMessageBox.critical(
                    self, "Processing Error", f"UV texture generation failed:\n{e}"
                ))
            finally:
                QTimer.singleShot(100, self._reset_source_uv_button)

        worker = Worker(worker_function)
        worker.signals.error.connect(self.update_log)
        self.thread_pool.start(worker)

    def export_source_obj(self):
        """Export OBJ file for source image"""
        if not hasattr(self, 'source_path'):
            QMessageBox.critical(self, "Input Error", "Please select a source image first.")
            return

        if not hasattr(self, 'output_path'):
            QMessageBox.critical(self, "Output Error", "Please select an output folder.")
            return

        self.source_obj_button.setEnabled(False)
        self.source_obj_button.setText("Exporting OBJ...")

        self.update_log(f"Starting OBJ export for source image...")

        def worker_function():
            try:
                # Export OBJ file for source image
                obj_path = self.controller.export_3d_obj(self.source_path)
                
                # Copy to output directory if specified
                if hasattr(self, 'output_path'):
                    import shutil
                    output_file = Path(self.output_path) / Path(obj_path).name
                    shutil.copy2(obj_path, output_file)
                    obj_path = str(output_file)
                
                self.update_log(f"✅ OBJ file exported successfully!")
                self.update_log(f"   OBJ file: {Path(obj_path).name}")
                
                QTimer.singleShot(100, lambda: QMessageBox.information(
                    self, "Success", f"OBJ file exported successfully!\nSaved to: {Path(obj_path).name}"
                ))
                    
            except Exception as e:
                self.update_log(f"❌ OBJ export failed: {e}")
                QTimer.singleShot(100, lambda: QMessageBox.critical(
                    self, "Processing Error", f"OBJ export failed:\n{e}"
                ))
            finally:
                QTimer.singleShot(100, self._reset_source_obj_button)

        worker = Worker(worker_function)
        worker.signals.error.connect(self.update_log)
        self.thread_pool.start(worker)

    def generate_uv_texture_only(self):
        """Generate UV textures for each frame"""
        if not self.target_path:
            QMessageBox.critical(self, "Input Error", "Please select a target video first.")
            return

        if not hasattr(self, 'output_path'):
            QMessageBox.critical(self, "Output Error", "Please select an output folder.")
            return

        self.generate_uv_button.setEnabled(False)
        self.generate_uv_button.setText("Generating UV Textures...")

        self.update_log(f"Starting UV texture generation for all frames...")

        def worker_function():
            try:
                # Generate UV textures for all frames
                uv_result = self.controller.generate_video_uv_textures(
                    video_path=self.target_path,
                    output_dir=self.output_path
                )
                
                self.update_log(f"✅ UV texture generation complete!")
                self.update_log(f"   UV textures: {len(uv_result.get('uv', []))} files")
                
                QTimer.singleShot(100, lambda: QMessageBox.information(
                    self, "Success", f"UV texture generation complete!\nCheck the output folder for results."
                ))
                    
            except Exception as e:
                self.update_log(f"❌ UV texture generation failed: {e}")
                QTimer.singleShot(100, lambda: QMessageBox.critical(
                    self, "Processing Error", f"UV texture generation failed:\n{e}"
                ))
            finally:
                QTimer.singleShot(100, self._reset_uv_button)

        worker = Worker(worker_function)
        worker.signals.error.connect(self.update_log)
        self.thread_pool.start(worker)

    def _extract_first_frame(self):
        """Extract first frame from video for texture generation"""
        try:
            cap = cv2.VideoCapture(self.target_path)
            success, frame = cap.read()
            cap.release()
            
            if success and frame is not None:
                temp_dir = Path("temp_dialog3d")
                temp_dir.mkdir(exist_ok=True)
                first_frame_path = temp_dir / "first_frame.jpg"
                cv2.imwrite(str(first_frame_path), frame)
                return str(first_frame_path)
        except Exception as e:
            self.update_log(f"⚠️ Could not extract first frame: {e}")
        return None



    
    
    

    def _run_controller_process(self, source_input, target_path, output_file, options):
        try:
            self.controller.process_video(source_input, target_path, output_file, options)
            self.update_log(f"✅ Video processing complete! Saved to: {output_file}")
            QTimer.singleShot(100, lambda: QMessageBox.information(
                self, "Success", f"Video processing complete!\nSaved to: {Path(output_file).name}"
            ))
        except Exception as e:
            self.update_log(f"❌ Processing failed: {e}")
            QTimer.singleShot(100, lambda: QMessageBox.critical(
                self, "Processing Error", f"A critical error occurred:\n{e}"
            ))
        finally:
            QTimer.singleShot(100, self._reset_export_button)

    def _reset_overlay_button(self):
        self.generate_overlay_button.setEnabled(True)
        self.generate_overlay_button.setText("Generate 3D Video Overlay")

    def _reset_maps_button(self):
        self.generate_maps_button.setEnabled(True)
        self.generate_maps_button.setText("Generate Depth Maps & Texture")

    def _reset_source_overlay_button(self):
        self.source_overlay_button.setEnabled(True)
        self.source_overlay_button.setText("Generate 3D Overlay")

    def _reset_source_maps_button(self):
        self.source_maps_button.setEnabled(True)
        self.source_maps_button.setText("Generate Maps")

    def _reset_source_uv_button(self):
        self.source_uv_button.setEnabled(True)
        self.source_uv_button.setText("Generate UV Texture")

    def _reset_source_obj_button(self):
        self.source_obj_button.setEnabled(True)
        self.source_obj_button.setText("Export OBJ File")

    def _reset_uv_button(self):
        self.generate_uv_button.setEnabled(True)
        self.generate_uv_button.setText("Generate UV Textures")






    def _choose_export_dir(self, dialog):
        path = QFileDialog.getExistingDirectory(dialog, "Select Export Folder")
        if path:
            self.export_dir_label.setText(Path(path).name)
            self.export_dir_path = path





    
    def _update_thumbnail(self, canvas, file_path, media_type):
        try:
            if media_type == 'image':
                img = Image.open(file_path)
            elif media_type == 'video':
                cap = cv2.VideoCapture(file_path)
                success, frame = cap.read()
                cap.release()
                if not success:
                    raise IOError("Failed to read first frame of video.")
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            self._update_thumbnail_from_image(canvas, img)
        except Exception as e:
            self.update_log(f"❌ Error creating thumbnail: {e}")
            
            canvas.set_pixmap(QPixmap())

    def _update_thumbnail_from_image(self, canvas, pil_image):

        if pil_image.mode == 'RGB':
            qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGB888)
        elif pil_image.mode == 'RGBA':
            qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGBA8888)
        else:

            pil_image = pil_image.convert('RGB')
            qimage = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage)
        canvas.set_pixmap(pixmap)





    def _update_video_frame(self, pil_image):
        self._update_thumbnail_from_image(self.target_canvas, pil_image)

    def set_source_image_from_frame(self, pil_image):
        """Set source image from a PIL Image object (used by main UI)"""
        try:
            # Update the source canvas with the PIL image
            self._update_thumbnail_from_image(self.source_canvas, pil_image)
            
            # Show clear button
            self.source_clear_button.setVisible(True)
            
            # Enable processing buttons if models are initialized
            if hasattr(self, 'controller') and self.controller.mesh_generator is not None:
                self.source_overlay_button.setEnabled(True)
                self.source_maps_button.setEnabled(True)
                self.source_uv_button.setEnabled(True)
                self.source_obj_button.setEnabled(True)
            
            self.update_log(f"✅ Source image set from frame")
            
        except Exception as e:
            self.update_log(f"❌ Error setting source image from frame: {e}")
            
            # Disable processing buttons on error
            self.source_overlay_button.setEnabled(False)
            self.source_maps_button.setEnabled(False)
            self.source_uv_button.setEnabled(False)
            self.source_obj_button.setEnabled(False)

    def closeEvent(self, event):

        self.stop_playback()


        if self.playback_cap:
            self.playback_cap.release()

        self.thread_pool.waitForDone(5000)

        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = FaceSwapUI()
    window.show()
    app.exec()
