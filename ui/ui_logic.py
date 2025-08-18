import os
import sys
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import onnxruntime as ort
import cv2


try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

from .aging import (
    FaceAgingWindow,
    apply_aging_to_video,
    detect_faces_opencv,
    find_model_path
)


from .enhancement import FaceEnhancementDialog

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. System monitoring will be limited.")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available. GPU monitoring will be limited.")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QGroupBox,
    QPushButton, QProgressBar, QTextEdit, QWidget, QSlider,
    QCheckBox, QComboBox
)
from PySide6.QtCore import (
    QSize, QEvent, Qt, QThread, Signal, QTimer,
    QPropertyAnimation, QEasingCurve, QUrl, QMutex, QMutexLocker,
    QCoreApplication, QDate, QDateTime, QLocale, QMetaObject,
    QObject, QPoint, QRect, QTime
)
from PySide6.QtGui import (
    QPixmap, QImage, QColor, QIcon, QAction, QBrush,
    QConicalGradient, QCursor, QFont, QFontDatabase, QGradient,
    QKeySequence, QLinearGradient, QPalette, QRadialGradient, QTransform
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

try:
    from .themes import apply_theme
except ImportError:
    from ui.themes import apply_theme

try:
    from .main_window import Ui_MainWindow
except ImportError:
    from ui.main_window import Ui_MainWindow

try:
    from .mask_app import MaskEditorWindow
except ImportError:
    from ui.mask_app import MaskEditorWindow

try:
    from .uitls import logger, SettingsDialog, ProcessingThread, CacheManager, safe_paint
except ImportError:
    from ui.uitls import logger, SettingsDialog, ProcessingThread, CacheManager, safe_paint

try:
    from core.process import FaceSwapProcessor
    from core import video_utils
except ImportError as e:
    print(f"Core component import error: {e}")
    FaceSwapProcessor = None
    video_utils = None

try:
    from sound_handler import RVCConverterGUI
except ImportError:
    print("Could not import RVCConverterGUI from sound_handler. Voice conversion UI will be unavailable.")
    RVCConverterGUI = None

from core.face_detector import FaceDetector

class EnhancedMainWindow(QMainWindow):

    def __init__(self):
        super().__init__(parent=None)

        if Ui_MainWindow is None:
            QMessageBox.critical(self, "Initialization Error",
                                "UI module (main_window.py) not found. The application cannot start.")
            sys.exit(1)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self._initialize_attributes()
        self._initialize_components()
        self.setup_ui_connections()
        self._setup_enhanced_ui()
        self.setup_preview_controls()

        apply_theme(self, self.current_theme)
        self.update_progress("Application initialized successfully")

        self._start_system_monitoring()
        QTimer.singleShot(100, self.initialize_models)

        self.update_progress("🎉 Welcome to PixelSwap!")
        if logger:
            logger.info("Application initialized successfully")

    def _initialize_attributes(self):
        """Initialize all instance attributes"""
        self.processor = None
        self.processing_thread = None
        self.is_processing = False
        self.source_image_path = None
        self.target_video_path = None
        self.current_mask = None
        self.cached_swapped_video = None
        self.video_player = None
        self.selected_face_index = -1
        self.detected_faces = []
        self.unique_faces_info = []
        self.original_target_layout = None
        self.current_theme = "dark"
        self.current_frame = None
        self.current_frame_cache = {'position': 0, 'frame': None}
        self.cache_manager = None
        self.tracked_mask_data = None
        self.is_playing = False
        self.aging_session = None
        self.voice_converter_window = None
        self.converted_audio_path = None
        self.source_face_embedding = None
        self.target_face_embeddings = []
        self.selected_target_face_embedding = None
        self.dialog_3d_instance = None

    def _initialize_components(self):
        """Initialize processors and threads"""
        try:
            if FaceSwapProcessor:
                try:
                    self.processor = FaceSwapProcessor()
                    self.processor.progress_updated.connect(self.update_progress)

                    if logger:
                        logger.info("Face swap processor initialized and connected via Signals/Slots.")
                except Exception as e:
                    logger.exception("CRITICAL: Failed to create FaceSwapProcessor instance!")
                    self.processor = None

            if ProcessingThread and self.processor:
                self.processing_thread = ProcessingThread(self.processor)
                self.processing_thread.progress.connect(self.update_progress)
                self.processing_thread.finished.connect(self.on_processing_finished)
                if logger:
                    logger.info("Processing thread initialized successfully")

            if CacheManager:
                self.cache_manager = CacheManager()
                if logger:
                    logger.info("CacheManager initialized successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Initialization Error",
                                f"Failed to initialize application components: {str(e)}")
            if logger:
                logger.critical(f"Application initialization failed: {e}")

    def _start_system_monitoring(self):
        """Start system resource monitoring"""
        self.system_monitor_timer = QTimer(self)
        self.system_monitor_timer.timeout.connect(self.update_system_resources)
        self.system_monitor_timer.start(2000)

    def initialize_models(self):
        """Initialize AI models in the background."""
        self.update_progress("🚀 Initializing AI models (This may take a moment)...")
        try:
            if self.processor:
                if hasattr(self.processor, 'initialize_models_async'):
                    self.processor.initialize_models_async(self.on_models_initialized)
                else:
                    self.processor.initialize_models()
                    self.on_models_initialized(True)
            else:
                self.update_progress("Face swap processor not available, skipping model initialization.")
                self.on_models_initialized(False)

            self._initialize_aging_model()

        except Exception as e:
            self.update_progress(f"❌ Error during model initialization: {e}")
            if logger:
                logger.error(f"Model initialization error: {e}", exc_info=True)
            self.on_models_initialized(False)
            self.aging_session = None

    def _initialize_aging_model(self):
        """Initialize face aging model"""
        self.update_progress("Searching for face aging model...")
        aging_model_path = find_model_path("styleganex_age.onnx")
        if aging_model_path:
            self.update_progress(f"✅ Found aging model at: {aging_model_path}")
            try:
                self.aging_session = ort.InferenceSession(
                    str(aging_model_path),
                    providers=['CUDAExecutionProvider']
                )
                self.update_progress("✅ Face aging model loaded successfully!")
            except Exception as e:
                self.update_progress(f"❌ Error loading aging model: {e}")
                self.aging_session = None
        else:
            self.update_progress("❌ CRITICAL: 'styleganex_age.onnx' not found in project folders.")
            self.update_progress("Please ensure the model file is in one of these locations:")
            self.update_progress("- ./Models/styleganex_age.onnx")
            self.update_progress("- ./models/styleganex_age.onnx")
            self.aging_session = None

    def on_models_initialized(self, success):
        """Called when model initialization is complete."""
        if success:
            self.update_progress("✅ Models initialized successfully!")
        else:
            self.update_progress("⚠️ Some models failed to initialize")

        self.setup_ui_states()

    def setup_ui_states(self):
        """Update UI states based on available models."""

        if hasattr(self.ui, "btnAgeEditor"):
            aging_available = (
                (hasattr(self, 'aging_session') and self.aging_session is not None)
            )

            self.ui.btnAgeEditor.setEnabled(aging_available)

            if not aging_available:
                self.ui.btnAgeEditor.setToolTip(
                    "Face Aging Not Available\n\n"
                    "Requirements:\n"
                    "• StyleGAN model: 'styleganex_age.onnx' in Models folder\n"
                    "• Valid face processor"
                )
            else:
                self.ui.btnAgeEditor.setToolTip(
                    "Launch Face Aging Editor\n\n"
                    "Supports StyleGANex aging with age adjustment"
                )

    def check_aging_requirements(self):
        """Check and report aging requirements status."""
        status = {
            'stylegan_available': hasattr(self, 'aging_session') and self.aging_session is not None,
            'processor_available': self.processor is not None
        }

        self.update_progress("🔍 Checking aging capabilities...")
        self.update_progress(f"  StyleGAN Model: {'✅' if status['stylegan_available'] else '❌'}")
        self.update_progress(f"  Face Processor: {'✅' if status['processor_available'] else '❌'}")

        if not status['stylegan_available']:
            self.update_progress("⚠️ No aging methods available!")
            self.update_progress("   Add styleganex_age.onnx to Models folder")

        return status

    def setup_ui_connections(self):
        """Setup all UI signal connections"""
        self._connect_main_buttons()
        self._connect_menu_actions()
        self._connect_combo_boxes()
        self._connect_switches_and_checkboxes()
        self._connect_sliders()
        self._connect_custom_widgets()
        self._connect_system_monitors()

    def _connect_main_buttons(self):
        """Connect main action buttons"""
        if hasattr(self.ui, 'detectButton_3'):
            self.ui.detectButton_3.clicked.connect(self.detect_faces)
        if hasattr(self.ui, 'alignButton_3'):
            self.ui.alignButton_3.clicked.connect(self.align_faces)
        if hasattr(self.ui, 'swapButton_3'):
            self.ui.swapButton_3.clicked.connect(self.perform_swap)
        if hasattr(self.ui, 'applyEnhanceButton_3'):
            self.ui.applyEnhanceButton_3.clicked.connect(self.open_enhancement_dialog)
        if hasattr(self.ui, 'exportButton_2'):
            self.ui.exportButton_2.clicked.connect(self.export_video)
        if hasattr(self.ui, "adjustmentsButton_01"):
            self.ui.adjustmentsButton_01.clicked.connect(self.launch_adjustment_Dialog)
        if hasattr(self.ui, "btnLandmarksEditor"):
            self.ui.btnLandmarksEditor.clicked.connect(self.landmark_launcher)
        if hasattr(self.ui, "btnAgeEditor"):
            self.ui.btnAgeEditor.clicked.connect(self.launch_aging_dialog)
        if hasattr(self.ui, "btnSoundProcessor"):
             self.ui.btnSoundProcessor.clicked.connect(self.launch_voice_converter)
        if hasattr(self.ui, "btnMeshGenerator"):
             self.ui.btnMeshGenerator.clicked.connect(self.dialog_3d_runer)

    def _connect_menu_actions(self):
        """Connect menu actions"""
        menu_connections = [
            ('actionprefrances', self.show_settings),
            ('actionsave', self.save_project),
            ('actionimport_image', lambda: self.select_source_image(None)),
            ('actionimport_video', lambda: self.select_target_video(None)),
            ('actiondelete_source', self.delete_source),
            ('actiondelete_target', self.delete_target)
        ]

        for attr_name, handler in menu_connections:
            if hasattr(self.ui, attr_name):
                getattr(self.ui, attr_name).triggered.connect(handler)

    def _connect_combo_boxes(self):
        """Connect combo box change events"""
        if hasattr(self.ui, 'comboBox_3'):
            self.ui.comboBox_3.currentTextChanged.connect(self.on_detector_changed)
        if hasattr(self.ui, 'comboBox_4'):
            self.ui.comboBox_4.currentTextChanged.connect(self.on_model_changed)
        if hasattr(self.ui, 'comboBox_2'):
            self.ui.comboBox_2.currentTextChanged.connect(self.on_export_format_changed)

    def _connect_switches_and_checkboxes(self):
        """Connect switches and checkboxes"""
        switch_connections = [
            ('alignmentEnableSwitch', self.on_alignment_switch_toggled),
            ('enhancementEnableSwitch', self.on_enhancement_switch_toggled),
            ('FaceAgeingEnableSwitch', self.on_3d_alignment_switch_toggled),
            ('autoAlignCheck', self.on_auto_align_toggled),
            ('preserveLightingCheck', self.on_preserve_lighting_toggled),
            ('smoothTransitionCheck', self.on_smooth_transition_toggled),
            ('Mask01', self.on_mask_selected),
            ('Auto01', self.auto_alignmentt_runner)
        ]

        for attr_name, handler in switch_connections:
            if hasattr(self.ui, attr_name):
                getattr(self.ui, attr_name).toggled.connect(handler)

    def _connect_sliders(self):
        """Connect sliders and their value displays"""
        sliders_and_values = {
            'precisionSlider_3': ('precisionValue_3', self.on_precision_changed),
            'blendSlider_3': ('blendValue_3', self.on_blend_changed),
            'edgeSlider_3': ('edgeValue_3', self.on_edge_changed),
            'strengthSlider_3': ('strengthValue_3', self.on_strength_changed),
            'skinSlider_3': ('skinValue_3', self.on_skin_changed),
            'qualitySlider_3': ('qualityValue_3', self.on_quality_changed)
        }

        for slider_name, (value_name, handler) in sliders_and_values.items():
            if hasattr(self.ui, slider_name) and hasattr(self.ui, value_name):
                slider = getattr(self.ui, slider_name)
                value_display = getattr(self.ui, value_name)
                slider.valueChanged.connect(lambda value, vd=value_display: vd.setText(str(value)))
                slider.valueChanged.connect(handler)
                value_display.setText(str(slider.value()))

    def _connect_custom_widgets(self):
        """Connect custom widgets"""
        if hasattr(self.ui, 'detected_faces_widget'):
            self.ui.detected_faces_widget.face_clicked.connect(self.on_face_clicked)

        if hasattr(self.ui, 'video_preview_widget'):
            self.ui.video_preview_widget.position_changed.connect(self.update_video_position)
            self.ui.video_preview_widget.duration_changed.connect(self.update_video_duration)

    def _connect_system_monitors(self):
        """Connect system resource progress bars"""
        monitor_connections = [
            ('cpu_progress', self.on_cpu_usage_changed),
            ('ram_progress', self.on_ram_usage_changed),
            ('gpu_progress', self.on_gpu_usage_changed),
            ('vram_progress', self.on_vram_usage_changed)
        ]

        for attr_name, handler in monitor_connections:
            if hasattr(self.ui, attr_name):
                getattr(self.ui, attr_name).valueChanged.connect(handler)

    def setup_preview_controls(self):
        """Setup connections for video preview radio buttons"""
        if hasattr(self.ui, 'swapped_preview_radio'):
            self.ui.swapped_preview_radio.setEnabled(False)
        if hasattr(self.ui, 'target_preview_radio'):
            self.ui.target_preview_radio.toggled.connect(self.on_preview_source_changed)
        if hasattr(self.ui, 'swapped_preview_radio'):
            self.ui.swapped_preview_radio.toggled.connect(self.on_preview_source_changed)

    def _setup_enhanced_ui(self):
        """Setup enhanced UI components"""
        if hasattr(self.ui, 'video_preview_widget'):
            self.video_preview = self.ui.video_preview_widget
        if hasattr(self.ui, 'detected_faces_widget'):
            self.detected_faces_widget = self.ui.detected_faces_widget
        self._setup_drag_drop()
        self._setup_menu_actions()
        self.setMinimumSize(1400, 900)
        self.setWindowTitle("PixelSwap")

        if hasattr(self.ui, 'main_splitter'):
            self.ui.main_splitter.setToolTip("Double-click on splitter handles to collapse/expand panels")

    def _setup_drag_drop(self):
        """Enhanced drag and drop setup for specific QLabel widgets"""
        if hasattr(self.ui, 'dropSource_4'):
            self.ui.dropSource_4.setAcceptDrops(True)
            self.ui.dropSource_4.installEventFilter(self)
            self.ui.dropSource_4.mousePressEvent = self.select_source_image
            self.ui.dropSource_4.setText(" Drop/Click to select Source Image ")

        if hasattr(self.ui, 'dropTarget_2'):
            self.ui.dropTarget_2.setAcceptDrops(True)
            self.ui.dropTarget_2.installEventFilter(self)
            self.ui.dropTarget_2.mousePressEvent = self.select_target_video
            self.ui.dropTarget_2.setText(" Drop/Click to select Target Video ")

    def _setup_menu_actions(self):
        """Setup menu actions"""
        menu_actions = [
            ('actionprefrances', self.show_settings),
            ('actionsave', self.save_project),
            ('actionimport_image', lambda: self.select_source_image(None)),
            ('actionimport_video', lambda: self.select_target_video(None)),
            ('actiondelete_source', self.delete_source),
            ('actiondelete_target', self.delete_target)
        ]

        for attr_name, handler in menu_actions:
            if hasattr(self.ui, attr_name):
                getattr(self.ui, attr_name).triggered.connect(handler)

    def on_detector_changed(self, text):
        self.update_progress(f"Detector changed to: {text}")
        if self.processor:
            self.processor.set_detector_model(text)

    def on_model_changed(self, text):
        self.update_progress(f"Model changed to: {text}")
        if self.processor:
            self.processor.set_swap_model(text)

    def on_alignment_switch_toggled(self, checked):
        self.update_progress(f"Alignment enabled: {checked}")
        controls = ['precisionSlider_3', 'blendSlider_3', 'edgeSlider_3', 'Mask01']
        for control in controls:
            if hasattr(self.ui, control):
                getattr(self.ui, control).setEnabled(checked)

    def on_enhancement_switch_toggled(self, checked):
        self.update_progress(f"Enhancement enabled: {checked}")

    def on_3d_alignment_switch_toggled(self, checked):
        self.update_progress(f"Face Ageing enabled: {checked}")

    def on_precision_changed(self, value):
        self.update_progress(f"Precision: {value}")

    def on_blend_changed(self, value):
        self.update_progress(f"Blend: {value}")

    def on_edge_changed(self, value):
        self.update_progress(f"Edge: {value}")

    def on_strength_changed(self, value):
        self.update_progress(f"Strength: {value}")

    def on_skin_changed(self, value):
        self.update_progress(f"Skin Smoothness: {value}")

    def on_quality_changed(self, value):
        self.update_progress(f"Quality: {value}")

    def on_enhancement_type_changed(self, text):
        self.update_progress(f"Enhancement Type: {text}")

    def on_mask_selected(self, checked):
        self.update_progress(f"Mask alignment selected: {checked}")

    def on_auto_align_toggled(self, checked):
        self.update_progress(f"Auto Align: {checked}")

    def on_preserve_lighting_toggled(self, checked):
        self.update_progress(f"Preserve Lighting: {checked}")

    def on_smooth_transition_toggled(self, checked):
        self.update_progress(f"Smooth Transition: {checked}")

    def on_face_clicked(self, index, face_info=None):
        self.selected_face_index = index
        if index < len(self.target_face_embeddings):
            self.selected_target_face_embedding = self.target_face_embeddings[index]
            self.update_progress(f"Selected face at index: {index}")
        else:
            self.selected_target_face_embedding = None
            self.update_progress(f"Error: Could not retrieve embedding for face index: {index}")

    def update_video_position(self, position):
        if hasattr(self.ui, 'video_preview_widget'):
            self.ui.video_preview_widget.update_position(position)

    def update_video_duration(self, duration):
        if hasattr(self.ui, 'video_preview_widget'):
            self.ui.video_preview_widget.update_duration(duration)

    def on_export_format_changed(self, format_type):
        self.update_progress(f"Export format changed to: {format_type}")

    def on_cpu_usage_changed(self, value): pass
    def on_ram_usage_changed(self, value): pass
    def on_gpu_usage_changed(self, value): pass
    def on_vram_usage_changed(self, value): pass

    def on_preview_source_changed(self):
        """Switch video preview based on selected radio button"""
        if hasattr(self.ui, 'target_preview_radio') and self.ui.target_preview_radio.isChecked():
            self.load_target_video_preview()
        elif hasattr(self.ui, 'swapped_preview_radio') and self.ui.swapped_preview_radio.isChecked():
            self.load_swapped_video_preview()

    def load_target_video_preview(self):
        """Load target video in preview"""
        if self.target_video_path and hasattr(self.ui, 'video_preview_widget'):
            try:
                self.ui.video_preview_widget.load_video(self.target_video_path)
                self.update_progress("Target video loaded in preview")
            except Exception as e:
                self.update_progress(f"Error loading target video preview: {e}")
        else:
            self.update_progress("No target video available or preview widget not found")

    def load_swapped_video_preview(self):
        """Load swapped video in preview"""
        if self.cached_swapped_video and hasattr(self.ui, 'video_preview_widget'):
            try:
                self.ui.video_preview_widget.load_video(self.cached_swapped_video)
                self.update_progress("Swapped video loaded in preview")
            except Exception as e:
                self.update_progress(f"Error loading swapped video preview: {e}")
        else:
            self.update_progress("No swapped video available or preview widget not found. Perform face swap first.")

    def update_preview_frame(self, frame):
        """Update video preview widget with modified frame"""
        if not hasattr(self.ui, 'video_preview_widget') or not self.ui.video_preview_widget:
            return

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.ui.video_preview_widget.display_frame(frame_rgb)
            self.ui.video_preview_widget.pause_video()
            self.update_progress("Preview updated with edited frame")
        except Exception as e:
            self.update_progress(f"Error updating preview frame: {e}")

    def get_current_preview_frame(self):
        """Capture the current frame from the video preview widget"""
        if not hasattr(self.ui, 'video_preview_widget') or not self.ui.video_preview_widget:
            return None

        try:
            current_position = 0

            try:
                if hasattr(self.ui.video_preview_widget, 'get_position'):
                    current_position = self.ui.video_preview_widget.get_position()
                elif hasattr(self.ui.video_preview_widget, 'current_position_ms'):
                    current_position = self.ui.video_preview_widget.current_position_ms
                elif hasattr(self.ui.video_preview_widget, 'media_player'):
                    if (self.ui.video_preview_widget.media_player and
                        hasattr(self.ui.video_preview_widget.media_player, 'position')):
                        current_position = self.ui.video_preview_widget.media_player.position()
            except:
                current_position = 0

            if (self.current_frame_cache['frame'] is not None and
                abs(current_position - self.current_frame_cache['position']) < 50):
                return self.current_frame_cache['frame']

            video_path = None
            if self.cached_swapped_video and os.path.exists(self.cached_swapped_video):
                video_path = self.cached_swapped_video
            elif self.target_video_path and os.path.exists(self.target_video_path):
                video_path = self.target_video_path
            else:
                return None

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            frame_number = int(current_position * fps / 1000)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                frame_number = max(0, min(frame_number, total_frames - 1))
            else:
                frame_number = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                self.current_frame_cache = {
                    'position': current_position,
                    'frame': frame.copy()
                }
                return frame

        except Exception as e:
            self.update_progress(f"Error capturing frame: {e}")
            if logger:
                logger.error(f"Frame capture error: {e}")

        return None

    def get_current_video_position(self):
        """Get current video position from video player if available."""
        try:
            if hasattr(self, 'video_player') and self.video_player:
                if hasattr(self.video_player, 'position'):
                    return self.video_player.position()
                elif hasattr(self.video_player, 'currentPosition'):
                    return self.video_player.currentPosition()

            if hasattr(self.ui, 'video_slider') and self.ui.video_slider:
                return self.ui.video_slider.value()

            if hasattr(self, 'current_frame_cache') and 'position' in self.current_frame_cache:
                return self.current_frame_cache['position']

            return 0
        except Exception as e:
            print(f"[DEBUG] Could not get video position: {e}")
            return 0

    def select_source_image(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Source Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.set_source_image(file_path)

    def select_target_video(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Target Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.set_target_video(file_path)

    def set_source_image(self, file_path: str):
        """Set source image and update UI"""
        if os.path.exists(file_path):
            self.source_image_path = file_path
            pixmap = QPixmap(file_path)
            if hasattr(self.ui, 'dropSource_4'):
                drop_size = self.ui.dropSource_4.size()
                scaled_pixmap = pixmap.scaled(
                    drop_size.width() - 10,
                    drop_size.height() - 10,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                frame_pixmap = QPixmap(drop_size)
                frame_pixmap.fill(Qt.GlobalColor.transparent)
                
                def do_painting(painter):
                    border_color = QColor("#0d7ae7")
                    painter.setPen(border_color)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(2, 2, drop_size.width()-4, drop_size.height()-4)
                    
                    x = (drop_size.width() - scaled_pixmap.width()) // 2
                    y = (drop_size.height() - scaled_pixmap.height()) // 2
                    painter.drawPixmap(x, y, scaled_pixmap)
                
                frame_pixmap = safe_paint(frame_pixmap, do_painting)
                
                self.ui.dropSource_4.setPixmap(frame_pixmap)
                self.ui.dropSource_4.setText("")
            
            source_img_np = cv2.imread(file_path)
            source_face_data = self.processor.detector.get_face(source_img_np)
            if source_face_data:
                self.source_face_embedding = source_face_data.normed_embedding
                self.update_progress("✅ Face embedding extracted from source image.")
            else:
                self.source_face_embedding = None
                self.update_progress("❌ No face found in source image.")
            
            self.update_progress(f"Source image loaded: {Path(file_path).name}")
            if hasattr(self.ui, 'overall_progress'):
                self.ui.overall_progress.setValue(20)
            if logger:
                logger.info(f"Source image set: {file_path}")

    def set_target_video(self, file_path: str):
        """Set target video and update UI"""
        if os.path.exists(file_path):
            self.target_video_path = file_path

            try:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if hasattr(self.ui, 'dropTarget_2'):
                        drop_size = self.ui.dropTarget_2.size()
                        height, width, channel = frame_rgb.shape
                        bytes_per_line = 3 * width
                        q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

                        pixmap = QPixmap.fromImage(q_image)
                        scaled_pixmap = pixmap.scaled(
                            drop_size.width() - 10,
                            drop_size.height() - 10,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation
                        )

                        frame_pixmap = QPixmap(drop_size)
                        frame_pixmap.fill(Qt.GlobalColor.transparent)

                        def do_painting(painter):
                            border_color = QColor("#0d7ae7")
                            painter.setPen(border_color)
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            painter.drawRect(2, 2, drop_size.width()-4, drop_size.height()-4)

                            x = (drop_size.width() - scaled_pixmap.width()) // 2
                            y = (drop_size.height() - scaled_pixmap.height()) // 2
                            painter.drawPixmap(x, y, scaled_pixmap)

                        frame_pixmap = safe_paint(frame_pixmap, do_painting)

                        self.ui.dropTarget_2.setPixmap(frame_pixmap)
                        self.ui.dropTarget_2.setText("")
                else:
                    if hasattr(self.ui, 'dropTarget_2'):
                        self.ui.dropTarget_2.setText(f"Loaded: {Path(file_path).name}\n(Cannot create thumbnail)")
                cap.release()
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to create video thumbnail: {e}")
                if hasattr(self.ui, 'dropTarget_2'):
                    self.ui.dropTarget_2.setText(f"Loaded: {Path(file_path).name}\n(Ready for processing)")

            if hasattr(self.ui, 'video_preview_widget') and self.ui.video_preview_widget:
                try:
                    self.ui.video_preview_widget.load_video(file_path)
                    self.update_progress(f"Target video loaded: {Path(file_path).name}")
                except Exception as e:
                    self.update_progress(f"Target video loaded: {Path(file_path).name} (Preview error: {e})")
            else:
                self.update_progress(f"Target video loaded: {Path(file_path).name} (Preview not available)")

            self.setup_ui_states()

            if hasattr(self.ui, 'overall_progress'):
                self.ui.overall_progress.setValue(30)
            if logger:
                logger.info(f"Target video set: {file_path}")

    def delete_source(self):
        """Delete source image"""
        if self.source_image_path:
            reply = QMessageBox.question(
                self,
                "Confirm Delete",
                "Are you sure you want to remove the source image?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self.source_image_path = None
                    if hasattr(self.ui, 'dropSource_4'):
                        self.ui.dropSource_4.clear()
                        self.ui.dropSource_4.setText("Drop/Click to select Source Face Image")
                    self.update_progress("Source image removed.")
                    if logger:
                        logger.info("Source image deleted by user.")
                except Exception as e:
                    if logger:
                        logger.error(f"Error removing source image: {e}")
                    self.update_progress(f"Error removing source image: {e}")
        else:
            QMessageBox.information(self, "Info", "No source image to delete.")

    def delete_target(self):
        """Delete target video and detected faces"""
        if self.target_video_path:
            reply = QMessageBox.question(
                self,
                "Confirm Delete",
                "Are you sure you want to remove the target video?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self.target_video_path = None
                    self.detected_faces = []
                    self.selected_face_index = -1
                    self.selected_target_face_embedding = None
                    self.target_face_embeddings = []

                    self._restore_original_preview()

                    if hasattr(self.ui, 'dropTarget_2'):
                        self.ui.dropTarget_2.clear()
                        self.ui.dropTarget_2.setText("Drop/Click to select Target Video")

                    if hasattr(self.ui, 'detected_faces_widget'):
                        self.ui.detected_faces_widget.clear_faces()

                    self.converted_audio_path = None
                    if hasattr(self.ui, "btnSoundProcessor"):
                        self.ui.btnSoundProcessor.setText("Launch Voice Converter")
                        self.ui.btnSoundProcessor.setStyleSheet("")

                    self.setup_ui_states()

                    self.update_progress("Target video and detected faces removed.")
                    if logger:
                        logger.info("Target video deleted by user.")

                except Exception as e:
                    if logger:
                        logger.error(f"Error removing target video: {e}")
                    self.update_progress(f"Error removing target video: {e}")
        else:
            QMessageBox.information(self, "Info", "No target video to delete.")

    def detect_faces(self):
        """Detect faces in a target video, store their embeddings, and display them for selection."""
        if not self.target_video_path:
            QMessageBox.warning(self, "Warning", "Please select a target video first!")
            return

        if not os.path.exists(self.target_video_path):
            QMessageBox.critical(self, "Error", "Target video file not found!")
            return

        if FaceDetector is None or self.processor is None or self.processor.detector is None:
            QMessageBox.critical(self, "Error", "FaceDetector module or processor's detector is not available. Cannot detect faces.")
            if logger:
                logger.error("FaceDetector is None. Cannot perform face detection.")
            return

        try:
            self.update_progress("Detecting faces in target video...")

            if not self.processor or not self.processor.detector:
                QMessageBox.critical(self, "Error", "Face processor or its detector is not available.")
                return

            detector_to_use = self.processor.detector

            if not detector_to_use.is_model_loaded():
                detector_to_use.load_model()

            cap = cv2.VideoCapture(self.target_video_path)
            if not cap.isOpened():
                raise FileNotFoundError("Cannot open target video")

            faces_detected_count = 0
            frame_count_processed = 0
            max_frames_to_check = 100
            unique_faces_data = []
            self.detected_faces = []
            self.unique_faces_info = []
            self.target_face_embeddings = []

            self.update_progress("Scanning video frames for faces...")

            if hasattr(self.ui, 'detected_faces_widget'):
                self.ui.detected_faces_widget.clear_faces()

            while cap.isOpened() and frame_count_processed < max_frames_to_check and faces_detected_count < 8:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count_processed += 1

                if frame_count_processed % 10 != 0:
                    continue

                faces = detector_to_use.detect_faces(frame)
                if faces:
                    for face in faces:
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox

                        padding = 10
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(frame.shape[1], x2 + padding)
                        y2 = min(frame.shape[0], y2 + padding)

                        face_img = frame[y1:y2, x1:x2]

                        is_unique = True
                        for existing_face_info in unique_faces_data:
                            ex_bbox = existing_face_info['bbox']

                            ix1 = max(x1, ex_bbox[0])
                            iy1 = max(y1, ex_bbox[1])
                            ix2 = min(x2, ex_bbox[2])
                            iy2 = min(y2, ex_bbox[3])

                            inter_width = max(0, ix2 - ix1 + 1)
                            inter_height = max(0, iy2 - iy1 + 1)
                            inter_area = inter_width * inter_height

                            area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
                            area2 = (ex_bbox[2] - ex_bbox[0] + 1) * (ex_bbox[3] - ex_bbox[1] + 1)
                            union_area = area1 + area2 - inter_area

                            if union_area > 0:
                                iou = inter_area / union_area
                                if iou > 0.5:
                                    is_unique = False
                                    break

                        if is_unique and face_img.size > 0:
                            faces_detected_count += 1

                            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                            face_info = {
                                'confidence': float(getattr(face, 'det_score', 0.9)),
                                'frame': frame_count_processed,
                                'bbox': [x1, y1, x2, y2],
                                'face_data': face,
                                'index': len(self.detected_faces)
                            }

                            unique_faces_data.append(face_info)
                            self.detected_faces.append(face)
                            self.target_face_embeddings.append(face.normed_embedding)
                            
                            self.update_progress(f"Found face #{faces_detected_count} (Frame {frame_count_processed})")

                            if hasattr(self.ui, 'detected_faces_widget'):
                                self.ui.detected_faces_widget.add_face(face_img_rgb, face_info)

                            if faces_detected_count >= 20:
                                break
                    if faces_detected_count >= 20:
                        break

            cap.release()
            self.unique_faces_info = unique_faces_data

            if faces_detected_count > 0:
                self.update_progress(f"Face detection complete! Found {faces_detected_count} unique faces")
                if faces_detected_count == 1:
                    self.selected_face_index = 0
                    self.selected_target_face_embedding = self.target_face_embeddings[0]
                    self.update_progress("Single face detected and auto-selected.")
                else:
                    self.update_progress("Multiple faces detected. Please select the target face in the 'Detected Faces' area.")

                if hasattr(self.ui, 'overall_progress'):
                    self.ui.overall_progress.setValue(40)

            else:
                self.update_progress("No faces detected in video.")
                if hasattr(self.ui, 'overall_progress'):
                    self.ui.overall_progress.setValue(0)
                QMessageBox.warning(self, "No Faces", "No faces were detected in the selected video.")

        except Exception as e:
            error_message = f"Face detection failed: {e}"
            self.update_progress(f"❌ {error_message}")
            if logger:
                logger.error(f"Face detection error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Face detection failed: {e}\nCheck logs for more details.")
            if hasattr(self.ui, 'overall_progress'):
                self.ui.overall_progress.setValue(0)

    def align_faces(self):
        """Align faces or draw mask based on user selection or automatically"""
        if not hasattr(self.ui, 'alignmentEnableSwitch') or not self.ui.alignmentEnableSwitch.isChecked():
            QMessageBox.warning(self, "Warning", "Please enable alignment settings first!")
            return

        if not self.target_video_path:
            QMessageBox.warning(self, "Warning", "Please select a target video first!")
            return

        self.update_progress("Starting face alignment process...")
        
        mask_mode = (hasattr(self.ui, 'Mask01') and self.ui.Mask01.isChecked())
        manual_mode = (hasattr(self.ui, 'manual_01') and self.ui.manual_01.isChecked())
        auto_mode = (hasattr(self.ui, 'Auto01') and self.ui.Auto01.isChecked())

        if mask_mode:
            self._handle_mask_alignment()
        elif manual_mode:
            self._handle_manual_alignment()
        elif auto_mode:
            self._handle_auto_alignment()
        else:
            QMessageBox.information(self, "Info", "Please select an alignment mode (Manual, Mask, or Auto).")
            self.update_progress("Alignment process aborted: No mode selected.")
            
    def _handle_auto_alignment(self):
        """Handle automatic landmark alignment"""
        try:
            self.update_progress("Opening automatic landmark alignment dialog...")
            
            if not isinstance(self.target_video_path, str) or not self.target_video_path:
                QMessageBox.warning(self, "Warning", "Target video path is invalid!")
                return

            if not self.detected_faces or self.selected_face_index == -1:
                QMessageBox.warning(self, "Warning", "Please detect faces and select a target face first.")
                return
            
            selected_face_embedding = self.detected_faces[self.selected_face_index].normed_embedding
            
            from ui.auto import AutoAlignmentDialog

            dialog = AutoAlignmentDialog(self.target_video_path, selected_face_embedding, parent=self)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.tracked_mask_data = dialog.get_auto_alignment_data()
                if self.tracked_mask_data:
                    self.update_progress(f"Auto alignment complete. Found landmarks for {len(self.tracked_mask_data)} frames.")
                    
                    if hasattr(self.processor.controller, 'tracked_data'):
                        self.processor.controller.tracked_data = self.tracked_mask_data
                    else:
                        self.update_progress("Warning: 'tracked_data' attribute not found in controller. Swapping might fail.")
                    
                    self.update_progress("Auto-tracked data stored for processing.")
                    
                    self.ui.swapButton_3.setEnabled(True)
                    self.ui.alignButton_3.setText("Auto Align ✓")
                    self.ui.alignButton_3.setStyleSheet("background-color: #28a745; color: white;")
                    
                else:
                    self.update_progress("Auto alignment failed or returned no data.")
            else:
                self.update_progress("Auto alignment cancelled by user.")

        except ImportError as e:
            QMessageBox.critical(self, "Error", "Automatic alignment module (Auto.py) is not available. Check your file paths.")
            if logger:
                logger.error(f"Auto.py import failed: {e}. Cannot open auto alignment dialog.")
            self.update_progress("Automatic alignment failed: missing module.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open auto alignment dialog: {e}")
            if logger:
                logger.error(f"Auto alignment dialog error: {e}", exc_info=True)
            self.update_progress(f"Automatic alignment failed: {e}")
    
    def _handle_manual_alignment(self):
        """Handle manual landmark editing and tracking"""
        try:
            self.update_progress("Opening manual landmark editor...")
            
            if not isinstance(self.target_video_path, str) or not self.target_video_path:
                QMessageBox.warning(self, "Warning", "Target video path is invalid!")
                return

            from ui.landmarksEditor import LandmarkEditorApp
            
            dialog = LandmarkEditorApp(self.target_video_path, parent=self)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.tracked_mask_data = dialog.get_tracked_landmarks()
                if self.tracked_mask_data:
                    self.update_progress(f"Manual landmark tracking complete. Found landmarks for {len(self.tracked_mask_data)} frames.")
                    self.processor.controller.tracked_data = self.tracked_mask_data
                    self.update_progress("Manual-tracked data stored for processing.")
                else:
                    self.update_progress("Manual landmark tracking failed or returned no data.")
            else:
                self.update_progress("Manual landmark editing cancelled by user.")

        except ImportError:
            QMessageBox.critical(self, "Error", "Manual alignment module (landmarksEditor.py) is not available.")
            if logger:
                logger.error("landmarksEditor.py import failed. Cannot open manual editor.")
            self.update_progress("Manual alignment failed: missing module.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open manual editor: {e}")
            if logger:
                logger.error(f"Manual editor dialog error: {e}", exc_info=True)
            self.update_progress(f"Manual alignment failed: {e}")
            
    def _handle_mask_alignment(self):
        """Handle mask drawing alignment"""
        try:
            self.update_progress("Opening mask editor for manual mask drawing...")

            if not isinstance(self.target_video_path, str) or not self.target_video_path:
                QMessageBox.warning(self, "Warning", "Target video path is invalid!")
                return

            if MaskEditorWindow is None:
                QMessageBox.critical(self, "Error", "Mask editor module is not available.")
                if logger:
                    logger.error("MaskEditorWindow is None. Cannot open mask editor.")
                return

            dialog = MaskEditorWindow(self, self.target_video_path, self.processor)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                mask = dialog.get_mask()
                if mask is not None:
                    if logger:
                        logger.info(f"Mask dimensions: {mask.shape}")
                    self.process_mask(mask)
                else:
                    self.update_progress("No mask was created.")
            else:
                self.update_progress("Mask creation cancelled by user.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open mask dialog: {e}")
            if logger:
                logger.error(f"Mask dialog error: {e}", exc_info=True)
            self.update_progress(f"Mask alignment failed: {e}")

    def process_mask(self, mask: np.ndarray):
        self.current_mask = mask
        if self.processor:
            self.processor.controller.custom_mask = mask
            self.processor.custom_mask = mask
        self.update_progress("Mask processed and ready for use!")
        self.update_progress(f"Mask integrated with alignment model (shape: {mask.shape if mask is not None else 'None'})")
        print(f"DEBUG: Mask received by process_mask -> shape: {mask.shape}, dtype: {mask.dtype}")
        if logger:
            logger.info(f"Mask processed: {mask.shape}")
            
    def set_tracked_mask_data(self, mask, bbox_cache):
        """Receives the mask and tracking data from the MaskEditorWindow."""
        self.current_mask = mask
        self.tracked_mask_data = bbox_cache
        self.selected_face_index = -1
        if hasattr(self.ui, 'alignButton_3'):
            self.ui.alignButton_3.setText("Mask Active ✓")
            self.ui.alignButton_3.setStyleSheet("background-color: #28a745; color: white;")

    def perform_swap(self):
        """Initiate the face swapping process with comprehensive validation and error handling."""
        if self.is_processing:
            QMessageBox.warning(self, "Warning", "Already processing! Please wait.")
            return

        if self.source_image_path is None or not os.path.exists(self.source_image_path):
            QMessageBox.warning(self, "Input Error", "Please select a source image.")
            return

        if self.tracked_mask_data is None:
            if not self.target_video_path or not os.path.exists(self.target_video_path):
                QMessageBox.warning(self, "Input Error", "Please select a target video.")
                return
            if not self.detected_faces:
                QMessageBox.warning(self, "Input Error", "Please run face detection first.")
                return
            if len(self.detected_faces) > 1 and self.selected_face_index == -1:
                QMessageBox.warning(self, "Input Error", "Multiple faces detected. Please select one.")
                return

        if not self.processor or not self.processing_thread:
            QMessageBox.critical(self, "Error", "Face swap processor not initialized!")
            return

        output_path = self._prepare_output_path()
        self._start_face_swap_processing(output_path)

    def _prepare_output_path(self):
        """Prepare the output path for the processed video."""
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"swapped_{timestamp}.mp4"
        return str(output_dir / output_filename)

    def _collect_processing_settings(self):
        """Collect all processing settings from the UI."""
        settings = {
            'blend_alpha': getattr(self.ui, 'blendSlider_3', type('', (), {'value': lambda: 84})).value() / 100.0,
            ########################################################################################################################################
            'edge_softness': getattr(self.ui, 'edgeSlider_3', type('', (), {'value': lambda: 48})).value() / 100.0,
            ########################################################################################################################################
            'quality': getattr(self.ui, 'qualitySlider_3', type('', (), {'value': lambda: 90})).value(),
            'model_name': getattr(self.ui, 'comboBox_4', type('', (), {'currentText': lambda: 'InSwapper_128'})).currentText(),
            'detector_name': getattr(self.ui, 'comboBox_3', type('', (), {'currentText': lambda: 'Yolo'})).currentText(),
            'alignment_enabled': getattr(self.ui, 'alignmentEnableSwitch', type('', (), {'isChecked': lambda: False})).isChecked(),
            'enhancement_enabled': getattr(self.ui, 'enhancementEnableSwitch', type('', (), {'isChecked': lambda: False})).isChecked(),
            '3d_alignment_enabled': getattr(self.ui, 'FaceAgeingEnableSwitch', type('', (), {'isChecked': lambda: False})).isChecked(),
            'enhancement_type': getattr(self.ui, 'typeCombo_3', type('', (), {'currentText': lambda: 'Quality'})).currentText(),
            'enhancement_strength': getattr(self.ui, 'strengthSlider_3', type('', (), {'value': lambda: 65})).value(),
            'skin_smoothness': getattr(self.ui, 'skinSlider_3', type('', (), {'value': lambda: 40})).value(),
            'custom_mask': self.current_mask,
            'tracked_mask_data': self.tracked_mask_data,
            'override_audio_path': self.converted_audio_path,
            'source_face_embedding': getattr(self, 'source_face_embedding', None),
            'selected_target_face_embedding': getattr(self, 'selected_target_face_embedding', None)
        }

        if self.tracked_mask_data is None:
            settings['selected_face_index'] = self.selected_face_index

        return settings

    def _start_face_swap_processing(self, output_path):
        """Start the face swapping processing thread."""
        try:
            self.is_processing = True
            self.update_progress("Starting face swap...")

            if hasattr(self.ui, 'overall_progress'):
                self.ui.overall_progress.setValue(10)

            settings = self._collect_processing_settings()

            self.processing_thread.setup(
                source_path = self.source_image_path,
                target_path = self.target_video_path,
                output_path = output_path,
                options = settings
            )

            self.processing_thread.start()

            if logger:
                logger.info(f"Face swap processing started: {self.source_image_path} -> {self.target_video_path}")

        except Exception as e:
            error_msg = f"Error starting processing thread: {str(e)}"
            self.update_progress(error_msg)
            self.is_processing = False

            if hasattr(self.ui, 'overall_progress'):
                self.ui.overall_progress.setValue(0)

            if logger:
                logger.error(f"Processing start failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Processing Error", error_msg)

    def on_processing_finished(self, success: bool):
        """Handle face swap processing completion"""
        self.is_processing = False
        output_path = self.processing_thread.output_path if self.processing_thread else None

        if success:
            if output_path:
                self.cached_swapped_video = output_path
                self.update_progress("Face swap completed successfully!")

                if hasattr(self.ui, 'swapped_preview_radio'):
                    self.ui.swapped_preview_radio.setEnabled(True)
                    self.ui.swapped_preview_radio.setChecked(True)
                self.load_swapped_video_preview()
                QMessageBox.information(self, "Success", "Face swap completed successfully!")
            else:
                self.update_progress("Face swap completed but output path not found!")
                if hasattr(self.ui, 'overall_progress'):
                    self.ui.overall_progress.setValue(0)
                QMessageBox.warning(self, "Warning", "Face swap completed but output path not found!")
        else:
            self.update_progress("Face swap processing failed!")
            if hasattr(self.ui, 'overall_progress'):
                self.ui.overall_progress.setValue(0)
            QMessageBox.critical(self, "Error", "Face swap processing failed!")

    def launch_voice_converter(self):
        """Launches the RVC Voice Converter GUI."""
        if RVCConverterGUI is None:
            QMessageBox.critical(self, "Error", "Voice Converter UI module (sound_handler.py) not found.")
            return

        if self.voice_converter_window is None:
            self.voice_converter_window = RVCConverterGUI(
                parent=self,
                target_video_path=self.target_video_path,
                stylesheet=self.styleSheet()
            )
            self.voice_converter_window.conversion_successful.connect(self.on_voice_conversion_complete)
        else:
            self.voice_converter_window.set_target_video(self.target_video_path)

        self.voice_converter_window.show()
        self.voice_converter_window.activateWindow()
        self.voice_converter_window.raise_()

    def on_voice_conversion_complete(self, audio_path: str):
        """Handles the signal from the voice converter when a new audio file is ready."""
        if os.path.exists(audio_path):
            self.converted_audio_path = audio_path
            self.update_progress(f"✅ New voice track is ready: {os.path.basename(audio_path)}")

            if hasattr(self.ui, "btnSoundProcessor"):
                self.ui.btnSoundProcessor.setText("Converted Audio Ready ✓")
                self.ui.btnSoundProcessor.setStyleSheet("background-color: #28a745; color: white;")

            reply = QMessageBox.question(
                self,
                "Apply New Audio",
                "The new voice track is ready.\n\n"
                "Do you want to merge it with the current video now? "
                "This will create a new video preview.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.finalize_video_with_audio()
            else:
                self.update_progress("🎤 New audio is ready and will be used the next time video is processed or finalized.")

        else:
            self.update_progress(f"❌ Received converted voice path, but file does not exist: {audio_path}")

    def landmark_launcher(self):
        """Open landmark editor with swapped video"""
        if not self.target_video_path:
            QMessageBox.warning(self, "Warning", "Please Add Target first!")
            return

        try:
            from .landmarksEditor import LandmarkEditorApp
            dialog = LandmarkEditorApp(self)
            dialog.set_video(self.target_video_path)
            result = dialog.exec()

            if result == QDialog.DialogCode.Accepted:
                self.update_progress("Landmark edits applied")
        except ImportError:
            QMessageBox.critical(self, "Error", "Landmark Editor module not found.")

    def launch_adjustment_Dialog(self):
        """Open adjustment dialog, get settings, and then process the whole video."""
        try:
            from .adjustments import FaceadjustmentsDialog

            if not self.target_video_path and not self.cached_swapped_video:
                QMessageBox.warning(self, "Warning", "Please Add Target first!")
                return

            current_frame = self.get_current_preview_frame()
            if current_frame is None:
                QMessageBox.warning(self, "Frame Error", "Could not capture a frame from the video for preview.")
                return

            dialog = FaceadjustmentsDialog(self, initial_image=current_frame)
            
            result = dialog.exec()

            if result == QDialog.DialogCode.Accepted:
                self.update_progress("Applying adjustments to the entire video...")
                adjustments = dialog.opencv_controls.get_current_adjustments()

                input_video_path = self.cached_swapped_video or self.target_video_path

                output_dir = Path("output")
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_video_path = str(output_dir / f"adjusted_{timestamp}.mp4")

                self.apply_adjustments_to_video(input_video_path, output_video_path, adjustments)
        except ImportError:
            QMessageBox.critical(self, "Error", "Adjustments module not found.")

    def open_enhancement_dialog(self):
        """Opens the face enhancement dialog with current frame preview"""
        if not self.cached_swapped_video and not self.target_video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first!")
            return

        if not self.processor:
            QMessageBox.critical(self, "Error", "Face processor is not available.")
            return

        try:
            from .enhancement import FaceEnhancementDialog
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

    def launch_aging_dialog(self):
        """
        Launches the advanced face aging dialog.
        This function prepares the data and handles the result after the dialog is closed.
        """
        if not self.processor:
            QMessageBox.critical(self, "Error", "Face processor is not available.")
            return

        current_frame = self.get_current_preview_frame()
        if current_frame is None:
            QMessageBox.warning(self, "Frame Error", "Could not get a frame for aging preview.")
            return

        current_frame = current_frame.copy()

        self.aging_window = FaceAgingWindow(
            processor=self.processor,
            initial_frame=current_frame,
            stylegan_session=self.aging_session,
            parent=self
        )

        self.aging_window.settings_accepted.connect(self.on_aging_window_accepted)
        self.aging_window.log_message.connect(self.update_progress)
        
        self.aging_window.show()

    def _merge_video_with_audio(self, video_path: str, audio_source_path: str, output_path: str) -> bool:
        """
        Helper function to merge a video with an audio track.
        This assumes a simple merge is needed.
        """
        try:
            if video_utils:
                self.update_progress("Extracting audio from original video...")
                temp_audio_path = Path(output_path).parent / "temp_audio.aac"
                if video_utils.extract_audio(audio_source_path, temp_audio_path, self.update_progress):
                    self.update_progress("Reconstructing video with audio...")
                    video_utils.reconstruct_video(video_path, output_path, None, str(temp_audio_path), self.update_progress)
                    os.remove(temp_audio_path)
                    return True
            return False
        except Exception as e:
            self.update_progress(f"❌ Error merging video with audio: {e}")
            return False

    def on_aging_window_accepted(self, settings):
        """
        This slot is called when the user accepts the aging settings in the FaceAgingWindow.
        This version processes frames, saves them, then reconstructs the video with audio.
        """
        target_age = settings.get('target_age')
        if target_age is None:
            QMessageBox.warning(self, "No Result", "Target age not found in settings.")
            return

        input_video_path = self.cached_swapped_video or self.target_video_path
        if not input_video_path:
            QMessageBox.warning(self, "Input Error", "No video available to apply aging to.")
            return

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_video_path = str(output_dir / f"aged_video_with_audio_{timestamp}.mp4")

        self.update_progress("Starting video aging process...")
        
        try:
            result = self._apply_aging_to_frames(input_video_path, output_video_path, settings)

            if result['success']:
                QMessageBox.information(
                    self,
                    "Aging Complete",
                    f"Video aging completed!\n\n"
                    f"Output: {result['output_path']}"
                )
                self.cached_swapped_video = result['output_path']
                self.load_swapped_video_preview()
            else:
                QMessageBox.critical(self, "Processing Error", f"Video aging failed:\n{result['error']}")

        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"An unexpected error occurred during aging: {e}")
            self.update_progress(f"❌ Aging process failed: {e}")

    def _apply_aging_to_frames(self, input_path: str, output_path: str, settings: dict) -> dict:
        """
        Processes video frame-by-frame, applies aging, and then reconstructs
        the final video with audio. This is the new, consistent approach.
        """
        try:
            processed_frames_dir, audio_path = self._prepare_video_processing(input_path, output_path)

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.update_progress(f"Applying aging to {frame_count} frames...")
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                from core.apply_age import apply_ageing
                
                aged_frame = apply_ageing(frame, settings['target_age'])
                
                if aged_frame is not None:
                    cv2.imwrite(str(processed_frames_dir / f"frame_{i:06d}.jpg"), aged_frame)

                if (i + 1) % 30 == 0:
                    self.update_progress(f"Processed frame {i + 1}/{frame_count}")

            cap.release()
            
            if video_utils:
                self.update_progress("Reconstructing video with audio...")
                video_utils.reconstruct_video(
                    frames_dir=processed_frames_dir,
                    output_path=output_path,
                    fps=fps,
                    audio_path=audio_path,
                    progress_callback=self.update_progress
                )
            
            self.update_progress("🧹 Cleaning up temporary files...")
            shutil.rmtree(processed_frames_dir.parent)
            
            return {'success': True, 'output_path': output_path}

        except Exception as e:
            error_msg = f"Video aging failed during frame processing: {e}"
            self.update_progress(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
        
    def _prepare_video_processing(self, input_path, output_path):
        """Helper to set up directories and audio for video processing."""
        temp_dir = Path(output_path).parent / f"temp_{Path(output_path).stem}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        processed_frames_dir = temp_dir / "processed"
        processed_frames_dir.mkdir(parents=True, exist_ok=True)

        audio_path_to_use = None
        temp_audio_path = temp_dir / "audio.aac"

        if self.converted_audio_path and Path(self.converted_audio_path).exists():
            self.update_progress("🎤 Using custom converted voice track for processing.")
            shutil.copy(self.converted_audio_path, temp_audio_path)
            audio_path_to_use = temp_audio_path
        elif video_utils and video_utils.extract_audio(input_path, temp_audio_path, self.update_progress):
            self.update_progress("🎵 Using original audio track for processing.")
            audio_path_to_use = temp_audio_path
        else:
            self.update_progress("⚠️ No audio track found or usable.")

        return processed_frames_dir, audio_path_to_use

    def apply_adjustments_to_video(self, input_path, output_path, adjustments):
        """Applies adjustments to each frame and saves result with audio."""
        try:
            from .adjustments import apply_opencv_to_aligned_face, FaceAligner

            processed_frames_dir, audio_path = self._prepare_video_processing(input_path, output_path)

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            aligner = FaceAligner()
            detector = self.processor.detector

            self.update_progress(f"Applying adjustments to {frame_count} frames...")

            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret: break

                faces = detector.detect_faces(frame)
                processed_frame = frame
                if faces:
                    for face in faces:
                        processed_frame = apply_opencv_to_aligned_face(frame, face, adjustments, aligner)

                cv2.imwrite(str(processed_frames_dir / f"frame_{i:06d}.jpg"), processed_frame)
                if (i + 1) % 30 == 0:
                    self.update_progress(f"Processed frame {i + 1}/{frame_count}")

            cap.release()

            if video_utils:
                video_utils.reconstruct_video(processed_frames_dir, output_path, fps, audio_path, self.update_progress)

            shutil.rmtree(processed_frames_dir.parent)

            self.update_progress(f"✅ Video adjustments complete! Saved to {output_path}")
            self.cached_swapped_video = output_path
            self.load_swapped_video_preview()
        except ImportError:
            QMessageBox.critical(self, "Error", "Adjustments module not found.")

    def apply_enhancement_to_video(self, input_path, output_path, settings):
        """
        Applies GFPGAN enhancement to each frame of a video based on user settings,
        and then reconstructs the final video with audio.
        """
        if not GFPGAN_AVAILABLE:
            self.update_progress("❌ Error: GFPGAN library not found.")
            return

        processed_frames_dir, audio_path = self._prepare_video_processing(input_path, output_path)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.update_progress(f"❌ Error: Could not open video file {input_path}")
            if processed_frames_dir.parent.exists():
                shutil.rmtree(processed_frames_dir.parent)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
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
            if processed_frames_dir.parent.exists():
                shutil.rmtree(processed_frames_dir.parent)
            return

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

            cv2.imwrite(str(processed_frames_dir / f"frame_{i:06d}.jpg"), processed_frame)

            if hasattr(self.ui, 'overall_progress') and i % 10 == 0:
                progress_percent = int(((i + 1) / frame_count) * 100)
                self.ui.overall_progress.setValue(progress_percent)

        cap.release()
        
        if video_utils:
            self.update_progress("Reconstructing video with audio...")
            video_utils.reconstruct_video(
                frames_dir=processed_frames_dir,
                output_path=output_path,
                fps=fps,
                audio_path=audio_path,
                progress_callback=self.update_progress
            )
        
        self.update_progress("🧹 Cleaning up temporary enhancement files...")
        shutil.rmtree(processed_frames_dir.parent)

        self.update_progress(f"✅ Video enhancement complete! Saved to {output_path}")
        QMessageBox.information(self, "Success", f"Enhanced video saved successfully to:\n{output_path}")
        
        self.cached_swapped_video = output_path
        if hasattr(self, 'load_swapped_video_preview'):
            self.load_swapped_video_preview()

    def finalize_video_with_audio(self):
        """
        Applies the selected audio (either converted or original) to the
        final swapped video. This function creates the final output by
        merging the video with the appropriate sound track.
        """
        self.update_progress("🔊 Finalizing audio for the video...")

        if not self.cached_swapped_video or not Path(self.cached_swapped_video).exists():
            QMessageBox.warning(self, "Processing Error", "No processed video found. Please perform a face swap first.")
            self.update_progress("❌ Audio finalization failed: No swapped video available.")
            return

        if not self.target_video_path and not self.converted_audio_path:
            QMessageBox.warning(self, "Audio Error", "No audio source available (neither original video nor converted audio).")
            self.update_progress("❌ Audio finalization failed: No audio source.")
            return

        input_video_path = self.cached_swapped_video
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_video_path = str(output_dir / f"final_video_{timestamp}.mp4")

        temp_dir = Path(final_video_path).parent / f"temp_audio_merge_{timestamp}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_to_use = None

        if self.converted_audio_path and Path(self.converted_audio_path).exists():
            self.update_progress(f"🎤 Using custom converted audio: {Path(self.converted_audio_path).name}")
            audio_to_use = self.converted_audio_path
        elif self.target_video_path and Path(self.target_video_path).exists():
            self.update_progress("🎵 Extracting audio from original target video...")
            temp_audio_path = temp_dir / "original_audio.aac"
            if video_utils and video_utils.extract_audio(self.target_video_path, temp_audio_path, self.update_progress):
                audio_to_use = str(temp_audio_path)
            else:
                self.update_progress("⚠️ Could not extract audio from the original video.")

        if not audio_to_use:
            self.update_progress("🔇 No audio track available. The final video will be silent.")

        try:
            self.update_progress("🔄 Merging video with audio... This may take a moment.")

            frames_dir = temp_dir / "frames"
            frames_dir.mkdir()

            if video_utils:
                fps, frame_count = video_utils.extract_frames(
                    input_video_path,
                    frames_dir,
                    self.update_progress
                )

                video_utils.reconstruct_video(
                    frames_dir=frames_dir,
                    output_path=final_video_path,
                    fps=fps,
                    audio_path=audio_to_use,
                    progress_callback=self.update_progress
                )

            self.cached_swapped_video = final_video_path
            self.update_progress(f"✅ Final video with audio is ready: {final_video_path}")
            QMessageBox.information(self, "Success", f"Final video created successfully at:\n{final_video_path}")

            self.load_swapped_video_preview()

        except Exception as e:
            error_msg = f"Failed to merge audio and video: {e}"
            self.update_progress(f"❌ {error_msg}")
            QMessageBox.critical(self, "Merge Error", error_msg)
            if logger:
                logger.error(error_msg, exc_info=True)
        finally:
            self.update_progress("🧹 Cleaning up temporary merge files...")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def save_project(self):
        """Save current project"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "project.json",
            "JSON Files (*.json)"
        )
        if file_path:
            project_data = {
                'source_image': self.source_image_path,
                'target_video': self.target_video_path,
                'selected_face_index': self.selected_face_index,
                'settings': {
                    'blend': getattr(self.ui, 'blendSlider_3', QSlider()).value() if hasattr(self.ui, 'blendSlider_3') else 84,
                    'edge': getattr(self.ui, 'edgeSlider_3', QSlider()).value() if hasattr(self.ui, 'edgeSlider_3') else 48,
                    'quality': getattr(self.ui, 'qualitySlider_3', QSlider()).value() if hasattr(self.ui, 'qualitySlider_3') else 90,
                    'model': getattr(self.ui, 'comboBox_4', QComboBox()).currentText() if hasattr(self.ui, 'comboBox_4') else 'InSwapper_128',
                    'detector': getattr(self.ui, 'comboBox_3', QComboBox()).currentText() if hasattr(self.ui, 'comboBox_3') else 'Yolo',
                    'alignment_enabled': getattr(self.ui, 'alignmentEnableSwitch', QCheckBox()).isChecked() if hasattr(self.ui, 'alignmentEnableSwitch') else False,
                    'enhancement_enabled': getattr(self.ui, 'enhancementEnableSwitch', QCheckBox()).isChecked() if hasattr(self.ui, 'enhancementEnableSwitch') else False,
                    '3d_alignment_enabled': getattr(self.ui, 'FaceAgeingEnableSwitch', QCheckBox()).isChecked() if hasattr(self.ui, 'FaceAgeingEnableSwitch') else False
                },
                'theme': self.current_theme
            }
            try:
                with open(file_path, 'w') as f:
                    json.dump(project_data, f, indent=4)
                self.update_progress(f"Project saved: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Success", f"Project saved successfully!\n\nLocation: {file_path}")
            except Exception as e:
                self.update_progress(f"Error saving project: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save project: {e}")
                if logger:
                    logger.error(f"Project save error: {e}", exc_info=True)

    def export_video(self):
        """Export the cached swapped video"""
        if not self.cached_swapped_video or not os.path.exists(self.cached_swapped_video):
            QMessageBox.warning(self, "Warning", "No swapped video available for export!")
            return

        format_type = getattr(self.ui, 'comboBox_2', QComboBox()).currentText()
        if not format_type:
            format_type = 'MP4'

        output_path_line_edit = getattr(self.ui, 'outputPath_2', None)
        initial_file_path = output_path_line_edit.text() if output_path_line_edit and output_path_line_edit.text() != 'Click Export to choose...' else ""

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Swapped Video",
            initial_file_path or f"swapped_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type.lower()}",
            f"{format_type} Files (*.{format_type.lower()})"
        )

        if file_path:
            try:
                shutil.copy2(self.cached_swapped_video, file_path)

                self.update_progress(f"Video exported successfully to: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Success", f"Video exported successfully!\n\nLocation: {file_path}")
                if output_path_line_edit:
                    output_path_line_edit.setText(file_path)

            except Exception as e:
                self.update_progress(f"Video export failed: {e}")
                QMessageBox.critical(self, "Error", f"Failed to export video: {e}")
                if logger:
                    logger.error(f"Video export error: {e}", exc_info=True)
        else:
            self.update_progress("Video export cancelled by user")

    def update_progress(self, message: str):
        """Update progress display and log the message"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] - {message}"

        if hasattr(self.ui, 'log_output'):
            self.ui.log_output.append(formatted_message)
            scrollbar = self.ui.log_output.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        if logger:
            logger.info(message)

        if hasattr(self.ui, 'overall_progress'):
            if "initializing" in message.lower():
                self.ui.overall_progress.setValue(10)
            elif "source image" in message.lower():
                self.ui.overall_progress.setValue(20)
            elif "extracting frames" in message.lower():
                self.ui.overall_progress.setValue(30)
            elif "detecting faces" in message.lower():
                self.ui.overall_progress.setValue(35)
            elif "face detection complete" in message.lower():
                self.ui.overall_progress.setValue(40)
            elif "selected face" in message.lower():
                self.ui.overall_progress.setValue(45)
            elif "alignment" in message.lower() and "complete" in message.lower():
                self.ui.overall_progress.setValue(50)
            elif "starting face swap" in message.lower():
                self.ui.overall_progress.setValue(60)
            elif "swapped frame" in message.lower():
                try:
                    if "(" in message and "%)" in message:
                        percent_str = message.split("(")[1].split("%")[0]
                        percent = int(percent_str)
                        self.ui.overall_progress.setValue(60 + int(percent * 0.2))
                except ValueError:
                    pass
            elif "reconstructing video" in message.lower():
                self.ui.overall_progress.setValue(80)
            elif "face swap completed" in message.lower():
                self.ui.overall_progress.setValue(90)
            elif "export" in message.lower():
                self.ui.overall_progress.setValue(95)
            elif "successfully" in message.lower() and "complete" in message.lower():
                self.ui.overall_progress.setValue(100)

    def update_system_resources(self):
        """Update system resource usage display"""
        try:
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=None)
                if hasattr(self.ui, 'cpu_progress'):
                    self.ui.cpu_progress.setValue(int(cpu_percent))
                    self.ui.cpu_progress.setFormat(f"CPU: {cpu_percent:.1f}%")

                memory = psutil.virtual_memory()
                ram_percent = memory.percent
                ram_used_gb = memory.used / (1024**3)
                ram_total_gb = memory.total / (1024**3)
                if hasattr(self.ui, 'ram_progress'):
                    self.ui.ram_progress.setValue(int(ram_percent))
                    self.ui.ram_progress.setFormat(f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f}GB ({ram_percent:.1f}%)")
            else:
                if hasattr(self.ui, 'cpu_progress'):
                    self.ui.cpu_progress.setValue(0)
                    self.ui.cpu_progress.setFormat("CPU: N/A (install psutil)")
                if hasattr(self.ui, 'ram_progress'):
                    self.ui.ram_progress.setValue(0)
                    self.ui.ram_progress.setFormat("RAM: N/A (install psutil)")

            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        gpu_percent = gpu.load * 100
                        gpu_temp = gpu.temperature if hasattr(gpu, 'temperature') and gpu.temperature else 0

                        if hasattr(self.ui, 'gpu_progress'):
                            self.ui.gpu_progress.setValue(int(gpu_percent))
                            self.ui.gpu_progress.setFormat(f"GPU: {gpu_percent:.1f}% ({gpu_temp}°C)")

                        vram_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                        vram_used_gb = gpu.memoryUsed / 1024
                        vram_total_gb = gpu.memoryTotal / 1024

                        if hasattr(self.ui, 'vram_progress'):
                            self.ui.vram_progress.setValue(int(vram_percent))
                            self.ui.vram_progress.setFormat(f"VRAM: {vram_used_gb:.1f}/{vram_total_gb:.1f}GB ({vram_percent:.1f}%)")
                    else:
                        if hasattr(self.ui, 'gpu_progress'):
                            self.ui.gpu_progress.setValue(0)
                            self.ui.gpu_progress.setFormat("GPU: Not Detected")
                        if hasattr(self.ui, 'vram_progress'):
                            self.ui.vram_progress.setValue(0)
                            self.ui.vram_progress.setFormat("VRAM: N/A")
                except Exception as e:
                    if logger:
                        logger.error(f"GPU monitoring failed: {e}")
                    if hasattr(self.ui, 'gpu_progress'):
                        self.ui.gpu_progress.setValue(0)
                        self.ui.gpu_progress.setFormat(f"GPU: Error")
                    if hasattr(self.ui, 'vram_progress'):
                        self.ui.vram_progress.setValue(0)
                        self.ui.vram_progress.setFormat("VRAM: Error")
            else:
                if hasattr(self.ui, 'gpu_progress'):
                    self.ui.gpu_progress.setValue(0)
                    self.ui.gpu_progress.setFormat("GPU: N/A (install GPUtil)")
                if hasattr(self.ui, 'vram_progress'):
                    self.ui.vram_progress.setValue(0)
                    self.ui.vram_progress.setFormat("VRAM: N/A (install GPUtil)")

        except KeyboardInterrupt:
            print("KeyboardInterrupt detected. Shutting down gracefully...")
            QApplication.instance().quit()

        except Exception as e:
            if logger:
                logger.warning(f"System monitoring error: {e}", exc_info=True)
            for attr in ['cpu_progress', 'ram_progress', 'gpu_progress', 'vram_progress']:
                if hasattr(self.ui, attr):
                    getattr(self.ui, attr).setFormat(f"Monitoring Error")

    def show_settings(self):
        """Show settings dialog"""
        if SettingsDialog:
            dialog = SettingsDialog(self)
            dialog.theme_changed.connect(self.apply_theme_with_feedback)

            if self.current_theme == "dark":
                dialog.dark_radio.setChecked(True)
            elif self.current_theme == "light":
                dialog.light_radio.setChecked(True)
            elif self.current_theme == "New Mode":
                dialog.new_radio.setChecked(True)

            if dialog.exec():
                self.update_progress("⚙️ Settings applied successfully!")
        else:
            QMessageBox.warning(self, "Warning", "Settings dialog module not available.")

    def apply_theme_with_feedback(self, theme: str):
        """Apply theme with user feedback"""
        old_theme = self.current_theme
        apply_theme(self, theme)

        self.current_theme = theme
        if old_theme != theme:
            self.update_progress(f"🎨 Theme changed from {old_theme} to {theme}")

            if theme == "light":
                self.update_progress("☀️ Light mode activated - Perfect for daytime work!")
            elif theme == "dark":
                self.update_progress("🌙 Dark mode activated - Easy on the eyes!")
            elif theme == "New Mode":
                self.update_progress("✨ New Mode activated - Stunning gradients and effects!")

        if logger:
            logger.info(f"Theme changed from {old_theme} to {theme}")

    def auto_alignmentt_runner(self):
        """Auto alignment runner - placeholder"""
        print("auto Enabled")
        return

    def eventFilter(self, obj, event):
        """Handle drag and drop events for dropSource_4 and dropTarget_2"""
        if obj in [getattr(self.ui, 'dropSource_4', None), getattr(self.ui, 'dropTarget_2', None)]:
            if event.type() == QEvent.Type.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Type.Drop:
                urls = event.mimeData().urls()
                if urls:
                    file_path = urls[0].toLocalFile()
                    if obj == getattr(self.ui, 'dropSource_4', None):
                        self.set_source_image(file_path)
                    else:
                        self.set_target_video(file_path)
                    return True
        return super().eventFilter(obj, event)

    def dialog_3d_runer(self):
        """Launches the 3D Face Swap UI dialog and passes current state and theme."""
        try:
            import sys
            import os
            import cv2
            from PySide6.QtWidgets import QMessageBox
            from PIL import Image

            print("DEBUG: Starting 3D dialog launch...")
            
            if not hasattr(self, 'dialog_3d_instance'):
                print("DEBUG: Initializing dialog_3d_instance to None")
                self.dialog_3d_instance = None
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alignment3d_path = os.path.join(current_dir, '..', 'alignment3D')
            alignment3d_path = os.path.normpath(alignment3d_path)
            
            print(f"DEBUG: Looking for alignment3D at: {alignment3d_path}")
            print(f"DEBUG: Path exists: {os.path.exists(alignment3d_path)}")
            
            if not os.path.exists(alignment3d_path):
                alignment3d_path = os.path.join(os.path.dirname(current_dir), 'alignment3D')
                alignment3d_path = os.path.normpath(alignment3d_path)
                print(f"DEBUG: Trying alternative path: {alignment3d_path}")
                print(f"DEBUG: Alternative path exists: {os.path.exists(alignment3d_path)}")
            
            if not os.path.exists(alignment3d_path):
                QMessageBox.critical(self, "Path Error", 
                                    f"Could not find alignment3D folder.\n"
                                    f"Looked in: {alignment3d_path}")
                return
            
            if alignment3d_path not in sys.path:
                sys.path.insert(0, alignment3d_path)
                print(f"DEBUG: Added to sys.path: {alignment3d_path}")
            
            print("DEBUG: Attempting import...")
            from alignment3D.dialog3d import FaceSwapUI
            print("DEBUG: Import successful")
            
            dialog_exists = (self.dialog_3d_instance is not None and 
                            hasattr(self.dialog_3d_instance, 'isVisible'))
            
            print(f"DEBUG: Dialog exists and has isVisible: {dialog_exists}")
            
            if dialog_exists:
                try:
                    is_visible = self.dialog_3d_instance.isVisible()
                    print(f"DEBUG: Dialog is visible: {is_visible}")
                    if is_visible:
                        self.dialog_3d_instance.activateWindow()
                        self.dialog_3d_instance.raise_()
                        print("DEBUG: Brought existing dialog to front")
                        return
                except Exception as e:
                    print(f"DEBUG: Error checking visibility: {e}")
                    self.dialog_3d_instance = None
            
            print("DEBUG: Creating new dialog instance and passing state...")
            
            target_video_path = getattr(self, 'target_video_path', None)
            cached_swapped_video = getattr(self, 'cached_swapped_video', None)
            current_theme = getattr(self, 'current_theme', 'dark')

            self.dialog_3d_instance = FaceSwapUI(
                target_video_path=target_video_path,
                cached_swapped_video = cached_swapped_video,
                theme_name=current_theme
            )


            source_image_path = getattr(self, 'source_image_path', None)
            
            if not source_image_path:
                current_frame = self.get_current_preview_frame()
                if current_frame is not None:
                    pil_image = Image.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
                    self.dialog_3d_instance.set_source_image_from_frame(pil_image)

            self.dialog_3d_instance.show()
            print("DEBUG: Dialog created and shown successfully")
            
        except Exception as e:
            print(f"DEBUG: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to launch 3D dialog: {e}")
            self.dialog_3d_instance = None

    def closeEvent(self, event):
        """Handle application close"""
        if self.is_processing:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Processing is still running. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        if hasattr(self, 'system_monitor_timer') and self.system_monitor_timer.isActive():
            self.system_monitor_timer.stop()
            if logger:
                logger.info("System monitor timer stopped.")

        if hasattr(self.ui.video_preview_widget, 'media_player') and self.ui.video_preview_widget.media_player:
            self.ui.video_preview_widget.media_player.stop()
        if self.video_player:
            self.video_player.stop()
            self.video_player.setSource(QUrl())
            if hasattr(self, 'audio_output'):
                self.audio_output.deleteLater()
            self.video_player.deleteLater()
            self.video_player = None
        if hasattr(self, 'video_widget') and self.video_widget:
            self.video_widget.deleteLater()
            del self.video_widget

        if hasattr(self, 'voice_converter_window') and self.voice_converter_window:
            self.voice_converter_window.close()
            self.voice_converter_window = None

        if hasattr(self, 'dialog_3d_instance') and self.dialog_3d_instance:
            self.dialog_3d_instance.close()
            self.dialog_3d_instance = None

        if self.cache_manager:
            try:
                self.cache_manager.clear()
                if logger:
                    logger.info("Application cache cleared.")
            except Exception as e:
                if logger:
                    logger.error(f"Error clearing cache: {e}")
        else:
            if logger:
                logger.warning("CacheManager not available, skipped cache clearing.")

        if logger:
            logger.info("Application closed.")
        event.accept()