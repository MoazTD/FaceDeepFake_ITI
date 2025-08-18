import sys
import os
import json
import logging
import cv2
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QRadioButton, QGroupBox,
    QPushButton, QProgressBar, QTextEdit, QWidget, QSlider,
    QSpinBox, QCheckBox, QComboBox, QStackedWidget, QTabWidget,
    QSizePolicy, QFormLayout, QFrame, QGridLayout, QSpacerItem,
    QSplitter, QSizeGrip, QMenuBar, QMenu, QScrollArea, QStyleOptionButton
)
from PySide6.QtCore import (
    QSize, QEvent, Qt, QThread, Signal, QTimer, 
    QPropertyAnimation, QEasingCurve, QUrl, QMutex, QMutexLocker,
    QCoreApplication, QDate, QDateTime, QLocale, QMetaObject, 
    QObject, QPoint, QRect, QTime
)
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QColor, QIcon, QAction, QBrush, 
    QConicalGradient, QCursor, QFont, QFontDatabase, QGradient,
    QKeySequence, QLinearGradient, QPalette, QRadialGradient, QTransform
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
# from PySide6.

from core.process import FaceSwapProcessor

class StreamHandlerUTF8(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)
        self.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_swap_app.log', encoding='utf-8'),
        StreamHandlerUTF8()
    ]
)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of processed frames and resources"""
    def __init__(self, cache_dir: Path = Path("cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_mutex = QMutex()
        self._cache_data = {}
        logger.info(f"Cache manager initialized at {cache_dir}")
    
    def get(self, key: str) -> Optional[Any]:
        with QMutexLocker(self.cache_mutex):
            return self._cache_data.get(key)
    
    def set(self, key: str, value: Any):
        with QMutexLocker(self.cache_mutex):
            self._cache_data[key] = value
    
    def clear(self):
        with QMutexLocker(self.cache_mutex):
            self._cache_data.clear()
        for file in self.cache_dir.glob("*"):
            try:
                file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {file}: {e}")


class ProcessingThread(QThread):
    """Thread for face swapping processing"""
    progress = Signal(str)
    finished = Signal(bool)
    frame_processed = Signal(np.ndarray)
    
    def __init__(self, processor: FaceSwapProcessor):
        super().__init__()
        self.processor = processor
        self.source_path = None
        self.target_path = None
        self.output_path = None
        self.options = {}
        self._is_running = True
    
    def setup(self, source_path: str, target_path: str, output_path: str, options: Dict):
        self.source_path = source_path
        self.target_path = target_path
        self.output_path = output_path
        self.options = options
    
    def run(self):
        try:
            logger.info("Starting face swap processing")
            success = self.processor.process_video(
                self.source_path,
                self.target_path,
                self.output_path,
                self.options
            )
            self.finished.emit(success)
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit(False)
    
    def stop(self):
        self._is_running = False


class SettingsDialog(QDialog):
    """Settings dialog for theme and preferences"""
    theme_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(400, 300)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        theme_group = QGroupBox("Theme Settings")
        theme_layout = QVBoxLayout()
        
        self.dark_radio = QRadioButton("Dark Mode")
        self.new_radio = QRadioButton("New Mode")
        self.light_radio = QRadioButton("Light Mode")
        self.dark_radio.setChecked(True)
        
        theme_layout.addWidget(self.dark_radio)
        theme_layout.addWidget(self.new_radio)
        theme_layout.addWidget(self.light_radio)
        theme_group.setLayout(theme_layout)
        
        proc_group = QGroupBox("Processing Settings")
        proc_layout = QVBoxLayout()
        
        self.gpu_check = QCheckBox("Use GPU Acceleration")
        self.gpu_check.setChecked(True)
        
        self.cache_check = QCheckBox("Enable Caching")
        self.cache_check.setChecked(True)
        
        proc_layout.addWidget(self.gpu_check)
        proc_layout.addWidget(self.cache_check)
        proc_group.setLayout(proc_layout)
        
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addWidget(theme_group)
        layout.addWidget(proc_group)
        layout.addStretch()
        layout.addLayout(button_layout)
        
        self.apply_btn.clicked.connect(self.apply_settings)
        self.cancel_btn.clicked.connect(self.reject)
        
    def apply_settings(self):
        if self.dark_radio.isChecked():
            theme = "dark"
        elif self.light_radio.isChecked():
            theme = "light"
        elif self.new_radio.isChecked():
            theme = "New Mode"
        self.theme_changed.emit(theme)
        self.accept()


def safe_paint(pixmap, paint_function):
    """Safely use a QPainter with proper cleanup even if exceptions occur
    
    Args:
        pixmap: The QPixmap to paint on
        paint_function: A function that takes a QPainter as argument and performs painting
    
    Returns:
        The pixmap after painting
    """
    painter = QPainter(pixmap)
    try:
        paint_function(painter)
    except Exception as e:
        print(f"Error during painting: {e}")
        raise
    finally:
        if painter.isActive():
            painter.end()
    return pixmap

