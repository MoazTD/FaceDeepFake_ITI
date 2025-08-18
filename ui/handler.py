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
    QPixmap, QImage, QColor, QIcon, QAction, QBrush, 
    QConicalGradient, QCursor, QFont, QFontDatabase, QGradient,
    QKeySequence, QLinearGradient, QPalette, QRadialGradient, QTransform
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget



from .uitls import logger


try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False




class VideoPreviewWidget(QWidget):
    """Enhanced video preview with controls"""
    position_changed = Signal(int)
    duration_changed = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_media_player()
        self.is_playing = False
        self.current_frame_cache = {'position': 0, 'frame': None}

    def get_position(self):
        """Get current playback position in milliseconds"""
        try:
            if hasattr(self, 'media_player') and self.media_player:
                if hasattr(self.media_player, 'position'):
                    return self.media_player.position()
            
            return getattr(self, 'current_position_ms', 0)
        except:
            return 0
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Video widget 
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(600, 400)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.video_widget, 1)  #
        
        # Controls container 
        controls_container = QWidget()
        controls_container.setMaximumHeight(120)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setSpacing(5)
        controls_layout.setContentsMargins(10, 5, 10, 5)
        
        # Timeline slider
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setObjectName("timeline_slider")
        controls_layout.addWidget(self.timeline_slider)
        
        # Time display and buttons row
        time_controls_layout = QHBoxLayout()
        time_controls_layout.setSpacing(10)
        
        # Time labels
        self.current_time_label = QLabel("00:00")
        self.current_time_label.setMinimumWidth(40)
        time_controls_layout.addWidget(self.current_time_label)
        
        # Control buttons (compact)
        self.prev_btn = QPushButton("⏪")
        self.prev_btn.setMaximumSize(35, 25)
        self.play_btn = QPushButton("")
        self.play_btn.setMaximumSize(50, 50)
        self.play_btn.setIcon(QIcon(r"Assets\Icons\pause.png"))
        self.next_btn = QPushButton("⏩")
        self.next_btn.setMaximumSize(35, 25)
        
        time_controls_layout.addWidget(self.prev_btn)
        time_controls_layout.addWidget(self.play_btn)
        time_controls_layout.addWidget(self.next_btn)
        
        time_controls_layout.addStretch()
        
        # Volume control 
        volume_label = QLabel("Vol:")
        volume_label.setMaximumWidth(30)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        
        time_controls_layout.addWidget(volume_label)
        time_controls_layout.addWidget(self.volume_slider)
        
        # Total time
        self.total_time_label = QLabel("00:00")
        self.total_time_label.setMinimumWidth(40)
        time_controls_layout.addWidget(self.total_time_label)
        
        controls_layout.addLayout(time_controls_layout)
        layout.addWidget(controls_container)
        
    def get_position(self):
        return self.media_player.position() if self.media_player else 0
        
    def setup_media_player(self):
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Connect signals
        self.media_player.positionChanged.connect(self.position_changed.emit)
        self.media_player.durationChanged.connect(self.duration_changed.emit)
        
        # Connect control signals
        self.prev_btn.clicked.connect(lambda: self.seek_backward(15))
        self.play_btn.clicked.connect(self.toggle_playback)
        self.next_btn.clicked.connect(lambda: self.seek_forward(15))
        self.timeline_slider.valueChanged.connect(self.set_position)
        self.volume_slider.valueChanged.connect(self.set_volume)
        
    def load_video(self, file_path):
        """Load video from file path"""
        if hasattr(self, 'media_player'):
            self.media_player.stop()
            self.media_player.setSource(QUrl())
        
        if not hasattr(self, 'media_player'):
            self.media_player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.media_player.setAudioOutput(self.audio_output)
            self.video_widget = QVideoWidget()
            self.media_player.setVideoOutput(self.video_widget)
            self.layout().addWidget(self.video_widget)
        
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.media_player.play()
        self.current_video_path = file_path
    def get_current_frame(self):
        """Capture current frame from video"""
        if not hasattr(self, 'media_player') or not self.media_player.isPlaying():
            return None
        
        return self.last_frame  
            
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
            self.is_playing = False
            self.play_btn.setText("")
            self.play_btn.setIcon(QIcon(r"Assets\Icons\pause.png"))
        else:
            self.media_player.play()
            self.is_playing = True
            self.play_btn.setText("")
            self.play_btn.setIcon(QIcon(r"Assets\Icons\play.png"))
        return self.is_playing
            
    def seek_forward(self, seconds=15):
        """Seek forward by seconds"""
        current_pos = self.media_player.position()
        new_pos = min(current_pos + (seconds * 1000), self.media_player.duration())
        self.media_player.setPosition(new_pos)
        
    def seek_backward(self, seconds=15):
        """Seek backward by seconds"""
        current_pos = self.media_player.position()
        new_pos = max(current_pos - (seconds * 1000), 0)
        self.media_player.setPosition(new_pos)
        
    def set_position(self, position):
        """Set playback position"""
        if self.media_player.duration() > 0:
            pos = int((position / 100) * self.media_player.duration())
            self.media_player.setPosition(pos)
        
    def set_volume(self, volume):
        """Set audio volume (0-100)"""
        self.audio_output.setVolume(volume / 100.0)
        
    def update_position(self, position):
        """Update position display"""
        if self.media_player.duration() > 0:
            slider_position = int((position / self.media_player.duration()) * 100)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(slider_position)
            self.timeline_slider.blockSignals(False)
            
            self.current_time_label.setText(self.format_time(position))
            
    def update_duration(self, duration):
        """Update duration display"""
        self.total_time_label.setText(self.format_time(duration))
        
    @staticmethod
    def format_time(ms):
        """Format milliseconds to MM:SS"""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"


class DetectedFacesWidget(QWidget):
    
    face_clicked = Signal(int)  
    from PySide6.QtWidgets import QScrollArea, QHBoxLayout, QSplitter
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QPixmap, QImage
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.face_images = []
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Title
        title_label = QLabel("Detected Faces (Click to select):")
        title_label.setStyleSheet("font-weight: bold; color: #0d7ae7;")
        main_layout.addWidget(title_label)
        
        # Create splitter for collapsible functionality
        self.faces_splitter = QSplitter(Qt.Orientation.Vertical)
        self.faces_splitter.setChildrenCollapsible(True)
        
        
        # Scroll area for faces
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMaximumHeight(120)
        scroll_area.setMinimumHeight(60)
        
        self.faces_container = QWidget()
        self.faces_layout = QHBoxLayout(self.faces_container)
        self.faces_layout.setSpacing(10)
        self.faces_layout.setContentsMargins(5, 5, 5, 5)
        
        scroll_area.setWidget(self.faces_container)
        
        self.faces_splitter.addWidget(scroll_area)
        
        empty_widget = QWidget()
        empty_widget.setMaximumHeight(0)
        self.faces_splitter.addWidget(empty_widget)
        
        main_layout.addWidget(self.faces_splitter)
        
    def add_face(self, face_image, face_info=None):
        """Add a detected face image"""
        face_index = len(self.face_images)
        
        face_label = ClickableLabel()
        face_label.setFixedSize(80, 80)
        face_label.setScaledContents(True)
        face_label.setStyleSheet("""
            QLabel {
                border: 2px solid #0d7ae7;
                border-radius: 8px;
                background-color: #2d2d2d;
            }
            QLabel:hover {
                border-color: #1a8aff;
                background-color: #3d3d3d;
                cursor: pointer;
            }
        """)
        
        face_label.face_index = face_index
        
        face_label.clicked.connect(lambda idx=face_index: self.on_face_clicked(idx))
        
        # Convert numpy array to QPixmap 
        if isinstance(face_image, np.ndarray):
            height, width, channel = face_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(face_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            face_label.setPixmap(pixmap)
        else:
            face_label.setPixmap(face_image)
            
        if face_info:
            tooltip_text = f"Face #{face_index + 1}\nConfidence: {face_info.get('confidence', 'N/A'):.2f}\nFrame: {face_info.get('frame', 'N/A')}\nClick to select"
            face_label.setToolTip(tooltip_text)
        else:
            face_label.setToolTip(f"Face #{face_index + 1}\nClick to select")
            
        self.faces_layout.addWidget(face_label)
        self.face_images.append(face_label)
        
        # Update container size
        self.faces_container.adjustSize()
        
    def on_face_clicked(self, face_index):
        """Handle face click"""
        print(f"✅ Selected Face Index: {face_index}")
        logger.info(f"User selected face at index {face_index}")

        for i, face_label in enumerate(self.face_images):
            if i == face_index:
                face_label.setStyleSheet("""
                    QLabel {
                        border: 3px solid #00ff00;
                        border-radius: 8px;
                        background-color: #4d4d4d;
                    }
                    QLabel:hover {
                        border-color: #00ff00;
                        background-color: #5d5d5d;
                        cursor: pointer;
                    }
                """)
            else:
                face_label.setStyleSheet("""
                    QLabel {
                        border: 2px solid #0d7ae7;
                        border-radius: 8px;
                        background-color: #2d2d2d;
                    }
                    QLabel:hover {
                        border-color: #1a8aff;
                        background-color: #3d3d3d;
                        cursor: pointer;
                    }
                """)
        
        # Emit signal
        self.face_clicked.emit(face_index)
        
    def clear_faces(self):
        """Clear all detected faces"""
        for face_label in self.face_images:
            face_label.setParent(None)
        self.face_images.clear()


class ClickableLabel(QLabel):
    """Clickable QLabel that emits signal when clicked"""
    clicked = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.face_index = 0
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.face_index)
        super().mousePressEvent(event)


class CollapsibleSplitter(QSplitter):
    """Custom splitter with collapsible functionality"""
    
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setChildrenCollapsible(True)
        self.setHandleWidth(12)
        self.left_collapsed = False
        self.right_collapsed = False
        self.saved_sizes = []
        
    def addCollapsibleWidget(self, widget):
        """Add widget that can be collapsed"""
        self.addWidget(widget)
        
    def mouseDoubleClickEvent(self, event):
        """Handle double click on splitter handle to collapse/expand"""
        handle_index = self.handleAt(event.position().toPoint())
        if handle_index >= 0:
            self.toggleSection(handle_index)
        else:
            super().mouseDoubleClickEvent(event)
            
    def handleAt(self, pos):
        """Find which handle was clicked"""
        for i in range(self.count() - 1):
            handle = self.handle(i + 1)
            if handle and handle.geometry().contains(pos):
                return i
        return -1
        
    def toggleSection(self, handle_index):
        """Toggle collapse/expand of section"""
        sizes = self.sizes()
        
        if handle_index == 0: 
            if not self.left_collapsed:
                # Collapse left
                self.saved_sizes = sizes.copy()
                new_sizes = [0, sizes[1] + sizes[0], sizes[2]]
                self.setSizes(new_sizes)
                self.left_collapsed = True
            else:
                if self.saved_sizes:
                    self.setSizes(self.saved_sizes)
                else:
                    total = sum(sizes)
                    new_sizes = [total//4, total//2, total//4]
                    self.setSizes(new_sizes)
                self.left_collapsed = False
                
        elif handle_index == 1:  
            if not self.right_collapsed:
                self.saved_sizes = sizes.copy()
                new_sizes = [sizes[0], sizes[1] + sizes[2], 0]
                self.setSizes(new_sizes)
                self.right_collapsed = True
            else:
                if self.saved_sizes:
                    self.setSizes(self.saved_sizes)
                else:
                    total = sum(sizes)
                    new_sizes = [total//4, total//2, total//4]
                    self.setSizes(new_sizes)
                self.right_collapsed = False
                
                
def main():
    
    from ui_logic import EnhancedMainWindow
    """Main application entry point with enhanced error handling"""
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Face Swapping Studio")
        app.setApplicationVersion("2.0.0")
        app.setOrganizationName("AI Studio Pro")
        
        try:
            if os.path.exists("assets/icon.png"):
                app.setWindowIcon(QIcon("assets/icon.png"))
        except Exception as e:
            logger.warning(f"Could not load application icon: {e}")
        
        app.setStyle('Fusion')
        
        main_window = EnhancedMainWindow()
        main_window.show()
        
        screen = app.primaryScreen().geometry()
        window_geometry = main_window.geometry()
        x = (screen.width() - window_geometry.width()) // 2
        y = (screen.height() - window_geometry.height()) // 2
        main_window.move(x, y)
        
        main_window.update_progress("🚀 Face Swapping Studio started successfully!")
        if not PSUTIL_AVAILABLE:
            main_window.update_progress("⚠️ Install 'psutil' for system monitoring: pip install psutil")
        if not GPUTIL_AVAILABLE:
            main_window.update_progress("⚠️ Install 'GPUtil' for GPU monitoring: pip install GPUtil")
        
        logger.info("Application started successfully")
        
        return app.exec()
        
    except ImportError as e:
        error_msg = f"Missing required dependencies: {e}\n\nPlease install required packages:\npip install PySide6 opencv-python numpy"
        logger.error(error_msg)
        if 'app' in locals():
            QMessageBox.critical(None, "Import Error", error_msg)
        else:
            print(f"CRITICAL ERROR: {error_msg}")
        return 1
        
    except Exception as e:
        error_msg = f"Application failed to start: {e}"
        logger.error(error_msg)
        if 'app' in locals():
            QMessageBox.critical(None, "Fatal Error", error_msg)
        else:
            print(f"CRITICAL ERROR: {error_msg}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)