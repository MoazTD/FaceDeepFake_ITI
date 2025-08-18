import sys
import os
import asyncio
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog,
    QProgressBar, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QFrame, QScrollArea, QMessageBox,
    QDialog, QDialogButtonBox, QFormLayout, QCheckBox, QSlider
)
from PySide6.QtCore import (
    Qt, QThread, QObject, Signal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, QRect, QUrl
)
from PySide6.QtGui import (
    QFont, QPalette, QColor, QIcon, QPixmap, QPainter, QBrush,
    QLinearGradient, QDragEnterEvent, QDropEvent
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from voice_handler import VoiceConversionWorkflow, RVCConverter

class ProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voice Conversion Progress")
        self.setModal(True)
        self.resize(600, 400)
        self.setup_ui()
        self.is_closed = False
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Initializing...")
        layout.addWidget(self.status_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
    
    def update_progress(self, message: str):
        if self.is_closed:
            return
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_text.append(f"{timestamp} {message}")
        self.status_label.setText(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def set_completed(self, success: bool, message: str = ""):
        if self.is_closed:
            return
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        if success:
            self.status_label.setText("✅ Conversion completed successfully!")
        else:
            self.status_label.setText(f"❌ Conversion failed: {message}")
        self.cancel_button.setText("Close")
    
    def closeEvent(self, event):
        self.is_closed = True
        event.accept()
    
    def reject(self):
        self.is_closed = True
        super().reject()

class ConversionWorker(QObject):
    progress = Signal(str)
    finished = Signal(dict)
    
    def __init__(self, converter: RVCConverter, params: Dict[str, Any]):
        super().__init__()
        self.converter = converter
        self.params = params
        self.is_canceled = False
    
    def run(self):
        try:
            def safe_progress_callback(message):
                if not self.is_canceled:
                    self.progress.emit(message)
            
            self.params['progress_callback'] = safe_progress_callback
            self.params['cancellation_check'] = lambda: self.is_canceled
            result = self.converter.convert_voice(**self.params)
            
            if not self.is_canceled:
                self.finished.emit(result)
        except Exception as e:
            if not self.is_canceled:
                self.finished.emit({"success": False, "error": f"Worker error: {e}"})

    def stop(self):
        self.is_canceled = True

class VideoMergeWorker(QObject):
    progress = Signal(str)
    finished = Signal(dict)

    def __init__(self, converter: RVCConverter, params: Dict[str, Any]):
        super().__init__()
        self.converter = converter
        self.params = params
        self.is_canceled = False

    def run(self):
        try:
            def safe_progress_callback(message):
                if not self.is_canceled:
                    self.progress.emit(message)
                    
            result = self.converter.merge_audio_to_video(
                video_path=self.params['video_path'],
                audio_path=self.params['audio_path'],
                output_path=self.params['output_path'],
                progress_callback=safe_progress_callback,
                cancellation_check=lambda: self.is_canceled
            )
            if not self.is_canceled:
                self.finished.emit(result)
        except Exception as e:
            if not self.is_canceled:
                self.finished.emit({"success": False, "error": f"Worker error during merge: {e}"})

    def stop(self):
        self.is_canceled = True

class AudioPlayerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_audio_path = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls_layout = QHBoxLayout()
        self.play_button = QPushButton("▶️ Play")
        self.play_button.clicked.connect(self.play_audio)
        self.pause_button = QPushButton("⏸️ Pause")
        self.pause_button.clicked.connect(self.pause_audio)
        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.clicked.connect(self.stop_audio)

        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.stop_button)
        layout.addLayout(controls_layout)

        progress_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.sliderMoved.connect(self.set_position)
        self.total_time_label = QLabel("00:00")

        progress_layout.addWidget(self.current_time_label)
        progress_layout.addWidget(self.progress_slider)
        progress_layout.addWidget(self.total_time_label)
        layout.addLayout(progress_layout)

        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)

        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.playbackStateChanged.connect(self.update_buttons)

        self.set_enabled_state(False)

    def set_enabled_state(self, enabled: bool):
        self.play_button.setEnabled(enabled)
        self.pause_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.progress_slider.setEnabled(enabled)

    def load_audio(self, file_path: str):
        self.media_player.stop()
        
        if not file_path or not Path(file_path).exists():
            self.current_audio_path = None
            self.media_player.setSource(QUrl())
            self.progress_slider.setRange(0, 0)
            self.current_time_label.setText("00:00")
            self.total_time_label.setText("00:00")
            self.set_enabled_state(False)
            return

        self.current_audio_path = file_path
        self.media_player.setSource(QUrl.fromLocalFile(str(Path(file_path).resolve())))
        self.set_enabled_state(True)
        self.audio_output.setVolume(1.0)

    def play_audio(self):
        if self.media_player.source().isEmpty():
            return
        self.media_player.play()

    def pause_audio(self):
        self.media_player.pause()

    def stop_audio(self):
        self.media_player.stop()

    def set_position(self, position: int):
        self.media_player.setPosition(position)

    def update_duration(self, duration: int):
        self.progress_slider.setRange(0, duration)
        self.total_time_label.setText(self.format_time(duration))

    def update_position(self, position: int):
        if not self.progress_slider.isSliderDown():
            self.progress_slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))

    def update_buttons(self, state: QMediaPlayer.PlaybackState):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        elif state == QMediaPlayer.PlaybackState.StoppedState:
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    def format_time(self, ms: int) -> str:
        seconds = int(ms / 1000)
        minutes = int(seconds / 60)
        seconds %= 60
        return f"{minutes:02}:{seconds:02}"
    
    def cleanup(self):
        """Clean up media player resources"""
        try:
            if self.media_player:
                self.media_player.stop()
                self.media_player.setSource(QUrl())
        except RuntimeError:
            pass

class RVCConverterGUI(QDialog):
    conversion_successful = Signal(str)
    
    def __init__(self, parent=None, target_video_path: Optional[str] = None, stylesheet: str = ""):
        super().__init__(parent)
        self.converter = None
        self.conversion_thread = None
        self.merge_thread = None
        self.conversion_worker = None
        self.merge_worker = None
        self.progress_dialog = None
        self.target_video_path = target_video_path
        self.converted_audio_path = None
        self._is_closing = False  
        
        self.setWindowTitle("RVC Voice Converter")
        self.setMinimumSize(800, 600)
        self.setModal(True)
        self.setup_ui()
            
        self.initialize_converter()
        
        if self.target_video_path:
            self.set_target_video(self.target_video_path)
    
    def set_target_video(self, video_path: Optional[str]):
        """Sets the input audio path from the target video."""
        if self._is_closing:
            return
            
        self.target_video_path = video_path
        if self.target_video_path and Path(self.target_video_path).exists():
            self.input_path_edit.setText(self.target_video_path)
            p = Path(self.target_video_path)
            self.output_path_edit.setText(str(p.parent / f"{p.stem}_voice_converted.wav"))
            self.log_message(f"Input automatically set from video: {p.name}")
            self.log_message("Audio will be extracted automatically during conversion")
        else:
            self.input_path_edit.clear()
            self.output_path_edit.clear()
            
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        title = QLabel("🎵 RVC Voice Converter")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        file_group = QGroupBox("1. File Selection")
        file_layout = QGridLayout(file_group)
        file_layout.addWidget(QLabel("Input Audio/Video:"), 0, 0)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select or drop an audio/video file...")
        self.input_path_edit.textChanged.connect(self.on_input_changed)
        file_layout.addWidget(self.input_path_edit, 0, 1)
        input_browse_btn = QPushButton("Browse...")
        input_browse_btn.clicked.connect(self.browse_input_file)
        file_layout.addWidget(input_browse_btn, 0, 2)
        
        self.file_type_label = QLabel("")
        self.file_type_label.setStyleSheet("color: #666; font-size: 10px;")
        file_layout.addWidget(self.file_type_label, 1, 1)
        
        file_layout.addWidget(QLabel("Output Audio:"), 2, 0)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Auto-generated or select custom path...")
        file_layout.addWidget(self.output_path_edit, 2, 1)
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(self.browse_output_file)
        file_layout.addWidget(output_browse_btn, 2, 2)
        main_layout.addWidget(file_group)
        
        model_group = QGroupBox("2. Model Configuration")
        model_layout = QGridLayout(model_group)
        model_layout.addWidget(QLabel("Voice Model:"), 0, 0)
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo, 0, 1)
        refresh_models_btn = QPushButton("Refresh Models")
        refresh_models_btn.clicked.connect(self.refresh_models)
        model_layout.addWidget(refresh_models_btn, 0, 2)
        
        setup_status_btn = QPushButton("Check Setup")
        setup_status_btn.clicked.connect(self.show_setup_status)
        model_layout.addWidget(setup_status_btn, 0, 3)
        
        self.model_path_label = QLabel()
        self.model_path_label.setWordWrap(True)
        self.model_path_label.setStyleSheet("color: #666; font-size: 10px;")
        model_layout.addWidget(self.model_path_label, 1, 0, 1, 4)
        
        main_layout.addWidget(model_group)
        
        params_group = QGroupBox("3. Conversion Parameters")
        params_layout = QGridLayout(params_group)
        params_layout.addWidget(QLabel("Pitch Adjustment (semitones):"), 0, 0)
        self.pitch_spinbox = QSpinBox()
        self.pitch_spinbox.setRange(-24, 24)
        self.pitch_spinbox.setValue(0)
        params_layout.addWidget(self.pitch_spinbox, 0, 1)
        
        params_layout.addWidget(QLabel("Index Rate (0.0 to 1.0):"), 1, 0)
        self.index_rate_spinbox = QDoubleSpinBox()
        self.index_rate_spinbox.setRange(0.0, 1.0)
        self.index_rate_spinbox.setValue(0.75)
        self.index_rate_spinbox.setSingleStep(0.05)
        params_layout.addWidget(self.index_rate_spinbox, 1, 1)
        
        params_layout.addWidget(QLabel("Protect Rate (0.0 to 0.5):"), 2, 0)
        self.protect_spinbox = QDoubleSpinBox()
        self.protect_spinbox.setRange(0.0, 0.5)
        self.protect_spinbox.setValue(0.33)
        self.protect_spinbox.setSingleStep(0.01)
        params_layout.addWidget(self.protect_spinbox, 2, 1)
        main_layout.addWidget(params_group)
        
        player_group = QGroupBox("4. Preview Converted Audio")
        player_layout = QVBoxLayout(player_group)
        self.audio_player_widget = AudioPlayerWidget(self)
        player_layout.addWidget(self.audio_player_widget)
        main_layout.addWidget(player_group)
        self.audio_player_widget.setVisible(False)
        
        log_group = QGroupBox("Logs")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

        self.convert_button = QPushButton("🎵 Convert Voice")
        self.convert_button.setObjectName("btn_convert_voice_primary")
        self.convert_button.setFixedHeight(40)
        self.convert_button.clicked.connect(self.start_conversion)
        
        self.merge_video_button = QPushButton("🎞️ Merge with Video")
        self.merge_video_button.setObjectName("btn_merge_video_secondary")
        self.merge_video_button.setFixedHeight(40)
        self.merge_video_button.clicked.connect(self.start_video_merge)
        self.merge_video_button.setEnabled(False)
        
        button_box_layout = QHBoxLayout()
        button_box_layout.addWidget(self.convert_button)
        button_box_layout.addWidget(self.merge_video_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        button_box_layout.addWidget(close_button)

        main_layout.addLayout(button_box_layout)
    
    def on_input_changed(self):
        """Update file type indicator when input changes"""
        if self._is_closing:
            return
            
        input_path = self.input_path_edit.text().strip()
        if input_path and Path(input_path).exists():
            if self._is_video_file(input_path):
                self.file_type_label.setText("🎬 Video file - audio will be extracted automatically")
                self.file_type_label.setStyleSheet("color: #007acc; font-size: 10px;")
                self.merge_video_button.setEnabled(False)
            elif self._is_audio_file(input_path):
                self.file_type_label.setText("🎵 Audio file")
                self.file_type_label.setStyleSheet("color: #28a745; font-size: 10px;")
                self.merge_video_button.setEnabled(False)
            else:
                self.file_type_label.setText("⚠️ Unsupported file type")
                self.file_type_label.setStyleSheet("color: #dc3545; font-size: 10px;")
                self.merge_video_button.setEnabled(False)
        else:
            self.file_type_label.setText("")
            self.merge_video_button.setEnabled(False)
        
        self.audio_player_widget.load_audio(None)
        self.audio_player_widget.setVisible(False)
        self.converted_audio_path = None

    def _is_video_file(self, file_path: str) -> bool:
        """Check if file is a video"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        return Path(file_path).suffix.lower() in video_extensions
    
    def _is_audio_file(self, file_path: str) -> bool:
        """Check if file is audio"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
        return Path(file_path).suffix.lower() in audio_extensions
    
    def initialize_converter(self):
        try:
            rvc_root = None  
            assets_path = None 
            
            self.converter = RVCConverter(rvc_root=rvc_root, assets_path=assets_path)
            
            setup_status = self.converter.validate_setup()
            
            self.log_message("✅ RVC Converter initialized.")
            self.log_message(f"📁 RVC Root: {self.converter.rvc_root}")
            self.log_message(f"📁 Assets Path: {self.converter.assets_path}")
            
            for warning in setup_status.get("warnings", []):
                self.log_message(f"⚠️ {warning}")
            
            for error in setup_status.get("errors", []):
                self.log_message(f"❌ {error}")
            
            self.model_path_label.setText(f"Models directory: {self.converter.assets_path}")
            
            self.refresh_models()
            
            if setup_status.get("errors") and not setup_status.get("models_found"):
                QMessageBox.warning(
                    self, 
                    "Setup Issues", 
                    f"RVC setup has some issues:\n\n" + 
                    "\n".join(setup_status["errors"]) +
                    f"\n\nExpected directory structure:\n"
                    f"Models/Sound/assets/weights/  (for .pth model files)\n"
                    f"Models/Sound/tools/  (for inference scripts)"
                )
                
        except Exception as e:
            self.log_message(f"⚠️ Failed to initialize converter: {e}")
            self.model_path_label.setText(f"Error: {e}")
            QMessageBox.critical(
                self, 
                "Initialization Error", 
                f"Could not initialize the RVC Converter.\n\n"
                f"Expected directory structure:\n"
                f"Models/Sound/assets/weights/  (for .pth model files)\n"
                f"Models/Sound/tools/  (for inference scripts)\n\n"
                f"Error: {e}"
            )

    def log_message(self, message: str):
        if self._is_closing:
            return
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.log_text.append(f"{timestamp} {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def browse_input_file(self):
        if self._is_closing:
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Input Audio or Video", 
            "", 
            "All Files (*.wav *.mp3 *.flac *.mp4 *.mov *.avi *.mkv);;Audio Files (*.wav *.mp3 *.flac);;Video Files (*.mp4 *.mov *.avi *.mkv)"
        )
        if file_path:
            self.input_path_edit.setText(file_path)
            if not self.output_path_edit.text():
                p = Path(file_path)
                self.output_path_edit.setText(str(p.parent / f"{p.stem}_converted.wav"))
    
    def browse_output_file(self):
        if self._is_closing:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output Audio", "", "WAV Files (*.wav)")
        if file_path:
            self.output_path_edit.setText(file_path)

    def refresh_models(self):
        if self._is_closing or not self.converter: 
            return
        
        try:
            models = self.converter.list_available_models()
            self.model_combo.clear()
            if models:
                self.model_combo.addItems(models)
                self.log_message(f"📁 Found {len(models)} models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
            else:
                self.model_combo.addItem("No models found")
                self.log_message("⚠️ No models found.")
                self.log_message(f"📂 Please place your RVC model files (.pth) in: {self.converter.assets_path}")
                
                if not self._is_closing:
                    QMessageBox.information(
                        self,
                        "No Models Found",
                        f"No RVC model files (.pth) were found.\n\n"
                        f"Please place your RVC model files in:\n{self.converter.assets_path}\n\n"
                        f"You can download RVC models from:\n"
                        f"- Official RVC repositories\n"
                        f"- HuggingFace model collections\n"
                        f"- Community model shares\n\n"
                        f"After adding models, click 'Refresh Models' to reload."
                    )
        except Exception as e:
            self.log_message(f"❌ Error loading models: {e}")
            self.model_combo.clear()
            self.model_combo.addItem("Error loading models")
    
    def show_setup_status(self):
        """Show detailed setup status"""
        if self._is_closing or not self.converter:
            QMessageBox.warning(self, "Error", "Converter not initialized")
            return
        
        status = self.converter.validate_setup()
        
        status_text = f"""
<h3>RVC Setup Status</h3>
<p><b>RVC Root:</b> {self.converter.rvc_root}</p>
<p><b>Root Exists:</b> {'✅' if status['rvc_root_exists'] else '❌'}</p>
<p><b>Assets Path:</b> {self.converter.assets_path}</p>
<p><b>Assets Exists:</b> {'✅' if status['assets_path_exists'] else '❌'}</p>
<p><b>Models Found:</b> {len(status['models_found'])}</p>
<p><b>Inference Script:</b> {'✅' if status['inference_script_found'] else '❌'}</p>
<p><b>FFmpeg Available:</b> {'✅' if status['ffmpeg_available'] else '❌'}</p>

{f"<p><b>Script Path:</b> {status['inference_script_path']}</p>" if status['inference_script_path'] else ""}

{f"<p><b>Models:</b><br>{'<br>'.join(status['models_found'][:10])}</p>" if status['models_found'] else ""}

{f"<p><b>Warnings:</b><br>{'<br>'.join(status['warnings'])}</p>" if status['warnings'] else ""}

{f"<p><b>Errors:</b><br>{'<br>'.join(status['errors'])}</p>" if status['errors'] else ""}
"""
        
        msg = QMessageBox(self)
        msg.setWindowTitle("RVC Setup Status")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(status_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()
    
    def start_conversion(self):
        if self._is_closing:
            return
            
        # Validation
        if not self.input_path_edit.text().strip():
            QMessageBox.warning(self, "Input Required", "Please select an input audio/video file.")
            return
            
        if not self.output_path_edit.text().strip():
            QMessageBox.warning(self, "Output Required", "Please specify an output file path.")
            return
            
        if not self.model_combo.currentText() or self.model_combo.currentText() in ["No models found", "Error loading models"]:
            QMessageBox.warning(self, "Model Required", "Please select a valid voice model.")
            return
        
        if not self.converter:
            QMessageBox.critical(self, "Error", "Converter not initialized.")
            return

        input_path = self.input_path_edit.text().strip()
        if not Path(input_path).exists():
            QMessageBox.warning(self, "File Not Found", f"Input file not found:\n{input_path}")
            return
        
        self._cleanup_conversion_thread()
        
        temp_output_dir = Path(self.converter.temp_base_dir) / "preview_audio"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        temp_output_file = temp_output_dir / f"temp_converted_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"

        params = {
            'input_path': input_path,
            'model_name': self.model_combo.currentText(),
            'output_path': str(temp_output_file),
            'f0_up_key': self.pitch_spinbox.value(),
            'index_rate': self.index_rate_spinbox.value(),
            'protect': self.protect_spinbox.value()
        }
        
        self.progress_dialog = ProgressDialog(self)
        self.progress_dialog.show()
        
        self.conversion_thread = QThread()
        self.conversion_worker = ConversionWorker(self.converter, params)
        self.conversion_worker.moveToThread(self.conversion_thread)
        
        self.progress_dialog.cancel_button.clicked.connect(self._cancel_conversion)
        self.conversion_thread.started.connect(self.conversion_worker.run)
        self.conversion_worker.progress.connect(self.progress_dialog.update_progress)
        self.conversion_worker.finished.connect(self.on_conversion_finished)
        
        self.conversion_worker.finished.connect(self.conversion_worker.deleteLater)
        self.conversion_thread.finished.connect(self.conversion_thread.deleteLater)
        
        self.convert_button.setEnabled(False)
        self.merge_video_button.setEnabled(False)
        self.convert_button.setText("Converting...")
        self.audio_player_widget.load_audio(None)
        self.audio_player_widget.setVisible(False)
        self.converted_audio_path = None

        self.conversion_thread.start()
        self.log_message(f"🎵 Starting conversion for: {Path(params['input_path']).name}")
    
    def _cancel_conversion(self):
        """Cancel the conversion process"""
        if self.conversion_worker:
            self.conversion_worker.stop()
        if self.progress_dialog:
            self.progress_dialog.reject()
    
    def _cleanup_conversion_thread(self):
        """Clean up conversion thread and worker with proper error handling"""
        if self.conversion_worker:
            self.conversion_worker.stop()
            self.conversion_worker = None
        
        if self.conversion_thread:
            try:
                if hasattr(self.conversion_thread, 'isRunning') and self.conversion_thread.isRunning():
                    self.conversion_thread.quit()
            except RuntimeError:
                pass
            finally:
                self.conversion_thread = None
    
    def on_conversion_finished(self, result: Dict[str, Any]):
        if self._is_closing:
            return
            
        self.convert_button.setEnabled(True)
        self.convert_button.setText("🎵 Convert Voice")
        
        if self.conversion_thread:
            try:
                if hasattr(self.conversion_thread, 'quit'):
                    self.conversion_thread.quit()
            except RuntimeError:
                pass
            
        if self.progress_dialog and not self.progress_dialog.is_closed:
            self.progress_dialog.set_completed(result['success'], result.get('error', ''))
        
        if result['success']:
            self.converted_audio_path = result['output_path']
            self.log_message(f"✅ Conversion successful. Converted audio saved to temporary file: {self.converted_audio_path}")
            
            self.audio_player_widget.load_audio(self.converted_audio_path)
            self.audio_player_widget.setVisible(True)
            self.log_message("🎧 Converted audio ready for preview.")

            if self.target_video_path and self._is_video_file(self.input_path_edit.text().strip()):
                self.merge_video_button.setEnabled(True)
                self.log_message("Click 'Merge with Video' to combine with original video, or 'Close' to finish.")
            else:
                self.log_message("Converted audio is ready. No video to merge with (original input was not a video).")
                if not self._is_closing:
                    QMessageBox.information(
                        self, 
                        "Conversion Complete", 
                        f"Voice conversion complete!\n\nConverted audio is ready for preview.\n"
                        f"Output saved to temporary file: {self.converted_audio_path}\n\n"
                        f"Since the input was not a video, no merging option is available."
                    )

        else:
            error = result.get('error', 'Unknown error')
            self.log_message(f"❌ Conversion failed: {error}")
            if not self._is_closing:
                QMessageBox.critical(self, "Conversion Failed", f"An error occurred during conversion:\n\n{error}")
            self.audio_player_widget.load_audio(None)
            self.audio_player_widget.setVisible(False)
            self.merge_video_button.setEnabled(False)

    def start_video_merge(self):
        if self._is_closing:
            return
            
        if not self.converted_audio_path or not Path(self.converted_audio_path).exists():
            QMessageBox.warning(self, "Error", "No converted audio found to merge.")
            return

        original_video_path = self.input_path_edit.text().strip()
        if not original_video_path or not self._is_video_file(original_video_path):
            QMessageBox.warning(self, "Error", "Original input was not a video file. Cannot merge.")
            return

        default_output_video_name = Path(original_video_path).stem + "_voice_merged.mp4"
        output_video_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Merged Video", 
            str(Path(original_video_path).parent / default_output_video_name), 
            "MP4 Files (*.mp4);;All Files (*.*)"
        )

        if not output_video_path:
            self.log_message("Video merge cancelled by user.")
            return

        self._cleanup_merge_thread()

        self.log_message(f"🎬 Starting video merge: {Path(original_video_path).name} + {Path(self.converted_audio_path).name}")
        self.log_message(f"Final output video: {output_video_path}")

        self.progress_dialog = ProgressDialog(self)
        self.progress_dialog.setWindowTitle("Video Merging Progress")
        self.progress_dialog.show()

        merge_params = {
            'video_path': original_video_path,
            'audio_path': self.converted_audio_path,
            'output_path': output_video_path,
        }

        self.merge_thread = QThread()
        self.merge_worker = VideoMergeWorker(self.converter, merge_params)
        self.merge_worker.moveToThread(self.merge_thread)

        self.progress_dialog.cancel_button.clicked.connect(self._cancel_merge)
        self.merge_thread.started.connect(self.merge_worker.run)
        self.merge_worker.progress.connect(self.progress_dialog.update_progress)
        self.merge_worker.finished.connect(self.on_video_merge_finished)

        self.merge_worker.finished.connect(self.merge_worker.deleteLater)
        self.merge_thread.finished.connect(self.merge_thread.deleteLater)

        self.convert_button.setEnabled(False)
        self.merge_video_button.setEnabled(False)
        self.merge_video_button.setText("Merging...")
        self.audio_player_widget.set_enabled_state(False)

        self.merge_thread.start()
    
    def _cancel_merge(self):
        """Cancel the merge process"""
        if self.merge_worker:
            self.merge_worker.stop()
        if self.progress_dialog:
            self.progress_dialog.reject()
    
    def _cleanup_merge_thread(self):
        """Clean up merge thread and worker with proper error handling"""
        if self.merge_worker:
            self.merge_worker.stop()
            self.merge_worker = None
        
        if self.merge_thread:
            try:
                if hasattr(self.merge_thread, 'isRunning') and self.merge_thread.isRunning():
                    self.merge_thread.quit()
            except RuntimeError:
                pass
            finally:
                self.merge_thread = None

    def on_video_merge_finished(self, result: Dict[str, Any]):
        if self._is_closing:
            return
            
        self.convert_button.setEnabled(True)
        self.merge_video_button.setText("🎞️ Merge with Video")
        self.audio_player_widget.set_enabled_state(True)

        if self.merge_thread:
            try:
                if hasattr(self.merge_thread, 'quit'):
                    self.merge_thread.quit()
            except RuntimeError:
                pass

        if self.progress_dialog and not self.progress_dialog.is_closed:
            self.progress_dialog.set_completed(result['success'], result.get('error', ''))

        if result['success']:
            final_video_path = result['output_path']
            self.log_message(f"✅ Video merged successfully: {final_video_path}")
            
            if not self._is_closing:
                reply = QMessageBox.question(
                    self, 
                    "Success", 
                    f"Video merge complete!\n\nOutput saved to:\n{final_video_path}\n\nWould you like to open the output folder?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import platform
                    import subprocess
                    output_dir = Path(final_video_path).parent
                    if platform.system() == "Windows":
                        os.startfile(output_dir)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.run(["open", str(output_dir)])
                    else:  
                        subprocess.run(["xdg-open", str(output_dir)])
            
            self.converted_audio_path = None
            self.audio_player_widget.load_audio(None)
            self.audio_player_widget.setVisible(False)
            self.merge_video_button.setEnabled(False)

        else:
            error = result.get('error', 'Unknown error')
            self.log_message(f"❌ Video merge failed: {error}")
            if not self._is_closing:
                QMessageBox.critical(self, "Merge Failed", f"An error occurred during video merging:\n\n{error}")
            self.merge_video_button.setEnabled(True)

    def closeEvent(self, event):
        """Clean up when dialog is closed with proper error handling"""
        self._is_closing = True
        
        try:
            self.audio_player_widget.cleanup()
        except (RuntimeError, AttributeError):
            pass

        try:
            self._cleanup_conversion_thread()
        except (RuntimeError, AttributeError):
            pass
        
        try:
            self._cleanup_merge_thread()
        except (RuntimeError, AttributeError):
            pass
        
        try:
            if self.progress_dialog and not self.progress_dialog.is_closed:
                self.progress_dialog.close()
        except (RuntimeError, AttributeError):
            pass
        
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RVCConverterGUI()
    window.show()
    sys.exit(app.exec())