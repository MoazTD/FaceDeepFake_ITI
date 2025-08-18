import sys
import cv2
import dlib
import numpy as np
import os
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox, QGraphicsView, QGraphicsScene,
    QLabel, QProgressBar, QDialogButtonBox, QSlider, QFrame
)
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QBrush, QPainter
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize

# Assuming FaceDetector is from core and is capable of face recognition.
from core.face_detector import FaceDetector 

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')

class FullVideoProcessor(QThread):
    progress_updated = Signal(int, int, str)
    processing_completed = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, video_path: str, detector, predictor, selected_face_embedding: Optional[np.ndarray] = None):
        super().__init__()
        self.video_path = video_path
        self.detector = detector
        self.predictor = predictor
        self.selected_face_embedding = selected_face_embedding
        self.is_cancelled = False

    def run(self):
        logging.info(f"Starting full video processing for: {self.video_path}")
        all_landmarks = {}
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_occurred.emit("Could not open video file.")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_updated.emit(0, frame_count, "Starting video processing...")

        for i in range(frame_count):
            if self.is_cancelled:
                logging.info("Video processing was cancelled by the user.")
                cap.release()
                return

            ret, frame = cap.read()
            if not ret:
                break

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                all_faces = self.detector.detect_faces(frame)
                target_face = None

                if self.selected_face_embedding is not None and all_faces:
                    # Use face recognition to find the selected face
                    min_distance = float('inf')
                    for face in all_faces:
                        # Assuming the face object has a 'normed_embedding' attribute
                        distance = np.linalg.norm(self.selected_face_embedding - face.normed_embedding)
                        if distance < min_distance:
                            min_distance = distance
                            target_face = face
                elif all_faces:
                    # Fallback to finding the largest face
                    target_face = max(all_faces, key=lambda face: (face.bbox[2]-face.bbox[0]) * (face.bbox[3]-face.bbox[1]))

                if target_face:
                    # Get landmarks for the identified target face
                    rect = dlib.rectangle(
                        int(target_face.bbox[0]), int(target_face.bbox[1]),
                        int(target_face.bbox[2]), int(target_face.bbox[3])
                    )
                    shape = self.predictor(gray, rect)
                    landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)
                    all_landmarks[i] = landmarks
            
            except Exception as e:
                logging.warning(f"Could not process landmarks for frame {i}: {e}")

            if (i + 1) % 10 == 0:
                self.progress_updated.emit(i + 1, frame_count, f"Processing frame {i+1}/{frame_count}")
        
        cap.release()
        logging.info(f"Finished video processing. Found landmarks in {len(all_landmarks)} frames.")
        self.processing_completed.emit(all_landmarks)

    def cancel(self):
        self.is_cancelled = True

class AutoAlignmentDialog(QDialog):
    def __init__(self, video_path: str, selected_face_embedding: Optional[np.ndarray] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automatic Landmark Alignment")
        self.setGeometry(150, 150, 800, 650)
        self.setModal(True)

        self.video_path = video_path
        self.frames = []
        self.preview_frame = None
        self.preview_landmarks = None
        self.full_video_landmarks = {}
        self.scale_factor = 1.0
        self.canvas_max_dim = 550
        self.full_processor_thread = None
        self.selected_face_embedding = selected_face_embedding
        self.current_frame_index = 0

        self.detector = FaceDetector()
        self.predictor = None
        if not self.init_dlib():
            QTimer.singleShot(0, self.reject)
            return

        self._load_all_frames()
        self.init_ui()
        self.detect_preview_landmarks()

    def get_auto_alignment_data(self) -> Optional[Dict[int, np.ndarray]]:
        return self.full_video_landmarks

    def init_dlib(self) -> bool:
        try:
            dlib_model_path = self._find_dlib_model()
            logging.info(f"Loading dlib model from: {dlib_model_path}")
            self.predictor = dlib.shape_predictor(dlib_model_path)
            return True
        except Exception as e:
            QMessageBox.critical(self, "dlib Model Error", 
                                 f"Failed to initialize dlib predictor.\n"
                                 f"Ensure 'shape_predictor_68_face_landmarks.dat' is in the 'Models' folder.\n\n"
                                 f"Error: {e}")
            return False

    def _find_dlib_model(self) -> str:
        model_name = "shape_predictor_68_face_landmarks.dat"
        possible_paths = [
            Path("models") / model_name,
            Path("Models") / model_name,
            Path(__file__).parent.parent / "models" / model_name,
            Path(__file__).parent.parent / "Models" / model_name,
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        raise FileNotFoundError(f"Dlib model '{model_name}' not found.")

    def _load_all_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file.")
            self.reject()
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()

        if not self.frames:
            QMessageBox.critical(self, "Error", "No frames loaded from video.")
            self.reject()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        self.status_label = QLabel("Previewing landmarks on the first frame. Click 'Apply' to process the full video.")
        self.status_label.setWordWrap(True)
        main_layout.addWidget(self.status_label)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        main_layout.addWidget(self.view)
        
        if self.frames:
            self.frame_slider = QSlider(Qt.Horizontal)
            self.frame_slider.setRange(0, len(self.frames) - 1)
            self.frame_slider.valueChanged.connect(self.seek_frame)
            main_layout.addWidget(self.frame_slider)

            self.frame_label = QLabel("Frame 0 / {}".format(len(self.frames) - 1))
            main_layout.addWidget(self.frame_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        button_layout = QHBoxLayout()
        self.button_box = QDialogButtonBox()

        self.process_all_button = self.button_box.addButton("Process All Frames", QDialogButtonBox.ButtonRole.AcceptRole)
        self.process_all_button.clicked.connect(self.run_full_video_processing)

        self.apply_button = self.button_box.addButton("Apply & Exit", QDialogButtonBox.ButtonRole.AcceptRole)
        self.apply_button.clicked.connect(self.accept)
        self.apply_button.setEnabled(False)

        self.cancel_button = self.button_box.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.button_box)
        main_layout.addLayout(button_layout)


    def detect_preview_landmarks(self):
        try:
            if not self.frames:
                self.on_processing_error("No frames available.")
                return

            frame = self.frames[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            target_rect = None
            all_faces = self.detector.detect_faces(frame)

            if self.selected_face_embedding is not None and all_faces:
                min_distance = float('inf')
                for face in all_faces:
                    distance = np.linalg.norm(self.selected_face_embedding - face.normed_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        target_rect = dlib.rectangle(
                            int(face.bbox[0]), int(face.bbox[1]),
                            int(face.bbox[2]), int(face.bbox[3])
                        )
            elif all_faces:
                largest_face = max(all_faces, key=lambda face: (face.bbox[2]-face.bbox[0]) * (face.bbox[3]-face.bbox[1]))
                target_rect = dlib.rectangle(
                    int(largest_face.bbox[0]), int(largest_face.bbox[1]),
                    int(largest_face.bbox[2]), int(largest_face.bbox[3])
                )
            
            if target_rect is None:
                self.on_processing_error("No face detected in the preview frame.")
                return
            
            shape = self.predictor(gray, target_rect)
            self.preview_landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)
            
            self.draw_frame(0)
            self.process_all_button.setEnabled(True)
            self.status_label.setText("Preview successful. Ready to process the full video.")
        except Exception as e:
            self.on_processing_error(f"Error during preview detection: {e}")

    def draw_frame(self, frame_index: int):
        if not self.frames:
            return
            
        self.scene.clear()
        
        frame = self.frames[frame_index].copy()
        h, w = frame.shape[:2]
        scale = min(self.canvas_max_dim / w, self.canvas_max_dim / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)

        img_disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        if self.full_video_landmarks and frame_index in self.full_video_landmarks:
            landmarks_to_draw = self.full_video_landmarks[frame_index]
            if landmarks_to_draw is not None and landmarks_to_draw.any():
                for x, y in landmarks_to_draw:
                    cv2.circle(img_disp, (int(x * scale), int(y * scale)), 2, (0, 255, 255), -1)
        elif self.preview_landmarks is not None and self.preview_landmarks.any() and frame_index == 0:
            for x, y in self.preview_landmarks:
                cv2.circle(img_disp, (int(x * scale), int(y * scale)), 2, (0, 255, 255), -1)
        
        img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        q_image = QImage(img_rgb.data, disp_w, disp_h, img_rgb.strides[0], QImage.Format_RGB888)
        self.scene.addPixmap(QPixmap.fromImage(q_image))
        self.view.setSceneRect(0, 0, disp_w, disp_h)

    def seek_frame(self, value):
        self.current_frame_index = value
        if hasattr(self, 'frame_label'):
            self.frame_label.setText("Frame {} / {}".format(value, len(self.frames) - 1))
        self.draw_frame(value)

    def run_full_video_processing(self):
        self.process_all_button.setEnabled(False)
        self.cancel_button.setText("Cancel Processing")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.full_processor_thread = FullVideoProcessor(
            self.video_path,
            self.detector,
            self.predictor,
            self.selected_face_embedding
        )
        self.full_processor_thread.progress_updated.connect(self.update_full_progress)
        self.full_processor_thread.processing_completed.connect(self.on_full_processing_complete)
        self.full_processor_thread.error_occurred.connect(self.on_processing_error)
        self.full_processor_thread.start()

    def update_full_progress(self, current, total, message):
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

    def on_full_processing_complete(self, landmark_data: dict):
        logging.info(f"AutoAlignmentDialog: Received {len(landmark_data)} landmark sets from the processing thread.")
        
        if landmark_data:
            self.full_video_landmarks = landmark_data
            self.status_label.setText(f"Processing complete! Found landmarks in {len(landmark_data)} frames.")
            QMessageBox.information(self, "Success", "Automatic landmark tracking is complete. You can now scrub the video to preview the results.")
            self.apply_button.setText("Apply")
            self.apply_button.setEnabled(True)
        else:
            logging.warning("AutoAlignmentDialog: Landmark data dictionary is empty. Cannot preview results.")
            self.status_label.setText("Processing complete, but no landmarks were found.")
            QMessageBox.warning(self, "No Landmarks Found", "Automatic landmark tracking is complete, but no faces were tracked in the video.")
            self.full_video_landmarks = {}

        self.progress_bar.setVisible(False)
        self.draw_frame(self.current_frame_index)

    def on_processing_error(self, error_message: str):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.warning(self, "Processing Error", error_message)
        self.apply_button.setEnabled(False)
        self.cancel_button.setText("Close")

    def reject(self):
        if self.full_processor_thread and self.full_processor_thread.isRunning():
            self.full_processor_thread.cancel()
            self.full_processor_thread.wait() 
        super().reject()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = AutoAlignmentDialog("test_video.mp4")
    dialog.exec()
    sys.exit(app.exec())