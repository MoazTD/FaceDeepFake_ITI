import sys
import cv2
import dlib
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox, QGraphicsView, QGraphicsScene,
    QLabel, QFrame, QProgressBar, QDialogButtonBox
)
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QBrush, QPainter
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from pathlib import Path
import logging

class LandmarkTrackingThread(QThread):
    """Tracks landmarks across all video frames."""
    progress_updated = Signal(int, int)
    tracking_complete = Signal(dict)

    def __init__(self, frames, detector, predictor, initial_landmarks):
        super().__init__()
        self.frames = frames
        self.detector = detector
        self.predictor = predictor
        self.initial_landmarks = initial_landmarks
        self.is_cancelled = False

    def run(self):
        tracked_landmarks = {0: self.initial_landmarks}
        prev_landmarks = self.initial_landmarks

        for i, frame_pil in enumerate(self.frames[1:], start=1):
            if self.is_cancelled:
                break
            
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use previous landmarks to create a bounding box for the next frame
            x_min, y_min = np.min(prev_landmarks, axis=0)
            x_max, y_max = np.max(prev_landmarks, axis=0)
            rect = dlib.rectangle(int(x_min), int(y_min), int(x_max), int(y_max))

            shape = self.predictor(gray, rect)
            current_landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)
            
            tracked_landmarks[i] = current_landmarks
            prev_landmarks = current_landmarks
            
            self.progress_updated.emit(i + 1, len(self.frames))
            
        self.tracking_complete.emit(tracked_landmarks)

    def cancel(self):
        self.is_cancelled = True

class LandmarkEditorApp(QDialog):
    """A QDialog for editing and tracking facial landmarks on a video."""
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Landmark Editor & Tracker")
        self.setGeometry(100, 100, 1200, 800)
        self.setModal(True)

        self.video_path = video_path
        self.frames = []
        self.current_frame_index = 0
        self.original_landmarks = None
        self.modified_landmarks = None
        self.tracked_landmark_data = None
        self.scale_factor = 1.0
        self.canvas_max_dim = 700
        self.dragged_landmark_index = None
        self.landmark_radius = 8

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = self._get_predictor()
        if not self.predictor:
            QTimer.singleShot(0, self.reject)
            return

        self._load_frames()
        self.init_ui()
        self._process_initial_frame()

    def get_tracked_landmarks(self):
        return self.tracked_landmark_data

    def _get_predictor(self):
        model_name = "shape_predictor_68_face_landmarks.dat"
        possible_paths = [Path("models") / model_name, Path("Models") / model_name]
        for path in possible_paths:
            if path.exists():
                return dlib.shape_predictor(str(path))
        QMessageBox.critical(self, "dlib Error", f"Model file '{model_name}' not found.")
        return None

    def _load_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            self.frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.status_label = QLabel("Edit landmarks on the first frame, then click 'Track & Apply'.")
        main_layout.addWidget(self.status_label)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.mousePressEvent = self.on_mouse_press
        self.view.mouseMoveEvent = self.on_mouse_drag
        self.view.mouseReleaseEvent = self.on_mouse_release
        main_layout.addWidget(self.view)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.button_box = QDialogButtonBox()
        self.track_button = self.button_box.addButton("Track & Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        self.reset_button = self.button_box.addButton("Reset Points", QDialogButtonBox.ButtonRole.ResetRole)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        
        self.track_button.clicked.connect(self.start_tracking)
        self.reset_button.clicked.connect(self.reset_landmarks)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

    def _process_initial_frame(self):
        if not self.frames:
            QMessageBox.critical(self, "Error", "No frames loaded from video.")
            self.reject()
            return

        frame = self.frames[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        rects = self.detector(gray, 1)
        if not rects:
            QMessageBox.warning(self, "Error", "No face detected in the first frame.")
            return

        shape = self.predictor(gray, rects[0])
        self.original_landmarks = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)
        self.modified_landmarks = self.original_landmarks.copy()
        self.draw_canvas()

    def draw_canvas(self):
        if self.modified_landmarks is None: return
        self.scene.clear()
        frame = self.frames[self.current_frame_index]
        h, w, _ = frame.shape
        self.scale_factor = min(self.canvas_max_dim / w, self.canvas_max_dim / h, 1.0)
        disp_w, disp_h = int(w * self.scale_factor), int(h * self.scale_factor)

        img_disp = cv2.resize(frame, (disp_w, disp_h))
        q_image = QImage(img_disp.data, disp_w, disp_h, img_disp.strides[0], QImage.Format_RGB888)
        self.scene.addPixmap(QPixmap.fromImage(q_image))
        self.view.setSceneRect(0, 0, disp_w, disp_h)

        pen = QPen(QColor("lime"), 2)
        brush = QBrush(QColor("red"))
        for x, y in self.modified_landmarks:
            cx, cy = x * self.scale_factor, y * self.scale_factor
            self.scene.addEllipse(cx - 3, cy - 3, 6, 6, pen, brush)

    def on_mouse_press(self, event):
        if self.modified_landmarks is None: return
        pos = self.view.mapToScene(event.pos())
        img_x, img_y = pos.x() / self.scale_factor, pos.y() / self.scale_factor
        
        distances = np.linalg.norm(self.modified_landmarks - np.array([img_x, img_y]), axis=1)
        if np.min(distances) < self.landmark_radius:
            self.dragged_landmark_index = np.argmin(distances)

    def on_mouse_drag(self, event):
        if self.dragged_landmark_index is not None:
            pos = self.view.mapToScene(event.pos())
            self.modified_landmarks[self.dragged_landmark_index] = [pos.x() / self.scale_factor, pos.y() / self.scale_factor]
            self.draw_canvas()

    def on_mouse_release(self, event):
        self.dragged_landmark_index = None

    def reset_landmarks(self):
        if self.original_landmarks is not None:
            self.modified_landmarks = self.original_landmarks.copy()
            self.draw_canvas()

    def start_tracking(self):
        self.track_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Tracking landmarks through video...")
        
        self.tracking_thread = LandmarkTrackingThread(self.frames, self.detector, self.predictor, self.modified_landmarks)
        self.tracking_thread.progress_updated.connect(lambda cur, total: self.progress_bar.setValue(int(cur/total*100)))
        self.tracking_thread.tracking_complete.connect(self.on_tracking_complete)
        self.tracking_thread.start()

    def on_tracking_complete(self, tracked_data):
        self.tracked_landmark_data = tracked_data
        QMessageBox.information(self, "Success", f"Landmark tracking complete for {len(tracked_data)} frames.")
        self.accept()
