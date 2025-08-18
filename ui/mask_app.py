import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QCheckBox, QFrame, QProgressBar, QMessageBox, QSizePolicy,QSpinBox
)
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import Qt, QThread, Signal

from PySide6.QtCore import Signal

from PIL import Image, ImageDraw

class MaskEditorWindow(QDialog):
    
    tracking_complete = Signal()
    
    def __init__(self, parent, video_path, controller):
        super().__init__(parent)
        self.parent = parent
        self.video_path = video_path
        self.controller = controller
        self.setWindowTitle("Manual Mask Editor")
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #212529;")
        
        
        self.current_theme = "dark"
        
        
        self.brush_size = 20
        self.is_erasing = False
        
        self.frames = []
        self.bbox_cache = {}
        self.current_frame_index = 0
        self.scale_factor = 1.0
        self.canvas_offset = (0, 0)
        self.is_tracking_complete = False
        
        
        self._ensure_detector_loaded()
        
        self._load_video_frames()
        
        
        self.mask_image = Image.new("L", self.frames[0].size, 0)
        self.draw = ImageDraw.Draw(self.mask_image)
        self.current_mask = np.zeros(self.frames[0].size[::-1], dtype=np.uint8)

        self._create_widgets()
        self._update_canvas()

        
        self.tracking_complete.connect(self._on_tracking_complete)

    def _ensure_detector_loaded(self):
        try:
            if hasattr(self.controller, 'detector') and self.controller.detector:
                
                if not hasattr(self.controller.detector, '_model_loaded') or not self.controller.detector._model_loaded:
                    self.controller.detector.load_model()
            else:
                
                from core.face_detector import FaceDetector
                self.controller.detector = FaceDetector()
                self.controller.detector.load_model()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load face detector model: {e}")
            self.reject()

    def _load_video_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        while True:
            success, frame = cap.read()
            if not success:
                break
            self.frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA"))
        cap.release()
        if not self.frames:
            QMessageBox.critical(self, "Error", "Could not load any frames from the video.")
            self.reject()

    def _create_widgets(self):
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        
        self.canvas = QLabel()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setStyleSheet("background-color: #212529;")
        main_layout.addWidget(self.canvas, 1)
        
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.frames))
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        
        toolbar = QFrame()
        toolbar.setStyleSheet("background-color: #343a40; padding: 5px;")
        toolbar_layout = QHBoxLayout(toolbar)
        
        
        self.prev_button = QPushButton("<")
        self.prev_button.setFixedSize(30, 30)
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self.prev_frame)
        toolbar_layout.addWidget(self.prev_button)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, len(self.frames)-1)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        toolbar_layout.addWidget(self.frame_slider, 1)
        
        self.next_button = QPushButton(">")
        self.next_button.setFixedSize(30, 30)
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.next_frame)
        toolbar_layout.addWidget(self.next_button)
        
        
        self.track_button = QPushButton("Track Mask")
        self.track_button.clicked.connect(self.start_tracking)
        toolbar_layout.addWidget(self.track_button)
        
        
        toolbar_layout.addWidget(QLabel("Brush:"))
        
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(5, 100)
        self.brush_size_spin.setValue(self.brush_size)
        self.brush_size_spin.valueChanged.connect(self.set_brush_size)
        toolbar_layout.addWidget(self.brush_size_spin)
        
        self.eraser_check = QCheckBox("Eraser")
        self.eraser_check.stateChanged.connect(self.set_eraser_mode)
        toolbar_layout.addWidget(self.eraser_check)
        
        
        self.apply_button = QPushButton("Apply Mask")
        self.apply_button.clicked.connect(self.apply_mask)
        toolbar_layout.addWidget(self.apply_button)
        
        main_layout.addWidget(toolbar)
        
        
        self.canvas.setMouseTracking(True)
        self.canvas.mousePressEvent = self._mouse_press
        self.canvas.mouseMoveEvent = self._mouse_move
        self.canvas.mouseReleaseEvent = self._mouse_release

    def resizeEvent(self, event):
        self._update_canvas()

    def _mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.position()
            self._paint(event.position())

    def _mouse_move(self, event):
        if event.buttons() & Qt.LeftButton:
            self._paint(event.position())
            self.last_mouse_pos = event.position()

    def _mouse_release(self, event):
        self.last_mouse_pos = None

    def _paint(self, pos):
        if not self.frames or self.current_frame_index != 0:
            return
            
        x = (pos.x() - self.canvas_offset[0]) / self.scale_factor
        y = (pos.y() - self.canvas_offset[1]) / self.scale_factor
        
        if x < 0 or y < 0 or x >= self.frames[0].width or y >= self.frames[0].height:
            return
            
        size = self.brush_size / self.scale_factor
        color = 0 if self.is_erasing else 255
        
        self.draw.ellipse((x-size, y-size, x+size, y+size), fill=color)
        self._update_canvas()

    def _update_canvas(self):
        if not self.frames:
            return
            
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        if canvas_width <= 1 or canvas_height <= 1:
            return

        bg_image = self.frames[self.current_frame_index]
        img_aspect = bg_image.width / bg_image.height
        canvas_aspect = canvas_width / canvas_height
        
        if img_aspect > canvas_aspect:
            self.scale_factor = canvas_width / bg_image.width
            new_height = int(self.scale_factor * bg_image.height)
            self.canvas_offset = (0, (canvas_height - new_height) // 2)
            resized_bg = bg_image.resize((canvas_width, new_height), Image.Resampling.LANCZOS)
        else:
            self.scale_factor = canvas_height / bg_image.height
            new_width = int(self.scale_factor * bg_image.width)
            self.canvas_offset = ((canvas_width - new_width) // 2, 0)
            resized_bg = bg_image.resize((new_width, canvas_height), Image.Resampling.LANCZOS)

        display_image = resized_bg.copy().convert("RGBA")

        
        final_mask_for_display = Image.new("L", resized_bg.size)

        if self.is_tracking_complete and self.current_frame_index > 0:
            bbox = self._get_bbox_for_current_frame()
            if bbox is not None:
                
                x1, y1, x2, y2 = (
                    int(bbox[0] * self.scale_factor),
                    int(bbox[1] * self.scale_factor),
                    int(bbox[2] * self.scale_factor),
                    int(bbox[3] * self.scale_factor),
                )
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    
                    master_bbox = self.bbox_cache.get(0)
                    if master_bbox is not None:
                        mx1, my1, mx2, my2 = [int(v) for v in master_bbox]
                        mask_template = self.mask_image.crop((mx1, my1, mx2, my2))
                        
                        resized_user_mask = mask_template.resize((w, h), Image.Resampling.NEAREST)
                        
                        final_mask_for_display.paste(resized_user_mask, (x1, y1))
        else:
            final_mask_for_display = self.mask_image.resize(resized_bg.size, Image.Resampling.NEAREST)

        
        overlay = Image.new("RGBA", resized_bg.size, (255, 0, 0, 0))
        alpha_mask = final_mask_for_display.point(lambda p: 100 if p > 0 else 0)
        overlay.putalpha(alpha_mask)
        display_image = Image.alpha_composite(display_image, overlay)

        
        qimage = QImage(
            display_image.tobytes(), 
            display_image.width, 
            display_image.height, 
            QImage.Format_RGBA8888
        )
        pixmap = QPixmap.fromImage(qimage)
        self.canvas.setPixmap(pixmap)

    def _get_bbox_for_current_frame(self):
        if self.current_frame_index in self.bbox_cache:
            return self.bbox_cache[self.current_frame_index]
        return None

    def set_brush_size(self, size):
        self.brush_size = size

    def set_eraser_mode(self, state):
        self.is_erasing = state == Qt.Checked

    

    def start_tracking(self):
        if not np.any(np.array(self.mask_image)):
            QMessageBox.warning(self, "Mask Empty", "Please draw a mask on the first frame before tracking.")
            return

        
        if hasattr(self.parent, 'update_progress'):
            self.parent.update_progress("Finding face under the mask...")
        
        first_frame_cv = cv2.cvtColor(np.array(self.frames[0]), cv2.COLOR_RGBA2BGR)
        all_faces = self.controller.detector.detect_faces(first_frame_cv)

        if not all_faces:
            QMessageBox.critical(self, "Error", "No faces were detected in the first frame.")
            return

        mask_np = np.array(self.mask_image)
        best_face = None
        max_overlap = 0

        for face in all_faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_mask_region = mask_np[y1:y2, x1:x2]
            overlap = np.sum(face_mask_region > 0)
            if overlap > max_overlap:
                max_overlap = overlap
                best_face = face
        
        if best_face is None:
            QMessageBox.warning(self, "No Overlap", "The drawn mask does not overlap with any detected face.")
            return
            
        self.bbox_cache[0] = best_face.bbox
        
        
        if hasattr(self.parent, 'update_progress'):
            self.parent.update_progress(f"Face identified under mask. Starting tracking...")
        
        self.track_button.setEnabled(False)
        self.track_button.setText("Tracking...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.tracking_thread = TrackingThread(
            self.frames,
            self.controller.detector,
            self.progress_bar
        )
        self.tracking_thread.progress_updated.connect(self.update_tracking_progress)
        self.tracking_thread.tracking_complete.connect(self.on_tracking_complete)
        self.tracking_thread.start()

    def update_tracking_progress(self, frame_idx, bbox):
        self.bbox_cache[frame_idx] = bbox
        self.progress_bar.setValue(frame_idx + 1)

    def on_tracking_complete(self):
        self.tracking_complete.emit()

    def _on_tracking_complete(self):
        self.is_tracking_complete = True
        self.track_button.setText("Tracking Complete")
        self.frame_slider.setEnabled(True)
        self.next_button.setEnabled(True)
        self.prev_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._update_canvas()
        QMessageBox.information(self, "Tracking", 
            "Face tracking is complete. You can now scrub the timeline to preview the tracked mask.")

    def seek_frame(self, value):
        self.current_frame_index = value
        self._update_canvas()

    def next_frame(self):
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.frame_slider.setValue(self.current_frame_index)

    def prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.frame_slider.setValue(self.current_frame_index)



    def apply_mask(self):
        
        if not self.is_tracking_complete:
            QMessageBox.warning(self, "Tracking Not Complete", "Please track the mask before applying.")
            return

        
        bbox_frame_0 = self.bbox_cache.get(0)

        
        if bbox_frame_0 is None:
            if hasattr(self.parent, 'update_progress'):
                self.parent.update_progress("Cache for frame 0 was missing, re-detecting...")
            
            first_frame_cv = cv2.cvtColor(np.array(self.frames[0]), cv2.COLOR_RGBA2BGR)
            face = self.controller.detector.get_face(first_frame_cv)
            
            if face and face.bbox is not None:
                bbox_frame_0 = face.bbox
                self.bbox_cache[0] = bbox_frame_0  
            else:
                
                QMessageBox.critical(self, "Detection Failed", "Could not find a face in the first frame to apply the mask to.")
                return

        
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox_frame_0]
            
            
            if x1 >= x2 or y1 >= y2:
                QMessageBox.critical(self, "Error", "The tracked bounding box is invalid (zero area).")
                return

            mask_template = self.mask_image.crop((x1, y1, x2, y2))
            mask_array = np.array(mask_template, dtype=np.uint8)

            
            if hasattr(self.parent, 'set_tracked_mask_data'):
                self.parent.set_tracked_mask_data(mask_array, self.bbox_cache)
                if hasattr(self.parent, 'update_progress'):
                    self.parent.update_progress("✅ Custom mask and tracking data applied.")
            
            
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Mask Crop Error", f"Failed to process the mask with the found bounding box. Error: {e}")
            return

    def get_mask(self):
        if hasattr(self, 'mask_image'):
            return np.array(self.mask_image, dtype=np.uint8)
        return None

    def launch_face_swapper(self):
        from .enhancement import FaceSwapperDialog
        dialog = FaceSwapperDialog(self) 
        result = dialog.exec()           
        
        if result == QDialog.Accepted:

            result_image = dialog.get_result_image()
            if result_image is not None:

                pass

class TrackingThread(QThread):
    progress_updated = Signal(int, object)  
    tracking_complete = Signal()
    
    def __init__(self, frames, detector, progress_bar):
        super().__init__()
        self.frames = frames
        self.detector = detector
        self.progress_bar = progress_bar
        self.cancel_flag = False
        
    def run(self):
        
        try:
            if not hasattr(self.detector, '_model_loaded') or not self.detector._model_loaded:
                self.detector.load_model()
        except Exception as e:
            print(f"Failed to load detector model in tracking thread: {e}")
            self.tracking_complete.emit()
            return
        
        for i, pil_frame in enumerate(self.frames[1:], start=1):
            if self.cancel_flag:
                break
                
            cv_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGBA2BGR)
            try:
                face = self.detector.get_face(cv_frame)
            except Exception as e:
                print(f"Face detection failed in frame {i}: {e}")
                face = None
            bbox = face.bbox if face else None
            self.progress_updated.emit(i, bbox)
            
        self.tracking_complete.emit()
        
    def cancel(self):
        self.cancel_flag = True
