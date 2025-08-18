import insightface
import numpy as np
import os
import cv2
from pathlib import Path

class FaceDetector:
    """A wrapper class for the insightface face detection model."""
    def __init__(self):
        self.model = None
        self._setup_model_path()
        self.load_model()

    def _setup_model_path(self):
        """Set up the model path for insightface models."""
        possible_model_dirs = [
            Path("models"),
            Path("Models"),
            Path(__file__).parent.parent / "models",
            Path(__file__).parent.parent / "Models",
        ]
        
        for model_dir in possible_model_dirs:
            if model_dir.exists() and (model_dir / "buffalo_l").exists():
                os.environ['INSIGHTFACE_HOME'] = str(model_dir.parent)
                print(f"Using insightface models from: {model_dir}")
                return True
        
        print("Using default insightface model location (will download if needed)")
        return False

    def load_model(self):
        """Loads the face analysis model."""
        try:
            self.model = insightface.app.FaceAnalysis(
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            print("Face detection model loaded successfully")
        except Exception as e:
            print(f"Error loading face detection model: {e}")
            raise
        
    def is_model_loaded(self):
        """Check if the model is loaded."""
        return self.model is not None
    
    def get_face(self, img: np.ndarray):
        """
        Detects faces in an image and returns the largest one.
        
        Returns:
            The largest face object or None if no face is detected.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        try:
            faces = self.model.get(img)
            if not faces:
                return None
            return sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return None

    def detect_faces(self, img: np.ndarray):
        """
        Detects all faces in an image and returns them as a list.
        
        Returns:
            List of face objects or empty list if no faces are detected.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        try:
            faces = self.model.get(img)
            if not faces:
                return []
            return sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []

    def draw_faces(self, image_frame, faces):
        """
        Draws rectangles around detected faces on the image frame.
        """
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(image_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image_frame