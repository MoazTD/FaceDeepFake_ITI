import insightface
import numpy as np

class FaceDetector:
    """A wrapper class for the insightface face detection model."""
    def __init__(self):
        self.model = None

    def load_model(self):
        """Loads the face analysis model."""
        self.model = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=(640, 640))

    def get_face(self, img: np.ndarray):
        """
        Detects faces in an image and returns the largest one.
        
        Returns:
            The largest face object or None if no face is detected.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        faces = self.model.get(img)
        if not faces:
            return None
        
        # Return the face with the largest bounding box area
        return sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[-1]
