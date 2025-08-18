import insightface
import numpy as np
from pathlib import Path
import cv2
import os
from typing import Optional, List, Tuple
from insightface.app import FaceAnalysis
import logging
from ui import enhancement
# from ui.ui_logic import EnhancedMainWindow

# blend_value = EnhancedMainWindow._collect_processing_settings()["edge_softness"].values[0]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceSwapperModel:
    def __init__(self):
        self.model = None
        self._model_loaded = False
        
    def _find_model_path(self) -> str:
        possible_paths = [
            Path("models/inswapper_128.onnx"),
            Path("Models/inswapper_128.onnx"),
            Path("model/inswapper_128.onnx"),
            
            Path(__file__).parent / "models" / "inswapper_128.onnx",
            Path(__file__).parent / "Models" / "inswapper_128.onnx",
            Path(__file__).parent.parent / "models" / "inswapper_128.onnx",
            Path(__file__).parent.parent / "Models" / "inswapper_128.onnx",
            
            Path.home() / ".insightface" / "models" / "inswapper_128.onnx",
            Path.home() / "insightface" / "models" / "inswapper_128.onnx",
            
            Path("/usr/local/share/insightface/models/inswapper_128.onnx"),
            Path("/opt/insightface/models/inswapper_128.onnx"),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Model found at: {path}")
                return str(path)
        
        error_msg = (
            f"inswapper_128.onnx model not found in any of these locations:\n" +
            "\n".join(f"  - {path}" for path in possible_paths) +
            "\n\nPlease ensure the model file exists in the 'models' directory.\n"
            "You can download the model from: https://github.com/deepinsight/insightface/releases"
        )
        raise FileNotFoundError(error_msg)
        
    def load_model(self) -> bool:
        try:
            if self._model_loaded:
                logger.info("Model already loaded")
                return True
                
            model_path = self._find_model_path()
            logger.info(f"Loading face swap model from: {model_path}")
            
            self.model = insightface.model_zoo.get_model(model_path)
            self._model_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def swap(self, target_img: np.ndarray, source_face, target_face) -> np.ndarray:
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        
        try:
            return self.model.get(target_img, target_face, source_face, paste_back=True)
        except Exception as e:
            logger.error(f"Error swapping face: {e}")
            return target_img


class FaceProcessor:
    def __init__(self, gpu_ctx_id: int = 0, det_size: Tuple[int, int] = (640, 640)):
        self.gpu_ctx_id = gpu_ctx_id
        self.det_size = det_size
        self.app = None
        self.face_swapper = None
        self._initialized = False
        
        self._initialize()

    def _initialize(self) -> bool:
        try:
            logger.info("Initializing FaceProcessor...")
            
            self.app = FaceAnalysis(
                name='buffalo_l', 
                allowed_modules=['detection', 'recognition']
            )
            self.app.prepare(ctx_id=self.gpu_ctx_id, det_size=self.det_size)
            
            swapper_model = FaceSwapperModel()
            if swapper_model.load_model():
                self.face_swapper = swapper_model.model
                self._initialized = True
                logger.info("FaceProcessor initialized successfully")
                return True
            else:
                raise RuntimeError("Failed to load face swapper model")
                
        except Exception as e:
            logger.error(f"Error initializing FaceProcessor: {e}")
            if self.gpu_ctx_id >= 0:
                logger.info("Trying with CPU...")
                self.gpu_ctx_id = -1
                return self._initialize()
            return False

    def get_source_face(self, image: np.ndarray, face_index: int = 0):
        if not self._initialized or self.app is None:
            logger.error("FaceProcessor not initialized")
            return None
            
        try:
            faces = self.app.get(image)
            if not faces:
                logger.warning("No face found in source image")
                return None
            
            faces_sorted = sorted(
                faces, 
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )
            
            if face_index >= len(faces_sorted):
                logger.warning(f"Face index {face_index} not available. Available count: {len(faces_sorted)}")
                face_index = 0
            
            selected_face = faces_sorted[face_index]
            logger.info(f"Source face analyzed and stored (index: {face_index})")
            return selected_face
            
        except Exception as e:
            logger.error(f"Error extracting source face: {e}")
            return None

    def process_frame(self, 
                     target_frame: np.ndarray, 
                     source_face, 
                     alignment_enabled: bool = True, 
                     blend_amount: int = 90, 
                     edge_softness: int = 5,
                     quality_mode: bool = True) -> np.ndarray:
        if not self._initialized or source_face is None:
            return target_frame

        try:
            target_faces = self.app.get(target_frame)
            if not target_faces:
                return target_frame

            result_frame = target_frame.copy()

            for target_face in target_faces:
                result_frame = self._swap_single_face(
                    result_frame, target_frame, target_face, source_face,
                    alignment_enabled, blend_amount, edge_softness, quality_mode
                )

            return result_frame

        except Exception as e:
            logger.error(f"Error during frame processing: {e}")
            return target_frame

    def _swap_single_face(self, 
                         result_frame: np.ndarray,
                         original_frame: np.ndarray,
                         target_face, 
                         source_face,
                         alignment_enabled: bool,
                         blend_amount: int,
                         edge_softness: int,
                         quality_mode: bool) -> np.ndarray:
        
        if not alignment_enabled:
            result_frame = self.face_swapper.get(
                result_frame, target_face, source_face, paste_back=True
            )
            
            if blend_amount < 100:
                alpha = blend_amount / 100.0
                result_frame = cv2.addWeighted(
                    result_frame, alpha, original_frame, 1 - alpha, 0
                )
        else:
            swapped_region = self.face_swapper.get(
                original_frame, target_face, source_face, paste_back=False
            )
            
            mask = self._create_face_mask(target_face, original_frame.shape[:2], edge_softness)
            
            bbox = target_face.bbox.astype(np.int32)
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            if quality_mode:
                result_frame = cv2.seamlessClone(
                    swapped_region, result_frame, mask, center, cv2.NORMAL_CLONE
                )
            else:
                mask_norm = mask.astype(np.float32) / 255.0
                mask_3d = np.expand_dims(mask_norm, axis=2)
                result_frame = (swapped_region * mask_3d + 
                              result_frame * (1 - mask_3d)).astype(np.uint8)
        
        return result_frame

    def _create_face_mask(self, face, frame_shape: Tuple[int, int], softness: int) -> np.ndarray:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        
        if hasattr(face, 'kps') and face.kps is not None:
            kps = face.kps.astype(np.int32)
            hull = cv2.convexHull(kps)
            cv2.fillConvexPoly(mask, hull, 255)
        else:
            bbox = face.bbox.astype(np.int32)
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 255, -1)
        
        if softness > 0:
            blur_amount = max(1, int(softness / 2) * 2 + 1)
            mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
        
        return mask

    def get_face_count(self, image: np.ndarray) -> int:
        if not self._initialized:
            return 0
        
        try:
            faces = self.app.get(image)
            return len(faces)
        except Exception:
            return 0

    def is_initialized(self) -> bool:
        return self._initialized


class HaarFaceDetector:
    
    
    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        else:
            self.cascade_path = cascade_path

        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Error: Could not load Haar cascade from: {self.cascade_path}")

    def detect_faces(self, image_frame: np.ndarray, 
                    scale_factor: float = 1.1,
                    min_neighbors: int = 5,
                    min_size: Tuple[int, int] = (30, 30)) -> List[Tuple[int, int, int, int]]:
        if image_frame is None:
            return []
        
        try:
            gray_image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size
            )
            return faces.tolist()
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

