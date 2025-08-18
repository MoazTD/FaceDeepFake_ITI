import threading
from pathlib import Path
import cv2
from PIL import Image
from typing import Dict, Any, Callable, Optional, Union
import logging
from PySide6.QtCore import QObject, Signal
import shutil
from core import video_utils
import numpy as np 
from core import processors

from core.controller import FaceSwapController
from core.face_detector import FaceDetector

logger = logging.getLogger(__name__)


class FaceSwapProcessor(QObject):
    
    progress_updated = Signal(str)
    
    
    def __init__(self):
        super().__init__()

        self.controller = FaceSwapController(processor_instance=self)

        self.is_initialized = False
        self.is_processing = False
        self._processing_lock = threading.Lock()
        
        self.default_options = {
            'mode': 'default',
            'color_correction': True,
            'alpha': 0.8,
            'softness': 15.0,
            'selected_face_index': -1
        }

        self.detector = FaceDetector()
        
        try:
            self.detector.load_model()
        except Exception as e:
            logger.warning(f"Failed to load detector model during initialization: {e}")
        
        self.custom_mask = None
        self.current_mask = None
    
    def initialize_models(self) -> bool:
        try:
            self.progress_updated.emit("Initializing face swap models...")
            
            success = self.controller.initialize_models()
            
            if success:
                self.is_initialized = True
                self.progress_updated.emit("Models initialized successfully!")
            else:
                self.progress_updated.emit("Failed to initialize models")
                
            return success
        except Exception as e:
            self.progress_updated.emit(f"Error initializing models: {str(e)}")
            logger.error(f"Model initialization error: {e}")
            return False
        
    def initialize_models_async(self, completion_callback: Optional[Callable[[bool], None]] = None):
        def init_thread():
            success = self.initialize_models()
            if completion_callback:
                completion_callback(success)
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def validate_inputs(self, source_path: str, target_path: str, output_path: str) -> Dict[str, Any]:
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not Path(source_path).exists():
            result['valid'] = False
            result['errors'].append(f"Source image not found: {source_path}")
        
        if not Path(target_path).exists():
            result['valid'] = False
            result['errors'].append(f"Target video not found: {target_path}")
        
        output_dir = Path(output_path).parent
        if not output_dir.exists():
            result['valid'] = False
            result['errors'].append(f"Output directory not found: {output_dir}")
        
        source_ext = Path(source_path).suffix.lower()
        if source_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            result['valid'] = False
            result['errors'].append(f"Unsupported source image format: {source_ext}")
        
        target_ext = Path(target_path).suffix.lower()
        if target_ext not in ['.mp4', '.mov', '.avi', '.mkv']:
            result['valid'] = False
            result['errors'].append(f"Unsupported target video format: {target_ext}")
        
        return result
    
    def get_media_info(self, file_path: str) -> Dict[str, Any]:
        info = {
            'exists': False,
            'type': None,
            'dimensions': None,
            'duration': None,
            'fps': None,
            'frame_count': None,
            'error': None
        }
        
        try:
            path = Path(file_path)
            if not path.exists():
                info['error'] = "File not found"
                return info
            
            info['exists'] = True
            ext = path.suffix.lower()
            
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                info['type'] = 'image'
                img = Image.open(file_path)
                info['dimensions'] = img.size
                img.close()
            
            elif ext in ['.mp4', '.mov', '.avi', '.mkv']:
                info['type'] = 'video'
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    info['dimensions'] = (width, height)
                    info['fps'] = fps
                    info['frame_count'] = frame_count
                    info['duration'] = frame_count / fps if fps > 0 else None
                cap.release()
            
        except Exception as e:
            info['error'] = str(e)
            logger.error(f"Error getting media info: {e}")
        
        return info
    
    def create_thumbnail(self, file_path: str, max_size: tuple = (200, 200)) -> Optional[Image.Image]:
        try:
            media_info = self.get_media_info(file_path)
            
            if not media_info['exists']:
                return None
            
            if media_info['type'] == 'image':
                img = Image.open(file_path)
            elif media_info['type'] == 'video':
                cap = cv2.VideoCapture(file_path)
                success, frame = cap.read()
                cap.release()
                if not success:
                    return None
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return None
            
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return img
            
        except Exception as e:
            self.progress_callback(f"Error creating thumbnail: {str(e)}")
            logger.error(f"Thumbnail creation error: {e}")
            return None
    
    def process_video(self, source_path: str, target_path: str, output_path: str,
                    options: Optional[Dict[str, Any]] = None) -> bool:
        """
        Process a video with face swapping.
        """
        with self._processing_lock:
            if self.is_processing:
                self.progress_updated.emit("Error: Already processing another video")
                return False
            self.is_processing = True

        try:
            if not self.is_initialized:
                self.progress_updated.emit("Error: Models not initialized")
                return False
            validation = self.validate_inputs(source_path, target_path, output_path)
            if not validation['valid']:
                for error in validation['errors']:

                    self.progress_updated.emit(f"Error: {error}")
                return False

            final_options = {**self.default_options, **(options or {})}

            video_info = self.get_media_info(target_path)
            if video_info.get('frame_count', 0) > 0:

                self.progress_updated.emit(f"Video info: {video_info['frame_count']} frames @ {video_info['fps']:.2f} FPS")

            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.controller.process_video(source_path, target_path, output_path, final_options)

            return True 

        except Exception as e:

            self.progress_updated.emit(f"Error during processing: {str(e)}")
            logger.error(f"Processing error: {e}", exc_info=True) 
            
            return False
        finally:
            self.is_processing = False
        
    def get_processing_status(self) -> Dict[str, Any]:
        return {
            'initialized': self.is_initialized,
            'processing': self.is_processing,
            'ready': self.is_initialized and not self.is_processing
        }
    
    def set_default_options(self, options: Dict[str, Any]):
        self.default_options.update(options)
    
    def get_supported_formats(self) -> Dict[str, list]:
        return {
            'images': ['.jpg', '.jpeg', '.png', '.bmp'],
            'videos': ['.mp4', '.mov', '.avi', '.mkv']
        }