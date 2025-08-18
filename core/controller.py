import cv2
import shutil
from pathlib import Path
import numpy as np
import gc
import logging
from core.face_detector import FaceDetector
from core.face_swapper import FaceSwapperModel
from core import video_utils, processors
        
logger = logging.getLogger(__name__)

class FaceSwapController:
    def __init__(self, processor_instance=None):
        self.processor = processor_instance
        self.detector = FaceDetector()
        self.swapper = FaceSwapperModel()
        self.custom_mask = None
        self.current_mask = None
        self.tracked_data = None

    def initialize_models(self) -> bool:
        self.processor.progress_updated.emit("🚀 Initializing models...")
        try:
            self.detector.load_model()
            self.swapper.load_model()
            self.processor.progress_updated.emit("✅ Models initialized successfully.")
            return True
        except Exception as e:
            self.processor.progress_updated.emit(f"❌ Error initializing models: {e}")
            return False

    def get_face_data_from_image(self, image_path: str):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        return self.detector.get_face(image)

    def get_face_data_from_video_frame(self, video_path: str):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video at {video_path}")
        success, frame = cap.read()
        cap.release()
        if not success:
            raise IOError(f"Could not read first frame from {video_path}")
        return self.detector.get_face(frame)

    def process_video(self,
                      source_img_path: str,
                      target_vid_path: str,
                      output_path: str,
                      options: dict):
        try:
            self.processor.progress_updated.emit("🖼️ Processing source image...")
            source_img = cv2.imread(str(source_img_path))
            if source_img is None:
                raise FileNotFoundError("Cannot load source image.")
            source_face = self.detector.get_face(source_img)
            if source_face is None:
                raise ValueError("No face detected in source image.")
            self.processor.progress_updated.emit("👍 Source face processed.")

            source_face_embedding = options.get('source_face_embedding')
            selected_target_face_embedding = options.get('selected_target_face_embedding')

            if source_face_embedding is None or selected_target_face_embedding is None:
                self.processor.progress_updated.emit("⚠️ Face embeddings not available. Swapping will be based on the selected face index only.")

            temp_dir = Path(output_path).parent / "temp_frames_processing"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            original_frames_dir = temp_dir / "original"
            swapped_frames_dir = temp_dir / "swapped"
            original_frames_dir.mkdir(parents=True)
            swapped_frames_dir.mkdir(parents=True)

            self.processor.progress_updated.emit("🎵 Checking for audio track...")
            temp_audio_path = temp_dir / "original_audio.aac"
            has_audio = video_utils.extract_audio(
                target_vid_path,
                temp_audio_path,
                self.processor.progress_updated.emit
            )
            audio_path_to_use = temp_audio_path if has_audio else None

            fps, frame_count = video_utils.extract_frames(
                target_vid_path,
                original_frames_dir,
                self.processor.progress_updated.emit
            )

            self.processor.progress_updated.emit("🎭 Performing face swap...")
            frame_paths = sorted(original_frames_dir.glob("*.jpg"))

            custom_mask = options.get("custom_mask")
            tracked_data = options.get('tracked_mask_data')

            for i, frame_path in enumerate(frame_paths):
                target_frame = cv2.imread(str(frame_path))
                if target_frame is None:
                    continue

                final_frame = target_frame.copy()
                target_face = None

                # Use tracked data if available and the current frame has a landmark set
                if tracked_data and i in tracked_data:
                    # Find the face that corresponds to the tracked landmarks for swapping
                    all_faces_in_frame = self.detector.detect_faces(target_frame)
                    if all_faces_in_frame:
                        # Simple logic: find the closest face to the tracked landmarks
                        tracked_landmarks = tracked_data[i]
                        tracked_bbox_center = np.mean(tracked_landmarks, axis=0)
                        min_dist = float('inf')
                        for face_in_frame in all_faces_in_frame:
                            face_center = np.array([(face_in_frame.bbox[0] + face_in_frame.bbox[2]) / 2, (face_in_frame.bbox[1] + face_in_frame.bbox[3]) / 2])
                            dist = np.linalg.norm(tracked_bbox_center - face_center)
                            if dist < min_dist:
                                min_dist = dist
                                target_face = face_in_frame
                else:
                    # Fallback to standard detection and selection if no tracked data
                    all_faces = self.detector.detect_faces(target_frame)
                    if all_faces:
                        if source_face_embedding is not None and selected_target_face_embedding is not None:
                            for face_in_frame in all_faces:
                                target_embedding = face_in_frame.normed_embedding
                                similarity = np.dot(selected_target_face_embedding, target_embedding)
                                if similarity > 0.5:
                                    target_face = face_in_frame
                                    break
                        else:
                            selected_index = options.get('selected_face_index', 0)
                            if all_faces and len(all_faces) > selected_index:
                                target_face = all_faces[selected_index]
                
                if target_face:
                    swapped_face_img = self.swapper.swap(
                        target_frame, source_face, target_face
                    )
                    if custom_mask is not None and hasattr(processors, 'manual_blend'):
                        final_frame = processors.manual_blend(
                            original=target_frame,
                            swapped=swapped_face_img,
                            bbox=target_face.bbox,
                            alpha=options.get("blend_alpha", 0.9),
                            softness=options.get("edge_softness", 40),
                            custom_mask=custom_mask
                        )
                    else:
                        final_frame = swapped_face_img

                cv2.imwrite(str(swapped_frames_dir / frame_path.name), final_frame)
                if (i + 1) % 30 == 0:
                    self.processor.progress_updated.emit(f"🎭 Swapped frame {i + 1}/{len(frame_paths)}")

            video_utils.reconstruct_video(
                frames_dir=swapped_frames_dir,
                output_path=output_path,
                fps=fps,
                audio_path=audio_path_to_use,
                progress_callback=self.processor.progress_updated.emit
            )

            self.processor.progress_updated.emit("🧹 Cleaning up temporary files...")
            shutil.rmtree(temp_dir)
            self.processor.progress_updated.emit(f"\n✅ Face swap complete! Video saved to:\n'{output_path}'")
        except Exception as e:
            self.processor.progress_updated.emit(f"❌ An error occurred: {e}")
            if 'temp_dir' in locals() and temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise