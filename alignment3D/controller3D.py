import cv2
import shutil
from pathlib import Path
import numpy as np
import zipfile


from .core3d.mesh_generator3d import MeshGenerator

class MeshController:

    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.mesh_generator = None  # Initialize as None, will be set in initialize_models
        
        # Try to initialize mesh generator immediately, but don't crash if it fails
        try:
            self.mesh_generator = MeshGenerator(
                config_file="configs/mb1_120x120.yml",
                onnx=True,  
                alpha=0.6,
                out_dir=str(Path(__file__).parent / "3d_results")
            )
            if self.progress_callback:
                self.progress_callback("✅ Mesh generator initialized successfully")
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(f"⚠️ Mesh generator initialization failed: {e}")
            self.mesh_generator = None
        
    def initialize_models(self):
        try:
            self._update_progress("Initializing 3D mesh generation models...")
            
            # If mesh generator is None, try to initialize it
            if self.mesh_generator is None:
                try:
                    self.mesh_generator = MeshGenerator(
                        config_file="configs/mb1_120x120.yml",
                        onnx=True,  
                        alpha=0.6,
                        out_dir=str(Path(__file__).parent / "3d_results")
                    )
                    self._update_progress("✅ 3D Mesh Generator initialized successfully.")
                except Exception as init_error:
                    self._update_progress(f"❌ 3D Mesh Generator initialization failed: {init_error}")
                    self.mesh_generator = None
                    return False
            
            if self.mesh_generator:
                self._update_progress("✅ 3D Mesh Generator is ready.")
                return True
            else:
                self._update_progress("❌ 3D Mesh Generator failed to initialize.")
                return False
        except Exception as e:
            self._update_progress(f"❌ Error during 3D model initialization: {e}")
            return False

    def get_face_data_from_image(self, image_path: str):
        self._update_progress(f"🔍 Detecting face in image: {Path(image_path).name}")
        try:
            # Check if mesh generator is properly initialized
            if not hasattr(self, 'mesh_generator') or self.mesh_generator is None:
                self._update_progress("⚠️ Mesh generator not initialized, using OpenCV fallback")
                face_data = self._detect_faces_opencv(image_path)
            else:
                # First try to use the mesh generator's face detection if available
                if hasattr(self.mesh_generator, 'detect_faces'):
                    try:
                        face_data = self.mesh_generator.detect_faces(str(image_path))
                        if face_data:
                            self._update_progress("✅ Face detected successfully using mesh generator.")
                            return face_data
                    except Exception as mesh_error:
                        self._update_progress(f"⚠️ Mesh generator face detection failed: {mesh_error}")
                
                # Fallback to OpenCV face detection
                face_data = self._detect_faces_opencv(image_path)
            
            if not face_data:
                raise ValueError("No face detected in the image.")
            
            self._update_progress("✅ Face detected successfully using OpenCV.")
            return face_data
        except Exception as e:
            self._update_progress(f"❌ Face detection failed: {e}")
            raise

    def _detect_faces_opencv(self, image_path):
        try:
            self._update_progress("   Using OpenCV as fallback face detector...")
            
            # Try to load the cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if face_cascade.empty():
                self._update_progress(f"❌ Could not load OpenCV cascade classifier from: {cascade_path}")
                return None
            
            # Load the image
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not read image file: {image_path}")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with different parameters for better detection
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                self._update_progress("   No faces detected with OpenCV")
                return None
            
            # Convert to the expected format
            face_data = [{'bbox': f} for f in faces]
            self._update_progress(f"   OpenCV detected {len(faces)} face(s)")
            return face_data
            
        except Exception as e:
            self._update_progress(f"❌ OpenCV face detection fallback failed: {e}")
            return None

    def _update_progress(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)

    def generate_3d_mesh(self, image_path: str) -> str:
        self._update_progress("🔍 Generating 3D mesh...")
        try:
            # Check if mesh generator is available
            if self.mesh_generator is None:
                raise RuntimeError("3D Mesh Generator is not initialized. Please wait for model initialization to complete.")
            
            mesh_path = self.mesh_generator.generate(str(image_path))
            self._update_progress(f"✅ 3D mesh saved to: {mesh_path}")
            return mesh_path
        except Exception as e:
            self._update_progress(f"❌ 3D mesh generation failed: {e}")
            raise

    

    def _process_and_merge_frames(self, video_path: str, export_maps: bool, onnx: bool, alpha: float, output_video_path: str, export_dir: str):
        self._update_progress("🎥 Extracting video frames...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_frames_dir = Path(export_dir) / "temp_frames"
        temp_frames_dir.mkdir(parents=True, exist_ok=True)
        
        processed_frames_paths = []
        frame_num = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            temp_frame_path = temp_frames_dir / f"frame_{frame_num:06d}.jpg"
            cv2.imwrite(str(temp_frame_path), frame)

            try:
                # Generate 3D overlay for this frame
                processed_frame_path = self.mesh_generator.generate(str(temp_frame_path))
                
                                # Verify the processed frame exists and is readable
                if Path(processed_frame_path).is_file():
                    # Test if we can read the image
                    test_img = cv2.imread(processed_frame_path)
                    if test_img is not None and test_img.size > 0:
                        processed_frames_paths.append(processed_frame_path)
                        self._update_progress(f"✅ Processed frame {frame_num} successfully.")
                    else:
                        self._update_progress(f"❌ Warning: Processed frame {frame_num} is corrupted. Using original frame.")
                        processed_frames_paths.append(str(temp_frame_path))
                else:
                    self._update_progress(f"❌ Warning: Processed frame {frame_num} not found. Using original frame.")
                    processed_frames_paths.append(str(temp_frame_path))

            except Exception as e:
                self._update_progress(f"❌ Warning: Failed to process frame {frame_num}: {e}. Using original frame.")
                processed_frames_paths.append(str(temp_frame_path))

            # Clean up temp frame
            try:
                temp_frame_path.unlink() 
            except FileNotFoundError:
                pass
                
            frame_num += 1
            if (frame_num) % 30 == 0:
                self._update_progress(f"Processed {frame_num} / {frame_count} frames...")

        cap.release()
        
        if not processed_frames_paths:
            raise Exception("No frames were processed successfully.")

        self._update_progress("Merging processed frames back into a video...")

        final_video_path = output_video_path if output_video_path else Path(export_dir) / f"{Path(video_path).stem}_3d_overlay.mp4"
        
        # Use H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(final_video_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to create video writer for {final_video_path}")
        
        frames_written = 0
        for path in processed_frames_paths:
            try:
                frame = cv2.imread(path)
                if frame is not None and frame.size > 0:
                    # Ensure frame dimensions match video dimensions
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
                    frames_written += 1
                else:
                    self._update_progress(f"❌ Warning: Could not read frame from {Path(path).name}")
            except Exception as e:
                self._update_progress(f"❌ Warning: Error writing frame {Path(path).name}: {e}")
            
            # Clean up processed frame
            try:
                Path(path).unlink() 
            except FileNotFoundError:
                pass
            
        out.release()
        
        # Clean up temp directory
        try:
            shutil.rmtree(temp_frames_dir) 
        except Exception as e:
            self._update_progress(f"⚠️ Warning: Could not clean up temp directory: {e}")

        if frames_written == 0:
            raise RuntimeError("No frames were successfully written to the video")
            
        self._update_progress(f"✅ Video created with {frames_written} frames")

        return str(final_video_path)



    
    def export_3d_obj(self, image_path: str) -> str:
        self._update_progress("🔄 Exporting OBJ file...")
        try:
            # Check if mesh generator is available
            if self.mesh_generator is None:
                raise RuntimeError("3D Mesh Generator is not initialized. Please wait for model initialization to complete.")
            
            obj_path = self.mesh_generator.export_obj(str(image_path))
            self._update_progress(f"✅ OBJ file saved to: {obj_path}")
            return obj_path
        except Exception as e:
            self._update_progress(f"❌ OBJ export failed: {e}")
            raise

    def generate_3d_maps(self, image_path: str):
        self._update_progress("🔄 Generating 3D maps (texture, depth, PNCC)...")
        try:
            # Check if mesh generator is available
            if self.mesh_generator is None:
                raise RuntimeError("3D Mesh Generator is not initialized. Please wait for model initialization to complete.")
            
            tex_path, depth_path, pncc_path = self.mesh_generator.generate_3d_maps(str(image_path))
            self._update_progress(f"✅ 3D maps saved to: {tex_path}, {depth_path}, {pncc_path}")
            return tex_path, depth_path, pncc_path
        except Exception as e:
            self._update_progress(f"❌ 3D map generation failed: {e}")
            raise

    def generate_uv_texture(self, image_path: str, output_dir: str = None) -> str:
        self._update_progress("🔄 Generating UV texture...")
        try:
            # Check if mesh generator is available
            if self.mesh_generator is None:
                raise RuntimeError("3D Mesh Generator is not initialized. Please wait for model initialization to complete.")
            
            uv_tex_path = self.mesh_generator.generate_uv_texture(str(image_path), output_dir)
            self._update_progress(f"✅ UV texture saved to: {uv_tex_path}")
            return uv_tex_path
        except Exception as e:
            self._update_progress(f"❌ UV texture generation failed: {e}")
            raise


    def generate_video_overlay(self, video_path: str, output_dir: str = None,
                           onnx=True, alpha=0.6) -> str:
        self._update_progress("🎬 Processing video for 3D mesh overlay...")
        
        try:
            # Check if mesh generator is available
            if self.mesh_generator is None:
                raise RuntimeError("3D Mesh Generator is not initialized. Please wait for model initialization to complete.")
            
            # Use the more robust overlay_mesh_on_video method
            output_fp = output_dir and str(Path(output_dir) / f"{Path(video_path).stem}_3d_overlay.mp4")
            
            output_video = self.mesh_generator.overlay_mesh_on_video(
                video_fp=video_path,
                output_fp=output_fp,
                export_maps=False,
                export_dir=None,
                onnx=onnx,
                alpha=alpha
            )
            
            self._update_progress(f"✅ Overlay video generated: {output_video}")
            return output_video
        except Exception as e:
            self._update_progress(f"❌ Overlay generation failed: {e}")
            # Fallback to frame-by-frame processing if the direct method fails
            self._update_progress("🔄 Trying fallback frame-by-frame processing...")
            try:
                output_fp = output_dir and str(Path(output_dir) / f"{Path(video_path).stem}_3d_overlay.mp4")
                
                output_video = self._process_and_merge_frames(
                    video_path=video_path,
                    export_maps=False,
                    onnx=onnx,
                    alpha=alpha,
                    output_video_path=output_fp,
                    export_dir=output_dir or str(Path(self.mesh_generator.out_dir) / "temp")
                )
                
                self._update_progress(f"✅ Overlay video generated (fallback): {output_video}")
                return output_video
            except Exception as fallback_error:
                self._update_progress(f"❌ Fallback processing also failed: {fallback_error}")
                raise
    

    def generate_video_maps(self, video_path: str, output_dir: str = None, **kwargs) -> dict:
        self._update_progress("🗺️ Generating per-frame maps...")
        base = Path(video_path).stem
        root = Path(output_dir) / f"{base}_maps" if output_dir else Path(self.mesh_generator.out_dir) / f"{base}_maps"


        subfolders = {s: root / s for s in ("depth", "texture", "pncc")}
        for folder in subfolders.values():
            folder.mkdir(parents=True, exist_ok=True)


        self.mesh_generator.overlay_mesh_on_video(
            video_fp=video_path,
            export_maps=True,
            export_dir=str(root),
            onnx=kwargs.get("onnx", True),
            alpha=kwargs.get("alpha", self.mesh_generator.alpha)
        )

        
        moved = set()
        for file in root.iterdir():
            if not file.is_file():
                continue
            for suffix, subdir in subfolders.items():
                if file.stem.endswith(f"_{suffix}"):
                    shutil.move(str(file), str(subdir / file.name))
                    moved.add(file.name)
                    break

        
        for file in root.iterdir():
            if file.is_file() and file.name not in moved:
                file.unlink()

        
        results = {s: sorted(str(p) for p in subdir.glob("*")) for s, subdir in subfolders.items()}
        for suffix, files in results.items():
            self._update_progress(f"✅ {len(files)} '{suffix}' files organized in: {subfolders[suffix]}")
        return results

    def generate_video_uv_textures(self, video_path: str, output_dir: str = None, **kwargs) -> dict:
        self._update_progress("🗺️ Generating per-frame UV textures...")
        base = Path(video_path).stem
        root = Path(output_dir) / f"{base}_uv_maps" if output_dir else Path(self.mesh_generator.out_dir) / f"{base}_uv_maps"

        # Create UV folder
        uv_folder = root / "uv"
        uv_folder.mkdir(parents=True, exist_ok=True)

        # Process video and generate UV textures
        self.mesh_generator.overlay_mesh_on_video(
            video_fp=video_path,
            export_uv_maps=True,
            export_dir=str(root),
            onnx=kwargs.get("onnx", True),
            alpha=kwargs.get("alpha", self.mesh_generator.alpha)
        )

        # Move UV files to uv subfolder
        moved = set()
        for file in root.iterdir():
            if not file.is_file():
                continue
            if file.stem.endswith("_uv"):
                shutil.move(str(file), str(uv_folder / file.name))
                moved.add(file.name)

        # Clean up any remaining files
        for file in root.iterdir():
            if file.is_file() and file.name not in moved:
                file.unlink()

        # Get results
        uv_files = sorted(str(p) for p in uv_folder.glob("*"))
        self._update_progress(f"✅ {len(uv_files)} 'uv' files organized in: {uv_folder}")
        
        return {"uv": uv_files}
    
    def generate_video_overlay_and_maps(self, video_path: str, output_dir: str = None,
                                    onnx=True, alpha=0.6) -> tuple:
        self._update_progress("🎬🗺️ Generating overlay AND per-frame maps...")

        base = Path(video_path).stem
        maps_root = (Path(output_dir) / f"{base}_maps") if output_dir else (Path(self.mesh_generator.out_dir) / f"{base}_maps")
        maps_root.mkdir(parents=True, exist_ok=True)

        
        output_vid = self.mesh_generator.overlay_mesh_on_video(
            video_fp=video_path,
            output_fp=(str(Path(output_dir) / f"{base}_3d_overlay.mp4") if output_dir else None),
            export_maps=True,
            export_dir=str(maps_root),
            onnx=onnx,
            alpha=alpha
        )

        self._update_progress(f"✅ Overlay video saved to: {output_vid}")

        
        subfolders = {s: maps_root / s for s in ("depth", "texture", "pncc")}
        for folder in subfolders.values():
            folder.mkdir(parents=True, exist_ok=True)

        moved = set()
        for file in maps_root.iterdir():
            if file.is_file():
                for suffix, subdir in subfolders.items():
                    if file.stem.endswith(f"_{suffix}"):
                        shutil.move(str(file), str(subdir / file.name))
                        moved.add(file.name)
                        break

        
        for file in maps_root.iterdir():
            if file.is_file() and file.name not in moved:
                file.unlink()

        self._update_progress("✅ Maps organized into subfolders.")

        return output_vid, str(maps_root)
    


