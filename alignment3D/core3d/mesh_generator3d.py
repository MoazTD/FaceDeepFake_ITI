# File: alignment3D/core3d/mesh_generator3d.py

import sys
import os
import cv2
import yaml
from pathlib import Path
import numpy as np

# 1) Compute 3DDFA_V2 root dynamically
try:
    THREEDFAV2_ROOT = Path(__file__).parents[2] / "DDFA_V2"
    if not THREEDFAV2_ROOT.exists():
        raise FileNotFoundError(f"DDFA_V2 folder not found at {THREEDFAV2_ROOT}")
    sys.path.insert(0, str(THREEDFAV2_ROOT))
except:
    THREEDFAV2_ROOT = Path(__file__).parents[1] / "DDFA_V2"
    if not THREEDFAV2_ROOT.exists():
        raise FileNotFoundError(f"DDFA_V2 folder not found at {THREEDFAV2_ROOT}")
    sys.path.insert(0, str(THREEDFAV2_ROOT))

print(f"DDFA_V2 root: {THREEDFAV2_ROOT}")

# 2) Now import the 3DDFA_V2 pipeline bits
from DDFA_V2.FaceBoxes.FaceBoxes import FaceBoxes
from DDFA_V2.TDDFA import TDDFA
from DDFA_V2.utils.render import render
from DDFA_V2.utils.depth import depth
from DDFA_V2.utils.pncc import pncc
from DDFA_V2.utils.serialization import ser_to_obj


class MeshGenerator:
    """
    Wraps the 3DDFA_V2 pipeline to:
      1) detect a face
      2) reconstruct a dense 3D mesh
      3) render it as a semi-transparent overlay
      4) save to disk, returning the saved JPEG path
    """

    def __init__(self,
                 config_file: str = "configs/mb1_120x120.yml",
                 onnx: bool = False,
                 alpha: float = 0.6,
                 out_dir: str = None):
        # 1) Load config
        cfg_path = THREEDFAV2_ROOT / config_file
        cfg = yaml.load(open(cfg_path, "rb"), Loader=yaml.SafeLoader)
        # Patch bfm_fp to absolute path if needed
        if 'bfm_fp' in cfg and not os.path.isabs(cfg['bfm_fp']):
            cfg['bfm_fp'] = str(THREEDFAV2_ROOT / cfg['bfm_fp'])
        # Patch checkpoint_fp to absolute path if needed
        if 'checkpoint_fp' in cfg and not os.path.isabs(cfg['checkpoint_fp']):
            cfg['checkpoint_fp'] = str(THREEDFAV2_ROOT / cfg['checkpoint_fp'])

        # 2) Init FaceBoxes + TDDFA
        # Try different initialization approaches based on ONNX flag
        if onnx:
            try:
                # Try ONNX version first if available
                from DDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
                from DDFA_V2.TDDFA_ONNX import TDDFA_ONNX
                self.face_boxes = FaceBoxes_ONNX()
                self.tddfa = TDDFA_ONNX(**cfg)
            except ImportError:
                # Fallback to regular versions
                self.face_boxes = FaceBoxes()
                self.tddfa = TDDFA(gpu_mode=False, **cfg)
        else:
            self.face_boxes = FaceBoxes()
            self.tddfa = TDDFA(gpu_mode=not onnx, **cfg)
            
        self.tri   = self.tddfa.tri  # triangle list for render()

        self.alpha = alpha
        # Default output dir next to InsightFaceTest
        self.out_dir = Path(out_dir or Path(__file__).parents[1] / "3d_results")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, img_path: str) -> str:
        try:
            # a) Load image
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")

            # b) Detect face(s)
            boxes = self.face_boxes(img)
            if len(boxes) == 0:
                # Try to provide more helpful error message
                raise RuntimeError(f"No face detected in image: {Path(img_path).name}. Make sure the image contains a clear, front-facing face.")

            # c) Get 3DMM parameters & ROI boxes
            param_lst, roi_box_lst = self.tddfa(img, boxes)

            # d) Reconstruct dense vertices
            ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

            # e) Build clean output path
            base = Path(img_path).stem
            out_file = f"{base}_3d.jpg"
            out_path = self.out_dir / out_file

            # f) Render & save
            render(img, ver_lst, self.tri,
                   alpha=self.alpha,
                   show_flag=False,
                   wfp=str(out_path))

            # Verify the output file was created
            if not out_path.exists():
                raise RuntimeError(f"Failed to save 3D overlay image to {out_path}")

            return str(out_path)
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"3D mesh generation failed for {Path(img_path).name}: {str(e)}")

    def export_obj(self, img_path: str) -> str:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        boxes = self.face_boxes(img)
        if len(boxes) == 0:
            raise RuntimeError("No face detected in image")
        param_lst, roi_box_lst = self.tddfa(img, boxes)
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        base = Path(img_path).stem
        out_file = f"{base}_mesh.obj"
        out_path = self.out_dir / out_file
        ser_to_obj(img, ver_lst, self.tri, height=img.shape[0], wfp=str(out_path))
        return str(out_path)

    def generate_3d_maps(self, img_path: str):
        # Generate and save texture map, depth map, and PNCC map
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        boxes = self.face_boxes(img)
        if len(boxes) == 0:
            raise RuntimeError("No face detected in image")
        param_lst, roi_box_lst = self.tddfa(img, boxes)
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        base = Path(img_path).stem
        # Texture map (rendered overlay)
        tex_path = self.out_dir / f"{base}_texture.jpg"
        render(img, ver_lst, self.tri, alpha=1.0, show_flag=False, wfp=str(tex_path))
        # Depth map
        depth_path = self.out_dir / f"{base}_depth.png"
        depth(img, ver_lst, self.tri, show_flag=False, wfp=str(depth_path), with_bg_flag=True)
        # PNCC map
        pncc_path = self.out_dir / f"{base}_pncc.png"
        pncc(img, ver_lst, self.tri, show_flag=False, wfp=str(pncc_path), with_bg_flag=True)
        return str(tex_path), str(depth_path), str(pncc_path)

    def overlay_mesh_on_video(self, video_fp: str, output_fp: str = None, 
                            export_maps: bool = False, export_uv_maps: bool = False, export_dir: str = None,
                            onnx: bool = True, alpha: float = None) -> str:
        """
        Process a video and generate a 3D mesh overlay video.
        Optionally export per-frame maps as well.
        """
        if alpha is None:
            alpha = self.alpha
            
        # Set up output paths
        if output_fp is None:
            video_name = Path(video_fp).stem
            output_fp = str(self.out_dir / f"{video_name}_3d_overlay.mp4")
            
        if export_maps and export_dir is None:
            export_dir = str(self.out_dir)
            
        # Load video
        cap = cv2.VideoCapture(video_fp)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_fp}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get first frame to determine video dimensions
        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from video")
        height, width = first_frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_fp, fourcc, fps, (width, height))
        
        if not out_writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_fp}")
        
        frame_idx = 0
        frames_processed = 0
        frames_with_faces = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Detect faces and process
                boxes = self.face_boxes(frame)
                if len(boxes) > 0:
                    print(f"Frame {frame_idx}: Found {len(boxes)} face(s)")
                    
                    # Get 3DMM parameters and ROI boxes
                    param_lst, roi_box_lst = self.tddfa(frame, boxes)
                    
                    # Reconstruct dense vertices
                    ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
                    
                    # Verify we have valid vertices
                    if ver_lst and len(ver_lst) > 0:
                        # Create overlay frame
                        overlay_frame = frame.copy()
                        
                        # Render mesh overlay and capture the result
                        rendered_frame = render(overlay_frame, ver_lst, self.tri, 
                                              alpha=alpha, show_flag=False, wfp=None)
                        
                        # Export maps if requested
                        if export_maps and export_dir:
                            frame_name = f"frame_{frame_idx:05d}"
                            
                            # Depth map
                            depth_path = Path(export_dir) / f"{frame_name}_depth.png"
                            depth(frame, ver_lst, self.tri, show_flag=False, 
                                 wfp=str(depth_path), with_bg_flag=True)
                            
                            # Texture map
                            tex_path = Path(export_dir) / f"{frame_name}_texture.jpg"
                            render(frame, ver_lst, self.tri, alpha=1.0, 
                                  show_flag=False, wfp=str(tex_path))
                            
                            # PNCC map
                            pncc_path = Path(export_dir) / f"{frame_name}_pncc.png"
                            pncc(frame, ver_lst, self.tri, show_flag=False, 
                                 wfp=str(pncc_path), with_bg_flag=True)
                        
                        # Export UV maps if requested
                        if export_uv_maps and export_dir:
                            frame_name = f"frame_{frame_idx:05d}"
                            
                            # UV texture map
                            uv_path = Path(export_dir) / f"{frame_name}_uv.jpg"
                            from DDFA_V2.utils.uv import uv_tex
                            uv_tex(frame, ver_lst, self.tri, show_flag=False, wfp=str(uv_path))
                        
                        # Write the rendered frame with 3D overlay
                        out_writer.write(rendered_frame)
                        frames_with_faces += 1
                        print(f"Frame {frame_idx}: 3D overlay applied successfully")
                    else:
                        print(f"Frame {frame_idx}: No valid vertices generated, using original frame")
                        out_writer.write(frame)
                else:
                    # No face detected, write original frame
                    print(f"Frame {frame_idx}: No faces detected")
                    out_writer.write(frame)
                    
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                # Write original frame on error
                out_writer.write(frame)
                
            frames_processed += 1
            frame_idx += 1
            
            # Progress reporting every 30 frames
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames ({frames_with_faces} with faces detected)")
            
        cap.release()
        out_writer.release()
        
        # Print final summary
        print(f"Video processing complete:")
        print(f"  - Total frames processed: {frames_processed}")
        print(f"  - Frames with faces detected: {frames_with_faces}")
        print(f"  - Output video: {output_fp}")
        
        # Verify the output file was created
        if not Path(output_fp).exists():
            raise RuntimeError(f"Failed to create output video: {output_fp}")
            
        return output_fp

    def generate_uv_texture(self, img_path: str, output_dir: str = None) -> str:
        """Generate UV texture map from an image"""
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            
            # Detect face(s)
            boxes = self.face_boxes(img)
            if len(boxes) == 0:
                raise RuntimeError(f"No face detected in image: {Path(img_path).name}")
            
            # Get 3DMM parameters & ROI boxes
            param_lst, roi_box_lst = self.tddfa(img, boxes)
            
            # Reconstruct dense vertices
            ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            
            # Build output path - use user-specified directory if provided
            base = Path(img_path).stem
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                uv_tex_path = output_path / f"{base}_uv_tex.jpg"
            else:
                uv_tex_path = self.out_dir / f"{base}_uv_tex.jpg"
            
            # Generate UV texture map
            from DDFA_V2.utils.uv import uv_tex
            
            # UV texture map
            uv_tex(img, ver_lst, self.tri, show_flag=False, wfp=str(uv_tex_path))
            
            return str(uv_tex_path)
            
        except Exception as e:
            raise RuntimeError(f"UV texture generation failed for {Path(img_path).name}: {str(e)}")