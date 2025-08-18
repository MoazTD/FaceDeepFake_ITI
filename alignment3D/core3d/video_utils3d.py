import cv2
import glob
from pathlib import Path

def extract_frames(video_path: Path, output_dir: Path, progress_callback=None):
    """
    Extracts all frames from a video file and saves them to a directory.
    
    Returns:
        Tuple[float, int]: The original FPS and the total number of frames.
    """
    if progress_callback: progress_callback(f"🎞️ Extracting frames from video...")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frame_count):
        success, frame = cap.read()
        if not success:
            break
        cv2.imwrite(str(output_dir / f"frame_{i:05d}.jpg"), frame)
        if progress_callback and i % 30 == 0:
            progress_callback(f"🎞️ Extracted frame {i+1}/{frame_count}")
    
    cap.release()
    if progress_callback: progress_callback("👍 Frames extracted.")
    return original_fps, frame_count

def reconstruct_video(frames_dir: Path, output_path: Path, fps: float, progress_callback=None):
    """
    Stitches all frames from a directory back into a video file.
    """
    if progress_callback: progress_callback("🎬 Reconstructing video...")
    
    frame_paths = sorted(glob.glob(str(frames_dir / '*.jpg')))
    if not frame_paths:
        raise FileNotFoundError("No frames found to reconstruct.")

    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame_path in frame_paths:
        out.write(cv2.imread(frame_path))
    
    out.release()
    if progress_callback: progress_callback("👍 Video reconstructed.")
