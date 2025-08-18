import cv2
import glob
from pathlib import Path
import gc
from typing import Optional, Union
import shutil

try:
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio import fx as afx
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

    MOVIEPY_AVAILABLE = False
    print("--------------------------------------------------------------------")
    print("WARNING: moviepy library not found. Audio features will be disabled.")
    print("         To enable audio, open the Anaconda Prompt and run:")
    print("         conda install -c conda-forge moviepy")
    print("--------------------------------------------------------------------")


def extract_frames(video_path: Union[str, Path], output_dir: Union[str, Path], progress_callback=None):
    if progress_callback: 
        progress_callback(f"🎞️ Extracting frames from video...")
    
    
    if isinstance(video_path, str):
        video_path = Path(video_path)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    batch_size = 100
    extracted = 0

    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(batch_start + batch_size, frame_count)
        
        for i in range(batch_start, batch_end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if not success:
                break
                
            output_file = output_dir / f"frame_{i:05d}.jpg"
            cv2.imwrite(str(output_file), frame)
            extracted += 1
            
            if progress_callback and extracted % 30 == 0:
                progress_callback(f"🎞️ Extracted frame {extracted}/{frame_count}")
        
        gc.collect()
    
    cap.release()
    if progress_callback: 
        progress_callback(f"👍 Frames extracted. Total: {extracted}")
    return original_fps, extracted

def extract_audio(video_path: Union[str, Path], output_audio_path: Union[str, Path], progress_callback=None):
    if not MOVIEPY_AVAILABLE:
        if progress_callback:
            progress_callback("⚠️ Audio features disabled: moviepy not installed.")
        return False

    
    if isinstance(video_path, str):
        video_path = Path(video_path)
    if isinstance(output_audio_path, str):
        output_audio_path = Path(output_audio_path)

    if progress_callback:
        progress_callback("🎵 Extracting audio track...")
    try:
        video_clip = VideoFileClip(str(video_path))
        if video_clip.audio is None:
            if progress_callback:
                progress_callback("⚠️ Video has no audio track.")
            video_clip.close()
            return False

        
        
        video_clip.audio.write_audiofile(str(output_audio_path), codec='aac', logger=None)

        video_clip.close()
        
        if progress_callback:
            progress_callback("👍 Audio extracted successfully.")
        return True
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Error extracting audio: {e}")
        return False

def reconstruct_video(frames_dir: Union[str, Path], output_path: Union[str, Path], fps: float, audio_path: Optional[Union[str, Path]] = None, progress_callback=None):
    if progress_callback:
        progress_callback("🎬 Reconstructing video from frames...")

    if isinstance(frames_dir, str):
        frames_dir = Path(frames_dir)
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if isinstance(audio_path, str) and audio_path:
        audio_path = Path(audio_path)

    frame_paths = sorted(glob.glob(str(frames_dir / '*.jpg')))
    if not frame_paths:
        raise FileNotFoundError("No frames found to reconstruct.")

    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Cannot read first frame: {frame_paths[0]}")
    
    height, width, _ = first_frame.shape

    temp_video_path = output_path.with_suffix(".temp.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        temp_video_path = output_path.with_suffix(".temp.avi")
        out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("Failed to initialize video writer with any codec")
    
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
        if progress_callback and (i + 1) % 30 == 0:
            progress_callback(f"🎬 Writing frame {i + 1}/{len(frame_paths)}")
    out.release()
    
    use_audio = audio_path and audio_path.exists() if audio_path else False

    if use_audio and MOVIEPY_AVAILABLE:
        if progress_callback:
            progress_callback("🔗 Merging video and audio...")
        try:
            final_clip = VideoFileClip(str(temp_video_path))
            audio_clip = AudioFileClip(str(audio_path))
            
            if audio_clip.duration < final_clip.duration:
                audio_clip = afx.audio_loop(audio_clip, duration=final_clip.duration)

            final_clip.audio = audio_clip
            
            final_clip.write_videofile(
                str(output_path), 
                codec="libx264", 
                audio_codec="aac", 
                logger=None,
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=fps
            )
            
            final_clip.close()
            audio_clip.close()
            
            if temp_video_path.exists():
                temp_video_path.unlink()
            
            if progress_callback:
                progress_callback("👍 Video reconstruction complete with audio.")
            return

        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ Could not attach audio due to error: {e}. Saving silent video.")
            try:
                if 'final_clip' in locals():
                    final_clip.close()
                if 'audio_clip' in locals():
                    audio_clip.close()
            except:
                pass
    
    if use_audio and not MOVIEPY_AVAILABLE:
        if progress_callback:
            progress_callback("⚠️ moviepy not found. Saving silent video.")

    try:
        if temp_video_path.exists():
            shutil.move(str(temp_video_path), str(output_path))
        if progress_callback:
            progress_callback("👍 Video reconstruction complete (silent).")
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Error moving temporary video: {e}")
        raise

def has_audio_track(video_path: Union[str, Path]):
    if not MOVIEPY_AVAILABLE:
        return False
    
    try:
        
        if isinstance(video_path, str):
            video_path = Path(video_path)
            
        video_clip = VideoFileClip(str(video_path))
        has_audio = video_clip.audio is not None
        video_clip.close()
        return has_audio
    except Exception:
        return False

def get_video_info(video_path: Union[str, Path]):
    if isinstance(video_path, str):
        video_path = Path(video_path)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        'has_audio': has_audio_track(video_path)
    }
    
    cap.release()
    return info
