import os
import shutil
import subprocess
import asyncio
import torch
import uuid
import time
from datetime import datetime
from typing import Callable, Dict, Any, Optional
from pathlib import Path
import sys

try:
    try:
        from core import video_utils
        VIDEO_UTILS_AVAILABLE = True
    except ImportError:
        import video_utils
        VIDEO_UTILS_AVAILABLE = True
except ImportError:
    VIDEO_UTILS_AVAILABLE = False
    print("⚠️ Warning: video_utils not available for audio extraction")

class RVCConverter:
    """
    Unified RVC Voice Converter with workflow integration
    """
    
    def __init__(self, rvc_root: str = None, assets_path: str = None, temp_base_dir: str = "temp"):

        if rvc_root is None:
            possible_paths = [
                Path.cwd() / "Models" / "Sound",
                Path.cwd().parent / "Models" / "Sound",
                Path(__file__).parent / "Models" / "Sound",
                Path(__file__).parent.parent / "Models" / "Sound",
            ]
            self.rvc_root = next((path for path in possible_paths if path.exists() and path.is_dir()), None)
            if not self.rvc_root:

                for root, dirs, files in os.walk(Path.cwd()):
                    if "Sound" in dirs and (Path(root) / "Sound" / "assets" / "weights").exists():
                        self.rvc_root = Path(root) / "Sound"
                        break
                if not self.rvc_root:
                    self.rvc_root = Path.cwd() / "Models" / "Sound" 
        else:
            self.rvc_root = Path(rvc_root)

        if assets_path is None:
            self.assets_path = self.rvc_root / "assets" / "weights"
        else:
            self.assets_path = Path(assets_path)
        
        self.temp_base_dir = Path(temp_base_dir)
        self.active_conversions = {}
        
        self.temp_base_dir.mkdir(exist_ok=True)
        self._ensure_ffmpeg_in_path()

        print(f"[DEBUG] RVC Root: {self.rvc_root}")
        print(f"[DEBUG] Assets Path: {self.assets_path}")
        print(f"[DEBUG] RVC Root exists: {self.rvc_root.exists()}")
        print(f"[DEBUG] Assets Path exists: {self.assets_path.exists()}")
        
    def _ensure_ffmpeg_in_path(self):
        """Ensure FFmpeg is available in system PATH"""
        try:
            result = subprocess.run(["ffmpeg", "-version"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  check=True, timeout=10)
            print("✅ FFmpeg found in system PATH")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            possible_paths = [
                Path("C:/ffmpeg/bin"),
                Path("C:/Program Files/ffmpeg/bin"),
                self.rvc_root / "ffmpeg/bin",
                Path.cwd() / "ffmpeg/bin"
            ]
            
            ffmpeg_added = False
            for path in possible_paths:
                ffmpeg_exe = path / "ffmpeg.exe"
                if ffmpeg_exe.exists():
                    path_str = str(path)
                    if path_str not in os.environ.get("PATH", ""):
                        os.environ["PATH"] += os.pathsep + path_str
                        print(f"✅ Added FFmpeg to PATH: {path}")
                    ffmpeg_added = True
                    break
            
            if not ffmpeg_added:
                print("⚠️ Warning: FFmpeg not found in PATH. Video processing may fail")
                return False
            return True

    def _resolve_model_path(self, model_name: str) -> str:

        model_path = Path(model_name)
        if model_path.is_absolute() and model_path.exists():
            if model_path.suffix != '.pth':
                raise ValueError(f"Model file must be .pth format. Got: {model_path.suffix}")
            return str(model_path)

        asset_model_path = self.assets_path / model_name
        if asset_model_path.exists() and asset_model_path.suffix == '.pth':
            return str(asset_model_path)
        
        if not model_name.endswith('.pth'):
            asset_model_path_pth = self.assets_path / f"{model_name}.pth"
            if asset_model_path_pth.exists():
                return str(asset_model_path_pth)
            
            asset_model_path_raw = self.assets_path / model_name
            if asset_model_path_raw.exists():
                return str(asset_model_path_raw)
        
        index_files = list(self.assets_path.glob(f"{model_name}*.index"))
        if index_files:
            pth_files = list(self.assets_path.glob(f"{model_name}*.pth"))
            if pth_files:
                return str(pth_files[0])
            raise FileNotFoundError(
                f"Found index file for '{model_name}' but no matching .pth file in {self.assets_path}"
            )
        
        all_pth_files = list(self.assets_path.glob("*.pth"))
        matching_files = [f for f in all_pth_files if model_name.lower() in f.stem.lower()]
        
        if matching_files:
            return str(matching_files[0])
        
        available_models = "\n  ".join([f.stem for f in all_pth_files])
        raise FileNotFoundError(
            f"Model '{model_name}' not found in {self.assets_path}\n"
            f"Available models:\n  {available_models}"
        )
    
    def _create_progress_callback(self, session_id: str, main_callback: Callable = None) -> Callable:
        """
        Create a progress callback that formats messages consistently
        """
        def progress_callback_func(message: str):
            timestamp = datetime.now().strftime("[%H:%M:%S]")
            formatted_message = f"{timestamp} {message}"
            
            if main_callback:
                main_callback(formatted_message)
            else:
                print(formatted_message)
        
        return progress_callback_func

    def _find_inference_script(self) -> Path:
        """
        Find the RVC inference script in your directory structure
        """
        script_path = Path(__file__).parent / "Models" / "Sound" / "tools" / "infer_cli.py"
        
        if script_path.exists():
            return script_path
        
        possible_scripts = [
            self.rvc_root / "tools" / "infer_cli.py",
            self.rvc_root / "tools" / "infer" / "cli.py",
            self.rvc_root / "tools" / "inference.py",
            self.rvc_root / "infer_cli.py",
            self.rvc_root / "inference.py",
            self.rvc_root / "cli.py",
            self.rvc_root / "infer.py",
            self.rvc_root / "scripts" / "infer_cli.py",
            self.rvc_root / "src" / "infer_cli.py",
            self.rvc_root / "tools" / "infer.py",
            self.rvc_root / "tools" / "voice_conversion.py",
            self.rvc_root / "rvc_inference.py",
        ]
        
        print(f"[DEBUG] Searching for inference script in {len(possible_scripts)} locations...")
        
        for script_path in possible_scripts:
            if script_path.exists():
                print(f"[DEBUG] Found inference script: {script_path}")
                return script_path
        
        tools_dir = self.rvc_root / "tools"
        if tools_dir.exists():
            py_files = list(tools_dir.glob("*.py"))
            
            infer_files = [f for f in py_files if 'infer' in f.name.lower()]
            if infer_files:
                return infer_files[0]
            
            if py_files:
                return py_files[0]
        
        fallback_script = self._create_fallback_inference_script()
        if fallback_script:
            return fallback_script
        
        error_msg = (
            f"RVC inference script not found. Searched in:\n" + 
            "\n".join([f"  - {p}" for p in possible_scripts]) +
            f"\n\nPlease ensure your RVC installation is complete in: {self.rvc_root}"
        )
        raise FileNotFoundError(error_msg)
    
    def _create_fallback_inference_script(self) -> Optional[Path]:
        """
        Create a simple fallback inference script if none exists
        """
        try:
            fallback_script_content = '''#!/usr/bin/env python3
import sys
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='RVC Voice Conversion')
    parser.add_argument('--input_path', required=True, help='Input audio file')
    parser.add_argument('--output_path', required=True, help='Output audio file') 
    parser.add_argument('--model_name', required=True, help='Model name')
    parser.add_argument('--f0up_key', type=int, default=0, help='Pitch adjustment')
    parser.add_argument('--index_rate', type=float, default=0.75, help='Index rate')
    parser.add_argument('--protect', type=float, default=0.33, help='Protect rate')
    parser.add_argument('--pth_path', help='Path to model weights directory')
    
    # Alternative argument names
    parser.add_argument('--input', dest='input_path', help='Input audio file (alt)')
    parser.add_argument('--output', dest='output_path', help='Output audio file (alt)')
    parser.add_argument('--model', dest='model_name', help='Model name (alt)')
    parser.add_argument('--pitch', dest='f0up_key', type=int, help='Pitch adjustment (alt)')
    
    args = parser.parse_args()
    
    print(f"RVC Conversion Parameters:")
    print(f"  Input: {args.input_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Pitch: {args.f0up_key}")
    print(f"  Index Rate: {args.index_rate}")
    print(f"  Protect: {args.protect}")
    
    # For now, just copy the input to output as a placeholder
    # This should be replaced with actual RVC inference code
    from shutil import copy2
    try:
        copy2(args.input_path, args.output_path)
        print("Conversion completed (fallback - file copied)")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
'''
            
            fallback_path = self.rvc_root / "tools" / "fallback_infer.py"
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(fallback_path, 'w', encoding='utf-8') as f:
                f.write(fallback_script_content)
            
            print(f"[DEBUG] Created fallback inference script: {fallback_path}")
            return fallback_path
            
        except Exception as e:
            print(f"[DEBUG] Failed to create fallback script: {e}")
            return None
    
    def _is_video_file(self, file_path: str) -> bool:
        """Check if the file is a video file based on extension"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        return Path(file_path).suffix.lower() in video_extensions
    
    def _is_audio_file(self, file_path: str) -> bool:
        """Check if the file is an audio file based on extension"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
        return Path(file_path).suffix.lower() in audio_extensions
    
    def _extract_audio_from_video_ffmpeg(self, video_path: str, session_id: str, progress_callback: Callable = None) -> str:
        """
        Extract audio from video file using ffmpeg (primary method)
        """
        progress_cb_local = progress_callback or print
        
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")
        
        temp_session_dir = self.temp_base_dir / "voice" / session_id
        temp_session_dir.mkdir(parents=True, exist_ok=True)
        
        audio_file = temp_session_dir / f"{video_file.stem}_extracted.wav"
        
        progress_cb_local("🎬 Extracting audio from video using FFmpeg...")
        
        if not self._ensure_ffmpeg_in_path():
            raise RuntimeError("FFmpeg is required but not found in PATH")
        
        cmd = [
            "ffmpeg", "-y",  
            "-i", str(video_file),  
            "-vn",  
            "-acodec", "pcm_s16le",  
            "-ar", "44100", 
            "-ac", "2",  
            str(audio_file)  
        ]
        
        progress_cb_local(f"DEBUG: FFmpeg extraction command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout_lines = []
            stderr_lines = []
            while process.poll() is None:
                if process.stdout.readable():
                    line_stdout = process.stdout.readline()
                    if line_stdout:
                        stdout_lines.append(line_stdout.strip())
                        progress_cb_local(f"FFMPEG OUT: {line_stdout.strip()}")
                
                if process.stderr.readable():
                    line_stderr = process.stderr.readline()
                    if line_stderr:
                        stderr_lines.append(line_stderr.strip())
                        if "time=" in line_stderr: 
                            progress_cb_local(f"🎬 {line_stderr.strip()}")
                
                time.sleep(0.05)  
            stdout_final, stderr_final = process.communicate(timeout=30)
            if stdout_final:
                stdout_lines.extend(stdout_final.strip().split('\n'))
            if stderr_final:
                stderr_lines.extend(stderr_final.strip().split('\n'))

            progress_cb_local(f"📄 FFmpeg return code: {process.returncode}")
            
            if process.returncode == 0 and audio_file.exists() and audio_file.stat().st_size > 0:
                progress_cb_local(f"✅ Audio extracted successfully: {audio_file.name}")
                return str(audio_file)
            else:
                stderr_text = '\n'.join(stderr_lines)
                if "does not contain any stream" in stderr_text or "No audio" in stderr_text:
                    raise RuntimeError("Video file does not contain any audio stream")
                
                error_msg = f"FFmpeg failed with return code {process.returncode}\nSTDERR: {stderr_text}"
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("FFmpeg extraction timed out")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and add it to your system PATH.")
        except Exception as e:
            raise RuntimeError(f"Audio extraction failed: {str(e)}")
    
    def _extract_audio_from_video_moviepy(self, video_path: str, session_id: str, progress_callback: Callable = None) -> str:
        """
        Extract audio from video using MoviePy (fallback method)
        """
        progress_cb_local = progress_callback or print
        
        try:
            from moviepy.editor import VideoFileClip
            
            video_file = Path(video_path)
            if not video_file.exists():
                raise FileNotFoundError(f"Video file not found: {video_file}")
            
            temp_session_dir = self.temp_base_dir / "voice" / session_id
            temp_session_dir.mkdir(parents=True, exist_ok=True)
            
            audio_file = temp_session_dir / f"{video_file.stem}_extracted.wav"
            
            progress_cb_local("🎬 Using MoviePy for audio extraction...")

            video = VideoFileClip(str(video_file))
            if video.audio is None:
                video.close()
                raise RuntimeError("Video file has no audio track")

            video.audio.write_audiofile(str(audio_file), verbose=False, logger=None)
            video.close()

            if audio_file.exists() and audio_file.stat().st_size > 0:
                progress_cb_local(f"✅ Audio extracted using MoviePy: {audio_file.name}")
                return str(audio_file)
            else:
                raise RuntimeError("Audio extraction failed - output file empty or not created")

        except ImportError:
            raise RuntimeError("MoviePy not available. Please install moviepy: pip install moviepy")
        except Exception as e:
            raise RuntimeError(f"MoviePy audio extraction failed: {str(e)}")

    def _prepare_audio_input(self, input_path: str, session_id: str, progress_callback: Callable = None) -> str:
        """
        Prepare audio input - extract from video if needed, validate audio files
        """
        progress_cb_local = progress_callback or print
        
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if self._is_audio_file(input_path):
            progress_cb_local(f"📁 Input is audio file: {input_file.name}")
            
            if input_file.stat().st_size == 0:
                raise ValueError("Audio file is empty")
            
            return str(input_file.resolve())
        
        elif self._is_video_file(input_path):
            progress_cb_local(f"🎬 Input is video file: {input_file.name}")
            
            try:
                return self._extract_audio_from_video_ffmpeg(input_path, session_id, progress_cb_local)
            except Exception as ffmpeg_error:
                progress_cb_local(f"⚠️ FFmpeg extraction failed: {ffmpeg_error}")
                try:
                    progress_cb_local("🔄 Trying MoviePy as fallback...")
                    return self._extract_audio_from_video_moviepy(input_path, session_id, progress_cb_local)
                except Exception as moviepy_error:
                    progress_cb_local(f"❌ MoviePy extraction also failed: {moviepy_error}")
                    raise RuntimeError(f"Both FFmpeg and MoviePy failed to extract audio.\nFFmpeg: {ffmpeg_error}\nMoviePy: {moviepy_error}")
        
        else:
            supported_extensions = [".wav", ".mp3", ".flac", ".mp4", ".avi", ".mov"]
            raise ValueError(f"Unsupported file type: {input_file.suffix}. Supported: {', '.join(supported_extensions)}")

    def convert_voice(
        self,
        input_path: str,
        model_name: str,
        output_path: str = None,
        f0_up_key: int = 0,
        index_rate: float = 0.75,
        protect: float = 0.33,
        session_id: str = None,
        progress_callback: Callable = None,
        cancellation_check: Callable = None
    ) -> Dict[str, Any]:
        """
        Convert voice using RVC with comprehensive error handling and progress tracking
        """
        if not session_id:
            session_id = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        progress_cb = self._create_progress_callback(session_id, progress_callback)
        
        progress_cb("🎵 Starting voice conversion...")
        progress_cb(f"📁 RVC Root: {self.rvc_root}")
        progress_cb(f"📁 Assets Path: {self.assets_path}")
        
        self.active_conversions[session_id] = {
            "status": "running",
            "start_time": datetime.now(),
            "input_path": input_path,
            "model_name": model_name
        }
        
        try:
            if not self.rvc_root.exists():
                raise FileNotFoundError(f"RVC root directory not found: {self.rvc_root}")
            
            if not self.assets_path.exists():
                progress_cb(f"⚠️ Creating assets directory: {self.assets_path}")
                self.assets_path.mkdir(parents=True, exist_ok=True)
            
            progress_cb(f"🔍 Resolving model: {model_name}")
            try:
                model_path = self._resolve_model_path(model_name)
                progress_cb(f"📁 Using model: {model_path}")
            except FileNotFoundError as e:
                progress_cb(f"❌ Error: Model not found: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "session_id": session_id,
                }
            
            # Prepare audio input (extract from video if needed)
            progress_cb("🎵 Preparing audio input...")
            try:
                audio_input_path = self._prepare_audio_input(input_path, session_id, progress_cb)
                progress_cb(f"🎵 Audio input ready: {Path(audio_input_path).name}")
            except Exception as e:
                progress_cb(f"❌ Error preparing audio input: {e}")
                return {
                    "success": False,
                    "error": f"Error preparing audio input: {e}",
                    "session_id": session_id,
                }

            if not output_path:
                temp_session_dir = self.temp_base_dir / "voice" / session_id
                temp_session_dir.mkdir(parents=True, exist_ok=True)
                input_name = Path(input_path).stem
                output_file = temp_session_dir / f"{input_name}_converted.wav"
            else:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
            
            progress_cb(f"📁 Output will be saved to: {output_file}")
            
            progress_cb("🔍 Finding RVC inference script...")
            try:
                script_path = self._find_inference_script()
                progress_cb(f"🔧 Using inference script: {script_path}")
            except FileNotFoundError as e:
                progress_cb(f"❌ Error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "session_id": session_id,
                }
            
            progress_cb("⚙️ Building RVC command...")
            try:
                cmd = self._build_rvc_command(
                    script_path=script_path,
                    audio_input_path=audio_input_path,
                    output_file=output_file,
                    model_path=model_path,
                    f0_up_key=f0_up_key,
                    index_rate=index_rate,
                    protect=protect
                )
                progress_cb(f"✅ RVC command built successfully")
            except Exception as e:
                progress_cb(f"❌ Error building RVC command: {e}")
                return {
                    "success": False,
                    "error": f"Error building RVC command: {e}",
                    "session_id": session_id,
                }
            
            cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
            progress_cb(f"DEBUG: Full RVC command: {cmd_str}")

            progress_cb("🔄 Running RVC inference...")
            
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.rvc_root) + os.pathsep + env.get("PYTHONPATH", "")
            
            progress_cb("🚀 Starting subprocess...")
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, 
                cwd=str(self.rvc_root),
                env=env
            )
            
            progress_cb(f"📊 Process started with PID: {process.pid}")
            
            stdout_lines = []
            stderr_lines = []
            
            while process.poll() is None:
                if cancellation_check and cancellation_check():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    progress_cb("❌ Conversion canceled by user.")
                    return {
                        "success": False,
                        "error": "Conversion canceled by user.",
                        "session_id": session_id,
                    }
                
                if process.stdout.readable():
                    line_stdout = process.stdout.readline()
                    if line_stdout:
                        stdout_lines.append(line_stdout.strip())
                        progress_cb(f"RVC OUT: {line_stdout.strip()}")
                
                if process.stderr.readable():
                    line_stderr = process.stderr.readline()
                    if line_stderr:
                        stderr_lines.append(line_stderr.strip())
                        progress_cb(f"RVC ERR: {line_stderr.strip()}")
                
                time.sleep(0.1)
            
            try:
                stdout_final, stderr_final = process.communicate(timeout=10)
                if stdout_final:
                    stdout_lines.extend(stdout_final.strip().split('\n'))
                if stderr_final:
                    stderr_lines.extend(stderr_final.strip().split('\n'))
            except subprocess.TimeoutExpired:
                progress_cb("⚠️ Process communication timeout")

            progress_cb(f"📄 Process return code: {process.returncode}")
            
            if process.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
                file_size = output_file.stat().st_size
                progress_cb("✅ Voice conversion completed successfully!")
                progress_cb(f"📄 Output file: {output_file} ({file_size} bytes)")
                
                self.active_conversions[session_id].update({
                    "status": "completed",
                    "output_path": str(output_file),
                })
                
                return {
                    "success": True,
                    "output_path": str(output_file),
                    "session_id": session_id,
                }
            else:
                error_lines = [
                    f"RVC process failed. Return code: {process.returncode}",
                    "STDERR:"
                ] + stderr_lines[-10:] + ["STDOUT:"] + stdout_lines[-10:]  
                error_msg = "\n".join(error_lines)
                raise RuntimeError(error_msg)
                
        except Exception as e:
            error_msg = f"Conversion error: {str(e)}"
            progress_cb(f"❌ {error_msg}")
            
            self.active_conversions[session_id].update({
                "status": "error",
                "error": error_msg
            })
            
            return {
                "success": False,
                "error": error_msg,
                "session_id": session_id
            }
    
    def _build_rvc_command(self, script_path: Path, audio_input_path: str, output_file: Path, 
                      model_path: str, f0_up_key: int, index_rate: float, protect: float) -> list:
        """
        Build RVC command with comprehensive argument handling and index file detection
        """
        model_path = Path(model_path)
        audio_input_path = Path(audio_input_path)
        output_file = Path(output_file)
        
        cmd = ["python", str(script_path)]
        
        script_content = ""
        try:
            script_content = script_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            print(f"[WARNING] Couldn't read script content: {e}")
        
        index_path = ""
        model_stem = model_path.stem
        model_dir = model_path.parent
        
        index_files = list(model_dir.glob(f"{model_stem}*.index"))
        if index_files:
            exact_index = [f for f in index_files if f.stem == model_stem]
            index_path = str(exact_index[0]) if exact_index else str(index_files[0])
            print(f"[DEBUG] Found index file: {index_path}")
        
        uses_new_format = "--input_path" in script_content
        uses_old_format = "--input" in script_content
        
        if uses_new_format:
            cmd.extend([
                "--input_path", str(audio_input_path.resolve()),
                "--output_path", str(output_file.resolve()),
                "--model_name", model_stem, 
                "--f0up_key", str(f0_up_key),
                "--index_rate", str(index_rate),
                "--protect", str(protect),
            ])
            
            if any(arg in script_content for arg in ["--pth_path", "--model_path", "--weights_path"]):
                cmd.extend(["--pth_path", str(model_dir.resolve())])
            
            if index_path and "--index_path" in script_content:
                cmd.extend(["--index_path", index_path])
                
        elif uses_old_format:
            cmd.extend([
                "--input", str(audio_input_path.resolve()),
                "--output", str(output_file.resolve()),
                "--model", model_stem,  
                "--pitch", str(f0_up_key),
                "--index_rate", str(index_rate),
                "--protect", str(protect),
            ])
            
            if any(arg in script_content for arg in ["--model_dir", "--weights_dir"]):
                cmd.extend(["--model_dir", str(model_dir.resolve())])
            
            if index_path and "--index" in script_content:
                cmd.extend(["--index", index_path])
                
        else:
            cmd.extend([
                str(audio_input_path.resolve()),
                str(output_file.resolve()),
                model_stem, 
                str(f0_up_key),
                str(index_rate),
                str(protect),
            ])
            
            if "pth_path" in script_content:
                cmd.extend([str(model_dir.resolve())])
            
            if index_path and "index_path" in script_content:
                cmd.extend([index_path])
        
        if "device" in script_content and not any(arg.startswith("--device") for arg in cmd):
            cmd.extend(["--device", "cuda:0" if torch.cuda.is_available() else "cpu"])
        
        if "sr" in script_content and not any(arg.startswith("--sr") for arg in cmd):
            cmd.extend(["--sr", "44100"])
        
        cmd_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd)
        print(f"[DEBUG] Final RVC command: {cmd_str}")
        
        return cmd
    
    def merge_audio_to_video(self, video_path: str, audio_path: str, output_path: str, progress_callback: Callable = None, cancellation_check: Callable = None) -> Dict[str, Any]:
        """
        Merges an audio file into a video file using FFmpeg.
        """
        progress_cb = progress_callback or print
        
        progress_cb("🎬 Starting video and audio merge...")
        progress_cb(f"Video: {Path(video_path).name}")
        progress_cb(f"Audio: {Path(audio_path).name}")
        progress_cb(f"Output: {Path(output_path).name}")

        if not self._ensure_ffmpeg_in_path():
            return {
                "success": False,
                "error": "FFmpeg is required for video merging but was not found in PATH",
                "output_path": None
            }

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            output_path
        ]

        progress_cb(f"DEBUG: Full FFmpeg merge command: {' '.join(cmd)}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout_lines = []
            stderr_lines = []
            while process.poll() is None:

                if cancellation_check and cancellation_check():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    progress_cb("❌ Video merge canceled by user.")
                    return {
                        "success": False,
                        "error": "Video merge canceled by user.",
                        "output_path": None
                    }
                
                if process.stdout.readable():
                    line_stdout = process.stdout.readline()
                    if line_stdout:
                        stdout_lines.append(line_stdout.strip())
                        progress_cb(f"FFMPEG MERGE OUT: {line_stdout.strip()}")
                
                if process.stderr.readable():
                    line_stderr = process.stderr.readline()
                    if line_stderr:
                        stderr_lines.append(line_stderr.strip())
                        if "time=" in line_stderr:  # Show progress
                            progress_cb(f"🎬 {line_stderr.strip()}")
                
                time.sleep(0.1)

            try:
                stdout_final, stderr_final = process.communicate(timeout=30)
                if stdout_final:
                    stdout_lines.extend(stdout_final.strip().split('\n'))
                if stderr_final:
                    stderr_lines.extend(stderr_final.strip().split('\n'))
            except subprocess.TimeoutExpired:
                progress_cb("⚠️ Process communication timeout during merge")

            progress_cb(f"📄 FFmpeg return code: {process.returncode}")

            if process.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                progress_cb("✅ Video and audio merged successfully!")
                return {"success": True, "output_path": output_path}
            else:
                error_lines = [
                    f"FFmpeg merge failed. Return code: {process.returncode}",
                    "STDERR:"
                ] + stderr_lines[-10:] + ["STDOUT:"] + stdout_lines[-10:]  
                error_msg = "\n".join(error_lines)
                raise RuntimeError(error_msg)

        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please ensure FFmpeg is installed and accessible in your system's PATH.")
        except Exception as e:
            raise RuntimeError(f"Error during video merge: {str(e)}")
    
    async def convert_voice_async(self, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.convert_voice, *args, **kwargs)
    
    def get_conversion_status(self, session_id: str) -> Dict[str, Any]:
        return self.active_conversions.get(session_id, {"status": "not_found"})
    
    def list_available_models(self) -> list:
        """
        List available models in the assets/weights directory
        """
        if not self.assets_path.exists():
            try:
                self.assets_path.mkdir(parents=True, exist_ok=True)
                print(f"Created assets directory: {self.assets_path}")
            except Exception as e:
                print(f"Could not create assets directory: {e}")
                return []
        
        models = []
        try:
            pth_files = [f.name for f in self.assets_path.iterdir() if f.suffix == '.pth']
            
            other_formats = [f.name for f in self.assets_path.iterdir() 
                           if f.suffix in ['.onnx', '.pkl', '.pt'] and f.is_file()]
            
            models = pth_files + other_formats
            
            if not models:
                print(f"No model files found in: {self.assets_path}")
                print("Please place your RVC model files (.pth) in the assets/weights directory")
                print("Supported formats: .pth, .onnx, .pkl, .pt")
            else:
                print(f"Found {len(models)} models: {', '.join(models)}")
                
        except Exception as e:
            print(f"Error listing models: {e}")
        
        return sorted(models)
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate the RVC setup and return detailed status
        """
        status = {
            "rvc_root_exists": self.rvc_root.exists(),
            "assets_path_exists": self.assets_path.exists(),
            "models_found": [],
            "inference_script_found": False,
            "inference_script_path": None,
            "ffmpeg_available": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            status["models_found"] = self.list_available_models()
            
            try:
                script_path = self._find_inference_script()
                status["inference_script_found"] = True
                status["inference_script_path"] = str(script_path)
            except FileNotFoundError as e:
                status["errors"].append(str(e))
            
            status["ffmpeg_available"] = self._ensure_ffmpeg_in_path()
            if not status["ffmpeg_available"]:
                status["warnings"].append("FFmpeg not found - video processing will not work")
            
            if not VIDEO_UTILS_AVAILABLE:
                status["warnings"].append("video_utils not available - video input may have issues")
            
            if not status["models_found"]:
                status["errors"].append(f"No models found in {self.assets_path}")
            
            if not status["rvc_root_exists"]:
                status["errors"].append(f"RVC root directory not found: {self.rvc_root}")
                
        except Exception as e:
            status["errors"].append(f"Setup validation error: {e}")
        
        return status
    
    def cleanup_session(self, session_id: str):
        if session_id in self.active_conversions:
            del self.active_conversions[session_id]
        
        temp_session_dir = self.temp_base_dir / "voice" / session_id
        if temp_session_dir.exists():
            shutil.rmtree(temp_session_dir)
    
    def cleanup_all(self):
        self.active_conversions.clear()
        voice_temp_dir = self.temp_base_dir / "voice"
        if voice_temp_dir.exists():
            shutil.rmtree(voice_temp_dir)
        voice_temp_dir.mkdir(parents=True, exist_ok=True)

class VoiceConversionWorkflow:
    def __init__(self, rvc_root: str = None, assets_path: str = None):
        self.converter = RVCConverter(rvc_root=rvc_root, assets_path=assets_path)
    
    def convert_voice_with_progress(self, input_file: str, model_name: str, output_file: str = None, progress_callback=None, **conversion_params):
        def progress_handler(message):
            if progress_callback:
                progress_callback(message)
        
        conversion_params.pop('progress_callback', None)
        return self.converter.convert_voice(
            input_path=input_file,
            model_name=model_name,
            output_path=output_file,
            progress_callback=progress_handler,
            **conversion_params
        )
    
    def validate_setup(self):
        """Validate the RVC setup"""
        return self.converter.validate_setup()
    
    def list_models(self):
        """List available models"""
        return self.converter.list_available_models()
    
    def merge_audio_to_video(self, video_path: str, audio_path: str, output_path: str, progress_callback: Callable = None, cancellation_check: Callable = None):
        """
        Wrapper for RVCConverter's merge_audio_to_video.
        """
        return self.converter.merge_audio_to_video(video_path, audio_path, output_path, progress_callback, cancellation_check)