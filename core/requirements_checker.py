import sys
import os
import subprocess
import platform
from pathlib import Path
import json
import urllib.request
import hashlib
from typing import List, Dict, Tuple, Optional


REQUIRED_PACKAGES = {
    'PySide6': '6.0.0',
    'numpy': '1.19.0',
    'opencv-python': '4.5.0',
    'insightface': '0.7.0',
    'onnxruntime': '1.12.0',
    
}


REQUIRED_MODELS = {
    'inswapper_128.onnx': {
        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx',
        'size': 554253681,  
        'md5': 'a3a155b90354160350efd66fed6b3d80',
        'path': 'models/inswapper_128.onnx'
    },
    'buffalo_l': {
        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip',
        'size': 281906913,
        'md5': '68e9b8e116c7b3e9a06a7b3d52ba0c83',
        'path': 'models/buffalo_l',
        'is_zip': True
    }
}


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class RequirementsChecker:
    def __init__(self):
        self.missing_packages = []
        self.missing_models = []
        self.errors = []
        self.warnings = []
        
    def print_header(self):
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Face Swapping Application - Requirements Checker{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")
        
    def check_python_version(self) -> bool:
        print(f"{Colors.BOLD}Checking Python version...{Colors.RESET}")
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"{Colors.GREEN}✓ Python {version.major}.{version.minor}.{version.micro} - OK{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}✗ Python {version.major}.{version.minor}.{version.micro} - "
                  f"Requires Python 3.8 or higher{Colors.RESET}")
            self.errors.append("Python 3.8 or higher is required")
            return False
            
    def check_package(self, package_name: str, min_version: str) -> bool:
        try:
            
            if package_name == 'opencv-python':
                import cv2
                installed_version = cv2.__version__
            elif package_name == 'Pillow':
                import PIL
                installed_version = getattr(PIL, '__version__', 'Unknown')
            else:
                module = __import__(package_name.replace('-', '_'))
                installed_version = getattr(module, '__version__', 'Unknown')
                
            
            if self._compare_versions(installed_version, min_version):
                print(f"{Colors.GREEN}✓ {package_name} ({installed_version}) - OK{Colors.RESET}")
                return True
            else:
                print(f"{Colors.YELLOW}⚠ {package_name} ({installed_version}) - "
                      f"Requires version {min_version} or higher{Colors.RESET}")
                self.warnings.append(f"{package_name} version {min_version} or higher recommended")
                return True  
                
        except ImportError:
            print(f"{Colors.RED}✗ {package_name} - Not installed{Colors.RESET}")
            self.missing_packages.append(package_name)
            return False
            
    def _compare_versions(self, installed: str, required: str) -> bool:
        try:
            inst_parts = [int(x) for x in installed.split('.')[:3]]
            req_parts = [int(x) for x in required.split('.')[:3]]
            
            for i in range(len(req_parts)):
                if i >= len(inst_parts):
                    return False
                if inst_parts[i] > req_parts[i]:
                    return True
                elif inst_parts[i] < req_parts[i]:
                    return False
            return True
        except:
            return True  
            
    def check_packages(self):
        print(f"\n{Colors.BOLD}Checking required packages...{Colors.RESET}")
        for package, version in REQUIRED_PACKAGES.items():
            self.check_package(package, version)
            
    def check_cuda(self) -> bool:
        print(f"\n{Colors.BOLD}Checking GPU/CUDA support...{Colors.RESET}")
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in providers:
                print(f"{Colors.GREEN}✓ CUDA support available{Colors.RESET}")
                
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        print(f"{Colors.GREEN}✓ CUDA {torch.version.cuda} detected{Colors.RESET}")
                except:
                    pass
                return True
            else:
                print(f"{Colors.YELLOW}⚠ CUDA not available - Will use CPU (slower){Colors.RESET}")
                self.warnings.append("GPU acceleration not available")
                return False
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Could not check CUDA: {e}{Colors.RESET}")
            return False
            
    def check_model_file(self, model_name: str, model_info: Dict) -> bool:
        model_path = Path(model_info['path'])
        
        if model_path.exists():
            if model_info.get('is_zip'):
                
                if model_path.is_dir() and list(model_path.glob('*')):
                    print(f"{Colors.GREEN}✓ {model_name} - Found{Colors.RESET}")
                    return True
            else:
                
                size = model_path.stat().st_size
                if size == model_info['size']:
                    print(f"{Colors.GREEN}✓ {model_name} - OK{Colors.RESET}")
                    return True
                else:
                    print(f"{Colors.YELLOW}⚠ {model_name} - Size mismatch{Colors.RESET}")
                    
        print(f"{Colors.RED}✗ {model_name} - Not found{Colors.RESET}")
        self.missing_models.append(model_name)
        return False
        
    def check_models(self):
        print(f"\n{Colors.BOLD}Checking required models...{Colors.RESET}")
        
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        for model_name, model_info in REQUIRED_MODELS.items():
            self.check_model_file(model_name, model_info)
            
    def download_model(self, model_name: str, model_info: Dict) -> bool:
        print(f"\n{Colors.BLUE}Downloading {model_name}...{Colors.RESET}")
        
        try:
            model_path = Path(model_info['path'])
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                bar_length = 40
                filled_length = int(bar_length * percent // 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                print(f'\rProgress: |{bar}| {percent:.1f}% ', end='', flush=True)
                
            
            temp_path = model_path.with_suffix('.tmp')
            urllib.request.urlretrieve(model_info['url'], temp_path, download_progress)
            print()  
            
            
            if model_info.get('is_zip'):
                print(f"Extracting {model_name}...")
                import zipfile
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(model_path.parent)
                temp_path.unlink()  
            else:
                temp_path.rename(model_path)
                
            print(f"{Colors.GREEN}✓ {model_name} downloaded successfully{Colors.RESET}")
            return True
            
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to download {model_name}: {e}{Colors.RESET}")
            return False
            
    def install_packages(self):
        if not self.missing_packages:
            return True
            
        print(f"\n{Colors.BOLD}Installing missing packages...{Colors.RESET}")
        
        for package in self.missing_packages:
            print(f"\n{Colors.BLUE}Installing {package}...{Colors.RESET}")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package
                ])
                print(f"{Colors.GREEN}✓ {package} installed successfully{Colors.RESET}")
            except subprocess.CalledProcessError:
                print(f"{Colors.RED}✗ Failed to install {package}{Colors.RESET}")
                self.errors.append(f"Failed to install {package}")
                
    def download_models(self):
        if not self.missing_models:
            return True
            
        print(f"\n{Colors.BOLD}Downloading missing models...{Colors.RESET}")
        print(f"{Colors.YELLOW}Note: Model files are large and may take time to download{Colors.RESET}")
        
        for model_name in self.missing_models:
            if model_name in REQUIRED_MODELS:
                self.download_model(model_name, REQUIRED_MODELS[model_name])
                
    def create_directories(self):
        print(f"\n{Colors.BOLD}Creating required directories...{Colors.RESET}")
        
        directories = [
            'models',
            'output',
            'cache',
            'logs',
            'temp_frames_processing'
        ]
        
        for dir_name in directories:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
            print(f"{Colors.GREEN}✓ {dir_name}/ directory ready{Colors.RESET}")
            
    def print_summary(self):
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Summary{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")
        
        if self.errors:
            print(f"{Colors.RED}Errors:{Colors.RESET}")
            for error in self.errors:
                print(f"  • {error}")
            print()
            
        if self.warnings:
            print(f"{Colors.YELLOW}Warnings:{Colors.RESET}")
            for warning in self.warnings:
                print(f"  • {warning}")
            print()
            
        if not self.errors and not self.missing_packages and not self.missing_models:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ All requirements satisfied!{Colors.RESET}")
            print(f"{Colors.GREEN}The application is ready to run.{Colors.RESET}")
            return True
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ Some requirements are not satisfied.{Colors.RESET}")
            return False
            
    def run_checks(self, auto_install: bool = False) -> bool:
        self.print_header()
        
        
        if not self.check_python_version():
            self.print_summary()
            return False
            
        
        self.check_packages()
        
        
        self.check_cuda()
        
        
        self.check_models()
        
        
        self.create_directories()
        
        
        if auto_install:
            if self.missing_packages:
                response = input(f"\n{Colors.YELLOW}Install missing packages? (y/n): {Colors.RESET}")
                if response.lower() == 'y':
                    self.install_packages()
                    
            if self.missing_models:
                response = input(f"\n{Colors.YELLOW}Download missing models? (y/n): {Colors.RESET}")
                if response.lower() == 'y':
                    self.download_models()
                    
        
        return self.print_summary()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check requirements for Face Swapping Application')
    parser.add_argument('--auto-install', '-a', action='store_true',
                        help='Automatically install missing packages and download models')
    parser.add_argument('--run-app', '-r', action='store_true',
                        help='Run the application after checking requirements')
    
    args = parser.parse_args()
    
    checker = RequirementsChecker()
    success = checker.run_checks(auto_install=args.auto_install)
    
    if success and args.run_app:
        print(f"\n{Colors.BLUE}Starting Face Swapping Application...{Colors.RESET}\n")
        try:
            from ui.main_window import main as run_app
            run_app()
        except Exception as e:
            print(f"{Colors.RED}Failed to start application: {e}{Colors.RESET}")
            return 1
            
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
