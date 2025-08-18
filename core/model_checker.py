import os
from pathlib import Path
import urllib.request

def check_models():
    print("🔍 Checking Face Swap Models Setup...")
    print("=" * 50)
    
    
    required_models = {
        'inswapper_128.onnx': {
            'size_mb': 528,
            'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx'
        }
    }
    
    
    possible_dirs = [
        Path("models"),
        Path("Models"),
        Path(__file__).parent / "models",
        Path(__file__).parent / "Models",
    ]
    
    models_found = False
    model_dir = None
    
    for directory in possible_dirs:
        if directory.exists():
            print(f"📁 Found models directory: {directory}")
            model_dir = directory
            
            for model_name, info in required_models.items():
                model_path = directory / model_name
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    print(f"✅ Found {model_name} ({size_mb:.1f} MB)")
                    models_found = True
                else:
                    print(f"❌ Missing {model_name}")
            break
    
    if not model_dir:
        print("❌ No models directory found!")
        print("\n🛠️ Setup Instructions:")
        print("1. Create a 'models' directory in your project root")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        print(f"   Created: {models_dir.absolute()}")
        model_dir = models_dir
    
    if not models_found:
        print("\n🛠️ Model Download Instructions:")
        print("You need to download the following models:")
        
        for model_name, info in required_models.items():
            model_path = model_dir / model_name
            print(f"\n📥 {model_name}:")
            print(f"   URL: {info['url']}")
            print(f"   Size: ~{info['size_mb']} MB")
            print(f"   Save to: {model_path.absolute()}")
            
            
            response = input(f"\n❓ Download {model_name} now? (y/n): ").lower().strip()
            if response == 'y':
                print(f"📥 Downloading {model_name}...")
                try:
                    urllib.request.urlretrieve(info['url'], model_path)
                    print(f"✅ Downloaded {model_name} successfully!")
                except Exception as e:
                    print(f"❌ Download failed: {e}")
                    print(f"   Please download manually from: {info['url']}")
    
    
    print("\n🔍 Checking InsightFace Detection Models...")
    
    
    insightface_dirs = [
        Path.home() / ".insightface",
        Path("models") / "buffalo_l",
        Path("Models") / "buffalo_l",
    ]
    
    buffalo_found = False
    for directory in insightface_dirs:
        buffalo_path = directory / "models" / "buffalo_l" if "insightface" in str(directory) else directory
        if buffalo_path.exists() and any(buffalo_path.glob("*.onnx")):
            print(f"✅ Found buffalo_l models in: {buffalo_path}")
            buffalo_found = True
            break
    
    if not buffalo_found:
        print("❌ buffalo_l detection models not found")
        print("💡 These will be automatically downloaded when you first run face detection")
    
    
    print("\n" + "=" * 50)
    if models_found:
        print("🎉 Model setup looks good! You should be able to run face swapping.")
    else:
        print("⚠️  Please download the required models before running face swapping.")
    
    print("🚀 You can now run your face swapping application!")

if __name__ == "__main__":
    check_models()
