# DeepFake Studio Pro - Advanced Face Swapping Tool

## Overview
DeepFake Studio Pro is a comprehensive face swapping application that combines state-of-the-art AI models with an intuitive graphical interface. The tool provides advanced face detection, landmark alignment, and seamless face swapping capabilities for both images and videos.

## Features

### Core Capabilities
- **Advanced Face Detection**: Uses YOLOv8 model for high-accuracy face detection
- **Landmark Detection**: 68-point facial landmark detection for precise alignment
- **Multi-Method Face Swapping**: Supports both traditional and deep learning-based face swapping
- **Video Processing**: Full video face swapping with timeline controls
- **Image Processing**: High-quality image-to-image face swapping

### Advanced Face Swap Features
- **Seamless Cloning**: Advanced Poisson blending for natural-looking results
- **Multi-Scale Blending**: Pyramid-based blending for better edge preservation
- **Color Matching**: LAB color space transformation and histogram matching
- **Quality Enhancement**: Noise reduction, sharpening, and contrast enhancement
- **Edge Preservation**: Special algorithms to maintain natural edges in swapped faces

### UI Features
- **Step-by-Step Workflow**: 8-tab workflow from input to export
- **Real-time Previews**: Instant previews at each processing stage
- **Timeline Controls**: Frame-by-frame navigation for video processing
- **Parameter Adjustment**: Fine-tune all face swapping parameters

## System Requirements

### Minimum Requirements
- Python 3.8 or later
- Windows 10/11 or Linux (Ubuntu 20.04+ recommended)
- 8GB RAM
- 2GB VRAM (for GPU acceleration)
- 2GB free disk space

### Recommended Specifications
- NVIDIA GPU with CUDA support (for best performance)
- 16GB RAM or more
- 4GB+ VRAM
- SSD storage

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/deepfake-studio-pro.git
   cd deepfake-studio-pro
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models** (automatically downloaded on first run):
   - Face detection model (YOLOv8)
   - Landmark detection model (LBF)
   - Face swapping model (inswapper)

## Usage

1. **Launch the application**:
   ```bash
   python main.py
   ```

2. **Basic workflow**:
   1. Select input files (source image/video and target image/video)
   2. Configure detection parameters
   3. Detect facial landmarks
   4. Align faces
   5. Perform face swapping
   6. Export results

3. **Key controls**:
   - Use the timeline at the bottom to navigate video frames
   - Adjust sliders for fine-tuning swap parameters
   - Use the playback controls to preview results

## Configuration

The application includes several configuration parameters that can be adjusted in the code:

```python
# Advanced Face Swap Configuration
CONF_THRES = 0.45          # Confidence threshold for face detection
NMS_IOU = 0.35             # Non-maximum suppression IOU threshold
MODEL_SIZE = 640           # Input size for detection model
FEATHER_AMOUNT = 15        # Feather amount for mask edges
BLEND_RATIO = 0.85         # Blend ratio for final composition
FACE_PADDING = 0.25         # Padding around detected faces
MASK_BLUR_KERNEL = 21      # Kernel size for mask blurring
HISTOGRAM_MATCH_STRENGTH = 0.7  # Strength of histogram matching
SHARPEN_STRENGTH = 0.3     # Sharpening strength
```

## Troubleshooting

### Common Issues

1. **Models not downloading**:
   - Check your internet connection
   - Verify you have write permissions in the application directory
   - Manually download models from the URLs in the code

2. **Poor face swapping results**:
   - Ensure good lighting in source and target images
   - Adjust face alignment parameters
   - Try different blend ratios and mask settings

3. **Performance issues**:
   - Reduce the processing resolution
   - Disable GPU acceleration if using CPU
   - Close other memory-intensive applications

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is intended for ethical and legal use only. The developers are not responsible for any misuse of this technology. Always obtain proper consent before processing anyone's likeness.

---

For support or feature requests, please open an issue on the GitHub repository.