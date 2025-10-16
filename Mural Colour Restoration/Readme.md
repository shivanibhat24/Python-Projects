# 🎨 Mural Color Restoration System

A comprehensive web-based application for restoring and enhancing mural paintings using advanced image processing technology. This project implements the algorithms and methodologies from the research paper "Mural Color Restoration Algorithm Based on Image Processing Technology."

## 🎯 Overview

This application uses cutting-edge image processing techniques to restore and enhance the colors, contrast, and clarity of historical mural paintings. The system can analyze damaged or faded murals and apply intelligent restoration algorithms to recover visual details and improve color fidelity.

**Key Achievement**: Increases color gamut from 60-80 to 80-100 on the fidelity scale through advanced processing algorithms.

## ✨ Features

### Core Processing Capabilities

- **Color Fidelity Enhancement**: Improves color accuracy and vibrancy in murals
- **Contrast Restoration**: Enhances visual distinction between different elements
- **Denoising**: Removes environmental degradation and noise
- **Texture Reconstruction**: Preserves and enhances fine mural details
- **Brightness Adjustment**: Corrects uneven lighting conditions
- **Saturation Enhancement**: Restores color saturation in faded areas

### Analysis & Visualization

- **Color Fidelity Analysis**: Tracks color quality metrics before and after processing
- **Histogram Analysis**: Detailed RGB channel distribution visualization
- **Edge Detection**: Canny edge detection for detail comparison
- **Before/After Comparison**: Side-by-side visual comparison with difference maps
- **Processing Speed Metrics**: Monitors algorithm performance

### User Interface

- Interactive parameter adjustment via sidebar sliders
- Real-time image preview and processing
- Multi-tab analysis interface
- Download functionality for results and comparisons
- Responsive design compatible with desktop and tablet browsers

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step-by-Step Setup

1. **Clone or download the project**

```bash
git clone <repository-url>
cd mural-restoration
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Or manually install required packages:

```bash
pip install streamlit opencv-python pillow numpy scikit-image matplotlib
```

4. **Run the application**

```bash
streamlit run app.py
```

5. **Access the web interface**

Open your browser and navigate to `http://localhost:8501`

## 📖 Usage

### Basic Workflow

**Step 1: Upload a Mural Image**

Click the upload widget on the left side and select a JPG, JPEG, PNG, or BMP image from your computer. The original image will appear on the left panel.

**Step 2: Adjust Processing Parameters**

Use the left sidebar sliders to customize:

- **Contrast Enhancement**: Increase from 0.5 to 3.0 for more dramatic contrast
- **Brightness Adjustment**: Range from -50 to +50 to correct lighting
- **Saturation Enhancement**: Boost from 0.5 to 2.0 for color vibrancy
- **Blur Reduction Kernel**: Adjust from 3 to 15 for noise reduction
- **Denoise Strength**: Control denoising intensity from 0 to 20

**Step 3: View Results**

The processed image appears on the right panel with metrics showing image dimensions and processing stats.

**Step 4: Analyze Details**

Navigate through tabs for detailed analysis:

- **Color Fidelity**: View color quality metrics and trends
- **Histogram Analysis**: Examine RGB channel distributions
- **Edge Detection**: Compare edge details
- **Comparison**: View side-by-side comparison with difference map

**Step 5: Download Results**

Download the restored image as PNG, download a comparison visualization, or export high-quality outputs at 150 DPI.

### Example Scenarios

**Scenario 1: Faded Mural**

- Increase Saturation Enhancement to 1.5-2.0
- Boost Contrast Enhancement to 2.0-2.5
- Adjust Brightness as needed
- Result: Vibrant colors restored to original appearance

**Scenario 2: Noisy/Degraded Mural**

- Increase Denoise Strength to 15-20
- Adjust Blur Reduction Kernel to 7-11
- Moderate Contrast Enhancement (1.5-1.8)
- Result: Clean image with preserved details

**Scenario 3: Poorly Lit Mural**

- Adjust Brightness Adjustment (+10 to +30)
- Increase Contrast Enhancement (1.8-2.2)
- Fine-tune Denoise Strength (8-12)
- Result: Well-lit image with visible details

## 🔧 Technical Details

### Image Processing Pipeline

The application implements a sophisticated multi-stage processing pipeline:

**Stage 1: Denoising**

Non-Local Means (NLM) filter removes noise while preserving edges

**Stage 2: Brightness Adjustment**

Linear brightness correction applied uniformly across the image

**Stage 3: Contrast Enhancement**

Adaptive histogram equalization for local contrast improvement

**Stage 4: Saturation Enhancement**

HSV color space conversion and saturation scaling to restore color vibrancy

**Stage 5: Bilateral Filtering**

Edge-preserving smoothing with 9x9 kernel and sigma=75

**Stage 6: Sharpening**

Unsharp masking kernel for enhanced detail definition

### Mathematical Foundations

- **Color Space Conversion**: RGB ↔ BGR ↔ HSV transformations
- **Histogram Equalization**: Adaptive histogram equalization (CLAHE)
- **Edge-Aware Processing**: Bilateral filtering with Gaussian kernels
- **Noise Reduction**: Non-local similarity-based denoising
- **Contrast Metrics**: Standard deviation and histogram analysis

### Color Fidelity Calculation

Color fidelity is calculated as:

```
Fidelity = clip(std_dev(grayscale) / 2.56, 40, 100)
```

This metric ranges from 40-100, where higher values indicate better color preservation and distinction.

## 💻 System Requirements

### Minimum Requirements

- **OS**: Windows, macOS, or Linux
- **CPU**: Dual-core processor
- **RAM**: 4GB
- **Storage**: 500MB for application and dependencies
- **Display**: 1024x768 resolution minimum

### Recommended Requirements

- **OS**: Windows 10+, macOS 10.14+, or Ubuntu 18.04+
- **CPU**: Quad-core processor
- **RAM**: 8GB or higher
- **Storage**: 2GB available space
- **Display**: 1920x1080 or higher
- **GPU**: Optional (CUDA-enabled for faster processing)


## 🔌 API Reference

### Main Processing Function

```python
def process_mural(img, contrast, brightness, saturation, blur_k, denoise)
```

**Parameters:**

- `img` (np.ndarray): Input image in BGR format
- `contrast` (float): Contrast enhancement factor (0.5-3.0)
- `brightness` (int): Brightness adjustment (-50 to 50)
- `saturation` (float): Saturation enhancement factor (0.5-2.0)
- `blur_k` (int): Blur reduction kernel size (3-15, must be odd)
- `denoise` (int): Denoising strength (0-20)

**Returns:**

- `np.ndarray`: Processed image in BGR format (uint8)

### Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| Streamlit | Latest | Web interface framework |
| OpenCV (cv2) | 4.5+ | Image processing operations |
| NumPy | 1.20+ | Numerical computations |
| Pillow | 8.0+ | Image I/O and manipulation |
| scikit-image | 0.18+ | Advanced image restoration |
| Matplotlib | 3.3+ | Visualization and charting |

## 📸 Examples

### Example 1: Historical Mural Restoration

**Input**: Faded 16th-century mural with color degradation

**Parameters**: Contrast 2.2, Saturation 1.8, Denoise 12

**Output**: Restored colors, improved clarity, preserved historical details

### Example 2: Archaeological Fresco

**Input**: Partially damaged fresco with noise and missing sections

**Parameters**: Contrast 1.8, Denoise 15, Saturation 1.5

**Output**: Enhanced visibility, noise reduction, color recovery

### Example 3: Modern Street Mural

**Input**: Weather-exposed modern mural with fading

**Parameters**: Brightness +15, Saturation 1.6, Contrast 1.7

**Output**: Vibrant colors, improved lighting, preserved artistic intent

## 🐛 Troubleshooting

### Issue: Application won't start

**Solution**: Verify all dependencies are installed

```bash
pip install --upgrade streamlit opencv-python pillow numpy scikit-image matplotlib
```

### Issue: Slow processing on large images

**Solution**:

- Use images no larger than 4000x4000 pixels
- Reduce denoise strength (below 15)
- Lower contrast enhancement slightly
- Consider GPU acceleration with CUDA

### Issue: Downloaded image quality is poor

**Solution**:

- Check original image resolution (minimum 800x600 recommended)
- Adjust DPI in download (modify `dpi=150` in code to `dpi=300`)
- Verify color space settings

### Issue: Color fidelity not improving

**Solution**:

- Original image might have severe color loss
- Try increasing saturation enhancement progressively
- Adjust brightness and contrast together
- Experiment with denoise strength

### Issue: Memory error on very large images

**Solution**:

- Resize image before uploading (max 5000x5000)
- Close other applications
- Reduce processing intensity parameters
- Use system with more RAM

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 code style
- Add docstrings to new functions
- Include unit tests for new features
- Update README with new capabilities
- Test thoroughly before submitting

## 📞 Support

For issues, questions, or suggestions:

1. Check the Troubleshooting section
2. Review the Technical Details for algorithm explanation
3. Open an issue on GitHub with:
   - Description of the problem
   - Steps to reproduce
   - Screenshots/error messages
   - System information

## 🎓 Citation

If you use this application in your research, please cite:

```bibtex
@article{yang2024mural,
  title={Mural Color Restoration Algorithm Based on Image Processing Technology},
  author={Yang, Ning and Sun, Xin and Jiang, Tianyu},
  journal={2024 International Conference on Internet of Things, Robotics and Distributed Computing},
  year={2024},
  organization={IEEE}
}
```

## 🙏 Acknowledgments

- Original research paper authors and institutions
- Streamlit team for the excellent web framework
- OpenCV and scikit-image communities
- All contributors and users providing feedback

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Status**: Active & Maintained
