# Image Defogging Algorithm - Streamlit Web Interface

## Overview

This project implements a comprehensive image defogging algorithm based on the research paper "Image Defogging Algorithm for Smoke Conditions" by Zhiqi Cheng and Shuhua Liu from Shenyang Ligong University. The implementation provides a user-friendly web interface built with Streamlit that allows users to enhance foggy or hazy images using multiple defogging techniques and their combination.

## Research Paper Reference

**Title:** Image Defogging Algorithm for Smoke Conditions  
**Authors:** Zhiqi Cheng, Shuhua Liu  
**Institution:** School of Equipment Engineering, Shenyang Ligong University, Shenyang, China  
**Conference:** 2024 2nd International Conference on Signal Processing and Intelligent Computing (SPIC)

## Features

### Implemented Algorithms

1. **Dark Channel Prior Defogging**
   - Leverages the statistical property that most natural images have pixels with low intensity in at least one color channel
   - Estimates atmospheric light and transmission map
   - Effectively removes haze while preserving image details

2. **Single-Scale Retinex Algorithm**
   - Based on Retinex theory for color perception
   - Recovers realistic colors by simulating human visual perception
   - Balances illumination across the image

3. **Proposed Combined Algorithm** (Dark Channel + Retinex)
   - Applies Retinex first for color recovery
   - Then applies dark channel defogging for haze removal
   - Combines advantages of both methods for superior results
   - Addresses color distortion and improves saturation

4. **Homomorphic Filtering**
   - Processes illumination and reflectance components separately
   - Provides alternative approach to defogging

5. **Global Histogram Equalization**
   - Traditional contrast enhancement technique
   - Serves as baseline for comparison

### Quantitative Metrics

The application evaluates image quality using four key metrics:

- **Information Entropy:** Measures complexity and information content (higher is better)
- **Average Gradient:** Indicates texture complexity and detail clarity (higher is better)
- **Mean Square Error (MSE):** Measures disparity between images (lower is better)
- **Peak Signal-to-Noise Ratio (PSNR):** Reflects color distortion levels (higher is better)

### Web Interface Features

- **Interactive Parameter Adjustment**
  - Configurable window size for dark channel computation (5-25 pixels)
  - Adjustable weight parameter (w) for transmission estimation (0.8-0.99)
  - Tunable percentile for atmospheric light estimation

- **Real-time Processing**
  - Upload images in common formats (JPG, JPEG, PNG, BMP)
  - See results for all algorithms simultaneously
  - Compare visual outputs side-by-side

- **Comprehensive Analysis**
  - Visual comparison grid of all algorithms
  - Quantitative metrics comparison with visualizations
  - Detailed metrics table

- **Download Capabilities**
  - Export defogged images in PNG format
  - Save results from any algorithm

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd image-defogging
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements.txt Content

```
streamlit==1.28.0
opencv-python==4.8.1.78
numpy==1.24.3
pillow==10.0.0
matplotlib==3.7.2
scipy==1.11.2
```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Workflow

1. **Configure Parameters (Optional)**
   - Adjust settings in the sidebar if needed
   - Default values are optimized based on the research paper

2. **Upload an Image**
   - Click "Browse files" to select a foggy/hazy image
   - Supported formats: JPG, JPEG, PNG, BMP

3. **View Results**
   - Visual comparison grid displays all algorithm results
   - Metrics charts and table show quantitative analysis

4. **Download Results**
   - Use download buttons to save processed images

### Parameter Explanation

- **Window Size:** Controls the patch size for dark channel computation. Smaller values capture finer details; larger values are better for thick haze.

- **Weight Parameter (w):** Controls transmission estimation accuracy. Values closer to 1.0 preserve more details; lower values enhance contrast more.

- **Percentile:** Determines the proportion of brightest pixels used to estimate atmospheric light. Higher values use more pixels for more robust estimation.

## Algorithm Details

### Dark Channel Prior Defogging

Mathematical formulation:

```
Dark Channel: J_dark(x) = min(c ∈ C) min(p ∈ P(x)) min(c)

Atmospheric Scattering Model:
I(x) = J(x) * t(x) + A * (1 - t(x))

Transmission Estimation:
t(x) = 1 - w * min(c ∈ C) [I(c)(x) / A(c)(x)]

Dehazed Image:
J(x) = [I(x) - A(1 - t(x))] / t(x)
```

### Single-Scale Retinex

```
r_i(x,y) = log[I_i(x,y)] - log[I_i(x,y) * G(x,y)]

Where:
- r_i(x,y) is the processed reflection component
- I_i(x,y) is the original image
- G(x,y) is the Gaussian surround function
```

### Proposed Algorithm

1. Apply Retinex for initial color recovery
2. Apply dark channel defogging on Retinex output
3. Combine benefits of both approaches

## Expected Results

Based on the research paper, the proposed algorithm demonstrates:

- **Higher Information Entropy:** Better detail preservation (average: 7.76)
- **Higher Average Gradient:** Clearer texture and edges (average: 21.08)
- **Competitive MSE:** Reasonable error rates (average: 2553.44)
- **Balanced PSNR:** Good signal-to-noise ratio (average: 13.85)

### Limitations

- Performance varies across different image regions (sky areas may show artifacts)
- Extremely thick haze may require parameter tuning
- Processing time depends on image resolution

## Project Structure

```
image-defogging/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── sample_images/        # (Optional) Sample foggy images for testing
```

## Performance Considerations

- **Image Size:** Recommended maximum 2K resolution for optimal performance
- **Processing Time:** Typically 2-5 seconds for full analysis depending on image size
- **Memory:** Requires ~500MB RAM for standard sized images

## Troubleshooting

### Issue: Module not found error

**Solution:** Ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Application runs slowly

**Solution:** 
- Reduce image resolution before uploading
- Decrease window size parameter
- Close other applications to free up memory

### Issue: Downloaded image has wrong format

**Solution:** The application saves images in PNG format. Use image viewers or converters to change formats if needed.

## Customization

### Adding New Algorithms

To add a new defogging algorithm:

1. Create a new method in the `DefoggingAlgorithm` class
2. Add it to the processing loop
3. Display results in the appropriate section
4. Add metrics calculation

Example:
```python
def custom_algorithm(self, image):
    # Your implementation here
    return processed_image
```

### Adjusting Metrics

Modify the `calculate_metrics()` function to add custom evaluation metrics.

## Technical Stack

- **Frontend:** Streamlit (Python web framework)
- **Image Processing:** OpenCV, NumPy, SciPy
- **Visualization:** Matplotlib, Pillow
- **Language:** Python 3.8+

## References

1. He, K., Sun, J., & Tang, X. (2011). "Single Image Haze Removal Using Dark Channel Prior." IEEE TPAMI, 33(12), 2341-2353.

2. Land, E. H. (1977). "The Retinex Theory of Color Vision." Scientific American, 237(6), 108-128.

3. Cheng, Z., & Liu, S. (2024). "Image Defogging Algorithm for Smoke Conditions." 2024 2nd International Conference on Signal Processing and Intelligent Computing (SPIC).

## License

This project implements algorithms described in published research. Use in accordance with academic fair use policies.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{cheng2024defogging,
  title={Image Defogging Algorithm for Smoke Conditions},
  author={Cheng, Zhiqi and Liu, Shuhua},
  booktitle={2024 2nd International Conference on Signal Processing and Intelligent Computing (SPIC)},
  year={2024},
  organization={IEEE}
}
```

## Support

For issues, questions, or contributions, please refer to the original research paper or contact the development team.

## Acknowledgments

Special thanks to Zhiqi Cheng and Shuhua Liu for their pioneering work on image defogging algorithms for smoke conditions at Shenyang Ligong University.

---

**Last Updated:** 2024  
**Status:** Active Development
