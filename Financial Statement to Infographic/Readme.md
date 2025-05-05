# Financial Statement Animation Engine

A Python-based tool that transforms financial statement data from CSV files into dynamic, animated infographics. This tool analyzes financial data, creates appropriate visualizations, and compiles them into an MP4 video with smooth transitions, insights, and summaries.

## Features

- **Automatic Statement Detection**: Automatically identifies if your data is an income statement, balance sheet, or cash flow statement
- **Smart Visualization Selection**: Creates appropriate charts based on the type of financial data
- **Dynamic Animations**: Builds visualizations with engaging animation effects
- **Insight Generation**: Automatically extracts and displays key insights from the data
- **Summary Slides**: Creates comprehensive summary slides with key metrics
- **Web Interface**: Simple web-based UI for uploading CSV files and generating animations

## Requirements

- Python 3.7+
- pandas
- plotly
- numpy
- moviepy
- Flask (for web interface)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/financial-animation-engine.git
   cd financial-animation-engine
   ```

2. Install required packages:
   ```
   pip install pandas plotly numpy moviepy flask
   ```

## Usage

### Command Line Interface

```bash
python financial_animation_engine.py --csv your_financial_data.csv --output animation.mp4
```

#### Options:
- `--csv`: Path to your financial CSV file
- `--output`: Output path for the animation (default: financial_animation.mp4)
- `--sample`: Use a sample dataset ('income', 'balance', or 'cash_flow')
- `--duration`: Duration of the animation in seconds (default: 15)
- `--fps`: Frames per second for the animation (default: 24)

### Using Sample Data

```bash
python financial_animation_engine.py --sample income --output income_animation.mp4
```

### Web Interface

1. Start the web server:
   ```
   python web_interface.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Use the web interface to:
   - Upload your CSV file
   - Try sample financial statements
   - Configure animation parameters
   - Download the generated animation

## CSV Format

Your CSV file should contain financial data with:

1. **Income Statement Example**:
   - Date/period column (e.g., "Quarter", "Year")
   - Revenue/sales column
   - Various expense columns
   - Profit/earnings columns

2. **Balance Sheet Example**:
   - Date column
   - Asset columns (cash, inventory, receivables, etc.)
   - Liability columns (payables, debt, etc.)
   - Equity columns

3. **Cash Flow Example**:
   - Date/period column
   - Operating activities
   - Investing activities
   - Financing activities

## Using the Python API

You can also use the FinancialAnimator class directly in your Python code:

```python
from financial_animation_engine import FinancialAnimator

# Create animator from CSV file
animator = FinancialAnimator('your_financial_data.csv')

# Or from pandas DataFrame
import pandas as pd
df = pd.read_csv('your_financial_data.csv')
animator = FinancialAnimator(df=df)

# Prepare data for visualization
statement_type = animator.prepare_data()
print(f"Detected statement type: {statement_type}")

# Create animation
output_path = animator.create_animated_infographic(
    output_path='your_animation.mp4',
    fps=24,
    duration=15
)
```

## How It Works

1. **Data Loading & Analysis**: The engine loads your CSV data and automatically detects the type of financial statement.

2. **Visualization Generation**: Based on the statement type, it creates appropriate visualizations:
   - Income Statement: Revenue trends, expense breakdowns, profit margins
   - Balance Sheet: Asset composition, liability trends, debt ratios
   - Cash Flow: Component breakdowns, cumulative cash flow

3. **Animation Creation**: The visualizations are animated with smooth transitions and building effects.

4. **Insight Generation**: The engine analyzes the data to extract key insights and metrics.

5. **Video Compilation**: Everything is compiled into a polished MP4 video with title slides, charts, insights, and summary.

## Example Output

The generated animation will include:
- Title slides introducing each visualization
- Animated charts showing financial data trends
- Insight slides explaining key observations
- A comprehensive summary slide with key metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
