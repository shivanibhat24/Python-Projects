"""
Financial Statement Animation Engine
-----------------------------------
A Python application that converts CSV financial statements into dynamic animated infographics.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os
from moviepy.editor import *
import tempfile
import re
from datetime import datetime

class FinancialAnimator:
    def __init__(self, csv_path=None, df=None):
        """Initialize with either a CSV file path or a pandas DataFrame"""
        if df is not None:
            self.df = df
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either csv_path or df must be provided")
        
        # Clean column names
        self.df.columns = [col.strip() for col in self.df.columns]
        
        # Detect date columns and convert to datetime
        self._convert_date_columns()
        
        # Determine if statements are quarterly or annual
        self.period_type = self._determine_period_type()
        
        # Initialize storage for figures
        self.figures = {}
        
    def _convert_date_columns(self):
        """Convert date-like columns to datetime objects"""
        for col in self.df.columns:
            # Check if column name suggests it's a date
            if any(date_word in col.lower() for date_word in ['date', 'period', 'year', 'quarter']):
                try:
                    # Try to convert to datetime
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    # If more than 50% converted successfully, keep the conversion
                    if self.df[col].notna().mean() > 0.5:
                        continue
                    else:
                        # Revert to original if conversion mostly failed
                        self.df[col] = self.df[col].astype(str)
                except:
                    pass
    
    def _determine_period_type(self):
        """Determine if the data is quarterly or annual"""
        date_cols = [col for col in self.df.columns if pd.api.types.is_datetime64_dtype(self.df[col])]
        
        if date_cols:
            # Use the first date column
            date_col = self.df[date_cols[0]]
            if date_col.dt.day.nunique() <= 4 and date_col.dt.month.nunique() <= 4:
                return "quarterly"
            else:
                return "annual"
        else:
            # Default to annual if no date columns
            return "annual"
    
    def identify_statement_type(self):
        """Attempt to identify whether data is from income statement, balance sheet, or cash flow"""
        lower_cols = [col.lower() for col in self.df.columns]
        
        # Income statement indicators
        income_keywords = ['revenue', 'sales', 'income', 'profit', 'loss', 'earnings', 'ebitda', 'eps']
        
        # Balance sheet indicators
        balance_keywords = ['asset', 'liability', 'equity', 'debt', 'cash', 'inventory', 'receivable']
        
        # Cash flow indicators
        cashflow_keywords = ['cash flow', 'operating activities', 'investing activities', 
                            'financing activities', 'dividends paid']
        
        income_score = sum(any(kw in col for kw in income_keywords) for col in lower_cols)
        balance_score = sum(any(kw in col for kw in balance_keywords) for col in lower_cols)
        cashflow_score = sum(any(kw in col for kw in cashflow_keywords) for col in lower_cols)
        
        scores = {
            'income_statement': income_score,
            'balance_sheet': balance_score,
            'cash_flow': cashflow_score
        }
        
        return max(scores, key=scores.get)
    
    def prepare_data(self):
        """Prepare data for visualization based on statement type"""
        statement_type = self.identify_statement_type()
        
        # Find the date column, if any
        date_cols = [col for col in self.df.columns if pd.api.types.is_datetime64_dtype(self.df[col])]
        date_col = date_cols[0] if date_cols else None
        
        # Find numeric columns
        numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
        
        if statement_type == 'income_statement':
            self._prepare_income_statement(date_col, numeric_cols)
        elif statement_type == 'balance_sheet':
            self._prepare_balance_sheet(date_col, numeric_cols)
        elif statement_type == 'cash_flow':
            self._prepare_cash_flow(date_col, numeric_cols)
            
        return statement_type
        
    def _prepare_income_statement(self, date_col, numeric_cols):
        """Prepare income statement data"""
        if date_col:
            self.df = self.df.sort_values(by=date_col)
            
        # Identify key metrics
        revenue_col = self._find_column_by_keywords(['revenue', 'sales'], numeric_cols)
        expense_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in 
                      ['expense', 'cost', 'cogs', 'salaries', 'rent', 'marketing'])]
        profit_col = self._find_column_by_keywords(['net income', 'profit', 'earnings'], numeric_cols)
        
        # Create trending revenue figure
        if revenue_col:
            self.figures['revenue_trend'] = self._create_trend_chart(
                x=self.df[date_col] if date_col else range(len(self.df)),
                y=self.df[revenue_col],
                title=f"Revenue Trend ({self.period_type.capitalize()})",
                color='blue'
            )
        
        # Create expense breakdown
        if expense_cols:
            self.figures['expense_breakdown'] = self._create_pie_chart(
                self.df[expense_cols].iloc[-1],
                title="Latest Expense Breakdown"
            )
        
        # Create profit margin trend
        if revenue_col and profit_col:
            profit_margin = (self.df[profit_col] / self.df[revenue_col]) * 100
            self.figures['profit_margin'] = self._create_trend_chart(
                x=self.df[date_col] if date_col else range(len(self.df)),
                y=profit_margin,
                title="Profit Margin % Trend",
                color='green'
            )
            
        # Create revenue vs expenses comparison
        if revenue_col and expense_cols:
            total_expenses = self.df[expense_cols].sum(axis=1)
            self.figures['rev_vs_exp'] = self._create_comparison_chart(
                x=self.df[date_col] if date_col else range(len(self.df)),
                y1=self.df[revenue_col],
                y2=total_expenses,
                name1="Revenue",
                name2="Total Expenses",
                title="Revenue vs Expenses"
            )
    
    def _prepare_balance_sheet(self, date_col, numeric_cols):
        """Prepare balance sheet data"""
        if date_col:
            self.df = self.df.sort_values(by=date_col)
            
        # Identify key metrics
        asset_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in ['asset', 'cash', 'inventory', 'receivable'])]
        liability_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in ['liability', 'debt', 'payable'])]
        equity_col = self._find_column_by_keywords(['equity', 'shareholder'], numeric_cols)
        
        # Create asset trend
        if asset_cols:
            total_assets = self.df[asset_cols].sum(axis=1)
            self.figures['asset_trend'] = self._create_trend_chart(
                x=self.df[date_col] if date_col else range(len(self.df)),
                y=total_assets,
                title="Total Assets Trend",
                color='blue'
            )
        
        # Create asset composition
        if asset_cols:
            self.figures['asset_composition'] = self._create_area_chart(
                x=self.df[date_col] if date_col else range(len(self.df)),
                y_dict={col: self.df[col] for col in asset_cols},
                title="Asset Composition Over Time"
            )
        
        # Create liability to asset ratio
        if asset_cols and liability_cols:
            total_assets = self.df[asset_cols].sum(axis=1)
            total_liabilities = self.df[liability_cols].sum(axis=1)
            debt_to_asset = (total_liabilities / total_assets) * 100
            self.figures['debt_to_asset'] = self._create_trend_chart(
                x=self.df[date_col] if date_col else range(len(self.df)),
                y=debt_to_asset,
                title="Debt to Asset Ratio (%)",
                color='red'
            )
    
    def _prepare_cash_flow(self, date_col, numeric_cols):
        """Prepare cash flow data"""
        if date_col:
            self.df = self.df.sort_values(by=date_col)
            
        # Identify key metrics
        operating_col = self._find_column_by_keywords(['operating', 'operations'], numeric_cols)
        investing_col = self._find_column_by_keywords(['investing', 'investment'], numeric_cols)
        financing_col = self._find_column_by_keywords(['financing', 'finance'], numeric_cols)
        
        # Create cash flow components chart
        if operating_col or investing_col or financing_col:
            components = {}
            if operating_col:
                components['Operating'] = self.df[operating_col]
            if investing_col:
                components['Investing'] = self.df[investing_col]
            if financing_col:
                components['Financing'] = self.df[financing_col]
                
            self.figures['cash_flow_components'] = self._create_bar_chart(
                x=self.df[date_col] if date_col else range(len(self.df)),
                y_dict=components,
                title="Cash Flow Components"
            )
        
        # Create cumulative cash flow
        if operating_col or investing_col or financing_col:
            cumulative = pd.Series(0, index=self.df.index)
            if operating_col:
                cumulative += self.df[operating_col].fillna(0)
            if investing_col:
                cumulative += self.df[investing_col].fillna(0)
            if financing_col:
                cumulative += self.df[financing_col].fillna(0)
                
            self.figures['cumulative_cash'] = self._create_trend_chart(
                x=self.df[date_col] if date_col else range(len(self.df)),
                y=cumulative.cumsum(),
                title="Cumulative Cash Flow",
                color='green'
            )
    
    def _find_column_by_keywords(self, keywords, columns):
        """Find a column that contains any of the keywords"""
        for col in columns:
            if any(kw.lower() in col.lower() for kw in keywords):
                return col
        return None
    
    def _create_trend_chart(self, x, y, title, color='blue'):
        """Create a trend line chart"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, 
            y=y,
            mode='lines+markers',
            name=title,
            line=dict(color=color, width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Period",
            yaxis_title="Value",
            template="plotly_white"
        )
        return fig
    
    def _create_pie_chart(self, data, title):
        """Create a pie chart"""
        labels = data.index
        values = data.values
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            textinfo='label+percent',
            insidetextorientation='radial'
        )])
        
        fig.update_layout(
            title=title,
            template="plotly_white"
        )
        return fig
    
    def _create_comparison_chart(self, x, y1, y2, name1, name2, title):
        """Create a comparison chart with two lines"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, 
            y=y1,
            mode='lines+markers',
            name=name1,
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=x, 
            y=y2,
            mode='lines+markers',
            name=name2,
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Period",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )
        return fig
    
    def _create_area_chart(self, x, y_dict, title):
        """Create a stacked area chart"""
        fig = go.Figure()
        
        for name, values in y_dict.items():
            fig.add_trace(go.Scatter(
                x=x, 
                y=values,
                mode='lines',
                name=name,
                stackgroup='one',
                groupnorm='percent'
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title="Period",
            yaxis_title="Percentage (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )
        return fig
    
    def _create_bar_chart(self, x, y_dict, title):
        """Create a grouped bar chart"""
        fig = go.Figure()
        
        for name, values in y_dict.items():
            fig.add_trace(go.Bar(
                x=x,
                y=values,
                name=name
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title="Period",
            yaxis_title="Value",
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            template="plotly_white"
        )
        return fig
    
    def create_animated_infographic(self, output_path='financial_animation.mp4', fps=24, duration=15):
        """Create an animated infographic from all generated figures"""
        # Create temporary directory to store frames
        temp_dir = tempfile.mkdtemp()
        
        # Calculate how many frames for each chart
        num_charts = len(self.figures)
        if num_charts == 0:
            raise ValueError("No charts have been created. Run prepare_data() first.")
            
        frames_per_chart = int(fps * duration / num_charts)
        total_frames = frames_per_chart * num_charts
        
        # Create transition frames for each chart
        all_clips = []
        
        for i, (chart_name, fig) in enumerate(self.figures.items()):
            # Add title slide
            title = fig.layout.title.text
            title_clip = self._create_title_slide(title, temp_dir, f"title_{i}", fps=fps, duration=1)
            all_clips.append(title_clip)
            
            # Add animated chart
            chart_clip = self._animate_chart(fig, temp_dir, f"chart_{i}", fps=fps, duration=4)
            all_clips.append(chart_clip)
            
            # Add insights if appropriate
            if chart_name in ['revenue_trend', 'profit_margin', 'asset_trend', 'cumulative_cash']:
                insight = self._generate_insight(chart_name, fig)
                insight_clip = self._create_insight_slide(insight, temp_dir, f"insight_{i}", fps=fps, duration=2)
                all_clips.append(insight_clip)
        
        # Add summary slide at the end
        summary = self._generate_summary()
        summary_clip = self._create_summary_slide(summary, temp_dir, "summary", fps=fps, duration=3)
        all_clips.append(summary_clip)
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(all_clips)
        
        # Write to output file
        final_clip.write_videofile(output_path, fps=fps)
        
        return output_path
    
    def _animate_chart(self, fig, temp_dir, base_name, fps=24, duration=4):
        """Create animation frames for a chart"""
        # Calculate total frames
        total_frames = int(fps * duration)
        
        # Create a clip with increasing opacity/building effect
        frames = []
        
        for i in range(total_frames):
            progress = i / total_frames
            
            # Clone the figure to modify it
            anim_fig = go.Figure(fig)
            
            # Apply animation effect based on chart type
            if 'pie' in str(type(fig.data[0])).lower():
                # For pie charts, reveal slices one by one
                visible_slices = max(1, int(progress * len(fig.data[0].labels)))
                
                # Clone labels and values
                labels = list(fig.data[0].labels)
                values = list(fig.data[0].values)
                
                # Zero out values for slices that shouldn't be visible yet
                for j in range(len(values)):
                    if j >= visible_slices:
                        values[j] = 0
                
                anim_fig.data[0].values = values
                
            elif 'bar' in str(type(fig.data[0])).lower():
                # For bar charts, grow bars from bottom
                for trace_idx in range(len(anim_fig.data)):
                    original_y = np.array(fig.data[trace_idx].y)
                    anim_fig.data[trace_idx].y = original_y * progress
                    
            else:  # Line charts and others
                # For line charts, draw the line progressively
                for trace_idx in range(len(anim_fig.data)):
                    if hasattr(fig.data[trace_idx], 'x') and hasattr(fig.data[trace_idx], 'y'):
                        x_vals = np.array(fig.data[trace_idx].x)
                        y_vals = np.array(fig.data[trace_idx].y)
                        
                        if len(x_vals) > 1:
                            # Calculate how many points to show
                            points_to_show = max(2, int(len(x_vals) * progress))
                            
                            # Update the trace data
                            anim_fig.data[trace_idx].x = x_vals[:points_to_show]
                            anim_fig.data[trace_idx].y = y_vals[:points_to_show]
            
            # Save the frame
            frame_path = os.path.join(temp_dir, f"{base_name}_{i:04d}.png")
            anim_fig.write_image(frame_path, width=1280, height=720)
            
            # Add to frames list
            frames.append(frame_path)
        
        # Create clip from frames
        clip = ImageSequenceClip(frames, fps=fps)
        
        return clip
    
    def _create_title_slide(self, title, temp_dir, base_name, fps=24, duration=1):
        """Create a title slide"""
        # Create a figure with just the title
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=36, color="darkblue"),
                x=0.5,
                y=0.5
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=1280,
            height=720
        )
        
        # Save the image
        image_path = os.path.join(temp_dir, f"{base_name}.png")
        fig.write_image(image_path)
        
        # Create clip
        clip = ImageClip(image_path).set_duration(duration)
        
        return clip
    
    def _generate_insight(self, chart_name, fig):
        """Generate insights based on the chart"""
        if chart_name == 'revenue_trend':
            y_values = fig.data[0].y
            growth = ((y_values[-1] / y_values[0]) - 1) * 100 if y_values[0] != 0 else 0
            trend = "increasing" if growth > 0 else "decreasing"
            return f"Revenue has been {trend} with {abs(growth):.1f}% {'growth' if growth > 0 else 'decline'} over the period."
            
        elif chart_name == 'profit_margin':
            y_values = fig.data[0].y
            latest = y_values[-1]
            avg = np.mean(y_values)
            comparison = "above" if latest > avg else "below"
            return f"Current profit margin is {latest:.1f}%, which is {comparison} the period average of {avg:.1f}%."
            
        elif chart_name == 'asset_trend':
            y_values = fig.data[0].y
            growth = ((y_values[-1] / y_values[0]) - 1) * 100 if y_values[0] != 0 else 0
            return f"Total assets have {'grown' if growth > 0 else 'declined'} by {abs(growth):.1f}% over the period."
            
        elif chart_name == 'cumulative_cash':
            y_values = fig.data[0].y
            latest = y_values[-1]
            direction = "positive" if latest > 0 else "negative"
            return f"Cumulative cash flow is {direction}, ending at {latest:,.0f} for the period."
            
        return "Insight analysis for this chart type is not available."
    
    def _create_insight_slide(self, insight_text, temp_dir, base_name, fps=24, duration=2):
        """Create an insight slide with the given text"""
        # Create a figure with insight text
        fig = go.Figure()
        
        fig.update_layout(
            title=dict(
                text="Key Insight",
                font=dict(size=28, color="darkblue"),
                x=0.5,
                y=0.7
            ),
            annotations=[
                dict(
                    text=insight_text,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=20)
                )
            ],
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=1280,
            height=720
        )
        
        # Save the image
        image_path = os.path.join(temp_dir, f"{base_name}.png")
        fig.write_image(image_path)
        
        # Create clip
        clip = ImageClip(image_path).set_duration(duration)
        
        return clip
    
    def _generate_summary(self):
        """Generate a summary of key financial findings"""
        statement_type = self.identify_statement_type()
        
        if statement_type == 'income_statement':
            # Find revenue and profit columns
            numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
            revenue_col = self._find_column_by_keywords(['revenue', 'sales'], numeric_cols)
            profit_col = self._find_column_by_keywords(['net income', 'profit', 'earnings'], numeric_cols)
            
            if revenue_col and profit_col:
                latest_rev = self.df[revenue_col].iloc[-1]
                latest_profit = self.df[profit_col].iloc[-1]
                profit_margin = (latest_profit / latest_rev) * 100 if latest_rev != 0 else 0
                
                summary = (
                    f"Income Statement Summary\n\n"
                    f"Latest Revenue: {latest_rev:,.0f}\n"
                    f"Latest Profit: {latest_profit:,.0f}\n"
                    f"Profit Margin: {profit_margin:.1f}%"
                )
                return summary
            
        elif statement_type == 'balance_sheet':
            # Find asset and liability columns
            numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
            asset_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in ['asset', 'cash', 'inventory', 'receivable'])]
            liability_cols = [col for col in numeric_cols if any(kw in col.lower() for kw in ['liability', 'debt', 'payable'])]
            
            if asset_cols and liability_cols:
                total_assets = self.df[asset_cols].sum(axis=1).iloc[-1]
                total_liabilities = self.df[liability_cols].sum(axis=1).iloc[-1]
                equity = total_assets - total_liabilities
                debt_ratio = (total_liabilities / total_assets) * 100 if total_assets != 0 else 0
                
                summary = (
                    f"Balance Sheet Summary\n\n"
                    f"Total Assets: {total_assets:,.0f}\n"
                    f"Total Liabilities: {total_liabilities:,.0f}\n"
                    f"Equity: {equity:,.0f}\n"
                    f"Debt Ratio: {debt_ratio:.1f}%"
                )
                return summary
            
        elif statement_type == 'cash_flow':
            # Find cash flow columns
            numeric_cols = [col for col in self.df.columns if pd.api.types.is_numeric_dtype(self.df[col])]
            operating_col = self._find_column_by_keywords(['operating', 'operations'], numeric_cols)
            investing_col = self._find_column_by_keywords(['investing', 'investment'], numeric_cols)
            financing_col = self._find_column_by_keywords(['financing', 'finance'], numeric_cols)
            
            if operating_col or investing_col or financing_col:
                summary = "Cash Flow Summary\n\n"
                
                if operating_col:
                    latest_op = self.df[operating_col].iloc[-1]
                    summary += f"Operating Cash Flow: {latest_op:,.0f}\n"
                    
                if investing_col:
                    latest_inv = self.df[investing_col].iloc[-1]
                    summary += f"Investing Cash Flow: {latest_inv:,.0f}\n"
                    
                if financing_col:
                    latest_fin = self.df[financing_col].iloc[-1]
                    summary += f"Financing Cash Flow: {latest_fin:,.0f}\n"
                
                total = 0
                if operating_col:
                    total += self.df[operating_col].iloc[-1]
                if investing_col:
                    total += self.df[investing_col].iloc[-1]
                if financing_col:
                    total += self.df[financing_col].iloc[-1]
                    
                summary += f"Net Cash Flow: {total:,.0f}"
                return summary
            
        return "Financial Summary\n\nKey metrics visualization complete.\nThank you for using Financial Animation Engine!"
    
    def _create_summary_slide(self, summary_text, temp_dir, base_name, fps=24, duration=3):
        """Create a summary slide with multiple key metrics"""
        # Create a figure with summary text
        fig = go.Figure()
        
        # Split the summary text into title and body
        lines = summary_text.strip().split('\n')
        title = lines[0]
        body = '\n'.join(lines[2:])
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=32, color="darkblue"),
                x=0.5,
                y=0.9
            ),
            annotations=[
                dict(
                    text=body,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=24),
                    align="center"
                )
            ],
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=1280,
            height=720
        )
        
        # Save the image
        image_path = os.path.join(temp_dir, f"{base_name}.png")
        fig.write_image(image_path)
        
        # Create clip
        clip = ImageClip(image_path).set_duration(duration)
        
        return clip


# Example usage
def example_usage():
    """Example of how to use the Financial Animation Engine"""
    # Sample income statement data
    income_data = {
        'Quarter': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024'],
        'Revenue': [1200000, 1350000, 1450000, 1600000, 1750000],
        'Cost of Goods Sold': [720000, 810000, 870000, 960000, 1050000],
        'Gross Profit': [480000, 540000, 580000, 640000, 700000],
        'Operating Expenses': [350000, 375000, 390000, 430000, 460000],
        'Net Income': [130000, 165000, 190000, 210000, 240000]
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(income_data)
    
    # Create the animator
    animator = FinancialAnimator(df=df)
    
    # Prepare data for visualization
    statement_type = animator.prepare_data()
    print(f"Detected statement type: {statement_type}")
    
    # Create the animation
    output_path = animator.create_animated_infographic(
        output_path='income_statement_animation.mp4',
        fps=24,
        duration=15)
    
    print(f"Animation created: {output_path}")
    
    return animator, output_path


def create_sample_csv(output_path='sample_income_statement.csv'):
    """Create a sample income statement CSV file for testing"""
    # Sample income statement data
    income_data = {
        'Quarter': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024'],
        'Revenue': [1200000, 1350000, 1450000, 1600000, 1750000],
        'Cost of Goods Sold': [720000, 810000, 870000, 960000, 1050000],
        'Gross Profit': [480000, 540000, 580000, 640000, 700000],
        'Operating Expenses': [350000, 375000, 390000, 420000, 460000],
        'Net Income': [130000, 165000, 190000, 220000, 240000]
    }
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(income_data)
    df.to_csv(output_path, index=False)
    
    return output_path


def create_sample_balance_sheet(output_path='sample_balance_sheet.csv'):
    """Create a sample balance sheet CSV file for testing"""
    balance_data = {
        'Date': ['Dec 31 2022', 'Mar 31 2023', 'Jun 30 2023', 'Sep 30 2023', 'Dec 31 2023'],
        'Cash and Equivalents': [500000, 550000, 600000, 650000, 700000],
        'Accounts Receivable': [300000, 320000, 340000, 360000, 380000],
        'Inventory': [450000, 470000, 490000, 510000, 530000],
        'Property and Equipment': [1200000, 1180000, 1160000, 1140000, 1120000],
        'Total Assets': [2450000, 2520000, 2590000, 2660000, 2730000],
        'Accounts Payable': [200000, 210000, 220000, 230000, 240000],
        'Short-term Debt': [300000, 290000, 280000, 270000, 260000],
        'Long-term Debt': [800000, 780000, 760000, 740000, 720000],
        'Total Liabilities': [1300000, 1280000, 1260000, 1240000, 1220000],
        'Shareholder Equity': [1150000, 1240000, 1330000, 1420000, 1510000]
    }
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(balance_data)
    df.to_csv(output_path, index=False)
    
    return output_path


def create_sample_cash_flow(output_path='sample_cash_flow.csv'):
    """Create a sample cash flow statement CSV file for testing"""
    cash_flow_data = {
        'Quarter': ['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024'],
        'Operating Activities': [180000, 195000, 210000, 225000, 240000],
        'Investing Activities': [-120000, -150000, -90000, -180000, -130000],
        'Financing Activities': [-50000, -60000, -70000, -80000, -90000],
        'Net Cash Flow': [10000, -15000, 50000, -35000, 20000],
        'Beginning Cash Balance': [500000, 510000, 495000, 545000, 510000],
        'Ending Cash Balance': [510000, 495000, 545000, 510000, 530000]
    }
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(cash_flow_data)
    df.to_csv(output_path, index=False)
    
    return output_path


def main():
    """Main function to run the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Financial Statement Animation Engine')
    parser.add_argument('--csv', type=str, help='Path to CSV file with financial data')
    parser.add_argument('--output', type=str, default='financial_animation.mp4', 
                        help='Output path for the animation video')
    parser.add_argument('--sample', type=str, choices=['income', 'balance', 'cash_flow'], 
                        help='Create and use a sample dataset')
    parser.add_argument('--duration', type=int, default=15, 
                        help='Duration of the animation in seconds')
    parser.add_argument('--fps', type=int, default=24,
                        help='Frames per second for the animation')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.sample:
        if args.sample == 'income':
            csv_path = create_sample_csv()
            print(f"Created sample income statement at {csv_path}")
        elif args.sample == 'balance':
            csv_path = create_sample_balance_sheet()
            print(f"Created sample balance sheet at {csv_path}")
        elif args.sample == 'cash_flow':
            csv_path = create_sample_cash_flow()
            print(f"Created sample cash flow statement at {csv_path}")
    else:
        csv_path = args.csv
    
    # Check if CSV path is provided
    if not csv_path:
        print("Error: No CSV file specified. Use --csv or --sample option.")
        return
    
    # Create animator from CSV
    animator = FinancialAnimator(csv_path=csv_path)
    
    # Prepare data
    statement_type = animator.prepare_data()
    print(f"Detected statement type: {statement_type}")
    
    # Create animation
    output_path = animator.create_animated_infographic(
        output_path=args.output,
        fps=args.fps,
        duration=args.duration
    )
    
    print(f"Animation created: {output_path}")


if __name__ == "__main__":
    main()
