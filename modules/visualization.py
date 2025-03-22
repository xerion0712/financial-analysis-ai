import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

# Import Bokeh components
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter
from bokeh.layouts import gridplot

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Enable Bokeh output in the notebook; if running as a script, consider using output_file().
output_file("plots_test.html")

class BaseVisualizer(ABC):
    """
    Abstract base class for visualization modules.
    Defines a unified interface for creating charts from aggregated financial data.
    """
    @abstractmethod
    def create_charts(self, data: pd.DataFrame) -> dict:
        """
        Abstract method to create visualizations.

        Parameters:
            data (pd.DataFrame): Aggregated financial data with a DateTime index.

        Returns:
            dict: Dictionary containing chart objects keyed by chart names.
        """
        pass

class FinancialDataVisualizer(BaseVisualizer):
    """
    Concrete visualizer class for financial data using Bokeh.
    Implements the BaseVisualizer interface to create an interactive time series chart for price trends,
    with overlays for moving averages.
    """
    
    def create_charts(self, data: pd.DataFrame) -> dict:
        """
        Creates an interactive time series chart using Bokeh.
        The chart shows the price trend based on the "Average_Close" or "Close" column and overlays available moving averages.
        
        Parameters:
            data (pd.DataFrame): Aggregated financial data with a DateTime index.
            
        Returns:
            dict: A dictionary containing Bokeh Figure objects keyed by chart names.
        """
        charts = {}
        try:
            # Validate that the DataFrame is not empty
            if data.empty:
                logging.error("Input DataFrame is empty. Cannot create charts.")
                return charts

            # Ensure the DataFrame index is datetime type; if not, convert it.
            if not pd.api.types.is_datetime64_any_dtype(data.index):
                logging.info("Converting DataFrame index to datetime.")
                data.index = pd.to_datetime(data.index)
            
            # Select the appropriate price column: prefer "Average_Close", otherwise "Close"
            if "Average_Close" in data.columns:
                price_column = "Average_Close"
            elif "Close" in data.columns:
                price_column = "Close"
            else:
                logging.error("No valid price column ('Average_Close' or 'Close') found in the DataFrame.")
                return charts
            
            logging.info(f"Using '{price_column}' as the price column for the chart.")
            
            # Reset index to use date as a column and create a ColumnDataSource for Bokeh
            data_reset = data.reset_index().rename(columns={'index': 'Date'})
            source = ColumnDataSource(data_reset)
            
            # Create a Bokeh figure with datetime x-axis
            p = figure(title="Price Trend", x_axis_type="datetime", 
                       width=900, height=400,
                       tools="pan,wheel_zoom,box_zoom,reset,save")
            
            # Plot the price trend line
            p.line(x='Date', y=price_column, source=source, line_width=2, color="navy", legend_label=price_column)
            
            # Overlay moving average lines if available (columns starting with "moving_average_")
            moving_average_columns = [col for col in data.columns if col.startswith("moving_average_")]
            if moving_average_columns:
                logging.info(f"Overlaying moving averages: {moving_average_columns}")
                for ma_col in moving_average_columns:
                    p.line(x='Date', y=ma_col, source=source, line_width=2, legend_label=ma_col)
            else:
                logging.info("No moving average columns found for overlay.")
            
            # Configure x-axis to show dates in a readable format
            p.xaxis.formatter = DatetimeTickFormatter(days="%d %b %Y")
            p.xaxis.axis_label = "Date"
            p.yaxis.axis_label = "Price"
            
            # Add interactive hover tool to display date and price
            hover = HoverTool(
                tooltips=[
                    ("Date", "@Date{%F}"),
                    (price_column, f"@{price_column}{{0.2f}}")
                ],
                formatters={"@Date": "datetime"},
                mode="vline"
            )
            p.add_tools(hover)
            
            # Position the legend
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"  # allow hiding lines by clicking on legend entries
            
            # Save the figure in the charts dictionary
            charts["price_trend"] = p
            logging.info("Bokeh price trend chart created successfully.")
            
        except Exception as e:
            logging.exception("An error occurred while creating Bokeh charts: %s", e)
        
        return charts

# ---------------------------
# Example usage of the Visualization Module with Bokeh
# ---------------------------
if __name__ == "__main__":
    # Create sample aggregated financial data for demonstration purposes.
    # In a real scenario, this data would come from the Data Collection Module.
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    sample_data = pd.DataFrame({
        "Close": np.random.rand(100) * 50 + 150,            # Simulated closing prices
        "Average_Close": np.random.rand(100) * 50 + 152,      # Simulated average closing prices
        "moving_average_20": np.random.rand(100) * 50 + 149,
        "moving_average_50": np.random.rand(100) * 50 + 147
    }, index=dates)
    
    # Instantiate the visualizer and create charts using the sample data.
    visualizer = FinancialDataVisualizer()
    charts = visualizer.create_charts(sample_data)
    
    # Display the "price_trend" chart if it was created successfully.
    if "price_trend" in charts:
        show(charts["price_trend"])  # Opens an interactive Bokeh plot in the browser or notebook.
