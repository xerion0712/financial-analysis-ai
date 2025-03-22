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

# ---------------- Forecast Visualization Functions ----------------

def get_forecast_series(forecast_df: pd.DataFrame) -> pd.Series:
    """
    Returns the forecast series from the given DataFrame.
    
    Priority:
      1. Use the 'Forecast' column if it exists.
      2. Otherwise, use the 'yhat' column (commonly returned by Prophet).
      3. Otherwise, use the first column of the DataFrame.
    
    Parameters:
        forecast_df (pd.DataFrame): DataFrame with forecast data.
    
    Returns:
        pd.Series: A Pandas Series with forecast values.
    """
    if "Forecast" in forecast_df.columns:
        return forecast_df["Forecast"]
    elif "yhat" in forecast_df.columns:
        return forecast_df["yhat"]
    else:
        return forecast_df.iloc[:, 0]

def create_forecast_chart(forecast_df: pd.DataFrame, title: str):
    """
    Create a Bokeh line chart for a single forecast.
    
    Parameters:
        forecast_df (pd.DataFrame): DataFrame with forecast results.
        title (str): Title of the chart.
    
    Returns:
        Bokeh Figure.
    """
    p = figure(title=title, x_axis_type='datetime', width=600, height=400)
    forecast_values = get_forecast_series(forecast_df)
    p.line(forecast_df.index, forecast_values, line_width=2, legend_label=title)
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Forecast"
    return p

def create_combined_forecast_chart(prophet_forecast_df: pd.DataFrame, 
                                   arima_forecast_df: pd.DataFrame, 
                                   ensemble_forecast_df: pd.DataFrame, 
                                   title: str):
    """
    Create a combined Bokeh chart displaying forecast lines for Prophet, ARIMA, and Ensemble.
    
    Parameters:
        prophet_forecast_df (pd.DataFrame): DataFrame for Prophet forecast.
        arima_forecast_df (pd.DataFrame): DataFrame for ARIMA forecast.
        ensemble_forecast_df (pd.DataFrame): DataFrame for Ensemble forecast.
        title (str): Title of the chart.
    
    Returns:
        Bokeh Figure.
    """
    p = figure(title=title, x_axis_type='datetime', width=800, height=400)
    
    if prophet_forecast_df is not None:
        prophet_values = get_forecast_series(prophet_forecast_df)
        p.line(prophet_forecast_df.index, prophet_values, line_width=2, color="blue", legend_label="Prophet")
    
    if arima_forecast_df is not None:
        arima_values = get_forecast_series(arima_forecast_df)
        p.line(arima_forecast_df.index, arima_values, line_width=2, color="green", legend_label="ARIMA")
    
    if ensemble_forecast_df is not None:
        ensemble_values = get_forecast_series(ensemble_forecast_df)
        p.line(ensemble_forecast_df.index, ensemble_values, line_width=2, color="red", legend_label="Ensemble")
    
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Forecast"
    p.legend.location = "top_left"
    return p
