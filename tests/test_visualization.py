import sys
import os

# Add the parent directory to sys.path so that the test can import modules correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from modules.visualization import FinancialDataVisualizer, BaseVisualizer

@pytest.fixture
def visualizer():
    """
    Create a FinancialDataVisualizer instance for reuse across tests.
    """
    return FinancialDataVisualizer()

def test_empty_dataframe(visualizer, caplog):
    import logging
    # Ensure we capture ERROR logs from this specific logger or root logger
    caplog.set_level(logging.ERROR)  # or caplog.set_level(logging.ERROR, logger="modules.visualization")
    
    empty_df = pd.DataFrame()
    charts = visualizer.create_charts(empty_df)

    # 1) Expect empty dictionary
    assert charts == {}, "Expected an empty dictionary for empty DataFrame"
    
    # 2) Expect log message to contain 'empty'
    logged_messages = caplog.text.lower()
    print("DEBUG: Captured logs:", logged_messages)  # Debug print if needed
    assert "input dataframe is empty" in logged_messages, (
        "Expected an error log message about the empty DataFrame"
    )

def test_no_price_columns(visualizer, caplog):
    """
    Test that if neither 'Average_Close' nor 'Close' is present,
    the method returns an empty dictionary and logs an error.
    """
    caplog.set_level("ERROR")  # Ensure we capture error messages
    
    df = pd.DataFrame({
        "Open": [10, 11, 12],
        "High": [11, 12, 13],
        "Low": [9, 10, 11]
    }, index=pd.date_range("2023-01-01", periods=3, freq="D"))
    
    charts = visualizer.create_charts(df)
    
    # 1) We expect an empty dict
    assert charts == {}, "Expected an empty dictionary if no price columns are available"
    
    # 2) We expect a specific error message in the logs
    logs_lower = caplog.text.lower()
    print("DEBUG LOGS:", logs_lower)  # Debug print if needed
    
    assert "no valid price column" in logs_lower, (
        "Expected an error log message about missing price column"
    )
def test_non_datetime_index(visualizer, caplog):
    """
    Test that if the DataFrame index is not datetime, it is converted automatically.
    """
    # Устанавливаем уровень логирования на INFO, так как в коде 
    # обычно используется logging.info() для уведомления о конвертации
    caplog.set_level("INFO")

    # Create a DataFrame with numeric index
    df = pd.DataFrame({
        "Close": [150, 152, 153],
    }, index=[0, 1, 2])
    
    charts = visualizer.create_charts(df)

    
    assert charts != {}, "Expected a non-empty dictionary for valid data"

    
    assert "price_trend" in charts, "Expected 'price_trend' in charts"

    
    logs_lower = caplog.text.lower()
    print("DEBUG LOGS:", logs_lower)

    assert any("converting dataframe index to datetime" in message 
               for message in logs_lower.splitlines()), \
        "Expected a log about converting the index to datetime"


def test_valid_data_with_close(visualizer):
    """
    Test that a valid DataFrame with 'Close' column returns a dictionary
    containing a Bokeh figure. Also check if the figure is indeed from Bokeh.
    """
    
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Close": [150, 151, 152, 153, 154]
    }, index=dates)
    
    charts = visualizer.create_charts(df)
    assert "price_trend" in charts, "Expected 'price_trend' in charts for valid data"
    assert isinstance(charts["price_trend"], figure), "Expected a Bokeh Figure object"

def test_valid_data_with_average_close(visualizer):
    """
    Test that a valid DataFrame with 'Average_Close' column returns
    a dictionary containing a Bokeh figure.
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Average_Close": [155, 156, 155.5, 156.7, 157]
    }, index=dates)
    
    charts = visualizer.create_charts(df)
    assert "price_trend" in charts, "Expected 'price_trend' in charts for valid data"
    assert isinstance(charts["price_trend"], figure), "Expected a Bokeh Figure object"

def test_moving_averages_overlay(visualizer):
    """
    Test that if the DataFrame includes columns like 'moving_average_20' or 'moving_average_50',
    they are handled properly (the 'price_trend' chart should still be created).
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Close": [150, 152, 153, 155, 158],
        "moving_average_20": [151, 151.5, 152, 153, 154],
        "moving_average_50": [150.5, 151, 152, 152.5, 153]
    }, index=dates)
    
    charts = visualizer.create_charts(df)
    assert "price_trend" in charts, "Expected 'price_trend' in charts"
    # Optionally, you could analyze the figure's renderers or legends if needed,
    # but here we simply check that the chart is created successfully.



def test_abstract_base_class():
    """
    Ensures that BaseVisualizer cannot be instantiated directly.
    """
    with pytest.raises(TypeError):
        BaseVisualizer()