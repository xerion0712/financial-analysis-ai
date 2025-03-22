# tests/test_data_analysis.py

import sys
import os

# Ensure that the modules can be imported correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from modules.data_analysis import (
    AnalysisMethod,
    SummaryStatistics,
    MovingAverageAnalysis,
    DataAnalyzer
)


# =============================================================================
# Test Data Setup
# =============================================================================
@pytest.fixture
def sample_data():
    """
    Provides a sample DataFrame with 'Close' and 'Average_Close' columns.
    The DataFrame has a DateTime index and random values for testing.
    """
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Close": np.random.uniform(100, 200, size=10),
        "Average_Close": np.random.uniform(100, 200, size=10)
    }, index=dates)
    return df


@pytest.fixture
def sample_data_with_ma():
    """
    Provides a sample DataFrame with 'Average_Close' and pre-computed
    moving average columns for testing.
    """
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    close_data = np.random.uniform(100, 200, size=10)
    df = pd.DataFrame({
        "Average_Close": close_data,
        "moving_average_20": pd.Series([None]*9 + [np.mean(close_data)]),
    }, index=dates)
    return df


@pytest.fixture
def empty_data():
    """
    Provides an empty DataFrame to test how the module handles no data.
    """
    return pd.DataFrame()


# =============================================================================
# Tests for Abstract Base Class (AnalysisMethod)
# =============================================================================
def test_abstract_base_class():
    """
    Verifies that AnalysisMethod cannot be instantiated directly.
    """
    with pytest.raises(TypeError):
        AnalysisMethod()


# =============================================================================
# Tests for SummaryStatistics
# =============================================================================
def test_summary_statistics_with_average_close(sample_data):
    """
    Ensures that SummaryStatistics uses 'Average_Close' when present.
    """
    stats = SummaryStatistics()
    result = stats.analyze(sample_data)
    assert "mean" in result
    assert "median" in result
    assert "std" in result
    assert result["mean"] is not None

def test_summary_statistics_fallback_close():
    """
    Ensures that SummaryStatistics falls back to 'Close' if 'Average_Close' is absent.
    """
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    df_no_avg = pd.DataFrame({"Close": [10, 20, 30, 40, 50]}, index=dates)
    stats = SummaryStatistics()
    result = stats.analyze(df_no_avg)
    assert "mean" in result
    assert "median" in result
    assert "std" in result
    # Mean should be (10+20+30+40+50)/5 = 30
    assert result["mean"] == 30

def test_summary_statistics_empty_dataframe(empty_data):
    """
    Checks if SummaryStatistics gracefully handles an empty DataFrame.
    """
    stats = SummaryStatistics()
    result = stats.analyze(empty_data)
    assert result == {}  # Should return an empty dictionary due to exception


# =============================================================================
# Tests for MovingAverageAnalysis
# =============================================================================
def test_moving_average_basic(sample_data):
    """
    Verifies that MovingAverageAnalysis computes correct rolling means
    for the specified windows.
    """
    ma = MovingAverageAnalysis(windows=[2, 3])
    result = ma.analyze(sample_data)
    assert "moving_average_2" in result
    assert "moving_average_3" in result
    # First value for each MA should be NaN, second value should be the mean of the first two data points
    first_window_val_2 = sample_data["Average_Close"][:2].mean()
    # We check the second row of the resulting series
    second_index = sample_data.index[1]
    assert pytest.approx(result["moving_average_2"][second_index]) == first_window_val_2

def test_moving_average_fallback_close():
    """
    Verifies that MovingAverageAnalysis uses 'Close' when 'Average_Close' is unavailable.
    """
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    df_no_avg = pd.DataFrame({"Close": [10, 20, 30, 40, 50]}, index=dates)
    ma = MovingAverageAnalysis(windows=[2])
    result = ma.analyze(df_no_avg)
    assert "moving_average_2" in result
    # The second row's value should be the mean of the first two data points (10 and 20)
    second_index = df_no_avg.index[1]
    assert pytest.approx(result["moving_average_2"][second_index]) == 15

def test_moving_average_empty_dataframe(empty_data):
    """
    Ensures that MovingAverageAnalysis returns an empty dict when the DataFrame is empty.
    """
    ma = MovingAverageAnalysis()
    result = ma.analyze(empty_data)
    assert result == {}


# =============================================================================
# Tests for DataAnalyzer (integration of analysis methods)
# =============================================================================
def test_data_analyzer_register_methods(sample_data):
    """
    Checks if DataAnalyzer can register and execute multiple analysis methods.
    """
    analyzer = DataAnalyzer()
    analyzer.register_analysis_method(SummaryStatistics())
    analyzer.register_analysis_method(MovingAverageAnalysis(windows=[2]))
    
    final_results = analyzer.analyze(sample_data)
    assert "SummaryStatistics" in final_results
    assert "MovingAverageAnalysis" in final_results

def test_data_analyzer_with_precomputed_ma(sample_data_with_ma):
    """
    Ensures that if some columns (like moving averages) are already present,
    the analysis still runs without conflict.
    """
    analyzer = DataAnalyzer()
    analyzer.register_analysis_method(MovingAverageAnalysis(windows=[2]))
    
    # This shouldn't raise any errors; it just creates new keys for the newly computed MAs.
    final_results = analyzer.analyze(sample_data_with_ma)
    assert "MovingAverageAnalysis" in final_results
    # The new MAs should not overwrite existing columns in the DataFrame itself,
    # because each analysis method returns its results separately.

def test_data_analyzer_empty_data(empty_data):
    """
    Ensures that DataAnalyzer handles empty data gracefully.
    """
    analyzer = DataAnalyzer()
    analyzer.register_analysis_method(SummaryStatistics())
    analyzer.register_analysis_method(MovingAverageAnalysis())
    
    final_results = analyzer.analyze(empty_data)
    # We expect the results for each method to be empty dict or partial success
    assert final_results["SummaryStatistics"] == {}
    assert final_results["MovingAverageAnalysis"] == {}
