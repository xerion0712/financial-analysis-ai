# tests/test_data_collection.py

import sys
import os
# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch
import tempfile
from modules.data_collection import (
    YahooFinanceCollector,
    CSVDataCollector,
    DataAggregator,
    DataCollector
)
# ----------------------------------------------------------------------------
# Dummy collector to simulate a data source without external API calls
# ----------------------------------------------------------------------------
class DummyCollector(DataCollector):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        return self.df

# ----------------------------------------------------------------------------
# Pytest fixtures
# ----------------------------------------------------------------------------
@pytest.fixture
def multiindex_df():
    """
    Create a dummy DataFrame with a MultiIndex for columns
    simulating the output from yf.download.
    """
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    arrays = [
        ['Close', 'High', 'Low', 'Open', 'Volume'],
        ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL']
    ]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=["Price", "Ticker"])
    data = {
        'Close': [150, 152, 151, 153, 154],
        'High': [151, 153, 152, 154, 155],
        'Low': [149, 150, 150, 152, 153],
        'Open': [150, 151, 151, 153, 154],
        'Volume': [1000000, 1100000, 1050000, 1200000, 1150000]
    }
    df = pd.DataFrame(data, index=dates)
    df.columns = multi_index
    return df

@pytest.fixture
def temp_csv_file():
    """
    Create a temporary CSV file with sample data,
    yield its path, and then delete it after the test.
    """
    temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv")
    df_csv = pd.DataFrame(
        {
            "Close": [150, 152, 151],
            "Volume": [1000000, 1100000, 1050000]
        },
        index=pd.date_range(start="2023-01-01", periods=3, freq="D")
    )
    df_csv.to_csv(temp_file.name)
    temp_file.close()
    yield temp_file.name
    os.remove(temp_file.name)

@pytest.fixture
def dummy_dates():
    """Fixture returning a standard date range for aggregator tests."""
    return pd.date_range(start="2023-01-01", periods=5, freq="D")

# ----------------------------------------------------------------------------
# Test YahooFinanceCollector
# ----------------------------------------------------------------------------
def test_fetch_data_yahoo(multiindex_df):
    """Test that YahooFinanceCollector flattens the MultiIndex and returns expected columns."""
    with patch("modules.data_collection.yf.download", return_value=multiindex_df.copy()):
        collector = YahooFinanceCollector()
        df_result = collector.fetch_data("AAPL", "2023-01-01", "2023-01-05")
        assert "Close" in df_result.columns
        assert "High" in df_result.columns
        assert "Low" in df_result.columns
        assert "Open" in df_result.columns
        assert "Volume" in df_result.columns
        # Ensure no column name equals the ticker in any case
        for col in df_result.columns:
            assert col.lower() != "aapl"

def test_yahoo_finance_datetime_index(multiindex_df):
    """Test that the returned DataFrame has a DatetimeIndex."""
    with patch("modules.data_collection.yf.download", return_value=multiindex_df.copy()):
        collector = YahooFinanceCollector()
        df_result = collector.fetch_data("AAPL", "2023-01-01", "2023-01-05")
        assert isinstance(df_result.index, pd.DatetimeIndex)
        expected_first_date = pd.Timestamp("2023-01-01")
        assert df_result.index[0] == expected_first_date

# ----------------------------------------------------------------------------
# Test CSVDataCollector
# ----------------------------------------------------------------------------
def test_csv_collector_existing_file(temp_csv_file):
    """Test that CSVDataCollector loads data from an existing CSV file."""
    collector = CSVDataCollector(filepath=temp_csv_file)
    df_result = collector.fetch_data()
    assert not df_result.empty
    assert "Close" in df_result.columns
    # Check if the index is DatetimeIndex
    assert isinstance(df_result.index, pd.DatetimeIndex)

def test_csv_collector_file_not_found():
    """Test that CSVDataCollector returns an empty DataFrame if file not found."""
    collector = CSVDataCollector(filepath="non_existent_file.csv")
    df_result = collector.fetch_data()
    assert df_result.empty

# ----------------------------------------------------------------------------
# Test DataAggregator
# ----------------------------------------------------------------------------
def test_aggregate_data_success(dummy_dates):
    """
    Test that DataAggregator successfully aggregates data from multiple collectors
    and computes the average close.
    """
    df1 = pd.DataFrame({"Close": [150, 152, 151, 153, 154]}, index=dummy_dates)
    df2 = pd.DataFrame({"Close": [151, 153, 152, 154, 155]}, index=dummy_dates)

    collector1 = DummyCollector(df1)
    collector2 = DummyCollector(df2)

    aggregator = DataAggregator(collectors=[collector1, collector2])
    aggregated_df = aggregator.aggregate_data("AAPL", "2023-01-01", "2023-01-05")
    assert "Average_Close" in aggregated_df.columns

    expected_avg = (
        pd.Series([150, 152, 151, 153, 154], index=dummy_dates) +
        pd.Series([151, 153, 152, 154, 155], index=dummy_dates)
    ) / 2

    pd.testing.assert_series_equal(aggregated_df["Average_Close"], expected_avg, check_names=False)

def test_aggregate_data_no_data():
    """Test that DataAggregator raises ValueError when no collector returns data."""
    empty_df = pd.DataFrame()
    empty_collector = DummyCollector(empty_df)
    aggregator = DataAggregator(collectors=[empty_collector])
    with pytest.raises(ValueError):
        aggregator.aggregate_data("AAPL", "2023-01-01", "2023-01-05")

def test_aggregator_index_is_datetime(dummy_dates):
    """Test that the aggregator output uses a DatetimeIndex."""
    df1 = pd.DataFrame({"Close": [150, 152, 151, 153, 154]}, index=dummy_dates)
    df2 = pd.DataFrame({"Close": [151, 153, 152, 154, 155]}, index=dummy_dates)

    collector1 = DummyCollector(df1)
    collector2 = DummyCollector(df2)
    aggregator = DataAggregator(collectors=[collector1, collector2])

    aggregated_df = aggregator.aggregate_data("AAPL", "2023-01-01", "2023-01-05")
    assert isinstance(aggregated_df.index, pd.DatetimeIndex)
    pd.testing.assert_index_equal(aggregated_df.index, dummy_dates)