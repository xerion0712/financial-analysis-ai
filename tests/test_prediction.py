import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Adjust sys.path to include the parent directory so that tests can import our module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the forecasting classes from our module.
from modules.prediction import ProphetPredictor, ArimaPredictor, ForecastEvaluator

# Fixture to create sample data for testing
@pytest.fixture
def sample_data():
    """
    Creates a sample DataFrame with a DateTime index and valid values for testing.
    Ensure there are enough non-NaN rows for the Prophet model.
    """
    # Create a date range with 50 days
    dates = pd.date_range(start='2022-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        "Close": np.arange(50) + 100,
        "Average_Close": np.arange(50) + 100,  # Valid non-NaN values
        "moving_average_20": np.arange(50) + 95,
        "moving_average_50": np.arange(50) + 90,
        "Volume": [1000] * 50
    }, index=dates)
    return data

# Fixture to split data into training and holdout sets
@pytest.fixture
def train_holdout_data(sample_data):
    """
    Splits the sample data into training and holdout sets.
    Training data: all but the last 30 days.
    Holdout data: the last 30 days for evaluation.
    """
    train_data = sample_data.iloc[:-30]
    holdout_data = sample_data.iloc[-30:]
    return train_data, holdout_data

def test_prophet_predictor_structure(train_holdout_data):
    """
    Test that the ProphetPredictor returns a forecast with the expected structure.
    """
    train_data, _ = train_holdout_data
    forecast_horizon = 30
    predictor = ProphetPredictor(target_column="Average_Close")
    result = predictor.train_predict(train_data, forecast_horizon)
    
    # Check that the returned dictionary contains the forecast and metadata
    assert "Forecast" in result
    assert "Model_Metadata" in result
    
    forecast_df = result["Forecast"]
    # Verify that the DataFrame contains the required columns
    expected_columns = ['y', 'yhat', 'yhat_lower', 'yhat_upper']
    for col in expected_columns:
        assert col in forecast_df.columns

    # Verify that the forecast part (future dates) has exactly forecast_horizon rows
    future_rows = forecast_df[forecast_df['y'].isna()]
    assert len(future_rows) == forecast_horizon

def test_arima_predictor_structure(train_holdout_data):
    """
    Test that the ArimaPredictor returns a forecast with the expected structure.
    """
    train_data, _ = train_holdout_data
    forecast_horizon = 30
    predictor = ArimaPredictor(target_column="Average_Close")
    result = predictor.train_predict(train_data, forecast_horizon)
    
    # Check for keys in the returned result
    assert "Forecast" in result
    assert "Model_Metadata" in result
    
    forecast_df = result["Forecast"]
    # Verify required columns exist in the DataFrame
    expected_columns = ['y', 'yhat', 'yhat_lower', 'yhat_upper']
    for col in expected_columns:
        assert col in forecast_df.columns

    # Verify that forecasted values in 'yhat' are not all NaN for future dates
    future_rows = forecast_df[forecast_df['y'].isna()]
    assert future_rows['yhat'].notna().sum() > 0

def test_forecast_evaluator_evaluate_on_holdout(train_holdout_data):
    """
    Test that the ForecastEvaluator properly evaluates forecasts against holdout data.
    """
    train_data, holdout_data = train_holdout_data
    forecast_horizon = 30
    
    # Generate forecasts from both models
    prophet = ProphetPredictor(target_column="Average_Close")
    arima = ArimaPredictor(target_column="Average_Close")
    prophet_result = prophet.train_predict(train_data, forecast_horizon)
    arima_result = arima.train_predict(train_data, forecast_horizon)
    
    evaluator = ForecastEvaluator()
    evaluator.add_forecast("Prophet", prophet_result)
    evaluator.add_forecast("ARIMA", arima_result)
    
    metrics_df = evaluator.evaluate_on_holdout(holdout_data, "Average_Close")
    
    # Verify that evaluation metrics are returned and contain expected columns
    assert not metrics_df.empty
    for metric in ['Model', 'MAE', 'RMSE', 'MAPE', 'Sample_Size']:
        assert metric in metrics_df.columns

def test_forecast_evaluator_plot_comparison(train_holdout_data):
    """
    Test that the plot_comparison method returns a Bokeh figure.
    """
    train_data, _ = train_holdout_data
    forecast_horizon = 30
    # Generate forecasts from both models
    prophet = ProphetPredictor(target_column="Average_Close")
    arima = ArimaPredictor(target_column="Average_Close")
    prophet_result = prophet.train_predict(train_data, forecast_horizon)
    arima_result = arima.train_predict(train_data, forecast_horizon)
    
    evaluator = ForecastEvaluator()
    evaluator.add_forecast("Prophet", prophet_result)
    evaluator.add_forecast("ARIMA", arima_result)
    
    fig = evaluator.plot_comparison(historical_data=train_data.iloc[-60:], target_column="Average_Close")
    
    # Import Bokeh Figure type to verify the returned object
    from bokeh.plotting import figure
    assert isinstance(fig, figure)

def test_ensemble_forecast(train_holdout_data):
    """
    Test that the create_ensemble_forecast method creates an ensemble forecast.
    """
    train_data, _ = train_holdout_data
    forecast_horizon = 30
    
    # Generate forecasts from both models
    prophet = ProphetPredictor(target_column="Average_Close")
    arima = ArimaPredictor(target_column="Average_Close")
    prophet_result = prophet.train_predict(train_data, forecast_horizon)
    arima_result = arima.train_predict(train_data, forecast_horizon)
    
    evaluator = ForecastEvaluator()
    evaluator.add_forecast("Prophet", prophet_result)
    evaluator.add_forecast("ARIMA", arima_result)
    
    ensemble_result = evaluator.create_ensemble_forecast()
    assert "Forecast" in ensemble_result
    assert "Model_Metadata" in ensemble_result
    ensemble_df = ensemble_result["Forecast"]
    # Check that ensemble forecast DataFrame has the ensemble 'yhat' column
    assert 'yhat' in ensemble_df.columns
