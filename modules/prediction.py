from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from prophet import Prophet
from pmdarima import auto_arima

from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, Band
from bokeh.io import show

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Abstract base class for forecasting models
class BasePredictor(ABC):
    """
    Abstract base class for forecasting models in the analytics module.

    Attributes:
        target_column (str): The name of the target column for forecasting.
    """
    
    def __init__(self, target_column: str = 'Average_Close'):
        """
        Initialize the predictor with the target column.

        Args:
            target_column (str): The target column name. Default is 'Average_Close'.
        """
        self.target_column = target_column
    
    @abstractmethod
    def train_predict(self, data: pd.DataFrame, forecast_horizon: int) -> Dict[str, Any]:
        """
        Trains the model and generates forecasts.

        Args:
            data (pd.DataFrame): Historical data with a DateTime index.
            forecast_horizon (int): Number of periods to forecast into the future.

        Returns:
            Dict[str, Any]: A dictionary containing the forecast DataFrame and model metadata.
                Example:
                {
                    "Forecast": pd.DataFrame,  # DataFrame with historical and forecasted values
                    "Model_Metadata": dict     # Information about the model (parameters, training time, etc.)
                }

        Raises:
            ValueError: If the data or parameters are invalid.
        """
        pass

# Implementation of the Prophet model
class ProphetPredictor(BasePredictor):
    """
    Forecasting class using the Prophet model for time series forecasting.
    """
    
    def train_predict(self, data: pd.DataFrame, forecast_horizon: int) -> Dict[str, Any]:
        """
        Trains the Prophet model and generates forecasts.

        Args:
            data (pd.DataFrame): Historical data with a DateTime index.
            forecast_horizon (int): Number of periods to forecast.

        Returns:
            Dict[str, Any]: A dictionary containing the forecast DataFrame and model metadata.
        """
        try:
            # Validate input data
            if self.target_column not in data.columns:
                error_msg = f"Target column '{self.target_column}' not found in data: {data.columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if forecast_horizon <= 0:
                logger.error("Forecast horizon must be a positive number.")
                raise ValueError("Forecast horizon must be a positive number.")
            if data.empty:
                logger.error("Input data is empty.")
                raise ValueError("Input data cannot be empty.")
            
            logger.info(f"Training Prophet model for '{self.target_column}' with a horizon of {forecast_horizon} periods.")
            
            # Prepare data for Prophet: requires 'ds' (date) and 'y' (value)
            prophet_data = data[[self.target_column]].reset_index()
            prophet_data.columns = ['ds', 'y']
            
            # Initialize and train the Prophet model
            model = Prophet()
            model.fit(prophet_data)
            
            # Generate future dates
            future = model.make_future_dataframe(periods=forecast_horizon)
            forecast = model.predict(future)
            
            # Form the output DataFrame
            forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            historical = prophet_data[['ds', 'y']]
            output_df = pd.merge(historical, forecast_output, on='ds', how='outer')
            output_df.sort_values('ds', inplace=True)
            output_df.set_index('ds', inplace=True)
            
            # Model metadata
            metadata = {
                "model": "Prophet",
                "target_column": self.target_column,
                "forecast_horizon": forecast_horizon,
                "n_observations": len(prophet_data)
            }
            
            logger.info(f"Generated forecast with {len(output_df)} rows, including {forecast_horizon} future periods.")
            
            return {"Forecast": output_df, "Model_Metadata": metadata}
        
        except Exception as e:
            logger.error(f"Error during training or forecasting with Prophet: {str(e)}")
            raise

# Implementation of the ARIMA model
class ArimaPredictor(BasePredictor):
    """
    Forecasting class using the ARIMA model with automatic parameter selection.
    """
    
    def train_predict(self, data: pd.DataFrame, forecast_horizon: int, 
                      forecast_start_date: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        try:
            if self.target_column not in data.columns:
                error_msg = f"Target column '{self.target_column}' not found in data: {data.columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            if forecast_horizon <= 0:
                logger.error("Forecast horizon must be a positive number.")
                raise ValueError("Forecast horizon must be a positive number.")
            if data.empty:
                logger.error("Input data is empty.")
                raise ValueError("Input data cannot be empty.")
            
            # Ensure index is DateTimeIndex and sorted
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            
            logger.info(f"Training ARIMA model for '{self.target_column}' with a horizon of {forecast_horizon} periods.")
            
            # Convert column to numeric type
            data[self.target_column] = pd.to_numeric(data[self.target_column], errors='coerce')
            
            # Extract time series and drop NaN values
            series = data[self.target_column].dropna()
            
            # Determine frequency robustly
            freq = pd.infer_freq(series.index)
            if freq is None:
                time_diffs = series.index[1:] - series.index[:-1]
                most_common_diff = pd.Series(time_diffs).value_counts().index[0]
                if most_common_diff.days == 1:
                    freq = 'D'
                elif most_common_diff.days == 7:
                    freq = 'W'
                elif 28 <= most_common_diff.days <= 31:
                    freq = 'M'
                else:
                    freq = 'D'  # Default to daily if undetermined
                logger.warning(f"Frequency determined from time differences: '{freq}'")
            else:
                logger.info(f"Data frequency determined: '{freq}'")
            
            # Train the ARIMA model
            model = auto_arima(series, seasonal=False, suppress_warnings=True, error_action='ignore')
            
            # Generate forecasts and confidence intervals
            forecasts, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
            logger.debug(f"Forecasts: {forecasts[:5]}, type: {type(forecasts)}")
            
            offset = pd.tseries.frequencies.to_offset(freq)
            # Определяем дату начала прогноза
            if forecast_start_date is not None:
                if forecast_start_date <= series.index[-1]:
                    logger.warning("forecast_start_date is earlier than or equal to the last training date. Using training data end instead.")
                    last_date = series.index[-1]
                else:
                    last_date = forecast_start_date - offset
                    logger.info(f"Using forecast_start_date: {forecast_start_date}, setting last_date to: {last_date}")
            else:
                last_date = series.index[-1]
            
            future_dates = pd.date_range(start=last_date + offset, periods=forecast_horizon, freq=freq)
            
            # Создаем DataFrame с прогнозом для будущих дат
            forecast_data = {
                'y': [None] * forecast_horizon,
                'yhat': forecasts.tolist() if isinstance(forecasts, np.ndarray) else list(forecasts),
                'yhat_lower': conf_int[:, 0].tolist(),
                'yhat_upper': conf_int[:, 1].tolist()
            }
            future_forecast_df = pd.DataFrame(forecast_data, index=future_dates)
            
            # Создаем исторический DataFrame на основе обучающего ряда
            historical_df = pd.DataFrame({
                'y': series,
                'yhat': None,
                'yhat_lower': None,
                'yhat_upper': None
            })
            
            # Объединяем исторические данные и прогноз
            output_df = pd.concat([historical_df, future_forecast_df])
            
            # Приводим колонки прогноза к типу float
            future_forecast_df['yhat'] = future_forecast_df['yhat'].astype(float)
            future_forecast_df['yhat_lower'] = future_forecast_df['yhat_lower'].astype(float)
            future_forecast_df['yhat_upper'] = future_forecast_df['yhat_upper'].astype(float)
            
            metadata = {
                "model": "ARIMA",
                "target_column": self.target_column,
                "forecast_horizon": forecast_horizon,
                "n_observations": len(series),
                "arima_order": model.order,
                "seasonal_order": model.seasonal_order,
                "frequency": freq
            }
            
            logger.info(f"Generated forecast with {len(output_df)} rows, including {forecast_horizon} future periods.")
            logger.info(f"Training data last date: {series.index[-1]}")
            logger.info(f"Forecast start date: {future_dates[0]}")
            logger.info(f"Forecast end date: {future_dates[-1]}")
            logger.info(f"Forecast length: {len(future_dates)}")
            return {"Forecast": output_df, "Future_Forecast": future_forecast_df, "Model_Metadata": metadata}
        
        except Exception as e:
            logger.error(f"Error during training or forecasting with ARIMA: {str(e)}", exc_info=True)
            raise







# Forecast evaluation and comparison class
class ForecastEvaluator:
    """
    Class for evaluating and comparing forecasts from different models.
    """
    
    def __init__(self):
        """Initialize the forecast evaluator."""
        self.results = {}
        
    def add_forecast(self, name: str, forecast_result: Dict[str, Any]):
        """
        Adds a forecast result from a model for later comparison.
        
        Args:
            name (str): Name of the model/forecast.
            forecast_result (Dict[str, Any]): The forecast result.
        """
        self.results[name] = forecast_result
        
        # Added debugging information for date ranges
        forecast_df = forecast_result.get("Future_Forecast", forecast_result["Forecast"])
        logger.info(f"Model '{name}' forecast range: {forecast_df.index.min()} to {forecast_df.index.max()}")
        logger.info(f"Model '{name}' forecast index type: {type(forecast_df.index)}")
        if hasattr(forecast_df.index, 'tz') and forecast_df.index.tz is not None:
            logger.info(f"Model '{name}' forecast timezone: {forecast_df.index.tz}")
        
        logger.info(f"Added forecast for model '{name}' for evaluation.")
    
    def evaluate_on_holdout(self, holdout_data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Evaluates all added forecasts on the holdout dataset.
        """
        metrics = []
        
        # Ensure the index is datetime
        if not isinstance(holdout_data.index, pd.DatetimeIndex):
            holdout_data.index = pd.to_datetime(holdout_data.index)
        
        # For each added model
        for model_name, result in self.results.items():
            # Use 'Future_Forecast' if available; otherwise, use the full 'Forecast'
            forecast_df = result.get("Future_Forecast", result["Forecast"])
            
            # Merge forecast with actual data
            evaluation_df = holdout_data[[target_column]].copy()
            evaluation_df.columns = ['actual']
            
            # Find common dates between holdout and forecast
            common_dates = evaluation_df.index.intersection(forecast_df.index)
            logger.info(f"Number of common dates for model {model_name}: {len(common_dates)}")
            if len(common_dates) == 0:
                logger.warning(f"No common dates for evaluation for model {model_name}.")
                continue
            
            # Add forecast for common dates
            evaluation_df.loc[common_dates, 'predicted'] = forecast_df.loc[common_dates, 'yhat']
            evaluation_df = evaluation_df.dropna()
            
            if len(evaluation_df) == 0:
                logger.warning(f"No data available to calculate metrics for model {model_name}.")
                continue
            
            # Calculate metrics
            mae = np.mean(np.abs(evaluation_df['actual'] - evaluation_df['predicted']))
            rmse = np.sqrt(np.mean((evaluation_df['actual'] - evaluation_df['predicted'])**2))
            mape = np.mean(np.abs((evaluation_df['actual'] - evaluation_df['predicted']) / evaluation_df['actual'])) * 100
            
            metrics.append({
                'Model': model_name,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'Sample_Size': len(evaluation_df)
            })
            
            logger.info(f"For model {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%")
            logger.info(f"Holdout start date: {holdout_data.index.min()}")
            logger.info(f"Holdout end date: {holdout_data.index.max()}")
            logger.info(f"Forecast start date for {model_name}: {forecast_df.index.min()}")
            logger.info(f"Forecast end date for {model_name}: {forecast_df.index.max()}")
        
        metrics_df = pd.DataFrame(metrics)
        self.evaluation_results = metrics_df  # Save for later use
        return metrics_df



    
    def plot_comparison(self, historical_data: Optional[pd.DataFrame] = None, 
                    target_column: Optional[str] = None, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> Any:
        """
        Visualizes a comparison of forecasts from different models using Bokeh.

        Args:
            historical_data (pd.DataFrame, optional): Historical data to display.
            target_column (str, optional): The name of the target column.
            start_date (str, optional): The start date for display.
            end_date (str, optional): The end date for display.
            
        Returns:
            bokeh.plotting.figure: The Bokeh figure with the comparison plot.
        """
        # Create a Bokeh figure with width and height instead of plot_width/plot_height
        p = figure(x_axis_type="datetime", title="Comparison of Forecasts from Different Models",
                width=800, height=400)
        p.xaxis.axis_label = "Date"
        p.yaxis.axis_label = "Value"
        p.add_tools(HoverTool(tooltips=[("Date", "@x{%F}"), ("Value", "@y")],
                            formatters={'@x': 'datetime'}, mode='vline'))
        
        # Plot historical data if provided
        if historical_data is not None and target_column is not None:
            if not isinstance(historical_data.index, pd.DatetimeIndex):
                historical_data.index = pd.to_datetime(historical_data.index)
            source_hist = ColumnDataSource(data={
                'x': historical_data.index,
                'y': historical_data[target_column]
            })
            p.line('x', 'y', source=source_hist, legend_label="Historical Data",
                line_width=2, color="black")
        
        colors = ["blue", "red", "green", "orange", "purple", "brown"]
        color_idx = 0
        
        # For each model, plot the forecast and confidence intervals
        for model_name, result in self.results.items():
            forecast_df = result["Forecast"]
            # Filter forecast data (where actual 'y' is missing)
            forecast_only = forecast_df[forecast_df['y'].isna()]
            if forecast_only.empty:
                continue
            
            source_fc = ColumnDataSource(data={
                'x': forecast_only.index,
                'y': forecast_only['yhat'],
                'y_lower': forecast_only['yhat_lower'],
                'y_upper': forecast_only['yhat_upper']
            })
            color = colors[color_idx % len(colors)]
            
            # Plot forecast line
            p.line('x', 'y', source=source_fc, legend_label=f"Forecast {model_name}",
                line_dash="dashed", line_width=2, color=color)
            # Plot confidence band if available
            band = Band(base='x', lower='y_lower', upper='y_upper', source=source_fc,
                        level='underlay', fill_alpha=0.2, fill_color=color)
            p.add_layout(band)
            
            color_idx += 1
        
        # Set date range if provided
        if start_date is not None:
            p.x_range.start = pd.to_datetime(start_date)
        if end_date is not None:
            p.x_range.end = pd.to_datetime(end_date)
        
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        return p

    
    def create_ensemble_forecast(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Creates an ensemble forecast based on the added models.

        Args:
            weights (Dict[str, float], optional): A dictionary of weights for each model.
                If None, equal weights are used for all models with valid evaluation metrics.
                
        Returns:
            Dict[str, Any]: The ensemble forecast result.
        """
        if not self.results:
            error_msg = "No forecasts have been added for ensemble creation."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # If weights are not provided and evaluation_results exist, calculate based on inverse error
        if weights is None and hasattr(self, 'evaluation_results'):
            # Use inverse RMSE as weights (lower error = higher weight)
            weights = {}
            for _, row in self.evaluation_results.iterrows():
                model = row['Model']
                rmse = row['RMSE']
                # Only include models with valid metrics
                if pd.notna(rmse) and rmse > 0:
                    weights[model] = 1.0 / rmse
                else:
                    weights[model] = 0.0  # Zero weight for models with invalid metrics
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: w / total_weight for name, w in weights.items()}
            logger.info(f"Using weights based on inverse RMSE: {weights}")
        elif weights is None:
            # If no evaluation results and no weights provided, use equal weights
            weights = {name: 1.0 / len(self.results) for name in self.results.keys()}
            logger.info(f"Using equal weights for all models: {weights}")
        else:
            # Ensure all models have a weight
            for name in self.results.keys():
                if name not in weights:
                    weights[name] = 0.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: w / total_weight for name, w in weights.items()}
            logger.info(f"Using provided weights: {weights}")
        
        # Create a combined index for all forecasts
        all_indices = set()
        for result in self.results.values():
            all_indices.update(result["Forecast"].index)
        all_indices = sorted(all_indices)
        
        ensemble_df = pd.DataFrame(index=all_indices)
        
        # Copy the actual target variable from the first forecast if available
        first_model = list(self.results.values())[0]["Forecast"]
        if 'y' in first_model.columns:
            ensemble_df['y'] = first_model['y']
        
        # For each model, add forecast with weight
        for model_name, result in self.results.items():
            forecast_df = result["Forecast"]
            ensemble_df[f'{model_name}_yhat'] = forecast_df['yhat']
            if 'yhat_lower' in forecast_df.columns:
                ensemble_df[f'{model_name}_lower'] = forecast_df['yhat_lower']
            if 'yhat_upper' in forecast_df.columns:
                ensemble_df[f'{model_name}_upper'] = forecast_df['yhat_upper']
            if 'yhat' not in ensemble_df.columns:
                ensemble_df['yhat'] = 0.0
            mask = forecast_df['yhat'].notna()
            ensemble_df.loc[mask, 'yhat'] += weights[model_name] * forecast_df.loc[mask, 'yhat']
        
        metadata = {
            "model": "Ensemble",
            "weights": weights,
            "component_models": list(self.results.keys())
        }
        
        logger.info(f"Ensemble forecast generated with {len(ensemble_df)} rows.")
        return {"Forecast": ensemble_df, "Model_Metadata": metadata}
    
    def select_best_model(self, metric: str = 'RMSE') -> str:
        """
        Selects the best model based on evaluation metrics.

        Args:
            metric (str): Metric to compare ('MAE', 'RMSE', 'MAPE').

        Returns:
            str: The name of the best model.
        """
        if not hasattr(self, 'evaluation_results'):
            error_msg = "No evaluation results available. Run evaluate_on_holdout() first."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if metric not in ['MAE', 'RMSE', 'MAPE']:
            error_msg = f"Unknown metric: {metric}. Use 'MAE', 'RMSE', or 'MAPE'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Filter out rows with NaN values for the specified metric
        valid_results = self.evaluation_results.dropna(subset=[metric])
        
        if valid_results.empty:
            logger.warning(f"No valid {metric} values available. Returning first model.")
            return self.evaluation_results.iloc[0]['Model']
        
        best_model = valid_results.loc[valid_results[metric].idxmin(), 'Model']
        logger.info(f"Best model based on {metric}: {best_model}")
        return best_model


# Example usage
if __name__ == '__main__':
    # Simulate data
    dates = pd.date_range(start='2022-01-01', periods=300, freq='D')
    sample_data = pd.DataFrame({
        "Close": pd.Series(range(300)) + 100,
        "Average_Close": pd.Series(range(300)) + 100,
        "moving_average_20": pd.Series(range(300)) + 95,
        "moving_average_50": pd.Series(range(300)) + 90,
        "Volume": pd.Series([1000] * 300)
    }, index=dates)
    
    # Split data into training and holdout sets
    train_data = sample_data.iloc[:-30]  # All data except the last 30 days
    holdout_data = sample_data.iloc[-30:]  # Last 30 days for evaluation
    
    forecast_horizon = 30
    
    # Create forecasts using both models
    prophet_predictor = ProphetPredictor(target_column="Average_Close")
    prophet_results = prophet_predictor.train_predict(train_data, forecast_horizon)
    
    arima_predictor = ArimaPredictor(target_column="Average_Close")
    arima_results = arima_predictor.train_predict(train_data, forecast_horizon)
    
    # Evaluate and compare forecasts
    evaluator = ForecastEvaluator()
    evaluator.add_forecast("Prophet", prophet_results)
    evaluator.add_forecast("ARIMA", arima_results)
    
    # Evaluate on holdout data
    metrics = evaluator.evaluate_on_holdout(holdout_data, "Average_Close")
    print("Forecast evaluation metrics:")
    print(metrics)
    
    # Visualize forecast comparison using Bokeh
    # For example, use the last 60 days of training data and the forecast horizon
    output_file("forecast_comparison.html")
    comparison_fig = evaluator.plot_comparison(
        historical_data=train_data.iloc[-60:],
        target_column="Average_Close"
    )
    save(comparison_fig)  # Save the interactive Bokeh plot as HTML
    show(comparison_fig)  # Open the interactive plot in the browser
    
    # Create an ensemble forecast
    # Calculate weights based on evaluation metrics (lower RMSE gets higher weight)
    if len(metrics) > 0:
        rmse_values = metrics['RMSE'].values
        inverted_rmse = 1.0 / rmse_values
        total_inverted = np.sum(inverted_rmse)
        
        weights = {}
        for i, model in enumerate(metrics['Model'].values):
            weights[model] = inverted_rmse[i] / total_inverted
        
        ensemble_results = evaluator.create_ensemble_forecast(weights)
        
        print(f"\nEnsemble forecast created with weights: {weights}")
        print("Ensemble forecast metadata:", ensemble_results["Model_Metadata"])
        print("Ensemble forecast (last 5 rows):")
        print(ensemble_results["Forecast"].tail())
    else:
        print("\nNot enough data to create an ensemble forecast.")
