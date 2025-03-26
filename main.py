import logging
import pandas as pd
from typing import Optional, Dict, Any

# Import data collection modules
from modules.data_collection import YahooFinanceCollector, CSVDataCollector, DataAggregator
# Import data analysis modules
from modules.data_analysis import DataAnalyzer, SummaryStatistics, MovingAverageAnalysis
# Import the updated visualization module with additional forecast functions
from modules.visualization import FinancialDataVisualizer, create_forecast_chart, create_combined_forecast_chart, get_forecast_series
# Import predictive analytics modules
from modules.prediction import ProphetPredictor, ArimaPredictor, ForecastEvaluator
# Import reporting modules
from modules.reporting import HTMLReportGenerator, PlainTextReportGenerator

# Import Bokeh functions to embed charts as HTML
from bokeh.embed import file_html
from bokeh.resources import CDN

# Configure logging for the pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(ticker: str, start_date: str, end_date: str, forecast_horizon: int, enable_forecast: bool = False) -> Dict[str, Any]:
    """
    Runs the full pipeline:
      1. Data Collection: Aggregates financial data.
      2. Data Analysis: Generates summary statistics.
      3. Adaptive Data Splitting:
           - Training data: [start_date, end_date - forecast_horizon]
           - Holdout data: the last forecast_horizon days.
      4. Predictive Analytics (if enabled): Generates forecasts using Prophet and ARIMA.
      5. Forecast Evaluation and Ensemble (if enabled).
      6. Visualization: Creates charts for historical data and forecasts.
      7. Reporting: Generates an HTML report.
    
    :param ticker: Stock ticker symbol (e.g., "AAPL").
    :param start_date: Start date for data collection.
    :param end_date: End date for data collection.
    :param forecast_horizon: Number of days to forecast.
    :param enable_forecast: If True, run forecasting steps.
    :return: Dictionary with keys: 'ticker', 'analysis_summary', 'chart_html',
             'forecast_html', 'forecast_metadata', 'report_html'.
    """
    results = {}
    
    # 1. DATA COLLECTION
    try:
        logger.info("Starting data collection.")
        yahoo_collector = YahooFinanceCollector()
        csv_collector = CSVDataCollector(filepath="data/raw/aapl_data.csv")
        aggregator = DataAggregator(collectors=[yahoo_collector, csv_collector])
        full_data = aggregator.aggregate_data(ticker, start_date, end_date)
        logger.info("Data collection completed.")
    except Exception as e:
        logger.error("Data aggregation failed: %s", e)
        results["error"] = "Data aggregation failed."
        return results
    
    # 2. DATA ANALYSIS
    try:
        logger.info("Starting data analysis.")
        analyzer = DataAnalyzer()
        analyzer.register_analysis_method(SummaryStatistics())
        analyzer.register_analysis_method(MovingAverageAnalysis(windows=[20, 50]))
        analysis_summary = analyzer.analyze(full_data)
        logger.info("Data analysis completed.")
    except Exception as e:
        logger.error("Data analysis failed: %s", e)
        analysis_summary = "<p>Data analysis failed.</p>"
    
    # 3. ADAPTIVE DATA SPLITTING
    try:
        end_ts = pd.Timestamp(end_date)
        train_end_date = end_ts - pd.Timedelta(days=forecast_horizon)
        train_data = full_data.loc[:train_end_date]
        holdout_data = full_data.loc[train_end_date + pd.Timedelta(days=1): end_ts]
        logger.info(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
        logger.info(f"Holdout data range: {holdout_data.index.min()} to {holdout_data.index.max()}")
    except Exception as e:
        logger.error("Data splitting failed: %s", e)
        results["error"] = "Data splitting failed."
        return results
    
    ############################################
# 4. PREDICTIVE ANALYTICS
############################################
    forecast_results = {}
    if enable_forecast:
        try:
            logger.info("Starting predictive analytics with Prophet.")
            prophet_predictor = ProphetPredictor(target_column="Average_Close")
            # Prophet forecast
            prophet_results = prophet_predictor.train_predict(train_data, forecast_horizon)
            forecast_results["Prophet"] = prophet_results
            logger.info("Prophet forecast completed.")
        except Exception as e:
            logger.error("Prophet prediction failed: %s", e)
            prophet_results = None
            forecast_results["Prophet"] = {"error": "Prophet prediction failed."}
        
        try:
            logger.info("Starting predictive analytics with ARIMA.")
            arima_predictor = ArimaPredictor(target_column="Average_Close")
            # If you want to start from the last date in train_data + 1 day:
            forecast_start_date = train_data.index[-1] + pd.Timedelta(days=1)
            arima_results = arima_predictor.train_predict(
                train_data,
                forecast_horizon,
                forecast_start_date=forecast_start_date
            )
            forecast_results["ARIMA"] = arima_results
            logger.info("ARIMA forecast completed.")
        except Exception as e:
            logger.error("ARIMA prediction failed: %s", e)
            arima_results = None
            forecast_results["ARIMA"] = {"error": "ARIMA prediction failed."}
        
    # Evaluate & Ensemble (unchanged)
        try:
            logger.info("Starting forecast evaluation and ensemble creation.")
            evaluator = ForecastEvaluator()
            evaluator.add_forecast("Prophet", prophet_results)
            evaluator.add_forecast("ARIMA", arima_results)
            metrics = evaluator.evaluate_on_holdout(holdout_data, "Average_Close")
            logger.info("Forecast evaluation metrics: %s", metrics)
            if not metrics.empty:
                rmse_values = metrics["RMSE"].values
                inverted_rmse = 1.0 / rmse_values
                total_inverted = inverted_rmse.sum()
                weights = {model: inverted_rmse[i] / total_inverted for i, model in enumerate(metrics["Model"].values)}
                ensemble_results = evaluator.create_ensemble_forecast(weights)
                forecast_results["Ensemble"] = ensemble_results
            else:
                forecast_results["Ensemble"] = {...}
        except Exception as e:
            logger.error("Forecast evaluation or ensemble creation failed: %s", e)
            forecast_results["Ensemble"] = {"error": "Ensemble forecast not available."}
    else:
        logger.info("Forecasting skipped by user choice.")

############################################
# 6. VISUALIZATION (Updated)
############################################
    def filter_partial_history(forecast_df: pd.DataFrame, last_date: pd.Timestamp, days_of_history: int = 14) -> pd.DataFrame:
        """
        Filters the forecast DataFrame to include only rows from (last_date - days_of_history) onward.
        This retains a small portion of history before the forecast.
        """
        if forecast_df is None or forecast_df.empty:
            return pd.DataFrame()
        if not isinstance(forecast_df.index, pd.DatetimeIndex):
            raise ValueError("Forecast DataFrame must have a DatetimeIndex")
        cutoff_date = last_date - pd.Timedelta(days=days_of_history)
        return forecast_df.loc[forecast_df.index >= cutoff_date]

    try:
        logger.info("Starting visualization.")
        visualizer = FinancialDataVisualizer()
        charts = visualizer.create_charts(full_data)
        price_trend_html = "<p>Price trend chart not available.</p>"
        if "price_trend" in charts:
            try:
                price_trend_html = file_html(charts["price_trend"], CDN, "Price Trend")
            except Exception as e:
                logger.error(f"Error rendering price trend: {str(e)}")

        forecast_html_parts = []
        if enable_forecast:
            # Define last historical date and history window (e.g., last 14 days)
            last_date = train_data.index[-1]
            days_of_history = 14

            # Helper function: filter partial history + compute final forecast & trend.
            def compute_trend_and_filter(model_name):
                if model_name not in forecast_results or "Forecast" not in forecast_results[model_name]:
                    logger.warning(f"Missing forecast for {model_name}")
                    return None
                df = forecast_results[model_name]["Forecast"]
                partial_df = filter_partial_history(df, last_date, days_of_history)
                if partial_df.empty:
                    logger.warning(f"Empty forecast after partial filter for {model_name}")
                    return None
                # Use get_forecast_series() to extract the forecast series robustly.
                try:
                    forecast_series = get_forecast_series(partial_df)
                except Exception as e:
                    logger.error(f"Error getting forecast series for {model_name}: {str(e)}")
                    return None
                final_val = forecast_series.iloc[-1]
                last_close_price = train_data["Average_Close"].iloc[-1]
                diff = final_val - last_close_price
                trend_str = "No change"
                if diff > 0:
                    trend_str = f"Up by {diff:.2f}"
                elif diff < 0:
                    trend_str = f"Down by {abs(diff):.2f}"
                forecast_results[model_name]["FinalForecast"] = final_val
                forecast_results[model_name]["Trend"] = trend_str
                return partial_df

            prophet_future_df = compute_trend_and_filter("Prophet")
            arima_future_df   = compute_trend_and_filter("ARIMA")
            ensemble_future_df= compute_trend_and_filter("Ensemble")
            
            # Build individual charts (only if filtered forecast data is available)
            if prophet_future_df is not None:
                prophet_chart = create_forecast_chart(prophet_future_df, "Prophet Forecast (+14d History)")
                forecast_html_parts.append(file_html(prophet_chart, CDN, "Prophet Forecast"))
            if arima_future_df is not None:
                arima_chart = create_forecast_chart(arima_future_df, "ARIMA Forecast (+14d History)")
                forecast_html_parts.append(file_html(arima_chart, CDN, "ARIMA Forecast"))
            if ensemble_future_df is not None:
                ensemble_chart = create_forecast_chart(ensemble_future_df, "Ensemble Forecast (+14d History)")
                forecast_html_parts.append(file_html(ensemble_chart, CDN, "Ensemble Forecast"))
            
            # Combined chart: only if at least one model provides forecast data
            if (prophet_future_df is not None or arima_future_df is not None or ensemble_future_df is not None):
                combined_chart = create_combined_forecast_chart(
                    prophet_future_df,
                    arima_future_df,
                    ensemble_future_df,
                    "Combined Forecast (+14d History)"
                )
                forecast_html_parts.append(file_html(combined_chart, CDN, "Combined Forecast"))
            
            forecast_html = "".join(forecast_html_parts) or "<p>No forecast visualizations available.</p>"
        else:
            forecast_html = "<p>Forecasting was disabled by user choice.</p>"

        logger.info("Visualization completed.")
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        price_trend_html = "<p>Visualization error</p>"
        forecast_html = "<p>Forecast visualization failed</p>"


    
    # 7. REPORTING
    try:
        logger.info("Starting report generation.")
        html_report_generator = HTMLReportGenerator("financial_market_report.html")
        report_html = html_report_generator.generate_report(
            analysis_results=analysis_summary,
            forecast_results=forecast_results,
            visualizations={"Price Trend": price_trend_html, "Forecast": forecast_html}
        )
        logger.info("HTML report generated successfully.")
        
        text_report_generator = PlainTextReportGenerator("financial_market_report.txt")
        text_report = text_report_generator.generate_report(
            analysis_results=analysis_summary,
            forecast_results=forecast_results,
            visualizations={"Price Trend": price_trend_html, "Forecast": forecast_html}
        )
        logger.info("Plain text report generated successfully.")
    except Exception as e:
        logger.error("Report generation failed: %s", e)
        report_html = "<p>Report generation failed.</p>"
    
    # Compile final results
    results.update({
        "ticker": ticker,
        "analysis_summary": analysis_summary,
        "chart_html": price_trend_html,
        "forecast_html": forecast_html,
        "forecast_metadata": forecast_results.get("Prophet", {}).get("Model_Metadata", {}),
        "report_html": report_html
    })
    
    return results


