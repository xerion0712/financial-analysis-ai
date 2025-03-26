import logging
import pandas as pd
from typing import Optional, Dict, Any

# Import data collection modules
from modules.data_collection import YahooFinanceCollector, CSVDataCollector, DataAggregator
# Import data analysis modules
from modules.data_analysis import DataAnalyzer, SummaryStatistics, MovingAverageAnalysis
# Import the updated visualization module with additional forecast functions
from modules.visualization import FinancialDataVisualizer, create_forecast_chart, create_combined_forecast_chart
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
    
    # 4. PREDICTIVE ANALYTICS (Conditional)
    forecast_results = {}
    if enable_forecast:
        try:
            logger.info("Starting predictive analytics with Prophet.")
            prophet_predictor = ProphetPredictor(target_column="Average_Close")
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
            forecast_start_date = holdout_data.index[0]
            logger.info(f"Forecast start date for ARIMA: {forecast_start_date}")
            arima_results = arima_predictor.train_predict(train_data, forecast_horizon, forecast_start_date=forecast_start_date)
            forecast_results["ARIMA"] = arima_results
            logger.info("ARIMA forecast completed.")
        except Exception as e:
            logger.error("ARIMA prediction failed: %s", e)
            arima_results = None
            forecast_results["ARIMA"] = {"error": "ARIMA prediction failed."}
        
        # Log forecast date ranges
        prophet_forecast_df = prophet_results.get("Forecast") if prophet_results else None
        if prophet_forecast_df is not None:
            logger.info(f"Prophet forecast range: {prophet_forecast_df.index.min()} to {prophet_forecast_df.index.max()}")
        arima_forecast_df = arima_results.get("Forecast") if arima_results else None
        if arima_forecast_df is not None:
            logger.info(f"ARIMA forecast range: {arima_forecast_df.index.min()} to {arima_forecast_df.index.max()}")
        
        # 5. FORECAST EVALUATION AND ENSEMBLE
        try:
            logger.info("Starting forecast evaluation and ensemble creation.")
            evaluator = ForecastEvaluator()
            evaluator.add_forecast("Prophet", prophet_results)
            evaluator.add_forecast("ARIMA", arima_results)
            metrics = evaluator.evaluate_on_holdout(holdout_data, "Close")
            logger.info("Forecast evaluation metrics: %s", metrics)
            if not metrics.empty:
                rmse_values = metrics["RMSE"].values
                inverted_rmse = 1.0 / rmse_values
                total_inverted = inverted_rmse.sum()
                weights = {model: inverted_rmse[i] / total_inverted for i, model in enumerate(metrics["Model"].values)}
                ensemble_results = evaluator.create_ensemble_forecast(weights)
                forecast_results["Ensemble"] = ensemble_results
                logger.info("Ensemble forecast created with weights: %s", weights)
            else:
                forecast_results["Ensemble"] = {"forecast_summary": "Ensemble forecast not available", "Model_Metadata": {}}
                logger.info("No ensemble forecast created due to insufficient evaluation data.")
        except Exception as e:
            logger.error("Forecast evaluation or ensemble creation failed: %s", e)
            forecast_results["Ensemble"] = {"error": "Ensemble forecast not available."}
    else:
        logger.info("Forecasting skipped by user choice.")
    
    # 6. VISUALIZATION
    try:
        logger.info("Starting visualization.")
        visualizer = FinancialDataVisualizer()
        charts = visualizer.create_charts(full_data)
        if "price_trend" in charts:
            price_trend_html = file_html(charts["price_trend"], CDN, "Price Trend")
        else:
            price_trend_html = "<p>Price trend chart not available.</p>"
        
        forecast_html_parts = []
        if enable_forecast:
            if prophet_results is not None and "Forecast" in prophet_results:
                prophet_chart = create_forecast_chart(prophet_results["Forecast"], "Prophet Forecast")
                prophet_chart_html = file_html(prophet_chart, CDN, "Prophet Forecast")
                forecast_html_parts.append(prophet_chart_html)
            if arima_results is not None and "Forecast" in arima_results:
                arima_chart = create_forecast_chart(arima_results["Forecast"], "ARIMA Forecast")
                arima_chart_html = file_html(arima_chart, CDN, "ARIMA Forecast")
                forecast_html_parts.append(arima_chart_html)
            if "Ensemble" in forecast_results and forecast_results["Ensemble"] is not None and "Forecast" in forecast_results["Ensemble"]:
                ensemble_chart = create_forecast_chart(forecast_results["Ensemble"]["Forecast"], "Ensemble Forecast")
                ensemble_chart_html = file_html(ensemble_chart, CDN, "Ensemble Forecast")
                forecast_html_parts.append(ensemble_chart_html)
            if (prophet_results is not None and "Forecast" in prophet_results) or (arima_results is not None and "Forecast" in arima_results):
                combined_chart = create_combined_forecast_chart(
                    prophet_results["Forecast"] if prophet_results and "Forecast" in prophet_results else None,
                    arima_results["Forecast"] if arima_results and "Forecast" in arima_results else None,
                    forecast_results["Ensemble"]["Forecast"] if ("Ensemble" in forecast_results and forecast_results["Ensemble"] is not None and "Forecast" in forecast_results["Ensemble"]) else None,
                    "Combined Forecast"
                )
                combined_chart_html = file_html(combined_chart, CDN, "Combined Forecast")
            else:
                combined_chart_html = "<p>Combined forecast chart not available.</p>"
            forecast_html = "".join(forecast_html_parts) + combined_chart_html
        else:
            forecast_html = "<p>Forecasting was disabled by user choice.</p>"
        
        logger.info("Visualization completed.")
    except Exception as e:
        logger.error("Visualization failed: %s", e)
        price_trend_html = "<p>Visualization failed.</p>"
        forecast_html = "<p>Forecast visualization failed.</p>"
    
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


