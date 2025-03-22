# main.py

from modules.data_collection import YahooFinanceCollector, CSVDataCollector, DataAggregator
from modules.data_analysis import DataAnalyzer, SummaryStatistics, MovingAverageAnalysis
from modules.visualization import FinancialDataVisualizer
# Импорт прогнозных моделей из модуля предсказательной аналитики
from modules.prediction import ProphetPredictor, ArimaPredictor, ForecastEvaluator

# Импорт функций Bokeh для построения графиков
from bokeh.io import output_file, show
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.plotting import figure

import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_forecast_series(forecast_df):
    """
    Returns the forecast series from the given DataFrame.
    
    Priority:
      1. Use the 'Forecast' column if it exists.
      2. Otherwise, use the 'yhat' column (commonly returned by Prophet).
      3. Otherwise, use the first column of the DataFrame.
    
    :param forecast_df: DataFrame with forecast data.
    :return: A Pandas Series with forecast values.
    """
    if "Forecast" in forecast_df.columns:
        return forecast_df["Forecast"]
    elif "yhat" in forecast_df.columns:
        return forecast_df["yhat"]
    else:
        return forecast_df.iloc[:, 0]

def create_forecast_chart(forecast_df, title):
    """
    Create a Bokeh line chart for a single forecast.
    
    :param forecast_df: DataFrame with forecast results.
    :param title: Title of the chart.
    :return: Bokeh Figure.
    """
    p = figure(title=title, x_axis_type='datetime', width=600, height=400)
    forecast_values = get_forecast_series(forecast_df)
    p.line(forecast_df.index, forecast_values, line_width=2, legend_label=title)
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Forecast"
    return p

def create_combined_forecast_chart(prophet_forecast_df, arima_forecast_df, ensemble_forecast_df, title):
    """
    Create a combined Bokeh chart displaying forecast lines for Prophet, ARIMA, and Ensemble.
    
    :param prophet_forecast_df: DataFrame for Prophet forecast.
    :param arima_forecast_df: DataFrame for ARIMA forecast.
    :param ensemble_forecast_df: DataFrame for Ensemble forecast.
    :param title: Title of the chart.
    :return: Bokeh Figure.
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

def main():
    # ----- Parameters for Data Collection -----
    TICKER = "AAPL"
    START_DATE = "2022-01-01"
    # Ensure data is available up to at least 2024-01-28 for evaluation.
    END_DATE = "2024-01-28"  
    
    # ----- Instantiate Data Collectors -----
    yahoo_collector = YahooFinanceCollector()
    csv_collector = CSVDataCollector(filepath="data/raw/aapl_data.csv")
    
    # ----- Aggregate Data -----
    aggregator = DataAggregator(collectors=[yahoo_collector, csv_collector])
    try:
        aggregated_data = aggregator.aggregate_data(TICKER, START_DATE, END_DATE)
        print("Aggregated Data (first 5 rows):")
        print(aggregated_data.head())
    except Exception as e:
        print(f"Data aggregation failed: {e}")
        return
    
    # ----- Data Analysis -----
    analyzer = DataAnalyzer()
    analyzer.register_analysis_method(SummaryStatistics())
    analyzer.register_analysis_method(MovingAverageAnalysis(windows=[20, 50]))
    
    analysis_results = analyzer.analyze(aggregated_data)
    print("\nAnalysis Results:")
    print(analysis_results)
    
    # ----- Visualization -----
    visualizer = FinancialDataVisualizer()
    try:
        # Output charts to an HTML file.
        output_file("my_financial_dashboard.html")
        charts = visualizer.create_charts(aggregated_data)
        if "price_trend" in charts:
            show(charts["price_trend"])
        else:
            print("No chart named 'price_trend' was found.")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # ----- Convert Bokeh price trend chart to HTML string for the report -----
    visualizations_for_report = {}
    if "price_trend" in charts:
        price_trend_html = file_html(charts["price_trend"], CDN, "Price Trend")
        visualizations_for_report["Price Trend"] = price_trend_html
    
    # ----- Predictive Analytics -----
    results = {}
    forecast_horizon = 30
    
    try:
        # Run Prophet on the full dataset (for reference)
        prophet_predictor = ProphetPredictor(target_column="Average_Close")
        prophet_results_full = prophet_predictor.train_predict(aggregated_data, forecast_horizon)
        results["Prophet_full"] = prophet_results_full
        print("\nProphet (full data) Model Metadata:")
        print(prophet_results_full["Model_Metadata"])
        print("Prophet (full data) Forecast (last 5 rows):")
        print(prophet_results_full["Forecast"].tail())
    except Exception as e:
        results["Prophet_full"] = {"error": f"Prophet prediction failed: {str(e)}"}
    
    # ----- Adjust Data Splitting for Forecast Evaluation -----
    train_end_date = pd.Timestamp('2023-12-29')
    train_data = aggregated_data.loc[:train_end_date]
    holdout_data = aggregated_data.loc[train_end_date + pd.Timedelta(days=1) : train_end_date + pd.Timedelta(days=forecast_horizon)]
    
    logger.info(f"Training data range: {train_data.index.min()} to {train_data.index.max()}")
    logger.info(f"Holdout data range: {holdout_data.index.min()} to {holdout_data.index.max()}")
    
    forecast_start_date = holdout_data.index[0]
    
    try:
        # Forecast with Prophet on training data
        prophet_predictor = ProphetPredictor(target_column="Average_Close")
        prophet_results = prophet_predictor.train_predict(train_data, forecast_horizon)
        results["Prophet"] = prophet_results
        print("\nProphet Model Metadata (adjusted split):")
        print(prophet_results["Model_Metadata"])
        print("Prophet Forecast (last 5 rows, adjusted split):")
        print(prophet_results["Forecast"].tail())
    except Exception as e:
        results["Prophet"] = {"error": f"Prophet prediction failed: {str(e)}"}
    
    try:
        # Forecast with ARIMA on training data
        arima_predictor = ArimaPredictor(target_column="Average_Close")
        arima_results = arima_predictor.train_predict(train_data, forecast_horizon, forecast_start_date=forecast_start_date)
        results["ARIMA"] = arima_results
        print("\nARIMA Model Metadata (adjusted split):")
        print(arima_results["Model_Metadata"])
        print("ARIMA Forecast (last 5 rows, adjusted split):")
        print(arima_results["Forecast"].tail())
    except Exception as e:
        results["ARIMA"] = {"error": f"ARIMA prediction failed: {str(e)}"}
    
    arima_future_df = arima_results.get("Future_Forecast", arima_results["Forecast"])
    logger.info(f"ARIMA forecast date range: {arima_future_df.index.min()} to {arima_future_df.index.max()}")
    
    # ----- Forecast Evaluation and Ensemble Forecast -----
    ensemble_results = None
    try:
        evaluator = ForecastEvaluator()
        evaluator.add_forecast("Prophet", prophet_results)
        evaluator.add_forecast("ARIMA", arima_results)
        
        metrics = evaluator.evaluate_on_holdout(holdout_data, "Average_Close")
        print("Forecast Evaluation Metrics:")
        print(metrics)
        
        if not metrics.empty:
            rmse_values = metrics["RMSE"].values
            inverted_rmse = 1.0 / rmse_values
            total_inverted = inverted_rmse.sum()
            weights = {}
            for i, model in enumerate(metrics["Model"].values):
                weights[model] = inverted_rmse[i] / total_inverted
            
            ensemble_results = evaluator.create_ensemble_forecast(weights)
            print(f"\nEnsemble forecast created with weights: {weights}")
            print("Ensemble Forecast Metadata:")
            print(ensemble_results["Model_Metadata"])
            print("Ensemble Forecast (last 5 rows):")
            print(ensemble_results["Forecast"].tail())
        else:
            print("\nNot enough data to create an ensemble forecast.")
    except Exception as e:
        print(f"Forecast evaluation or ensemble creation failed: {e}")
    
    # ----- Prepare forecast_results dictionary with Prophet, ARIMA and Ensemble forecasts -----
    forecast_results = {"Prophet": prophet_results, "ARIMA": arima_results}
    if ensemble_results is not None:
        forecast_results["Ensemble"] = ensemble_results
    else:
        forecast_results["Ensemble"] = {
            "forecast_summary": "Ensemble forecast not available",
            "Model_Metadata": {}
        }
    
    # ----- Create individual forecast charts -----
    prophet_forecast_chart = create_forecast_chart(prophet_results["Forecast"], "Prophet Forecast")
    prophet_forecast_html = file_html(prophet_forecast_chart, CDN, "Prophet Forecast")
    visualizations_for_report["Prophet Forecast"] = prophet_forecast_html
    
    arima_forecast_chart = create_forecast_chart(arima_results["Forecast"], "ARIMA Forecast")
    arima_forecast_html = file_html(arima_forecast_chart, CDN, "ARIMA Forecast")
    visualizations_for_report["ARIMA Forecast"] = arima_forecast_html
    
    if ensemble_results is not None:
        ensemble_forecast_chart = create_forecast_chart(ensemble_results["Forecast"], "Ensemble Forecast")
        ensemble_forecast_html = file_html(ensemble_forecast_chart, CDN, "Ensemble Forecast")
        visualizations_for_report["Ensemble Forecast"] = ensemble_forecast_html
    
    # ----- Create a combined forecast chart with Prophet, ARIMA, and Ensemble on one graph -----
    combined_forecast_chart = create_combined_forecast_chart(
        prophet_results["Forecast"],
        arima_results["Forecast"],
        ensemble_results["Forecast"] if ensemble_results is not None else None,
        "Combined Forecast"
    )
    combined_forecast_html = file_html(combined_forecast_chart, CDN, "Combined Forecast")
    visualizations_for_report["Combined Forecast"] = combined_forecast_html
    
    # ----- Integrate Reporting Module -----
    try:
        from modules.reporting import HTMLReportGenerator, PlainTextReportGenerator

        # Генерация HTML-отчёта
        html_report_generator = HTMLReportGenerator("financial_market_report.html")
        report_html = html_report_generator.generate_report(
            analysis_results=analysis_results,
            forecast_results=forecast_results,
            visualizations=visualizations_for_report
        )
        print("\nHTML report generated successfully and saved as 'financial_market_report.html'")

        
        text_report_generator = PlainTextReportGenerator("financial_market_report.txt")
        text_report = text_report_generator.generate_report(
            analysis_results=analysis_results,
            forecast_results=forecast_results,
            visualizations=visualizations_for_report
        )
        print("Plain text report generated successfully and saved as 'financial_market_report.txt'")

    except Exception as e:
        print(f"Report generation failed: {e}")

    return results

if __name__ == '__main__':
    main()
