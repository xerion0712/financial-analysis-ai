#modules ui_module.py


import logging
from abc import ABC, abstractmethod
from flask import Flask, request, render_template
from typing import Dict, Any
import sys
import os
import pandas as pd
import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем функцию из общего пайплайна (убедитесь, что путь указан корректно)
from main import run_pipeline

# -------------------------------------------------------------------
# Logging configuration for debugging and traceability.
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Abstract Base Class for UI modules.
# -------------------------------------------------------------------
class BaseUI(ABC):
    """
    Abstract Base Class for UI modules.
    Defines the core API that all UI implementations must follow.
    """
    @abstractmethod
    def run(self) -> None:
        """Start the UI."""
        pass

    @abstractmethod
    def display_dashboard(self, **kwargs) -> str:
        """Render the dashboard with the given data."""
        pass

class WebUI(BaseUI):
    """
    Web-based UI using Flask.
    This class sets up a dashboard that displays outputs from the full data pipeline:
      - Data analysis summary
      - Interactive Bokeh chart (converted to HTML)
      - Forecast results and metadata
      - Combined HTML report
    """
    def __init__(self):
        self.app = Flask(__name__, template_folder='templates')
        self.setup_routes()

    def setup_routes(self) -> None:
        @self.app.route('/', methods=['GET', 'POST'])
        
        def index():
            if request.method == 'POST':
                mode = request.form.get('mode', 'historical')
                ticker = request.form.get('ticker', 'AAPL')

                # We'll read the form fields, but they might be empty if mode == 'forecast'
                start_date = request.form.get('start_date', '')
                end_date = request.form.get('end_date', '')
                forecast_horizon = int(request.form.get('forecast_horizon', 5))

                logger.info(f"Mode={mode}, Ticker={ticker}, Start={start_date}, End={end_date}, Horizon={forecast_horizon}")

                if mode == 'historical':
                    # Historical data only
                    pipeline_results = run_pipeline(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        forecast_horizon=0,
                        enable_forecast=False
                    )

                elif mode == 'forecast':
                    # Forecast only, but auto-set the last 2 years for data
                    # For example, from 2 years ago to today
                    today = pd.Timestamp.today().normalize()
                    two_years_ago = today - pd.Timedelta(days=730)

                    pipeline_results = run_pipeline(
                        ticker=ticker,
                        start_date=two_years_ago.strftime("%Y-%m-%d"),
                        end_date=today.strftime("%Y-%m-%d"),
                        forecast_horizon=forecast_horizon,
                        enable_forecast=True
                    )

                else:  # mode == 'both'
                    pipeline_results = run_pipeline(
                        ticker=ticker,
                        start_date=start_date,
                        end_date=end_date,
                        forecast_horizon=forecast_horizon,
                        enable_forecast=True
                    )

                if "error" in pipeline_results:
                    return render_template('error.html', message=pipeline_results["error"])

                return render_template(
                    'dashboard.html',
                    ticker=pipeline_results.get("ticker", ticker),
                    analysis_summary=pipeline_results.get("analysis_summary", ""),
                    chart_html=pipeline_results.get("chart_html", ""),
                    forecast_html=pipeline_results.get("forecast_html", ""),
                    forecast_metadata=pipeline_results.get("forecast_metadata", {}),
                    report_html=pipeline_results.get("report_html", "")
                )

            # GET request
            return render_template('index.html')
        @self.app.route('/report')
        def report():
            return "Report page placeholder."

    def run(self) -> None:
        logger.info("Starting the Web UI.")
        self.app.run(debug=True)

    def display_dashboard(self, **kwargs) -> str:
        logger.info("display_dashboard() called. Please visit the root URL of the web interface.")
        return "Dashboard is running. Please visit the web interface."

if __name__ == '__main__':
    ui = WebUI()
    ui.run()