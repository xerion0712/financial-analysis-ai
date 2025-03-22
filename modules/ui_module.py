import logging
from abc import ABC, abstractmethod
from flask import Flask, request, render_template
from typing import Dict, Any

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
    Abstract Base Class for User Interface modules.
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

# -------------------------------------------------------------------
# Concrete Web UI implementation using Flask.
# -------------------------------------------------------------------
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
        """
        Initialize the Flask app and set up the necessary routes.
        """
        self.app = Flask(__name__, template_folder='templates')
        self.setup_routes()

    def setup_routes(self) -> None:
        """
        Set up the Flask routes for the web dashboard.
        """
        @self.app.route('/', methods=['GET', 'POST'])
        def index():
            if request.method == 'POST':
                # Получаем параметры из формы
                ticker = request.form.get('ticker', 'AAPL')
                start_date = request.form.get('start_date', '2020-01-01')
                end_date = request.form.get('end_date', '2020-12-31')
                forecast_horizon = int(request.form.get('forecast_horizon', 5))

                logger.info("Received parameters: ticker=%s, start_date=%s, end_date=%s, forecast_horizon=%d",
                            ticker, start_date, end_date, forecast_horizon)

                try:
                    # Запуск общего пайплайна, который агрегирует данные, анализирует их,
                    # создает графики с помощью Bokeh, прогнозирует и генерирует отчёт.
                    pipeline_results: Dict[str, Any] = run_pipeline(ticker, start_date, end_date, forecast_horizon)
                except Exception as e:
                    logger.error("Error running pipeline: %s", e)
                    return render_template('error.html', message="An error occurred while processing the data.")

                # Если пайплайн вернул ошибку, отображаем сообщение
                if "error" in pipeline_results:
                    return render_template('error.html', message=pipeline_results["error"])

                # Рендерим шаблон dashboard.html с результатами
                return render_template(
                    'dashboard.html',
                    ticker=pipeline_results.get("ticker", ticker),
                    analysis_summary=pipeline_results.get("analysis_summary", ""),
                    chart_html=pipeline_results.get("chart_html", ""),
                    forecast_html=pipeline_results.get("forecast_html", ""),
                    forecast_metadata=pipeline_results.get("forecast_metadata", {}),
                    report_html=pipeline_results.get("report_html", "")
                )
            # Для GET-запроса отображаем форму ввода параметров.
            return render_template('index.html')

        @self.app.route('/report')
        def report():
            # Если необходимо, можно добавить отдельную страницу для отчёта.
            return "Report page placeholder."

    def run(self) -> None:
        """
        Start the Flask web server to run the UI.
        """
        logger.info("Starting the Web UI.")
        self.app.run(debug=True)

    def display_dashboard(self, **kwargs) -> str:
        """
        This method is part of the BaseUI interface.
        For this web UI, rendering is handled via Flask routes.
        """
        logger.info("display_dashboard() called. Please visit the root URL of the web interface.")
        return "Dashboard is running. Please visit the web interface."

# -------------------------------------------------------------------
# Entry point: Instantiate and run the Web UI.
# -------------------------------------------------------------------
if __name__ == '__main__':
    ui = WebUI()
    ui.run()
