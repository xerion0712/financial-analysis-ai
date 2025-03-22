import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

# Configure logging to display informative messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ================================
# Abstract Base Class Definition
# ================================
class BaseReportGenerator(ABC):
    @abstractmethod
    def generate_report(self,
                        analysis_results: Dict[str, Any],
                        forecast_results: Dict[str, Any],
                        visualizations: Dict[str, str]) -> str:
        pass

class HTMLReportGenerator(BaseReportGenerator):
    def __init__(self, output_path: str = "report.html"):
        self.output_path = output_path
        logging.info(f"HTMLReportGenerator initialized with output path: {self.output_path}")

    def generate_report(self,
                        analysis_results: Dict[str, Any],
                        forecast_results: Dict[str, Any],
                        visualizations: Dict[str, str]) -> str:
        logging.info("Starting HTML report generation...")

        # Generate analysis summary: if 'summary' is missing, use technical indicators.
        analysis_summary = analysis_results.get("summary", "").strip()
        if not analysis_summary:
            technical_indicators = analysis_results.get("technical_indicators", {})
            if technical_indicators:
                analysis_summary = "Technical indicators: " + ", ".join(
                    [f"{k}: {v}" for k, v in technical_indicators.items()])
            else:
                analysis_summary = "No analysis summary provided."

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Financial Market Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 40px; }}
                .section h2 {{ border-bottom: 2px solid #ccc; padding-bottom: 10px; }}
                .visualization {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Financial Market Analysis Report</h1>
            
            <!-- Analysis Section -->
            <div class="section">
                <h2>Data Analysis Summary</h2>
                <p>{analysis_summary}</p>
                <h3>Technical Indicators:</h3>
                <ul>
        """
        technical_indicators = analysis_results.get("technical_indicators", {})
        for key, value in technical_indicators.items():
            html_content += f"<li>{key}: {value}</li>"
        html_content += """
                </ul>
            </div>
            
            <!-- Forecast Section -->
            <div class="section">
                <h2>Forecast Results</h2>
        """
        # Check if forecast_results is empty and add default text
        if not forecast_results:
            html_content += "<p>No forecast summary provided.</p>"
        else:
            # Iterate over forecast_results and output block for each forecast
            for forecast_type, data in forecast_results.items():
                html_content += f"""
                <div class="forecast-block">
                    <h3>{forecast_type} Forecast</h3>
                    <p>{data.get("forecast_summary", "No forecast summary provided.")}</p>
                    <h4>Model Metadata:</h4>
                    <ul>
                """
                model_metadata = data.get("Model_Metadata", {})
                for key, value in model_metadata.items():
                    html_content += f"<li>{key}: {value}</li>"
                html_content += """
                    </ul>
                </div>
                """
        html_content += """
            </div>
            
            <!-- Visualization Section -->
            <div class="section">
                <h2>Visualizations</h2>
        """
        for viz_name, viz_html in visualizations.items():
            html_content += f"""
                <div class="visualization">
                    <h3>{viz_name}</h3>
                    {viz_html}
                </div>
            """
        html_content += """
            </div>
        </body>
        </html>
        """
        try:
            with open(self.output_path, "w", encoding="utf-8") as file:
                file.write(html_content)
            logging.info(f"HTML report successfully written to {self.output_path}")
        except Exception as e:
            logging.error(f"Failed to write HTML report: {e}")
            raise e

        return html_content

class PlainTextReportGenerator(BaseReportGenerator):
    def __init__(self, output_path: str = "report.txt"):
        self.output_path = output_path
        logging.info(f"PlainTextReportGenerator initialized with output path: {self.output_path}")

    def generate_report(self,
                        analysis_results: Dict[str, Any],
                        forecast_results: Dict[str, Any],
                        visualizations: Dict[str, str]) -> str:
        logging.info("Starting plain text report generation...")
        report_lines = []
        report_lines.append("Financial Market Analysis Report\n")
        report_lines.append("=== Data Analysis Summary ===")
        
        # Build analysis summary
        analysis_summary = analysis_results.get("summary", "").strip()
        if not analysis_summary:
            technical_indicators = analysis_results.get("technical_indicators", {})
            if technical_indicators:
                analysis_summary = "Technical indicators: " + ", ".join(
                    [f"{k}: {v}" for k, v in technical_indicators.items()])
            else:
                analysis_summary = "No analysis summary provided."
        report_lines.append(analysis_summary)
        
        report_lines.append("\nTechnical Indicators:")
        technical_indicators = analysis_results.get("technical_indicators", {})
        for key, value in technical_indicators.items():
            report_lines.append(f"  - {key}: {value}")
        
        report_lines.append("\n=== Forecast Results ===")
        # If forecast_results is empty, add default text
        if not forecast_results:
            report_lines.append("No forecast summary provided.")
        else:
            # Iterate over forecast_results dictionary (multiple models)
            for model_name, model_data in forecast_results.items():
                report_lines.append(f"\n--- {model_name} Forecast ---")
                forecast_summary = model_data.get("forecast_summary", "No forecast summary provided.")
                report_lines.append(f"Forecast Summary: {forecast_summary}")
                report_lines.append("Model Metadata:")
                model_metadata = model_data.get("Model_Metadata", {})
                for key, value in model_metadata.items():
                    report_lines.append(f"  - {key}: {value}")
        
        report_lines.append("\n=== Visualizations ===")
        for viz_name in visualizations.keys():
            report_lines.append(f"{viz_name}: See attached visualization output.")
        
        report_content = "\n".join(report_lines)
        try:
            with open(self.output_path, "w", encoding="utf-8") as file:
                file.write(report_content)
            logging.info(f"Plain text report successfully written to {self.output_path}")
        except Exception as e:
            logging.error(f"Failed to write plain text report: {e}")
            raise e

        return report_content
# =============================================
# Example Usage of the Reporting Module
# =============================================
if __name__ == "__main__":
    # Dummy data simulating outputs from Data Analysis, Predictive Analytics, and Visualization modules
    analysis_results = {
        "summary": "The stock exhibits a stable trend with moderate volatility.",
        "technical_indicators": {
            "moving_average_20": 150.25,
            "moving_average_50": 148.75,
            "RSI": 55
        }
    }
    forecast_results = {
        "forecast_summary": "A slight upward trend is expected over the next 30 days.",
        "model_metadata": {
            "model": "Prophet",
            "confidence_interval": "95%",
            "parameters": "default"
        }
    }
    # Visualizations provided as HTML strings (e.g., generated by Plotly or Bokeh)
    visualizations = {
        "Price Trend": "<div>HTML content for price trend chart</div>",
        "Moving Averages": "<div>HTML content for moving averages chart</div>"
    }

    # Generate an HTML report
    html_report_generator = HTMLReportGenerator("financial_report.html")
    html_report = html_report_generator.generate_report(analysis_results, forecast_results, visualizations)

    # Generate a plain text report
    text_report_generator = PlainTextReportGenerator("financial_report.txt")
    text_report = text_report_generator.generate_report(analysis_results, forecast_results, visualizations)

    # For demonstration, print the first few lines of each report
    print("HTML Report Preview:\n", html_report[:500])
    print("\nPlain Text Report Preview:\n", text_report[:500])
