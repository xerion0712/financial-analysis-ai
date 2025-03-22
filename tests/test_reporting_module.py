"""
test_reporting_module.py

Pytest tests for the Reporting Module (HTMLReportGenerator and PlainTextReportGenerator).
We ensure the module behaves correctly with various inputs, including edge cases.
"""

import sys
import os
import pytest
import tempfile
from pathlib import Path

# Allow imports from the parent directory (so we can access modules/reporting.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.reporting import HTMLReportGenerator, PlainTextReportGenerator, BaseReportGenerator


@pytest.fixture
def minimal_analysis_results():
    """
    Returns a minimal dictionary for analysis_results.
    """
    return {
        "summary": "Minimal analysis summary",
        "technical_indicators": {
            "MA20": 100.0
        }
    }

@pytest.fixture
def typical_analysis_results():
    """
    Returns a typical dictionary for analysis_results.
    """
    return {
        "summary": "The stock shows moderate volatility with an upward trend.",
        "technical_indicators": {
            "MA20": 150.0,
            "MA50": 148.5,
            "RSI": 55
        }
    }

@pytest.fixture
def minimal_forecast_results():
    """
    Returns a minimal dictionary for forecast_results.
    """
    return {
        "Prophet": {
            "forecast_summary": "Expect slight growth in the next 30 days.",
            "Model_Metadata": {
                "model": "Prophet",
                "params": "default"
            }
        }
    }

@pytest.fixture
def typical_forecast_results():
    """
    Returns a dictionary for forecast_results with multiple models.
    """
    return {
        "Prophet": {
            "forecast_summary": "Slight upward trend for the next 30 days.",
            "Model_Metadata": {
                "model": "Prophet",
                "params": "daily",
                "RMSE": 4.5
            }
        },
        "ARIMA": {
            "forecast_summary": "Stable forecast with minor fluctuations.",
            "Model_Metadata": {
                "model": "ARIMA",
                "params": "order=(2,1,2)",
                "RMSE": 5.2
            }
        }
    }

@pytest.fixture
def minimal_visualizations():
    """
    Returns a minimal dictionary for visualizations.
    """
    return {
        "Sample Chart": "<div>Mock chart HTML</div>"
    }

@pytest.fixture
def typical_visualizations():
    """
    Returns a typical dictionary for visualizations with multiple entries.
    """
    return {
        "Price Trend": "<div>Price Trend Chart</div>",
        "Moving Averages": "<div>MA Chart</div>",
        "Combined Forecast": "<div>Combined Forecast Chart</div>"
    }


def test_html_report_generator_minimal(minimal_analysis_results, minimal_forecast_results, minimal_visualizations):
    """
    Test HTMLReportGenerator with minimal valid data.
    """
    generator = HTMLReportGenerator()
    # Use a temporary file to avoid writing to the real filesystem
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_report.html")
        generator.output_path = output_path

        report_str = generator.generate_report(
            analysis_results=minimal_analysis_results,
            forecast_results=minimal_forecast_results,
            visualizations=minimal_visualizations
        )
        # Check that the report string is not empty
        assert len(report_str) > 0, "Expected non-empty HTML report string."
        # Check that file was created
        assert os.path.exists(output_path), "Expected output file to be created."


def test_html_report_generator_typical(typical_analysis_results, typical_forecast_results, typical_visualizations):
    """
    Test HTMLReportGenerator with typical data including multiple models and charts.
    """
    generator = HTMLReportGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "typical_report.html")
        generator.output_path = output_path

        report_str = generator.generate_report(
            analysis_results=typical_analysis_results,
            forecast_results=typical_forecast_results,
            visualizations=typical_visualizations
        )
        # Check that the report string is not empty
        assert "The stock shows moderate volatility" in report_str
        assert "Prophet" in report_str
        assert "ARIMA" in report_str
        assert "Price Trend" in report_str
        # Check file existence
        assert os.path.exists(output_path), "Expected HTML report file to be created."


def test_plain_text_report_generator_minimal(minimal_analysis_results, minimal_forecast_results, minimal_visualizations):
    """
    Test PlainTextReportGenerator with minimal data.
    """
    generator = PlainTextReportGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_report.txt")
        generator.output_path = output_path

        report_str = generator.generate_report(
            analysis_results=minimal_analysis_results,
            forecast_results=minimal_forecast_results,
            visualizations=minimal_visualizations
        )
        assert len(report_str) > 0, "Expected non-empty plain text report string."
        assert os.path.exists(output_path), "Expected text report file to be created."


def test_plain_text_report_generator_typical(typical_analysis_results, typical_forecast_results, typical_visualizations):
    """
    Test PlainTextReportGenerator with typical data.
    """
    generator = PlainTextReportGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "typical_report.txt")
        generator.output_path = output_path

        report_str = generator.generate_report(
            analysis_results=typical_analysis_results,
            forecast_results=typical_forecast_results,
            visualizations=typical_visualizations
        )
        assert "The stock shows moderate volatility" in report_str
        assert "Prophet" in report_str
        assert "ARIMA" in report_str
        assert os.path.exists(output_path), "Expected plain text report file to be created."


def test_missing_data_html():
    """
    Test HTMLReportGenerator with empty dictionaries to see if it handles missing data gracefully.
    """
    generator = HTMLReportGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "missing_data_report.html")
        generator.output_path = output_path

        report_str = generator.generate_report(
            analysis_results={},
            forecast_results={},
            visualizations={}
        )
        # The generator should handle empty data without errors
        assert "No analysis summary provided" in report_str
        assert "No forecast summary provided" in report_str
        assert os.path.exists(output_path), "Expected HTML report file to be created even with missing data."


def test_missing_data_text():
    """
    Test PlainTextReportGenerator with empty dictionaries to see if it handles missing data gracefully.
    """
    generator = PlainTextReportGenerator()
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "missing_data_report.txt")
        generator.output_path = output_path

        report_str = generator.generate_report(
            analysis_results={},
            forecast_results={},
            visualizations={}
        )
        # The generator should handle empty data without errors
        assert "No analysis summary provided" in report_str
        assert "No forecast summary provided" in report_str
        assert os.path.exists(output_path), "Expected text report file to be created even with missing data."


def test_inheritance():
    """
    Test that HTMLReportGenerator and PlainTextReportGenerator inherit from BaseReportGenerator.
    """
    assert issubclass(HTMLReportGenerator, BaseReportGenerator), "HTMLReportGenerator should inherit from BaseReportGenerator"
    assert issubclass(PlainTextReportGenerator, BaseReportGenerator), "PlainTextReportGenerator should inherit from BaseReportGenerator"


def test_invalid_output_path_html(minimal_analysis_results, minimal_forecast_results, minimal_visualizations):
    """
    Test HTMLReportGenerator with an invalid output path to ensure proper error handling.
    """
    generator = HTMLReportGenerator()
    # Use a directory path instead of a file path to cause an error
    with pytest.raises(Exception):
        generator.output_path = "/dev/null/invalid_file.html"
        generator.generate_report(
            analysis_results=minimal_analysis_results,
            forecast_results=minimal_forecast_results,
            visualizations=minimal_visualizations
        )


def test_invalid_output_path_text(minimal_analysis_results, minimal_forecast_results, minimal_visualizations):
    """
    Test PlainTextReportGenerator with an invalid output path to ensure proper error handling.
    """
    generator = PlainTextReportGenerator()
    with pytest.raises(Exception):
        generator.output_path = "/dev/null/invalid_file.txt"
        generator.generate_report(
            analysis_results=minimal_analysis_results,
            forecast_results=minimal_forecast_results,
            visualizations=minimal_visualizations
        )
