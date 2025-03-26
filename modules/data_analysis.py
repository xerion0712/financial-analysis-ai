# modules/data_analysis.py

import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod

# Configure logging for the module
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ======================================================================
# Abstract Base Class for analysis methods
# ======================================================================
class AnalysisMethod(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        """
        Performs analysis on the input data and returns the results as a dictionary.
        This method must be implemented by any subclass.
        """
        pass

# ======================================================================
# Class for computing summary statistics
# ======================================================================
class SummaryStatistics(AnalysisMethod):
    def analyze(self, data: pd.DataFrame) -> dict:
        """
        Computes basic summary statistics: mean, median, and standard deviation.
        Uses the 'Average_Close' column if available, otherwise falls back to 'Close'.
        """
        try:
            if 'Average_Close' in data.columns:
                series = data['Average_Close']
                logging.info("Using 'Average_Close' for summary statistics.")
            else:
                series = data['Close']
                logging.info("Using 'Close' for summary statistics.")
            
            result = {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std()
            }
            logging.info("Summary statistics computed successfully.")
            return result
        except Exception as e:
            logging.error(f"Error computing summary statistics: {e}")
            return {}

# ======================================================================
# Class for computing moving averages
# ======================================================================
class MovingAverageAnalysis(AnalysisMethod):
    def __init__(self, windows: list = [20, 50]):
        """
        Initializes the moving average analysis with specified window sizes.
        :param windows: List of integers representing the period of moving averages.
        """
        self.windows = windows

    def analyze(self, data: pd.DataFrame) -> dict:
        """
        Calculates moving averages for each specified window.
        Returns a dictionary where each key is of the form 'moving_average_<window>'.
        Uses the 'Average_Close' column if available; otherwise uses 'Close'.
        """
        result = {}
        try:
            if 'Average_Close' in data.columns:
                series = data['Average_Close']
                logging.info("Using 'Average_Close' for moving averages.")
            else:
                series = data['Close']
                logging.info("Using 'Close' for moving averages.")
            
            for window in self.windows:
                ma_series = series.rolling(window=window).mean()
                result[f'moving_average_{window}'] = ma_series
                logging.info(f"Computed {window}-day moving average.")
            
            logging.info("Moving averages computed successfully.")
            return result
        except Exception as e:
            logging.error(f"Error computing moving averages: {e}")
            return {}

# ======================================================================
# Main Data Analyzer class that aggregates different analysis methods
# ======================================================================
class DataAnalyzer:
    def __init__(self):
        """
        Initializes the DataAnalyzer with an empty list of analysis methods.
        """
        self.analysis_methods = []
        self.results = {}

    def register_analysis_method(self, method: AnalysisMethod):
        """
        Registers an analysis method (an instance of a subclass of AnalysisMethod).
        :param method: An object implementing the AnalysisMethod interface.
        """
        self.analysis_methods.append(method)
        logging.info(f"Registered analysis method: {method.__class__.__name__}")

    def analyze(self, data: pd.DataFrame) -> dict:
        """
        Applies all registered analysis methods on the input data.
        :param data: Aggregated data as a Pandas DataFrame with a DateTime index.
        :return: A dictionary containing results from each analysis method.
        """
        logging.info("Starting data analysis...")
        for method in self.analysis_methods:
            try:
                method_result = method.analyze(data)
                self.results[method.__class__.__name__] = method_result
            except Exception as e:
                logging.error(f"Error in {method.__class__.__name__}: {e}")
        logging.info("Data analysis completed.")
        return self.results

# ======================================================================
# Example usage
# ======================================================================
if __name__ == "__main__":
    # Create a dummy dataset for demonstration with a DateTime index
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Close': np.random.uniform(100, 200, size=100),
        'Average_Close': np.random.uniform(100, 200, size=100)
    }, index=dates)
    
    # Initialize the DataAnalyzer
    analyzer = DataAnalyzer()
    
    # Register analysis methods: SummaryStatistics and MovingAverageAnalysis
    analyzer.register_analysis_method(SummaryStatistics())
    analyzer.register_analysis_method(MovingAverageAnalysis(windows=[20, 50]))
    
    # Perform analysis on the aggregated data
    analysis_results = analyzer.analyze(df)
    logging.info(f"Analysis results:\n{analysis_results}")
