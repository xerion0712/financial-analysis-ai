# modules/data_collection.py

import os
import logging
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import yfinance as yf
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# STEP 1: Define an abstract base class for data collectors
# =============================================================================
class DataCollector(ABC):
    @abstractmethod
    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Abstract method to fetch data for a given ticker and time range.
        """
        pass

# =============================================================================
# STEP 2: Implement Yahoo Finance data collector using yfinance library
# =============================================================================
class YahooFinanceCollector(DataCollector):
    def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance for a specific ticker.
        """
        try:
            logging.info(f"Fetching data from Yahoo Finance for ticker: {ticker}")
            df = yf.download(ticker, start=start_date, end=end_date)
            logging.info(f"Successfully fetched data from Yahoo Finance for {ticker}")
            logging.info(f"Yahoo Finance raw columns: {df.columns}")
            
            # Convert MultiIndex columns to a flat index using the first level ('Price')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            logging.info(f"Yahoo Finance flattened columns: {df.columns}")
            
            # Normalize column names to Title Case for consistency
            normalized_columns = {col: col.title() for col in df.columns}
            df.rename(columns=normalized_columns, inplace=True)
            logging.info(f"Yahoo Finance normalized columns: {df.columns}")
            
            # Ensure we have a 'Close' column
            if 'Close' not in df.columns:
                if 'Adj Close' in df.columns:
                    df.rename(columns={'Adj Close': 'Close'}, inplace=True)
                else:
                    logging.warning(f"Yahoo Finance returned unexpected columns: {df.columns}")
            
            return df
        except Exception as e:
            logging.error(f"Error fetching data from Yahoo Finance: {e}")
            raise e




# =============================================================================
# STEP 3: Implement CSV data collector for manual data loading with graceful fallback
# =============================================================================
class CSVDataCollector(DataCollector):
    def __init__(self, filepath: str):
        """
        Initialize the CSV data collector with the path to the CSV file.
        """
        self.filepath = filepath

    def fetch_data(self, ticker: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load data from a CSV file. If file not found, return an empty DataFrame.
        """
        try:
            logging.info(f"Loading data from CSV file: {self.filepath}")
            df = pd.read_csv(self.filepath, index_col=0, parse_dates=True)
            logging.info(f"Successfully loaded data from CSV file: {self.filepath}")
            return df
        except FileNotFoundError:
            logging.warning(f"CSV file not found: {self.filepath}. Skipping CSV data collector.")
            return pd.DataFrame()  # Continue gracefully with an empty DataFrame
        except Exception as e:
            logging.error(f"Error loading data from CSV file: {e}")
            return pd.DataFrame()

# =============================================================================
# STEP 4: Implement Alpha Vantage - FM data collector with enhanced error handling
# =============================================================================
# class AlphaVantageCollector(DataCollector):
#     def __init__(self, api_key: str):
#         """
#         Initialize the Alpha Vantage collector with the provided API key.
#         """
#         self.api_key = api_key

#     def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
#         """
#         Fetch historical data from Alpha Vantage for a specific ticker.
#         """
#         try:
#             logging.info(f"Fetching data from Alpha Vantage for ticker: {ticker}")
#             base_url = "https://www.alphavantage.co/query"
#             params = {
#                 "function": "TIME_SERIES_DAILY_ADJUSTED",
#                 "symbol": ticker,
#                 "apikey": self.api_key,
#                 "outputsize": "full",
#                 "datatype": "json"
#             }
#             response = requests.get(base_url, params=params)
#             response.raise_for_status()  # Raises HTTPError for bad responses
#             data = response.json()
            
#             # Log the full response for debugging
#             logging.info(f"Alpha Vantage full response: {data}")
            
#             # Check for error messages or notes in the response
#             if "Time Series (Daily)" not in data:
#                 if "Note" in data:
#                     logging.warning(f"Alpha Vantage Note: {data['Note']}")
#                 elif "Error Message" in data:
#                     logging.warning(f"Alpha Vantage Error Message: {data['Error Message']}")
#                 else:
#                     logging.warning("Time Series (Daily) data not found in Alpha Vantage response.")
#                 return pd.DataFrame()  # Return empty DataFrame if expected data is missing

#             time_series = data["Time Series (Daily)"]
#             df = pd.DataFrame.from_dict(time_series, orient='index')
#             df.index = pd.to_datetime(df.index)
#             df = df.astype(float)
#             # Rename '4. close' to 'Close' for consistency
#             if '4. close' in df.columns:
#                 df.rename(columns={'4. close': 'Close'}, inplace=True)
#             # Filter DataFrame by date range
#             df = df.loc[start_date:end_date]
#             logging.info(f"Successfully fetched data from Alpha Vantage for {ticker}")
#             return df
#         except Exception as e:
#             logging.error(f"Error fetching data from Alpha Vantage: {e}")
#             return pd.DataFrame()

# class FinancialModelingPrepCollector(DataCollector):
#     def __init__(self, api_key: str):
#         """
#         Initialize the FMP collector with the provided API key.
#         """
#         self.api_key = api_key

#     def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
#         """
#         Fetch historical data from Financial Modeling Prep for a specific ticker.
#         """
#         try:
#             logging.info(f"Fetching data from Financial Modeling Prep for ticker: {ticker}")
#             # Build the endpoint URL for historical data
#             base_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
#             # When constructing the URL, requests.get() automatically handles query parameters.
#             params = {"apikey": self.api_key}
#             response = requests.get(base_url, params=params)
#             response.raise_for_status()
#             data = response.json()

#             # Log the full JSON response for debugging purposes
#             logging.info(f"Financial Modeling Prep full response: {data}")

#             if "historical" not in data:
#                 logging.warning("Historical data not found in Financial Modeling Prep response.")
#                 return pd.DataFrame()

#             historical_data = data["historical"]
#             df = pd.DataFrame(historical_data)
#             df["date"] = pd.to_datetime(df["date"])
#             df.set_index("date", inplace=True)
#             df.sort_index(inplace=True)

#             # Filter DataFrame by date range
#             df = df.loc[start_date:end_date]

#             # For consistency, rename the column 'close' to 'Close'
#             if "close" in df.columns:
#                 df.rename(columns={"close": "Close"}, inplace=True)

#             logging.info(f"Successfully fetched data from Financial Modeling Prep for {ticker}")
#             return df
#         except Exception as e:
#             logging.error(f"Error fetching data from Financial Modeling Prep: {e}")
#             return pd.DataFrame()
# modules/data_collection.py (добавляем новый класс)

# class PolygonCollector(DataCollector):
#     def __init__(self, api_key: str):
#         """
#         Initialize the Polygon.io collector with the provided API key.
#         """
#         self.api_key = api_key

#     def fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
#         """
#         Fetch historical aggregated daily data from Polygon.io for a specific ticker.
#         The endpoint used: 
#         https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}
#         """
#         try:
#             logging.info(f"Fetching data from Polygon.io for ticker: {ticker}")
#             base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
#             params = {"apiKey": self.api_key}
#             response = requests.get(base_url, params=params)
#             response.raise_for_status()
#             data = response.json()
            
#             # Log the full JSON response for debugging purposes
#             logging.info(f"Polygon.io full response: {data}")

#             if "results" not in data:
#                 logging.warning("Polygon.io response does not contain 'results'.")
#                 return pd.DataFrame()

#             results = data["results"]
#             df = pd.DataFrame(results)
#             if df.empty:
#                 logging.warning("Polygon.io returned an empty result set.")
#                 return df

#             # Convert the timestamp 't' (milliseconds since epoch) to datetime
#             if "t" in df.columns:
#                 df["date"] = pd.to_datetime(df["t"], unit="ms")
#                 df.set_index("date", inplace=True)
#                 df.sort_index(inplace=True)
#             else:
#                 logging.warning("Timestamp column 't' not found in Polygon.io data.")
            
#             # Rename the 'c' column (close price) to 'Close' for consistency
#             if "c" in df.columns:
#                 df.rename(columns={"c": "Close"}, inplace=True)
            
#             logging.info(f"Successfully fetched data from Polygon.io for {ticker}")
#             return df
#         except Exception as e:
#             logging.error(f"Error fetching data from Polygon.io: {e}")
#             return pd.DataFrame()

# =============================================================================
# STEP 5: Implement a data aggregator to compute average values from different sources
# =============================================================================
class DataAggregator:
    def __init__(self, collectors: List[DataCollector]):
        """
        Initialize the DataAggregator with a list of data collectors.
        """
        self.collectors = collectors

    def aggregate_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Aggregate data from all collectors and compute the average 'Close' price.
        """
        data_frames = []
        for collector in self.collectors:
            try:
                df = collector.fetch_data(ticker, start_date, end_date)
                # Ensure the DataFrame has the 'Close' column and is not empty
                if not df.empty and 'Close' in df.columns:
                    collector_name = collector.__class__.__name__
                    df = df[['Close']].rename(columns={'Close': f'Close_{collector_name}'})
                    data_frames.append(df)
            except Exception as e:
                logging.error(f"Error in collector {collector.__class__.__name__}: {e}")

        if not data_frames:
            raise ValueError("No data collected from any source.")

        # Align data on common dates by performing an inner join
        combined_df = pd.concat(data_frames, axis=1, join='inner')
        # Calculate the average of all Close columns
        close_columns = [col for col in combined_df.columns if str(col).startswith('Close')]
        combined_df['Average_Close'] = combined_df[close_columns].mean(axis=1)
        logging.info("Aggregated data and computed the average Close price from all sources.")
        return combined_df

# =============================================================================
# Example usage (this block can be executed from main.py)
# =============================================================================

if __name__ == "__main__":
    # Parameters for data fetching
    TICKER = "AAPL"              # Ticker symbol for Apple Inc.
    START_DATE = "2022-01-01"     # Start date for historical data
    END_DATE = "2023-12-31"       # End date for historical data

    # Filepath for CSV data (assumed to be located in the 'data/raw' directory)
    csv_filepath = os.path.join("data", "raw", "aapl_data.csv")
    
    # Instantiate individual data collectors
    yahoo_collector = YahooFinanceCollector()
    csv_collector = CSVDataCollector(filepath=csv_filepath)
    
    # Uncomment the next lines if you want to include Financial Modeling Prep in the future
    # fmp_api_key = os.getenv("FMP_API_KEY", "YOUR_FMP_API_KEY_HERE")
    # fmp_collector = FinancialModelingPrepCollector(api_key=fmp_api_key)
    
    # Create an aggregator with multiple sources (using Yahoo and CSV collectors)
    aggregator = DataAggregator(collectors=[yahoo_collector, csv_collector])
    
    # Aggregate data and compute average 'Close' prices
    try:
        aggregated_data = aggregator.aggregate_data(TICKER, START_DATE, END_DATE)
        print("Aggregated Data (first 5 rows):")
        print(aggregated_data.head())
    except Exception as e:
        logging.error(f"Failed to aggregate data: {e}")