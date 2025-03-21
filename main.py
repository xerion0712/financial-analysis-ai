# main.py
from modules.data_collection import YahooFinanceCollector, CSVDataCollector, DataAggregator
from modules.data_analysis import DataAnalyzer, SummaryStatistics, MovingAverageAnalysis

def main():
    # ----- Parameters for Data Collection -----
    TICKER = "AAPL"
    START_DATE = "2022-01-01"
    END_DATE = "2023-12-31"
    
    # ----- Instantiate Data Collectors -----
    # Collector from Yahoo Finance
    yahoo_collector = YahooFinanceCollector()
    # Collector from CSV file (укажите корректный путь к файлу CSV)
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
    # Инициализируем анализатор данных и регистрируем методы анализа
    analyzer = DataAnalyzer()
    analyzer.register_analysis_method(SummaryStatistics())
    analyzer.register_analysis_method(MovingAverageAnalysis(windows=[20, 50]))
    
    # Выполняем анализ агрегированных данных
    analysis_results = analyzer.analyze(aggregated_data)
    print("\nAnalysis Results:")
    print(analysis_results)
    
    return analysis_results

if __name__ == '__main__':
    main()
