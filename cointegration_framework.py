# Cointegration Analysis Framework
#
# This module provides a complete framework for studying cointegration
# between multiple financial symbols using data from Twelve Data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from twelvedata import TDClient
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Tests statistiques pour la cointegration
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.linear_model import LinearRegression

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CointegrationFramework:
    """Framework for analysing cointegration between multiple financial symbols."""
    
    def __init__(self, api_key: str):
        self.td_client = TDClient(apikey=api_key)
        self.data = {}
        self.prices_df = None
        self.returns_df = None
        self.cointegration_results = {}
        
    def download_data(self, symbols: List[str], interval: str = '1day', 
                     outputsize: int = 5000, start_date: str = None) -> Dict:
        """
        Downloads data for a list of symbols from Twelve Data
        
        Args:
            symbols: List of symbols to download
            interval: Time interval (1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week, 1month)
            outputsize: Number of data points to retrieve
            start_date: Start date (YYYY-MM-DD format)
        
        Returns:
            Dictionary containing data for each symbol
        """
        print(f"Downloading data for {len(symbols)} symbols…")
        
        for symbol in symbols:
            try:
                ts = self.td_client.time_series(
                    symbol=symbol,
                    interval=interval,
                    outputsize=outputsize,
                    start_date=start_date
                )
                
                df = ts.as_pandas()
                # The library already returns data with a sorted datetime index.
                
                self.data[symbol] = df
                print(f"✓ {symbol}: {len(df)} data points")
                
            except Exception as e:
                print(f"✗ Error while downloading {symbol}: {str(e)}")
        
        return self.data
    
    def prepare_price_matrix(self, price_column: str = 'close') -> pd.DataFrame:
        """
        Prepares an aligned price matrix for all symbols
        
        Args:
            price_column: Price column to use ('open', 'high', 'low', 'close')
        
        Returns:
            DataFrame with aligned prices
        """
        if not self.data:
            raise ValueError("No data available. Call download_data() first.")
        
        price_data = {}
        for symbol, df in self.data.items():
            if price_column in df.columns:
                price_data[symbol] = df[price_column]
        
        self.prices_df = pd.DataFrame(price_data)
        self.prices_df.dropna(inplace=True)
        
        print(f"Price matrix created: {self.prices_df.shape[0]} observations, {self.prices_df.shape[1]} symbols")
        return self.prices_df
    
    def calculate_returns(self, method: str = 'log') -> pd.DataFrame:
        """
        Calculates returns for all symbols
        
        Args:
            method: 'log' for logarithmic returns, 'simple' for simple returns
        
        Returns:
            DataFrame of returns
        """
        if self.prices_df is None:
            raise ValueError("Prepare the price matrix first with prepare_price_matrix().")
        
        if method == 'log':
            self.returns_df = np.log(self.prices_df / self.prices_df.shift(1))
        else:
            self.returns_df = self.prices_df.pct_change()
        
        self.returns_df.dropna(inplace=True)
        return self.returns_df

    def test_stationarity(self, series: pd.Series, name: str = "") -> Dict:
        """
        Stationarity test with Augmented Dickey-Fuller
        
        Args:
            series: Time series to test
            name: Name of the series for display
        
        Returns:
            Dictionary with test results
        """
        result = adfuller(series.dropna())
        
        output = {
            'name': name,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
        
        return output
    
    def test_all_stationarity(self) -> pd.DataFrame:
        """
        Tests the stationarity of all prices and returns
        
        Returns:
            DataFrame with test results
        """
        results = []
        
        # Price tests (levels)
        if self.prices_df is not None:
            for col in self.prices_df.columns:
                result = self.test_stationarity(self.prices_df[col], f"{col}_price")
                result['type'] = 'Price'
                results.append(result)
        
        # Returns tests
        if self.returns_df is not None:
            for col in self.returns_df.columns:
                result = self.test_stationarity(self.returns_df[col], f"{col}_return")
                result['type'] = 'Return'
                results.append(result)
        
        return pd.DataFrame(results)
        
    def engle_granger_test(self, y: pd.Series, x: pd.Series, 
                            symbol_y: str, symbol_x: str) -> Dict:
        """
        Engle-Granger cointegration test between two series
        
        Args:
            y: Dependent series
            x: Independent series
            symbol_y: Name of symbol y
            symbol_x: Name of symbol x
        
        Returns:
            Dictionary with test results
        """
        # Cointegration test
        coint_stat, p_value, critical_values = coint(y, x)
        
        # Regression to obtain residuals
        X = x.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y.values)
        residuals = y.values - reg.predict(X)
        
        # Stationarity test of residuals
        residuals_series = pd.Series(residuals, index=y.index)
        residuals_test = self.test_stationarity(residuals_series, "residuals")
        
        return {
            'pair': f"{symbol_y} ~ {symbol_x}",
            'coint_statistic': coint_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_cointegrated': p_value < 0.05,
            'beta': reg.coef_[0],
            'alpha': reg.intercept_,
            'residuals': residuals_series,
            'residuals_stationary': residuals_test['is_stationary']
        }

    def pairwise_cointegration(self) -> pd.DataFrame:
        """
        Tests cointegration for all symbol pairs (avoids duplicates)
        
        Returns:
            DataFrame with cointegration results
        """
        if self.prices_df is None:
            raise ValueError("Prepare the price matrix first.")
        
        results = []
        symbols = self.prices_df.columns.tolist()
        n_pairs = len(symbols) * (len(symbols) - 1) // 2
        
        print(f"Cointegration test for {len(symbols)} symbols ({n_pairs} pairs)...")
        
        for i, symbol_y in enumerate(symbols):
            for j, symbol_x in enumerate(symbols):
                # Avoid auto-cointegration and duplicates
                if i >= j:
                    continue
                
                try:
                    result = self.engle_granger_test(
                        self.prices_df[symbol_y], 
                        self.prices_df[symbol_x],
                        symbol_y, 
                        symbol_x
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error for pair {symbol_y}-{symbol_x}: {str(e)}")
        
        # Store results
        self.cointegration_results['pairwise'] = results
        
        # Create results DataFrame
        df_results = pd.DataFrame([
            {
                'pair': r['pair'],
                'coint_statistic': r['coint_statistic'],
                'p_value': r['p_value'],
                'is_cointegrated': r['is_cointegrated'],
                'beta': r['beta'],
                'alpha': r['alpha']
            } for r in results
        ])
        
        return df_results

    def johansen_test(self, max_lags: int = 12) -> Dict:
        """
        Johansen cointegration test for multivariate analysis
        
        Args:
            max_lags: Maximum number of lags to consider
        
        Returns:
            Dictionary with Johansen test results
        """
        if self.prices_df is None:
            raise ValueError("Prepare the price matrix first.")
        
        # Johansen test
        result = coint_johansen(self.prices_df.values, det_order=0, k_ar_diff=max_lags)
        
        # Extract results
        johansen_results = {
            'trace_stats': result.lr1,
            'max_eigen_stats': result.lr2,
            'critical_values_trace': result.cvt,
            'critical_values_max_eigen': result.cvm,
            'eigenvalues': result.eig,
            'eigenvectors': result.evec,
            'symbols': self.prices_df.columns.tolist()
        }
        
        # Determine the number of cointegration relationships
        n_coint_trace = 0
        n_coint_max_eigen = 0
        
        for i in range(len(result.lr1)):
            if result.lr1[i] > result.cvt[i, 1]:  # 5% critical value
                n_coint_trace = i + 1
            if result.lr2[i] > result.cvm[i, 1]:  # 5% critical value
                n_coint_max_eigen = i + 1
        
        johansen_results['n_coint_trace'] = n_coint_trace
        johansen_results['n_coint_max_eigen'] = n_coint_max_eigen
        
        self.cointegration_results['johansen'] = johansen_results
        
        return johansen_results

    def plot_prices(self, figsize: Tuple[int, int] = (15, 8), normalize: bool = True):
        """
        Price chart for all symbols
        
        Args:
            figsize: Figure size
            normalize: If True, normalizes prices to 100 at the beginning
        """
        if self.prices_df is None:
            raise ValueError("No price data available")
        
        plt.figure(figsize=figsize)
        
        if normalize:
            # Price normalization (base 100)
            normalized_prices = (self.prices_df / self.prices_df.iloc[0]) * 100
            for col in normalized_prices.columns:
                plt.plot(normalized_prices.index, normalized_prices[col], label=col, linewidth=2)
            plt.ylabel('Normalized Price (Base 100)')
            plt.title('Evolution of Normalized Prices')
        else:
            for col in self.prices_df.columns:
                plt.plot(self.prices_df.index, self.prices_df[col], label=col, linewidth=2)
            plt.ylabel('Price')
            plt.title('Evolution of Prices')
        
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data_type: str = 'prices', figsize: Tuple[int, int] = (10, 8)):
        """
        Correlation matrix
        
        Args:
            data_type: 'prices' or 'returns'
            figsize: Figure size
        """
        if data_type == 'prices' and self.prices_df is not None:
            data = self.prices_df
            title = 'Correlation Matrix - Prices'
        elif data_type == 'returns' and self.returns_df is not None:
            data = self.returns_df
            title = 'Correlation Matrix - Returns'
        else:
            raise ValueError(f"Data {data_type} not available")
        
        plt.figure(figsize=figsize)
        correlation_matrix = data.corr()
        
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f')
        
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_cointegration_heatmap(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Heatmap of cointegration p-values
        
        Args:
            figsize: Figure size
        """
        if 'pairwise' not in self.cointegration_results:
            raise ValueError("Perform pairwise cointegration test first")
        
        # Create p-value matrix
        symbols = self.prices_df.columns.tolist()
        n_symbols = len(symbols)
        p_value_matrix = np.ones((n_symbols, n_symbols))
        
        for result in self.cointegration_results['pairwise']:
            pair = result['pair'].split(' ~ ')
            y_idx = symbols.index(pair[0])
            x_idx = symbols.index(pair[1])
            p_value_matrix[y_idx, x_idx] = result['p_value']
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(p_value_matrix,
                   xticklabels=symbols,
                   yticklabels=symbols,
                   annot=True,
                   cmap='RdYlGn_r',
                   center=0.05,
                   fmt='.3f',
                   cbar_kws={'label': 'P-value'})
        
        plt.title('Cointegration P-values Heatmap\n(Green = Cointegrated, Red = Non Cointegrated)')
        plt.xlabel('Symbol X (Independent Variable)')
        plt.ylabel('Symbol Y (Dependent Variable)')
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, symbol_y: str, symbol_x: str, figsize: Tuple[int, int] = (15, 10)):
        """
        Residual plot for a cointegrated pair
        
        Args:
            symbol_y: Dependent symbol
            symbol_x: Independent symbol
            figsize: Figure size
        """
        if 'pairwise' not in self.cointegration_results:
            raise ValueError("Perform pairwise cointegration test first")
        
        # Search for results for this pair
        pair_name = f"{symbol_y} ~ {symbol_x}"
        result = None
        
        for r in self.cointegration_results['pairwise']:
            if r['pair'] == pair_name:
                result = r
                break
        
        if result is None:
            raise ValueError(f"Pair {pair_name} not found in results")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Price scatter plot
        axes[0, 0].scatter(self.prices_df[symbol_x], self.prices_df[symbol_y], alpha=0.6)
        axes[0, 0].set_xlabel(f'Price {symbol_x}')
        axes[0, 0].set_ylabel(f'Price {symbol_y}')
        axes[0, 0].set_title(f'Relationship {symbol_y} vs {symbol_x}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series of residuals
        axes[0, 1].plot(result['residuals'].index, result['residuals'], linewidth=1)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Time Series of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Histogram of residuals
        axes[1, 0].hist(result['residuals'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q plot of residuals
        from scipy import stats
        stats.probplot(result['residuals'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {pair_name}\nP-value: {result["p_value"]:.4f}, Cointegrated: {result["is_cointegrated"]}')
        plt.tight_layout()
        plt.show()

    def generate_summary_report(self) -> Dict:
        """
        Generates a comprehensive summary report
        
        Returns:
            Dictionary containing the summary report
        """
        report = {
            'data_summary': {},
            'stationarity_summary': {},
            'cointegration_summary': {},
            'recommendations': []
        }
        
        # Data summary
        if self.prices_df is not None:
            report['data_summary'] = {
                'n_symbols': len(self.prices_df.columns),
                'symbols': self.prices_df.columns.tolist(),
                'date_range': {
                    'start': self.prices_df.index.min().strftime('%Y-%m-%d'),
                    'end': self.prices_df.index.max().strftime('%Y-%m-%d')
                },
                'n_observations': len(self.prices_df)
            }
        
        # Stationarity test
        stationarity_results = self.test_all_stationarity()
        if not stationarity_results.empty:
            report['stationarity_summary'] = {
                'prices_stationary': stationarity_results[stationarity_results['type'] == 'Price']['is_stationary'].sum(),
                'returns_stationary': stationarity_results[stationarity_results['type'] == 'Return']['is_stationary'].sum(),
                'total_price_tests': len(stationarity_results[stationarity_results['type'] == 'Price']),
                'total_return_tests': len(stationarity_results[stationarity_results['type'] == 'Return'])
            }
        
        # Cointegration summary
        if 'pairwise' in self.cointegration_results:
            pairwise_results = self.cointegration_results['pairwise']
            cointegrated_pairs = [r for r in pairwise_results if r['is_cointegrated']]
            
            report['cointegration_summary'] = {
                'total_pairs_tested': len(pairwise_results),
                'cointegrated_pairs': len(cointegrated_pairs),
                'cointegration_rate': len(cointegrated_pairs) / len(pairwise_results) if pairwise_results else 0,
                'best_pairs': sorted(cointegrated_pairs, key=lambda x: x['p_value'])[:5]
            }
        
        # Recommendations
        if 'cointegration_summary' in report and report['cointegration_summary']['cointegrated_pairs'] > 0:
            report['recommendations'].append("Cointegration relationships detected - consider pairs trading strategies")
        
        if 'stationarity_summary' in report:
            if report['stationarity_summary']['prices_stationary'] > 0:
                report['recommendations'].append("Some prices are stationary - check data quality")
            if report['stationarity_summary']['returns_stationary'] < report['stationarity_summary']['total_return_tests']:
                report['recommendations'].append("Some returns are not stationary - consider additional transformations")
        
        return report
    
    
    def export_results(self, filename: str = 'cointegration_results.xlsx'):
        """
        Exports all results to an Excel file
        
        Args:
            filename: Output filename
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Prices
            if self.prices_df is not None:
                self.prices_df.to_excel(writer, sheet_name='Prices')
            
            # Returns
            if self.returns_df is not None:
                self.returns_df.to_excel(writer, sheet_name='Returns')
            
            # Stationarity tests
            stationarity_results = self.test_all_stationarity()
            if not stationarity_results.empty:
                stationarity_results.to_excel(writer, sheet_name='Stationarity', index=False)
            
            # Pairwise cointegration results
            if 'pairwise' in self.cointegration_results:
                pairwise_df = pd.DataFrame([
                    {
                        'pair': r['pair'],
                        'coint_statistic': r['coint_statistic'],
                        'p_value': r['p_value'],
                        'is_cointegrated': r['is_cointegrated'],
                        'beta': r['beta'],
                        'alpha': r['alpha']
                    } for r in self.cointegration_results['pairwise']
                ])
                pairwise_df.to_excel(writer, sheet_name='Cointegration_Pairwise', index=False)
            
            # Johansen results
            if 'johansen' in self.cointegration_results:
                johansen_summary = pd.DataFrame({
                    'Statistic_Type': ['Trace'] * len(self.cointegration_results['johansen']['trace_stats']) + 
                                     ['Max_Eigenvalue'] * len(self.cointegration_results['johansen']['max_eigen_stats']),
                    'Test_Statistic': list(self.cointegration_results['johansen']['trace_stats']) + 
                                     list(self.cointegration_results['johansen']['max_eigen_stats']),
                    'Critical_Value_5%': list(self.cointegration_results['johansen']['critical_values_trace'][:, 1]) + 
                                        list(self.cointegration_results['johansen']['critical_values_max_eigen'][:, 1])
                })
                johansen_summary.to_excel(writer, sheet_name='Johansen_Test', index=False)
        
        print(f"Results exported to {filename}")