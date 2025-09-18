#%%
import matplotlib.pyplot as plt
import os
from cointegration_framework import CointegrationFramework

# Global API key (set the environment variable TD_API_KEY or fill the string below)
API_KEY = ""  # <-- Your Twelve Data API key

# Matplotlib configuration for notebooks
# %matplotlib inline (uncomment if you are using Jupyter)

def example_tech_stocks():
    """
    Example 1: Cointegration analysis of technology stocks
    """
    print("=" * 60)
    print("EXAMPLE 1: TECHNOLOGY STOCKS")
    print("=" * 60)
    
    # Use the global API_KEY defined at the top of the script
    
    # Symbols to analyse
    TECH_SYMBOLS = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'TSLA',   # Tesla
        'NVDA',   # Nvidia
        'META',   # Meta
        'AMZN'    # Amazon
    ]
    
    framework = CointegrationFramework(API_KEY)
    
    print(f"Symbols to analyse: {TECH_SYMBOLS}")
    
    # Download data
    print("\nDownloading data...")
    data = framework.download_data(
        symbols=TECH_SYMBOLS,
        interval='1day',
        outputsize=1000  # Last 1000 days
    )
    
    if not data:
        print("❌ Error downloading data")
        return
    
    # Prepare data
    prices = framework.prepare_price_matrix('close')
    returns = framework.calculate_returns('log')
    
    # Visualisations
    framework.plot_prices(normalize=True)
    framework.plot_correlation_matrix('prices')
    framework.plot_correlation_matrix('returns')
    
    # Stationarity tests
    print("\n🔍 Stationarity tests...")
    stationarity_results = framework.test_all_stationarity()
    print(stationarity_results[['name', 'adf_statistic', 'p_value', 'is_stationary', 'type']])
    
    # Pairwise cointegration tests
    print("\n🔗 Pairwise cointegration tests...")
    pairwise_results = framework.pairwise_cointegration()
    
    # Display cointegrated pairs
    cointegrated_pairs = pairwise_results[pairwise_results['is_cointegrated']]
    print(f"\nCointegrated pairs found: {len(cointegrated_pairs)}")
    if len(cointegrated_pairs) > 0:
        print(cointegrated_pairs[['pair', 'p_value', 'beta']].sort_values('p_value'))
    
    # Cointegration heatmap
    framework.plot_cointegration_heatmap()
    
    # Johansen test
    print("\n�� Johansen test...")
    johansen_results = framework.johansen_test()
    print(f"Number of cointegration relationships detected:")
    print(f"  - Trace test: {johansen_results['n_coint_trace']}")
    print(f"  - Max eigenvalue test: {johansen_results['n_coint_max_eigen']}")
    
    # Residual analysis for the best pair
    if len(cointegrated_pairs) > 0:
        best_pair = cointegrated_pairs.iloc[0]['pair'].split(' ~ ')
        symbol_y, symbol_x = best_pair[0], best_pair[1]
        print(f"\n📈 Residual analysis for: {symbol_y} ~ {symbol_x}")
        framework.plot_residuals(symbol_y, symbol_x)
    
    
    return framework


def example_gold_bitcoin():
    """
    Example 2: Cointegration analysis of Bitcoin vs Ethereum
    """
    print("=" * 60)
    print("EXAMPLE 2: BITCOIN vs ETHEREUM")
    print("=" * 60)
    
    # Uses global API_KEY
    
    # Symbols for analysis - BTC and ETH as XAU/USD is not available with your plan
    CRYPTO_SYMBOLS = [
        'BTC/USD',   # Bitcoin in USD
        'ETH/USD'    # Ethereum in USD
    ]
    
    print(f"Symbols analyzed: {CRYPTO_SYMBOLS}")
    print("Hypothesis: Bitcoin and Ethereum, as major cryptocurrencies,")
    print("might exhibit a cointegration relationship.")
    
    framework = CointegrationFramework(API_KEY)
    
    print("\nDownloading data...")
    try:
        data = framework.download_data(
            symbols=CRYPTO_SYMBOLS,
            interval='1day',
            outputsize=730  # ~2 years of data
        )
        
        if not data:
            print("❌ Error downloading data")
            return
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nNote: If using the 'demo' key, data may be limited.")
        return
    
    # Prepare data
    prices = framework.prepare_price_matrix('close')
    returns = framework.calculate_returns('log')
    
    # Descriptive statistics
    print("\n📊 DESCRIPTIVE STATISTICS:")
    print("\nPrices:")
    print(prices.describe())
    print("\nReturns:")
    print(returns.describe())
    
    # Visualisations
    framework.plot_prices(normalize=True, figsize=(15, 8))
    
    # Graph with two Y axes
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Bitcoin on the left Y axis
    color = 'tab:orange'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Bitcoin Price (USD)', color=color)
    ax1.plot(prices.index, prices['BTC/USD'], color=color, linewidth=2, label='Bitcoin')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Ethereum Price (USD)', color=color)
    ax2.plot(prices.index, prices['ETH/USD'], color=color, linewidth=2, label='Ethereum')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Price Evolution: Bitcoin vs Ethereum (Separate Axes)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

    
    # Stationarity tests (background for the report)
    
    # Cointegration tests
    print("\n COINTEGRATION TEST: BITCOIN vs ETHEREUM")
    print("=" * 50)
    
    pairwise_results = framework.pairwise_cointegration()
    
    print("\nEngle-Granger Test Results:")
    print("-" * 80)
    
    for _, row in pairwise_results.iterrows():
        status = "✓ COINTEGRATED" if row['is_cointegrated'] else "❌ NON-COINTEGRATED"
        print(f"Pair: {row['pair']}")
        print(f"  Statistic: {row['coint_statistic']:8.4f}")
        print(f"  P-value: {row['p_value']:8.4f}")
        print(f"  Beta (slope): {row['beta']:8.4f}")
        print(f"  Alpha (intercept): {row['alpha']:8.4f}")
        print(f"  Status: {status}")
        print()
    
    # Analysis of results
    cointegrated_pairs = pairwise_results[pairwise_results['is_cointegrated']]
    
    print("📊 SUMMARY:")
    print(f"• Pairs tested: {len(pairwise_results)}")
    print(f"• Cointegrated pairs: {len(cointegrated_pairs)}")
    
    if len(cointegrated_pairs) > 0:
        print("\nINTERPRETATION:")
        print("✓ A cointegration relationship exists between Bitcoin and Ethereum!")
        print("  → Both cryptocurrencies tend towards a long-term equilibrium")
        print("  → Potential opportunities for pairs trading")
        
        # Residual analysis
        best_pair = cointegrated_pairs.iloc[0]['pair'].split(' ~ ')
        framework.plot_residuals(best_pair[0], best_pair[1])
        
    else:
        print("\n❌ INTERPRETATION:")
        print("• No cointegration relationship detected")
        print("  → Bitcoin and Ethereum evolve independently in the long term")
        print("  → Good diversification between these cryptocurrencies")
        
        # Rolling correlation analysis
        rolling_corr = prices.iloc[:, 0].rolling(30).corr(prices.iloc[:, 1])
        
        plt.figure(figsize=(15, 6))
        plt.plot(rolling_corr.index, rolling_corr, linewidth=2, color='purple')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Strong correlation')
        plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Strong negative correlation')
        plt.title('Rolling 30-day Correlation: Bitcoin vs Ethereum')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Final report
    framework.export_results('gold_bitcoin_cointegration.xlsx')
    
    return framework

#%%
def example_crypto_portfolio():
    """
    Example 3: Cointegration analysis of a cryptocurrency portfolio
    """
    print("=" * 60)
    print("EXAMPLE 3: CRYPTO PORTFOLIO")
    print("=" * 60)
    
    # Uses global API_KEY
    
    # Crypto symbols
    CRYPTO_SYMBOLS = [
        'BTC/USD',   # Bitcoin
        'ETH/USD',   # Ethereum
        'ADA/USD',   # Cardano
        'DOT/USD',   # Polkadot
        'LINK/USD'   # Chainlink
    ]
    
    print(f"Symbols analyzed: {CRYPTO_SYMBOLS}")
    
    # Initialization and analysis
    framework = CointegrationFramework(API_KEY)
    
    # Download data (hourly data for more points for better cointegration analysis)
    data = framework.download_data(
        symbols=CRYPTO_SYMBOLS,
        interval='1h',
        outputsize=2000
    )
    
    if not data:
        print("❌ Error downloading data")
        return
    
    prices = framework.prepare_price_matrix('close')
    returns = framework.calculate_returns('log')
    
    framework.plot_prices(normalize=True)
    framework.plot_correlation_matrix('returns')
    
    pairwise_results = framework.pairwise_cointegration()
    framework.plot_cointegration_heatmap()
    
    johansen_results = framework.johansen_test()
    print("\nJohansen test results:")
    print(f"  - Trace test: {johansen_results['n_coint_trace']}")
    print(f"  - Max eigenvalue test: {johansen_results['n_coint_max_eigen']}")
    
    framework.print_summary_report()
    framework.export_results('crypto_portfolio_cointegration.xlsx')
    
    return framework


def example_indices():
    """
    Example 4: Cointegration analysis of stock indices
    """
    print("=" * 60)
    print("EXAMPLE 4: STOCK INDICES")
    print("=" * 60)
    
    # Uses global API_KEY
    
    # Indices
    INDEX_SYMBOLS = [
        'SPY',   # S&P 500
        'QQQ',   # NASDAQ 100
        'IWM',   # Russell 2000
        'EFA',   # EAFE (Europe, Australia, Far East)
        'EEM'    # Emerging Markets
    ]
    
    print(f"Symbols analyzed: {INDEX_SYMBOLS}")
    
    # Analysis
    framework = CointegrationFramework(API_KEY)
    
    data = framework.download_data(
        symbols=INDEX_SYMBOLS,
        interval='1day',
        outputsize=1500
    )
    
    if not data:
        print("❌ Error downloading data")
        return
    
    # Complete analysis
    prices = framework.prepare_price_matrix('close')
    returns = framework.calculate_returns('log')
    
    framework.plot_prices(normalize=True)
    framework.plot_correlation_matrix('prices')
    
    pairwise_results = framework.pairwise_cointegration()
    framework.plot_cointegration_heatmap()
    
    johansen_results = framework.johansen_test()
    print("\nJohansen test results:")
    print(f"  - Trace test: {johansen_results['n_coint_trace']}")
    print(f"  - Max eigenvalue test: {johansen_results['n_coint_max_eigen']}")
    
    framework.print_summary_report()
    framework.export_results('indices_cointegration.xlsx')
    
    return framework


if __name__ == "__main__":
    """
    Example execution
    """
    print("🚀 COINTEGRATION ANALYSIS EXAMPLES")
    print("=" * 60)
    
    # Choose which example to run
    examples = {
        '1': ('Technology Stocks', example_tech_stocks),
        '2': ('Gold vs Bitcoin', example_gold_bitcoin),
        '3': ('Crypto Portfolio', example_crypto_portfolio),
        '4': ('Stock Indices', example_indices)
    }
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    # Default example: Gold vs Bitcoin 
    print("\n" + "=" * 60)
    print("EXECUTING DEFAULT EXAMPLE: GOLD vs BITCOIN")
    print("=" * 60)
    
    try:
        framework = example_gold_bitcoin()
        print("\n✅ Example finished successfully!")
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        print("Check your internet connection and API key its free btw.")
