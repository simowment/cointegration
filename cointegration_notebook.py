# %% [markdown]
# # Cointegration Analysis Framework
#
# This notebook provides a complete framework for studying cointegration between multiple financial symbols using data from Twelve Data.

# %%
# Import the framework
from cointegration_framework import CointegrationFramework

# Import example scripts
from cointegration_examples import (
    example_tech_stocks,
    example_gold_bitcoin, 
    example_crypto_portfolio,
    example_indices
)

# %%
# Configuration
API_KEY = ""  # Replace with your own API key

print("Cointegration Framework imported!")
print("Available examples:")
print("- example_tech_stocks()")
print("- example_gold_bitcoin()")
print("- example_crypto_portfolio()")
print("- example_indices()")

# %%
# Quick example: Gold vs Bitcoin analysis
print("Running Gold vs Bitcoin example...")
framework = example_gold_bitcoin()

# %%
# To use the framework directly:
# framework = CointegrationFramework(API_KEY)
# data = framework.download_data(['AAPL', 'MSFT'], interval='1day', outputsize=500)
# prices = framework.prepare_price_matrix('close')
# returns = framework.calculate_returns('log')
# results = framework.pairwise_cointegration()

# framework.print_summary_report()
