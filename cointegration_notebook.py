# %% [markdown]
# # Framework d'Analyse de Cointegration
# 
# Ce notebook fournit un framework complet pour étudier la cointegration entre plusieurs symboles financiers en utilisant les données de Twelve Data.

# %%
# Import du framework
from cointegration_framework import CointegrationFramework

# Import des exemples
from cointegration_examples import (
    example_tech_stocks,
    example_gold_bitcoin, 
    example_crypto_portfolio,
    example_indices
)

# %%
# Configuration
API_KEY = "939b755379b946c0b4ff05f7d30467a2"  # Remplacez par votre clé API

print("Framework de Cointegration importé!")
print("Exemples disponibles:")
print("- example_tech_stocks()")
print("- example_gold_bitcoin()")
print("- example_crypto_portfolio()")
print("- example_indices()")

# %%
# Exemple rapide: Analyse Or vs Bitcoin
print("Exécution de l'exemple Or vs Bitcoin...")
framework = example_gold_bitcoin()

# %%
# Pour utiliser le framework directement:
# framework = CointegrationFramework(API_KEY)
# data = framework.download_data(['AAPL', 'MSFT'], interval='1day', outputsize=500)
# prices = framework.prepare_price_matrix('close')
# returns = framework.calculate_returns('log')
# results = framework.pairwise_cointegration()
# framework.print_summary_report()