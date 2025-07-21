#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cointegration_framework import CointegrationFramework

# Configuration matplotlib pour les notebooks
# %matplotlib inline (décommentez si vous utilisez Jupyter)

def example_tech_stocks():
    """
    Exemple 1: Analyse de cointegration des actions technologiques
    """
    print("=" * 60)
    print("EXEMPLE 1: ACTIONS TECHNOLOGIQUES")
    print("=" * 60)
    
    # Configuration
    API_KEY = "939b755379b946c0b4ff05f7d30467a2"  # Remplacez par votre clé API
    
    # Symboles à analyser
    TECH_SYMBOLS = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Google
        'TSLA',   # Tesla
        'NVDA',   # Nvidia
        'META',   # Meta
        'AMZN'    # Amazon
    ]
    
    # Initialisation du framework
    framework = CointegrationFramework(API_KEY)
    
    print(f"Symboles à analyser: {TECH_SYMBOLS}")
    
    # Téléchargement des données
    print("\nTéléchargement des données...")
    data = framework.download_data(
        symbols=TECH_SYMBOLS,
        interval='1day',
        outputsize=1000  # Derniers 1000 jours
    )
    
    if not data:
        print("❌ Erreur lors du téléchargement des données")
        return
    
    # Préparation des données
    prices = framework.prepare_price_matrix('close')
    returns = framework.calculate_returns('log')
    
    # Visualisations
    framework.plot_prices(normalize=True)
    framework.plot_correlation_matrix('prices')
    framework.plot_correlation_matrix('returns')
    
    # Tests de stationnarité
    print("\n🔍 Tests de stationnarité...")
    stationarity_results = framework.test_all_stationarity()
    print(stationarity_results[['name', 'adf_statistic', 'p_value', 'is_stationary', 'type']])
    
    # Tests de cointegration
    print("\n🔗 Tests de cointegration pairwise...")
    pairwise_results = framework.pairwise_cointegration()
    
    # Affichage des paires cointegrées
    cointegrated_pairs = pairwise_results[pairwise_results['is_cointegrated']]
    print(f"\nPaires cointegrées trouvées: {len(cointegrated_pairs)}")
    if len(cointegrated_pairs) > 0:
        print(cointegrated_pairs[['pair', 'p_value', 'beta']].sort_values('p_value'))
    
    # Heatmap de cointegration
    framework.plot_cointegration_heatmap()
    
    # Test de Johansen
    print("\n📊 Test de Johansen...")
    johansen_results = framework.johansen_test()
    print(f"Nombre de relations de cointegration détectées:")
    print(f"  - Test de trace: {johansen_results['n_coint_trace']}")
    print(f"  - Test de valeur propre max: {johansen_results['n_coint_max_eigen']}")
    
    # Analyse des résidus pour la meilleure paire
    if len(cointegrated_pairs) > 0:
        best_pair = cointegrated_pairs.iloc[0]['pair'].split(' ~ ')
        symbol_y, symbol_x = best_pair[0], best_pair[1]
        print(f"\n📈 Analyse des résidus pour: {symbol_y} ~ {symbol_x}")
        framework.plot_residuals(symbol_y, symbol_x)
    
    
    return framework


def example_gold_bitcoin():
    """
    Exemple 2: Analyse de cointegration Bitcoin vs Ethereum
    """
    print("=" * 60)
    print("EXEMPLE 2: BITCOIN vs ETHEREUM")
    print("=" * 60)
    
    # Configuration - utilisation de votre vraie clé API
    API_KEY = "939b755379b946c0b4ff05f7d30467a2"
    
    # Symboles pour l'analyse - BTC et ETH car XAU/USD n'est pas disponible avec votre plan
    CRYPTO_SYMBOLS = [
        'BTC/USD',   # Bitcoin en USD
        'ETH/USD'    # Ethereum en USD
    ]
    
    print(f"Symboles analysés: {CRYPTO_SYMBOLS}")
    print("Hypothèse: Bitcoin et Ethereum, en tant que principales cryptomonnaies,")
    print("pourraient présenter une relation de cointegration.")
    
    # Initialisation du framework
    framework = CointegrationFramework(API_KEY)
    
    # Téléchargement des données
    print("\nTéléchargement des données...")
    try:
        data = framework.download_data(
            symbols=CRYPTO_SYMBOLS,
            interval='1day',
            outputsize=730  # ~2 ans de données
        )
        
        if not data:
            print("❌ Erreur lors du téléchargement des données")
            return
            
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        print("\nNote: Si vous utilisez la clé 'demo', les données peuvent être limitées.")
        return
    
    # Préparation des données
    prices = framework.prepare_price_matrix('close')
    returns = framework.calculate_returns('log')
    
    # Statistiques descriptives
    print("\n📊 STATISTIQUES DESCRIPTIVES:")
    print("\nPrix:")
    print(prices.describe())
    print("\nRendements:")
    print(returns.describe())
    
    # Visualisations
    framework.plot_prices(normalize=True, figsize=(15, 8))
    
    # Graphique avec deux axes Y
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Bitcoin sur l'axe gauche
    color = 'tab:orange'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Prix du Bitcoin (USD)', color=color)
    ax1.plot(prices.index, prices['BTC/USD'], color=color, linewidth=2, label='Bitcoin')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Ethereum sur l'axe droit
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Prix d\'Ethereum (USD)', color=color)
    ax2.plot(prices.index, prices['ETH/USD'], color=color, linewidth=2, label='Ethereum')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Évolution des Prix: Bitcoin vs Ethereum (Axes Séparés)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

    
    # Tests de stationnarité (en arrière-plan pour le rapport)
    stationarity_results = framework.test_all_stationarity()
    
    # Tests de cointegration
    print("\n🔗 TEST DE COINTEGRATION BITCOIN vs ETHEREUM")
    print("=" * 50)
    
    pairwise_results = framework.pairwise_cointegration()
    
    print("\nRésultats du test d'Engle-Granger:")
    print("-" * 80)
    
    for _, row in pairwise_results.iterrows():
        status = "✓ COINTEGRÉ" if row['is_cointegrated'] else "❌ NON-COINTEGRÉ"
        print(f"Paire: {row['pair']}")
        print(f"  Statistique: {row['coint_statistic']:8.4f}")
        print(f"  P-value: {row['p_value']:8.4f}")
        print(f"  Beta (pente): {row['beta']:8.4f}")
        print(f"  Alpha (intercept): {row['alpha']:8.4f}")
        print(f"  Statut: {status}")
        print()
    
    # Analyse des résultats
    cointegrated_pairs = pairwise_results[pairwise_results['is_cointegrated']]
    
    print("📊 SYNTHÈSE:")
    print(f"• Paires testées: {len(pairwise_results)}")
    print(f"• Paires cointegrées: {len(cointegrated_pairs)}")
    
    if len(cointegrated_pairs) > 0:
        print("\n🎯 INTERPRÉTATION:")
        print("✓ Une relation de cointegration existe entre Bitcoin et Ethereum!")
        print("  → Les deux cryptomonnaies tendent vers un équilibre à long terme")
        print("  → Opportunités potentielles de pairs trading")
        
        # Analyse des résidus
        best_pair = cointegrated_pairs.iloc[0]['pair'].split(' ~ ')
        framework.plot_residuals(best_pair[0], best_pair[1])
        
    else:
        print("\n❌ INTERPRÉTATION:")
        print("• Aucune relation de cointegration détectée")
        print("  → Bitcoin et Ethereum évoluent indépendamment à long terme")
        print("  → Bonne diversification entre ces cryptomonnaies")
        
        # Analyse de corrélation rolling
        rolling_corr = prices.iloc[:, 0].rolling(30).corr(prices.iloc[:, 1])
        
        plt.figure(figsize=(15, 6))
        plt.plot(rolling_corr.index, rolling_corr, linewidth=2, color='purple')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Corrélation forte')
        plt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Corrélation négative forte')
        plt.title('Corrélation Rolling 30 jours: Bitcoin vs Ethereum')
        plt.xlabel('Date')
        plt.ylabel('Corrélation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Rapport final
    framework.print_summary_report()
    framework.export_results('gold_bitcoin_cointegration.xlsx')
    
    return framework

#%%
def example_crypto_portfolio():
    """
    Exemple 3: Analyse de cointegration d'un portefeuille de cryptomonnaies
    """
    print("=" * 60)
    print("EXEMPLE 3: PORTEFEUILLE CRYPTO")
    print("=" * 60)
    
    # Configuration
    API_KEY = "939b755379b946c0b4ff05f7d30467a2"  # Remplacez par votre clé API
    
    # Symboles crypto
    CRYPTO_SYMBOLS = [
        'BTC/USD',   # Bitcoin
        'ETH/USD',   # Ethereum
        'ADA/USD',   # Cardano
        'DOT/USD',   # Polkadot
        'LINK/USD'   # Chainlink
    ]
    
    print(f"Symboles analysés: {CRYPTO_SYMBOLS}")
    
    # Initialisation et analyse
    framework = CointegrationFramework(API_KEY)
    
    # Téléchargement des données (données horaires pour plus de points)
    data = framework.download_data(
        symbols=CRYPTO_SYMBOLS,
        interval='1h',
        outputsize=2000
    )
    
    if not data:
        print("❌ Erreur lors du téléchargement des données")
        return
    
    # Analyse complète
    prices = framework.prepare_price_matrix('close')
    returns = framework.calculate_returns('log')
    
    # Visualisations
    framework.plot_prices(normalize=True)
    framework.plot_correlation_matrix('returns')
    
    # Tests de cointegration
    pairwise_results = framework.pairwise_cointegration()
    framework.plot_cointegration_heatmap()
    
    # Test de Johansen
    johansen_results = framework.johansen_test()
    
    # Rapport
    framework.print_summary_report()
    framework.export_results('crypto_portfolio_cointegration.xlsx')
    
    return framework


def example_indices():
    """
    Exemple 4: Analyse de cointegration des indices boursiers
    """
    print("=" * 60)
    print("EXEMPLE 4: INDICES BOURSIERS")
    print("=" * 60)
    
    # Configuration
    API_KEY = "939b755379b946c0b4ff05f7d30467a2"  # Remplacez par votre clé API
    
    # Indices
    INDEX_SYMBOLS = [
        'SPY',   # S&P 500
        'QQQ',   # NASDAQ 100
        'IWM',   # Russell 2000
        'EFA',   # EAFE (Europe, Australie, Extrême-Orient)
        'EEM'    # Marchés émergents
    ]
    
    print(f"Symboles analysés: {INDEX_SYMBOLS}")
    
    # Analyse
    framework = CointegrationFramework(API_KEY)
    
    data = framework.download_data(
        symbols=INDEX_SYMBOLS,
        interval='1day',
        outputsize=1500
    )
    
    if not data:
        print("❌ Erreur lors du téléchargement des données")
        return
    
    # Analyse complète
    prices = framework.prepare_price_matrix('close')
    returns = framework.calculate_returns('log')
    
    framework.plot_prices(normalize=True)
    framework.plot_correlation_matrix('prices')
    
    pairwise_results = framework.pairwise_cointegration()
    framework.plot_cointegration_heatmap()
    
    johansen_results = framework.johansen_test()
    
    framework.print_summary_report()
    framework.export_results('indices_cointegration.xlsx')
    
    return framework


if __name__ == "__main__":
    """
    Exécution des exemples
    """
    print("🚀 EXEMPLES D'ANALYSE DE COINTEGRATION")
    print("=" * 60)
    
    # Choix de l'exemple à exécuter
    examples = {
        '1': ('Actions Technologiques', example_tech_stocks),
        '2': ('Or vs Bitcoin', example_gold_bitcoin),
        '3': ('Portefeuille Crypto', example_crypto_portfolio),
        '4': ('Indices Boursiers', example_indices)
    }
    print("\nExemples disponibles:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")

    # Exemple par défaut: Or vs Bitcoin (utilise la clé demo)
    print("\n" + "=" * 60)
    print("EXÉCUTION DE L'EXEMPLE PAR DÉFAUT: OR vs BITCOIN")
    print("=" * 60)
    
    try:
        framework = example_gold_bitcoin()
        print("\n✅ Exemple terminé avec succès!")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {str(e)}")
        print("Vérifiez votre connexion internet et votre clé API.")