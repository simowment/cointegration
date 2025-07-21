# Framework d'Analyse de Cointegration
# 
# Ce module fournit un framework complet pour étudier la cointegration entre plusieurs symboles financiers 
# en utilisant les données de Twelve Data.

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
    """Framework pour l'analyse de cointegration de plusieurs symboles financiers"""
    
    def __init__(self, api_key: str):
        self.td_client = TDClient(apikey=api_key)
        self.data = {}
        self.prices_df = None
        self.returns_df = None
        self.cointegration_results = {}
        
    def download_data(self, symbols: List[str], interval: str = '1day', 
                     outputsize: int = 5000, start_date: str = None) -> Dict:
        """
        Télécharge les données pour une liste de symboles depuis Twelve Data
        
        Args:
            symbols: Liste des symboles à télécharger
            interval: Intervalle de temps (1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week, 1month)
            outputsize: Nombre de points de données à récupérer
            start_date: Date de début (format YYYY-MM-DD)
        
        Returns:
            Dictionnaire contenant les données pour chaque symbole
        """
        print(f"Téléchargement des données pour {len(symbols)} symboles...")
        
        for symbol in symbols:
            try:
                ts = self.td_client.time_series(
                    symbol=symbol,
                    interval=interval,
                    outputsize=outputsize,
                    start_date=start_date
                )
                
                df = ts.as_pandas()
                # La bibliothèque retourne déjà les données avec un index datetime trié.
                
                self.data[symbol] = df
                print(f"✓ {symbol}: {len(df)} points de données")
                
            except Exception as e:
                print(f"✗ Erreur lors du téléchargement de {symbol}: {str(e)}")
        
        return self.data
    
    def prepare_price_matrix(self, price_column: str = 'close') -> pd.DataFrame:
        """
        Prépare une matrice de prix alignée pour tous les symboles
        
        Args:
            price_column: Colonne de prix à utiliser ('open', 'high', 'low', 'close')
        
        Returns:
            DataFrame avec les prix alignés
        """
        if not self.data:
            raise ValueError("Aucune donnée disponible. Utilisez download_data() d'abord.")
        
        price_data = {}
        for symbol, df in self.data.items():
            if price_column in df.columns:
                price_data[symbol] = df[price_column]
        
        self.prices_df = pd.DataFrame(price_data)
        self.prices_df.dropna(inplace=True)
        
        print(f"Matrice de prix créée: {self.prices_df.shape[0]} observations, {self.prices_df.shape[1]} symboles")
        return self.prices_df
    
    def calculate_returns(self, method: str = 'log') -> pd.DataFrame:
        """
        Calcule les rendements pour tous les symboles
        
        Args:
            method: 'log' pour rendements logarithmiques, 'simple' pour rendements simples
        
        Returns:
            DataFrame des rendements
        """
        if self.prices_df is None:
            raise ValueError("Préparez d'abord la matrice de prix avec prepare_price_matrix()")
        
        if method == 'log':
            self.returns_df = np.log(self.prices_df / self.prices_df.shift(1))
        else:
            self.returns_df = self.prices_df.pct_change()
        
        self.returns_df.dropna(inplace=True)
        return self.returns_df

    def test_stationarity(self, series: pd.Series, name: str = "") -> Dict:
        """
        Test de stationnarité avec Augmented Dickey-Fuller
        
        Args:
            series: Série temporelle à tester
            name: Nom de la série pour l'affichage
        
        Returns:
            Dictionnaire avec les résultats du test
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
        Teste la stationnarité de tous les prix et rendements
        
        Returns:
            DataFrame avec les résultats des tests
        """
        results = []
        
        # Test des prix (niveaux)
        if self.prices_df is not None:
            for col in self.prices_df.columns:
                result = self.test_stationarity(self.prices_df[col], f"{col}_price")
                result['type'] = 'Price'
                results.append(result)
        
        # Test des rendements
        if self.returns_df is not None:
            for col in self.returns_df.columns:
                result = self.test_stationarity(self.returns_df[col], f"{col}_return")
                result['type'] = 'Return'
                results.append(result)
        
        return pd.DataFrame(results)
        
    def engle_granger_test(self, y: pd.Series, x: pd.Series, 
                            symbol_y: str, symbol_x: str) -> Dict:
        """
        Test de cointegration d'Engle-Granger entre deux séries
        
        Args:
            y: Série dépendante
            x: Série indépendante
            symbol_y: Nom du symbole y
            symbol_x: Nom du symbole x
        
        Returns:
            Dictionnaire avec les résultats du test
        """
        # Test de cointegration
        coint_stat, p_value, critical_values = coint(y, x)
        
        # Régression pour obtenir les résidus
        X = x.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y.values)
        residuals = y.values - reg.predict(X)
        
        # Test de stationnarité des résidus
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
        Teste la cointegration pour toutes les paires de symboles (évite les doublons)
        
        Returns:
            DataFrame avec les résultats de cointegration
        """
        if self.prices_df is None:
            raise ValueError("Préparez d'abord la matrice de prix")
        
        results = []
        symbols = self.prices_df.columns.tolist()
        n_pairs = len(symbols) * (len(symbols) - 1) // 2
        
        print(f"Test de cointegration pour {len(symbols)} symboles ({n_pairs} paires)...")
        
        for i, symbol_y in enumerate(symbols):
            for j, symbol_x in enumerate(symbols):
                # Éviter l'auto-cointegration et les doublons
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
                    print(f"Erreur pour la paire {symbol_y}-{symbol_x}: {str(e)}")
        
        # Stockage des résultats
        self.cointegration_results['pairwise'] = results
        
        # Création du DataFrame de résultats
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
        Test de cointegration de Johansen pour analyse multivariée
        
        Args:
            max_lags: Nombre maximum de lags à considérer
        
        Returns:
            Dictionnaire avec les résultats du test de Johansen
        """
        if self.prices_df is None:
            raise ValueError("Préparez d'abord la matrice de prix")
        
        # Test de Johansen
        result = coint_johansen(self.prices_df.values, det_order=0, k_ar_diff=max_lags)
        
        # Extraction des résultats
        johansen_results = {
            'trace_stats': result.lr1,
            'max_eigen_stats': result.lr2,
            'critical_values_trace': result.cvt,
            'critical_values_max_eigen': result.cvm,
            'eigenvalues': result.eig,
            'eigenvectors': result.evec,
            'symbols': self.prices_df.columns.tolist()
        }
        
        # Détermination du nombre de relations de cointegration
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
        Graphique des prix pour tous les symboles
        
        Args:
            figsize: Taille de la figure
            normalize: Si True, normalise les prix à 100 au début
        """
        if self.prices_df is None:
            raise ValueError("Aucune donnée de prix disponible")
        
        plt.figure(figsize=figsize)
        
        if normalize:
            # Normalisation des prix (base 100)
            normalized_prices = (self.prices_df / self.prices_df.iloc[0]) * 100
            for col in normalized_prices.columns:
                plt.plot(normalized_prices.index, normalized_prices[col], label=col, linewidth=2)
            plt.ylabel('Prix Normalisé (Base 100)')
            plt.title('Évolution des Prix Normalisés')
        else:
            for col in self.prices_df.columns:
                plt.plot(self.prices_df.index, self.prices_df[col], label=col, linewidth=2)
            plt.ylabel('Prix')
            plt.title('Évolution des Prix')
        
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data_type: str = 'prices', figsize: Tuple[int, int] = (10, 8)):
        """
        Matrice de corrélation
        
        Args:
            data_type: 'prices' ou 'returns'
            figsize: Taille de la figure
        """
        if data_type == 'prices' and self.prices_df is not None:
            data = self.prices_df
            title = 'Matrice de Corrélation - Prix'
        elif data_type == 'returns' and self.returns_df is not None:
            data = self.returns_df
            title = 'Matrice de Corrélation - Rendements'
        else:
            raise ValueError(f"Données {data_type} non disponibles")
        
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
        Heatmap des p-values de cointegration
        
        Args:
            figsize: Taille de la figure
        """
        if 'pairwise' not in self.cointegration_results:
            raise ValueError("Effectuez d'abord le test de cointegration pairwise")
        
        # Création de la matrice des p-values
        symbols = self.prices_df.columns.tolist()
        n_symbols = len(symbols)
        p_value_matrix = np.ones((n_symbols, n_symbols))
        
        for result in self.cointegration_results['pairwise']:
            pair = result['pair'].split(' ~ ')
            y_idx = symbols.index(pair[0])
            x_idx = symbols.index(pair[1])
            p_value_matrix[y_idx, x_idx] = result['p_value']
        
        # Création du heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(p_value_matrix,
                   xticklabels=symbols,
                   yticklabels=symbols,
                   annot=True,
                   cmap='RdYlGn_r',
                   center=0.05,
                   fmt='.3f',
                   cbar_kws={'label': 'P-value'})
        
        plt.title('Heatmap des P-values de Cointegration\n(Vert = Cointegré, Rouge = Non Cointegré)')
        plt.xlabel('Symbole X (Variable Indépendante)')
        plt.ylabel('Symbole Y (Variable Dépendante)')
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, symbol_y: str, symbol_x: str, figsize: Tuple[int, int] = (15, 10)):
        """
        Graphique des résidus pour une paire cointegree
        
        Args:
            symbol_y: Symbole dépendant
            symbol_x: Symbole indépendant
            figsize: Taille de la figure
        """
        if 'pairwise' not in self.cointegration_results:
            raise ValueError("Effectuez d'abord le test de cointegration pairwise")
        
        # Recherche des résultats pour cette paire
        pair_name = f"{symbol_y} ~ {symbol_x}"
        result = None
        
        for r in self.cointegration_results['pairwise']:
            if r['pair'] == pair_name:
                result = r
                break
        
        if result is None:
            raise ValueError(f"Paire {pair_name} non trouvée dans les résultats")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Scatter plot des prix
        axes[0, 0].scatter(self.prices_df[symbol_x], self.prices_df[symbol_y], alpha=0.6)
        axes[0, 0].set_xlabel(f'Prix {symbol_x}')
        axes[0, 0].set_ylabel(f'Prix {symbol_y}')
        axes[0, 0].set_title(f'Relation {symbol_y} vs {symbol_x}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Série temporelle des résidus
        axes[0, 1].plot(result['residuals'].index, result['residuals'], linewidth=1)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Résidus')
        axes[0, 1].set_title('Série Temporelle des Résidus')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Histogramme des résidus
        axes[1, 0].hist(result['residuals'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Résidus')
        axes[1, 0].set_ylabel('Fréquence')
        axes[1, 0].set_title('Distribution des Résidus')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q plot des résidus
        from scipy import stats
        stats.probplot(result['residuals'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot des Résidus')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Analyse des Résidus - {pair_name}\nP-value: {result["p_value"]:.4f}, Cointegré: {result["is_cointegrated"]}')
        plt.tight_layout()
        plt.show()

    def generate_summary_report(self) -> Dict:
        """
        Génère un rapport de synthèse complet
        
        Returns:
            Dictionnaire contenant le rapport de synthèse
        """
        report = {
            'data_summary': {},
            'stationarity_summary': {},
            'cointegration_summary': {},
            'recommendations': []
        }
        
        # Résumé des données
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
        
        # Test de stationnarité
        stationarity_results = self.test_all_stationarity()
        if not stationarity_results.empty:
            report['stationarity_summary'] = {
                'prices_stationary': stationarity_results[stationarity_results['type'] == 'Price']['is_stationary'].sum(),
                'returns_stationary': stationarity_results[stationarity_results['type'] == 'Return']['is_stationary'].sum(),
                'total_price_tests': len(stationarity_results[stationarity_results['type'] == 'Price']),
                'total_return_tests': len(stationarity_results[stationarity_results['type'] == 'Return'])
            }
        
        # Résumé de cointegration
        if 'pairwise' in self.cointegration_results:
            pairwise_results = self.cointegration_results['pairwise']
            cointegrated_pairs = [r for r in pairwise_results if r['is_cointegrated']]
            
            report['cointegration_summary'] = {
                'total_pairs_tested': len(pairwise_results),
                'cointegrated_pairs': len(cointegrated_pairs),
                'cointegration_rate': len(cointegrated_pairs) / len(pairwise_results) if pairwise_results else 0,
                'best_pairs': sorted(cointegrated_pairs, key=lambda x: x['p_value'])[:5]
            }
        
        # Recommandations
        if 'cointegration_summary' in report and report['cointegration_summary']['cointegrated_pairs'] > 0:
            report['recommendations'].append("Des relations de cointegration ont été détectées - considérez des stratégies de pairs trading")
        
        if 'stationarity_summary' in report:
            if report['stationarity_summary']['prices_stationary'] > 0:
                report['recommendations'].append("Certains prix sont stationnaires - vérifiez la qualité des données")
            if report['stationarity_summary']['returns_stationary'] < report['stationarity_summary']['total_return_tests']:
                report['recommendations'].append("Certains rendements ne sont pas stationnaires - considérez des transformations supplémentaires")
        
        return report
    
    
    def export_results(self, filename: str = 'cointegration_results.xlsx'):
        """
        Exporte tous les résultats vers un fichier Excel
        
        Args:
            filename: Nom du fichier de sortie
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Prix
            if self.prices_df is not None:
                self.prices_df.to_excel(writer, sheet_name='Prix')
            
            # Rendements
            if self.returns_df is not None:
                self.returns_df.to_excel(writer, sheet_name='Rendements')
            
            # Tests de stationnarité
            stationarity_results = self.test_all_stationarity()
            if not stationarity_results.empty:
                stationarity_results.to_excel(writer, sheet_name='Stationnarité', index=False)
            
            # Résultats de cointegration pairwise
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
            
            # Résultats de Johansen
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
        
        print(f"Résultats exportés vers {filename}")