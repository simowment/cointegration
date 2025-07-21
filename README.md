vibe coded cointegration framework
# Cointegration Analysis Framework
This project provides a comprehensive Python framework for analyzing cointegration relationships between multiple financial symbols, utilizing data from Twelve Data. It includes functionalities for data acquisition, statistical tests, and various visualizations to help identify long-term equilibrium relationships between asset prices.

## Features

*   **Data Download:** Seamlessly download historical price data for a list of financial symbols from the Twelve Data API.
*   **Data Preparation:** Prepare aligned price matrices and calculate returns (log or simple).
*   **Stationarity Tests:** Perform Augmented Dickey-Fuller (ADF) tests on individual price series and their returns.
*   **Engle-Granger Cointegration Test:** Conduct pairwise cointegration tests to identify stationary linear combinations.
*   **Johansen Cointegration Test:** Perform multivariate cointegration analysis to determine the number of cointegrating relationships within a portfolio of assets.
*   **Visualizations:** Generate plots for price evolution, correlation matrices (prices and returns), cointegration p-value heatmaps, and residual analysis.
*   **Summary Reports:** Generate and export comprehensive analysis reports to Excel.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/victorrr.git
    cd victorrr
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # On Windows
    source venv/bin/activate # On macOS/Linux
    ```
3.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn twelvedata statsmodels scikit-learn openpyxl
    ```

## Usage

The primary entry point for examples is `testing/cointegration_examples.py`.

1.  **Set your Twelve Data API Key:**    
    *   **Windows:**
        ```cmd
        set TD_API_KEY="YOUR_API_KEY_HERE"
        ```
    *   **macOS/Linux:**
        ```bash
        export TD_API_KEY="YOUR_API_KEY_HERE"
        ```

2.  **Run the examples:**
    Navigate to the project root and execute the `cointegration_examples.py` script:
    ```bash
    python testing/cointegration_examples.py
    ```
    The script will present a menu of available examples (Technology Stocks, Gold vs Bitcoin, Crypto Portfolio, Stock Indices) and will run a default example.

## Project Structure

*   `testing/cointegration_framework.py`: Contains the core `CointegrationFramework` class with all the statistical and plotting logic.
*   `testing/cointegration_examples.py`: Demonstrates how to use the `CointegrationFramework` with various financial assets.


