## Algorithmic Trading: Simulating Stock Prices (Direct vs. Inverse Transform)

This repository explores the simulation of stock prices using two sampling techniques: Direct Sampling and Inverse Transform Sampling. It investigates the limitations of these methods in capturing the complexities of real-world financial data.

**Key Findings:**

* Direct Sampling and Inverse Transform Sampling may not effectively capture the full characteristics of stock price data due to several limitations:
    * **Oversimplification of Underlying Distribution:** These techniques often rely on assuming a specific probability distribution for returns, such as a normal distribution (Gaussian). However, real-world stock returns often exhibit non-normality with features like skewness (asymmetry) and kurtosis (fat tails). This mismatch can lead to simulated data that doesn't accurately reflect the true behavior of the market.
    * **Neglecting Dependence Structure:** Direct Sampling and Inverse Transform Sampling typically treat price changes as independent and identically distributed (i.i.d.), meaning each price movement is considered independent of the previous ones. However, stock prices exhibit serial dependence, where past returns can influence future returns. These techniques fail to capture this crucial aspect.
    * **Limited Ability to Reproduce Extreme Events:** These methods might struggle to generate the extreme price movements (volatility spikes, crashes) that can occur in financial markets. This can lead to underestimating risk in algorithmic trading strategies relying on these simulations.

* While these techniques can be helpful for introductory explorations, they may not be suitable for developing robust algorithmic trading strategies that require a more realistic representation of market dynamics.

**Project Components:**

* `notebooks/simulating_stock_prices.ipynb` (Private): Jupyter Notebook demonstrating the sampling techniques.
* `streamlit_apps/interactive_sampling.py`: Streamlit app allowing users to experiment with different sample sizes and techniques.
* `streamlit_apps/monte_carlo_simulation.py`: Streamlit app showcasing Monte Carlo simulations using both sampling methods.

**Data Source:**

* The project utilizes data from the LOBSTER Dataset, specifically minute-aggregated best close mid-price returns.

**Running the Streamlit Applications:**

1. **Prerequisites:** Ensure you have Python 3.9, Streamlit (`pip install streamlit`), and any other necessary libraries installed (list dependencies in a separate file, e.g., `requirements.txt`).
2. **Clone the Repository:** Use `git clone https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/GitHub-URL-find-use-example` to clone this repository.
3. **Navigate to the Directory:** Run `cd Algorithmic-Trading-Simulations` (or your repository name) in your terminal.
4. **Run the First Streamlit App:** Execute `streamlit run streamlit_apps/interactive_sampling.py` to launch the interactive sampling app.
5. **Run the Second Streamlit App:** Execute `streamlit run streamlit_apps/monte_carlo_simulation.py` to launch the Monte Carlo simulation app.

**Disclaimer:**

The Jupyter Notebook is intentionally private to protect the course materials and my work. The provided Streamlit applications serve as a public demonstration of your learning and exploration.
