import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, shapiro
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import plotly.express as px

# Function to process data


def preprocess_data(file1, file2):
    lob_data = pd.read_csv(file1, header=None)
    messages = pd.read_csv(file2, header=None)
    messages.columns = ['Time', 'Type',
                        'OrderID', 'Size', 'Price', 'Direction']

    # Renaming columns for lob_data using a helper function
    def rename_lob_columns(data):
        cols = ['ask_price', 'ask_size', 'bid_price', 'bid_size']
        new_column_names = []
        num_levels = len(data.columns) // len(cols)  # how many group of 4
        for i in range(num_levels):
            new_column_names.extend(f"{name}_{i+1}" for name in cols)

        # Rename the columns
        data.columns = new_column_names
        return data
    lob_data = rename_lob_columns(lob_data)
    lob_data['mid_price'] = (lob_data['ask_price_1'] +
                             lob_data['bid_price_1']) / 2
    lob_data['mid_price'] = lob_data['mid_price'] / \
        10000  # to get the price in dollars
    lob_data['time'] = messages['Time']
    lob_data['time'] = pd.to_datetime(lob_data['time'], unit='s')
    lob_data.set_index('time', inplace=True)
    lob_data = lob_data.resample('1min').ohlc()
    price_data = lob_data['mid_price']
    price_data = price_data.add_suffix('_mid_price')
    price_data['returns'] = price_data['close_mid_price'].pct_change()
    return price_data

# MLE Computation function


def max_likelihood_estimation(price_data):
    mean_returns_mle, std_returns_mle = norm.fit(
        price_data['returns'].dropna())

    return mean_returns_mle, std_returns_mle

# Closed Form function


def closed_form_solution(price_data):
    mean_returns_closed_form = price_data['returns'].mean()
    std_returns_closed_form = price_data['returns'].std()

    return mean_returns_closed_form, std_returns_closed_form

# plot Distribution of returns


def plot_returns_distribution(price_data, estimation_func):
    returns = price_data['returns'].dropna()
    mean_returns, std_returns = estimation_func(price_data)

    # Create the histogram of returns
    hist_data = go.Histogram(
        x=returns, histnorm='probability density', name='Distribution of returns')

    # Generate points for the Gaussian fit
    x = np.linspace(min(returns), max(returns), 1000)
    pdf = norm.pdf(x, mean_returns, std_returns)

    # Create the line trace for the Gaussian fit
    gaussian_fit = go.Scatter(
        x=x, y=pdf, mode='lines', name='Fitted Gaussian PDF', line=dict(color='red', width=2))

    # Combine traces in one plot
    fig = go.Figure(data=[hist_data, gaussian_fit])

    # Update layout for a cleaner look
    fig.update_layout(
        title='Returns Data and Fitted Gaussian Distribution - AKA Empirical Distribution',
        xaxis_title='Returns',
        yaxis_title='Probability Density',
        bargap=0.2,  # Space between bars
        template='plotly_white'  # Cleaner template
    )

    # Show plot
    return fig

# plot the returns trend to show stationarity


def plot_returns_trend(price_data):
    returns_time = px.line(price_data, x=price_data.index,
                           y='returns', title='Returns Trend')
    returns_time.update_xaxes(title_text='Time')
    returns_time.update_yaxes(title_text='Returns')
    return returns_time

# Display the statistics of the returns (Saphiro test & Excess Kurtosis)


def display_returns_statistics(returns):
    # compute excess kurtosis & Saphiro-Wilk test
    excess_kurtosis = returns.kurtosis()
    shapiro_test = shapiro(returns)
    st.write(f"Excess Kurtosis: {excess_kurtosis:.4f}")
    if shapiro_test.pvalue < 0.05:
        st.write("The data is not normally distributed")
    st.write(f"Shapiro-Wilk Test Pvalue: {shapiro_test.pvalue}")

# Direct sampling method


def direct_sampling(sample_size: 390, estimation_func, price_data):
    mean_returns, std_returns = estimation_func(price_data)
    # Generate sample directly from the gaussian distribution
    sample_gaussian_returns = np.random.normal(
        mean_returns, std_returns, sample_size)

    # Simulate Stock Price trajectory
    # Start from the last price in the lob dataset
    start_price = price_data['close_mid_price'].iloc[0]  # -1

    # Initialize the simulated price list
    sim_prices = [start_price]

    for r in sample_gaussian_returns:
        # calculate the new price based on the previous price
        new_price = sim_prices[-1] * (1 + r)
        # append to the list
        sim_prices.append(new_price)

    return sample_gaussian_returns, sim_prices

# Inverse Transform Sampling method


def inverse_transform_sampling(sample_size: 390, estimation_func, price_data):
    mean_returns, std_returns = estimation_func(price_data)
    # we generate the samples from the uniform distribution
    uniform_samples = np.random.uniform(0, 1, sample_size)

    # Apply the inverse CDF of the Gaussian distribution to the uniform samples
    sample_gaussian_returns_inv = norm.ppf(
        uniform_samples, mean_returns, std_returns)

    # Simulate Stock Price trajectory
    # Start from the last price in the lob dataset
    start_price = price_data['close_mid_price'].iloc[0]  # -1

    # Initialize the simulated price list
    sim_prices = [start_price]

    for r in sample_gaussian_returns_inv:
        # calculate the new price based on the previous price
        new_price = sim_prices[-1] * (1 + r)
        # append to the list
        sim_prices.append(new_price)

    return sample_gaussian_returns_inv, sim_prices


def inverse_transform_sampling_v2(sample_size: 390, estimation_func, price_data):
    mean_returns, std_returns = estimation_func(price_data)
    returns = price_data['returns'].dropna()

    data_sorted = np.sort(returns)
    probs = np.linspace(1./len(returns), 1, len(returns))  # computing cdf

    inverse_cdf_function = interp1d(
        probs, data_sorted, bounds_error=False, fill_value="extrapolate")

    # we generate the samples from the uniform distribution
    uniform_samples = np.random.uniform(0, 1, sample_size)

    # Sample data from the inverse cdf
    returns_sampled_inv_trans_dist = inverse_cdf_function(uniform_samples)

    start_price = price_data['close_mid_price'].iloc[0]

    sim_prices = [start_price]

    for r in returns_sampled_inv_trans_dist:
        # calculate the new price based on the previous price
        new_price = sim_prices[-1] * (1 + r)
        # append to the list
        sim_prices.append(new_price)

    return returns_sampled_inv_trans_dist, sim_prices

# Ploting Sample


def plot_sample_vs_original(price_data, sim_prices):
    fig = go.Figure()

    # Add the historical price trace
    fig.add_trace(go.Scatter(
        x=list(range(len(price_data['close_mid_price']))),
        y=price_data['close_mid_price'],
        mode='lines',
        name='Historical Closing Prices'
    ))

    # Add the simulated price trace
    fig.add_trace(go.Scatter(x=list(range(len(sim_prices))),
                  y=sim_prices, mode='lines', name='Simulated Stock Prices'))

    # Update the layout to add titles and labels
    fig.update_layout(
        title='Simulated Stock Price Trajectory Using Direct Sampling',
        xaxis_title='Time Steps',
        yaxis_title='Price',
        template='plotly_dark'  # Using a dark theme for better visibility
    )

    # Show the interactive plot
    return fig

# Plot Stack historgrams


def plot_stack_histograms_sample_vs_original(data, sample_gaussian_returns):
    fig = go.Figure()

    # Create the histogram of original returns
    hist_data = go.Histogram(
        x=data['returns'].dropna(), histnorm='probability density', name='Original Returns')

    # Create the histogram of sampled returns
    hist_sample = go.Histogram(
        x=sample_gaussian_returns, histnorm='probability density', name='Sampled Returns')

    # Combine traces in one plot
    fig = go.Figure(data=[hist_data, hist_sample])

    # Update layout for a cleaner look
    fig.update_layout(
        title='Returns Data and Sampled Returns Distribution',
        xaxis_title='Returns',
        yaxis_title='Probability Density',
        barmode='overlay',  # Overlay histograms
        template='plotly_white'  # Cleaner template
    )

    # Show plot
    return fig


# Plot CDF for Inverse Transform Sampling
def plot_cdf(data, returns_sampled_inv_trans_dist):  # sample_gaussian_returns
    fig = go.Figure()
    sorted_returns = np.sort(data['returns'].dropna())
    sorted_samples = np.sort(returns_sampled_inv_trans_dist)

    # Create traces for CDF of original and sampled data
    fig.add_trace(go.Scatter(x=sorted_returns, y=np.linspace(
        0, 1, len(sorted_returns)), mode='lines', name='Original CDF'))
    fig.add_trace(go.Scatter(x=sorted_samples, y=np.linspace(
        0, 1, len(sorted_samples)), mode='lines', name='Sampled CDF'))

    # Layout updates
    fig.update_layout(title='CDF Comparison', xaxis_title='Returns',
                      yaxis_title='CDF', template='plotly_white')
    return fig

# Plot the inverse CDF of the Gaussian distribution


def plot_inverse_cdf(mean_returns, std_returns):
    x = np.linspace(-5, 5, 1000)
    y = norm.cdf(x, mean_returns, std_returns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='CDF'))
    fig.update_layout(title='Inverse CDF of the Gaussian Distribution',
                      xaxis_title='Returns', yaxis_title='CDF', template='plotly_white')
    return fig

# Function for Streamlit app


def streamlit_app():
    st.markdown('<center><h1>Price Simulation App</h1></center>',
                unsafe_allow_html=True)

    # File upload
    file1 = "LOBSTER/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"
    file2 = "LOBSTER/AAPL_2012-06-21_34200000_57600000_message_10.csv"

    # Processing Data
    data = preprocess_data(file1, file2)

    # Sidebar
    st.sidebar.title("Parameters")
    sample_size = st.sidebar.number_input(
        "Enter the sample size:", min_value=10, max_value=390*30, value=390, step=10)
    st.sidebar.info("390min = 1 Trading day")
    # method = st.sidebar.selectbox(
    #     "Choose the parameter estimation method:", ("Close Form", "MLE"))
    sampling_method = st.sidebar.selectbox(
        "Choose the sampling method:", ("Direct Sampling", "Inverse Transform Sampling"))

    # if method == "MLE":
    #     estimation_func = max_likelihood_estimation
    # else:
    estimation_func = closed_form_solution

    if st.sidebar.button("Run Simulation"):
        if sampling_method == "Direct Sampling":
            sample_gaussian_returns, sim_prices = direct_sampling(
                sample_size, estimation_func, data)
        elif sampling_method == "Inverse Transform Sampling":
            sample_gaussian_returns, sim_prices = inverse_transform_sampling_v2(
                sample_size, estimation_func, data)

    # Main Page

    # Display the simulated prices vs the original prices (corressponding to sidebar parameters)
    if 'sim_prices' in locals():
        st.write("Simulated vs Original Prices")
        sample_vs_original_plot = plot_sample_vs_original(data, sim_prices)
        st.plotly_chart(sample_vs_original_plot, use_container_width=True)
    else:
        st.write("Simulated vs Original Prices")
        st.write("Please run the simulation first.")

    # Add mean and std of the returns (sampled and original)
    st.write("Mean and Standard Deviation of Returns")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original Returns:")
        mean_original = data['returns'].mean()
        std_original = data['returns'].std()
        st.write("Mean:", mean_original)
        st.write("Standard Deviation:", std_original)
        st.write("Excess Kurtosis:", data['returns'].kurtosis())
    with col2:
        if 'sample_gaussian_returns' in locals():
            st.write("Sampled Returns:")
            mean_sampled = np.mean(sample_gaussian_returns)
            std_sampled = np.std(sample_gaussian_returns)
            st.write("Mean:", mean_sampled)
            st.write("Standard Deviation:", std_sampled)
            st.write("Excess Kurtosis:", pd.Series(
                sample_gaussian_returns).kurtosis())

    # Display the stack histograms of the returns
    if 'sample_gaussian_returns' in locals():
        if sampling_method == "Inverse Transform Sampling":
            st.write("Returns Distribution Comparison and CDF")
            col1, col2 = st.columns(2)
            with col1:
                stack_histograms_plot = plot_stack_histograms_sample_vs_original(
                    data, sample_gaussian_returns)
                st.plotly_chart(stack_histograms_plot,
                                use_container_width=True)
            with col2:
                cdf_plot = plot_cdf(data, sample_gaussian_returns)
                st.plotly_chart(cdf_plot, use_container_width=True)

                # inverse_cdf_plot = plot_inverse_cdf(mean_sampled, std_sampled)
                # st.plotly_chart(inverse_cdf_plot, use_container_width=True)
        else:
            st.write("Returns Distribution Comparison")
            stack_histograms_plot = plot_stack_histograms_sample_vs_original(
                data, sample_gaussian_returns)
            st.plotly_chart(stack_histograms_plot, use_container_width=True)
    else:
        st.write("Returns Distribution Comparison")
        st.write("Please run the simulation first.")

    # Bottom Statistics
    col1, col2 = st.columns([3, 2])
    with col1:
        st.write(
            "Processed Data Preview (OHLC + Returns of Mid_price aggregating by min interval):", data
        )
    with col2:
        st.markdown("""
        Returns are calculated as follows:

        $$
        r = \\frac{{P1 - P0}}{{P0}}
        $$
        To find the new_price we are using the returns formula as follows:
        $$
        P1 =P0×(1+r)
        $$

        Where:

        - $r$ is the return
        - $P1$ is the New Price
        - $P0$ is the Previous Price
        """, unsafe_allow_html=True)
    # plot the mid_close_price
    mid_close_price = px.line(data, x=data.index,
                              y='close_mid_price', title='Mid Close Price')
    mid_close_price.update_xaxes(title_text='Time (min)')
    mid_close_price.update_yaxes(title_text='Mid_Close_Price')
    st.plotly_chart(mid_close_price, use_container_width=True)

    # Plot returns distribution and returns trend side by side
    st.write("Returns Distribution and Trend")
    col1, col2 = st.columns([3, 2])
    with col1:
        returns_distribution_plot = plot_returns_distribution(
            data, estimation_func)
        st.plotly_chart(returns_distribution_plot, use_container_width=True)
    with col2:
        returns_trend_plot = plot_returns_trend(data)
        st.plotly_chart(returns_trend_plot, use_container_width=True)

    # Display statistics
    st.write("Returns Statistics")

    display_returns_statistics(data['returns'].dropna())


# Monte Carlo Simulation
def monte_carlo_simulation(estimation_func, sampling_func, data, num_paths, sample_size=390, progress_bar=None, status_text=None):
    """Perform Monte Carlo simulation and return simulated price paths, with progress updates."""
    paths = []
    for i in range(num_paths):
        _, sim_prices = sampling_func(sample_size, estimation_func, data)
        paths.append(sim_prices)
        if progress_bar:
            progress_bar.progress((i + 1) / num_paths)
        if status_text:
            status_text.text(f"Running simulation {i + 1} of {num_paths}")
    return paths


def plot_monte_carlo(paths, title):
    """Plot Monte Carlo simulation paths."""
    fig = go.Figure()
    for path in paths:
        fig.add_trace(go.Scatter(x=list(range(len(path))),
                      y=path, mode='lines', showlegend=False))
    fig.update_layout(title=title, xaxis_title='Time Steps',
                      yaxis_title='Simulated Price', template='plotly_dark')
    return fig


def plot_monte_carlo_distribution(sim_returns, title):
    """Plot the distribution of monte carlo"""
    fig = go.Figure()

    for path in sim_returns:
        # Calculate returns from prices
        returns = (np.array(path[1:]) -
                   np.array(path[:-1])) / np.array(path[:-1])
        fig.add_trace(go.Histogram(x=returns, histnorm='probability density',
                                   showlegend=False, opacity=0.5, nbinsx=50))
    fig.update_layout(title=title, xaxis_title='Return',  # Return
                      yaxis_title='Probability Density', template='plotly_white')
    return fig

    # for path in sim_returns:
    #     fig.add_trace(go.Histogram(x=path, histnorm='probability density',
    #                                showlegend=False, opacity=0.5, nbinsx=50))
    # fig.update_layout(title=title, xaxis_title='Price',  # Price
    #                   yaxis_title='Probability Density', template='plotly_white')
    # return fig


def streamlit_app_page2(data):
    st.title("Monte Carlo Simulations and Distribution Analysis")
    num_paths = st.number_input(
        "Enter the number of paths:", min_value=1, max_value=100, value=10, step=1)

    if st.button("Run Simulations"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Starting simulations for Direct Sampling...")
        paths_direct = monte_carlo_simulation(
            closed_form_solution, direct_sampling, data, num_paths)
        fig_direct = plot_monte_carlo(
            paths_direct, "Monte Carlo Simulation using Direct Sampling")
        dist_direct = plot_monte_carlo_distribution(
            paths_direct, "Distribution of Returns (Direct Sampling)")
        progress_bar.progress(50)

        status_text.text(
            "Starting simulations for Inverse Transform Sampling...")
        paths_inverse = monte_carlo_simulation(
            closed_form_solution, inverse_transform_sampling_v2, data, num_paths)
        fig_inverse = plot_monte_carlo(
            paths_inverse, "Monte Carlo Simulation using Inverse Transform Sampling")
        dist_inverse = plot_monte_carlo_distribution(
            paths_inverse, "Distribution of Returns (Inverse Transform Sampling)")
        progress_bar.progress(100)

        # Clear status text and progress bar after completion
        status_text.text("Simulations completed.")

        # Plotting simulations and distributions
        st.subheader("Direct Sampling Simulation")
        st.plotly_chart(fig_direct, use_container_width=True)
        st.plotly_chart(dist_direct, use_container_width=True)

        st.subheader("Inverse Transform Sampling Simulation")
        st.plotly_chart(fig_inverse, use_container_width=True)
        st.plotly_chart(dist_inverse, use_container_width=True)

        # Add average price and standard deviation
        st.write("Average Price and Standard Deviation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Original Returns:")
            mean_original = data['returns'].mean()
            std_original = data['returns'].std()
            st.write("Mean:", mean_original)
            st.write("Standard Deviation:", std_original)
            st.write("Excess Kurtosis:", data['returns'].kurtosis())

        with col2:
            st.write("Direct Sampling:")
            results_direct = []
            for i, path in enumerate(paths_direct):
                returns = np.diff(path) / path[:-1]
                average_return = np.mean(returns)
                stdev_return = np.std(returns)
                excess_kurtosis = pd.Series(returns).kurtosis()
                results_direct.append(
                    [average_return, stdev_return, excess_kurtosis])
            df_direct = pd.DataFrame(results_direct, columns=[
                                     "Average Return", "Standard Deviation", "Excess Kurtosis"])
            st.table(df_direct)

        with col3:
            st.write("Inverse Transform Sampling:")
            results_inverse = []
            for i, path in enumerate(paths_inverse):
                returns = np.diff(path) / path[:-1]
                average_return = np.mean(returns)
                stdev_return = np.std(returns)
                excess_kurtosis = pd.Series(returns).kurtosis()
                results_inverse.append(
                    [average_return, stdev_return, excess_kurtosis])
            df_inverse = pd.DataFrame(results_inverse, columns=[
                                      "Average Return", "Standard Deviation", "Excess Kurtosis"])
            st.table(df_inverse)


def main_app():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page:", ["Main Page", "Monte Carlo Simulations"])

    if page == "Main Page":
        streamlit_app()
    else:
        file1 = "LOBSTER/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"
        file2 = "LOBSTER/AAPL_2012-06-21_34200000_57600000_message_10.csv"
        data = preprocess_data(file1, file2)
        streamlit_app_page2(data)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main_app()
