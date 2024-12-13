import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Constants for option pricing
NUM_PATHS = 10000
NUM_STEPS = 50
TIME_TO_MATURITY = 1.0
RISK_FREE_RATE = 0.02
VOLATILITY = 0.4

# Step 1: Simulate Random Paths for Asset Prices
def simulate_paths(S0, num_paths, num_steps, T, r, sigma):
    dt = T / num_steps
    paths = np.zeros((num_steps + 1, num_paths))
    paths[0] = S0

    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_paths)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    return paths

# Simulate paths
S0 = 100  # Initial stock price
paths = simulate_paths(S0, NUM_PATHS, NUM_STEPS, TIME_TO_MATURITY, RISK_FREE_RATE, VOLATILITY)

# Step 2: Least Squares Monte Carlo (LSM) Algorithm
def least_squares_monte_carlo(paths, strike_price, r, dt):
    num_steps, num_paths = paths.shape
    cashflows = np.maximum(strike_price - paths[-1], 0)  # Payoff at maturity

    for t in range(num_steps - 1, 0, -1):
        itm = paths[t] < strike_price  # In-the-money paths
        X = paths[t, itm]
        Y = np.exp(-r * dt) * cashflows[itm]

        # Fit regression to continuation values
        if len(X) > 0:
            regression = np.polyfit(X, Y, deg=2)
            continuation_values = np.polyval(regression, X)

            exercise_values = strike_price - X
            cashflows[itm] = np.where(exercise_values > continuation_values, exercise_values, cashflows[itm])

    option_price = np.mean(np.exp(-r * dt) * cashflows)
    return option_price

# Calculate option price
strike_price = 100
option_price = least_squares_monte_carlo(paths, strike_price, RISK_FREE_RATE, TIME_TO_MATURITY / NUM_STEPS)


def fetch_historical_spy_options():
    """
    Fetch historical SPY options data with disk caching
    """
    cache_file = 'spy_options_history.pkl'
    
    # Try to load from cache first
    try:
        print("Loading cached options data...")
        all_options_df = pd.read_pickle(cache_file)
        print(f"Loaded {len(all_options_df)} options from cache")
        return all_options_df
    except FileNotFoundError:
        print("No cached data found. Generating new dataset...")
    
    # Generate new dataset
    spy = yf.Ticker("SPY")
    hist_prices = spy.history(period="5y")
    
    # Calculate rolling volatilities of different windows
    hist_prices['vol_30d'] = hist_prices['Close'].pct_change().rolling(30).std() * np.sqrt(252)
    hist_prices['vol_60d'] = hist_prices['Close'].pct_change().rolling(60).std() * np.sqrt(252)
    hist_prices['vol_90d'] = hist_prices['Close'].pct_change().rolling(90).std() * np.sqrt(252)
    
    # Rest of the function remains the same, but replace sigma calculation with:
    sigma = max(
        hist_prices['vol_30d'].iloc[-1],
        hist_prices['vol_60d'].iloc[-1],
        hist_prices['vol_90d'].iloc[-1],
        0.15  # minimum volatility floor
    )
    
    all_options = []
    print("Generating historical options data...")
    
    for date, row in hist_prices.iterrows():
        try:
            current_price = row['Close']
            strikes = np.linspace(current_price * 0.7, current_price * 1.3, 20)
            
            for days_to_expiry in [30, 60, 90, 180, 360]:
                expiry_date = date + pd.Timedelta(days=days_to_expiry)
                T = days_to_expiry / 365.0
                
                for strike in strikes:
                    hist_vol = hist_prices['Close'].pct_change().rolling(30).std() * np.sqrt(252)
                    sigma = hist_vol.iloc[-1] if len(hist_vol) > 0 else 0.2
                    
                    # Calculate for both calls and puts
                    for option_type in ['call', 'put']:
                        delta, gamma, theta, vega = calculate_option_greeks(
                            current_price, strike, T, RISK_FREE_RATE, sigma, option_type
                        )
                        
                        # Calculate theoretical price
                        if option_type == 'call':
                            intrinsic = max(0, current_price - strike)
                        else:
                            intrinsic = max(0, strike - current_price)
                        
                        time_value = sigma * np.sqrt(T) * current_price * 0.4  # Simplified time value
                        last_price = intrinsic + time_value
                        
                        option_data = {
                            'date': date,
                            'expirationDate': expiry_date,
                            'strike': strike,
                            'lastPrice': last_price,
                            'impliedVolatility': sigma,
                            'daysToExpiry': days_to_expiry,
                            'currentPrice': current_price,
                            'optionType': option_type,
                            'delta': delta,
                            'gamma': gamma,
                            'theta': theta,
                            'vega': vega,
                            'moneyness': current_price / strike,
                            'volume': 100,
                            'openInterest': 100
                        }
                        
                        all_options.append(option_data)
            
            if len(all_options) % 1000 == 0:
                print(f"Processed data points: {len(all_options)}")
                
        except Exception as e:
            print(f"Error processing date {date}: {e}")
    
    all_options_df = pd.DataFrame(all_options)
    
    # Save to cache
    print(f"\nSaving {len(all_options_df)} options to cache...")
    all_options_df.to_pickle(cache_file)
    
    return all_options_df

def calculate_option_greeks(S, K, T, r, sigma, option_type):
    """
    Calculate option Greeks using Black-Scholes formulas
    """
    if T <= 0:
        return 0, 0, 0, 0
    
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate Greeks
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * np.sqrt(T) * norm.pdf(d1)
    
    if option_type == 'put':
        delta = -delta
        gamma = -gamma
        theta = -theta
        vega = -vega
    
    return delta, gamma, theta, vega

def calculate_time_value(last_price, strike, current_price, option_type):
    """Calculate option's time value"""
    if option_type == 'call':
        intrinsic = max(0, current_price - strike)
    else:  # put
        intrinsic = max(0, strike - current_price)
    return max(last_price - intrinsic, 0.01)  # Minimum 0.01 to avoid zero values

def process_spy_option_data(options_data):
    """
    Enhanced data processing with better feature engineering
    """
    data = options_data[[
        "strike", "impliedVolatility", "expirationDate", 
        "lastPrice", "volume", "openInterest", "moneyness", 
        "daysToExpiry", "optionType"
    ]].copy()
    
    # Calculate current price (S) for each option
    current_price = options_data['strike'] * options_data['moneyness']
    
    # Calculate Greeks with sign adjustment for puts
    greeks = [calculate_option_greeks(
        S=current_price.iloc[i],
        K=data['strike'].iloc[i],
        T=max(data['daysToExpiry'].iloc[i] / 365, 0.001),
        r=RISK_FREE_RATE,
        sigma=data['impliedVolatility'].iloc[i],
        option_type=data['optionType'].iloc[i]
    ) for i in range(len(data))]
    
    # Add Greeks to dataframe
    data['delta'], data['gamma'], data['theta'], data['vega'] = zip(*greeks)
    
    # Add sophisticated features
    data['timeValue'] = data.apply(
        lambda x: calculate_time_value(x['lastPrice'], x['strike'], 
                                     current_price.iloc[x.name], x['optionType']),
        axis=1
    )
    data['volatilityTimeValue'] = data['impliedVolatility'] * np.sqrt(data['daysToExpiry'] / 365)
    
    # Add more features
    data['moneyness_squared'] = data['moneyness'] ** 2
    data['vol_time_interaction'] = data['impliedVolatility'] * data['daysToExpiry']
    data['delta_gamma_ratio'] = data['delta'] / (data['gamma'] + 1e-6)
    
    data = data.rename(columns={
        "strike": "Strike",
        "impliedVolatility": "Volatility",
        "daysToExpiry": "Time",
        "lastPrice": "BidPrice"
    })
    
    # Normalize time feature
    data["Time"] = data["Time"] / 365.0
    
    # Enhanced filtering
    data = data[
        (data['volume'] > 10) &
        (data['openInterest'] > 10) &
        (data['Volatility'] > 0.001) &
        (data['Volatility'] < 5) &
        (data['Time'] > 0) &
        (data['delta'].notna()) &
        (data['gamma'].notna()) &
        (data['theta'].notna()) &
        (data['vega'].notna()) &
        (abs(data['delta']) <= 1) &  # Reasonable delta values
        (data['gamma'] >= 0)  # Gamma should be positive
    ]
    
    print("\nProcessed data statistics:")
    print(data.describe())
    
    return data

def plot_prediction_analysis(y_test, rf_preds, rf_model, X_test):
    """
    Create comprehensive visualization of model performance with dark theme
    """
    # Set style
    plt.style.use('dark_background')
    
    # Create figure with transparent background
    fig = plt.figure(figsize=(15, 10), facecolor='none')
    fig.patch.set_alpha(0.0)
    
    # Plot 1: Predictions vs Actual
    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(y_test, rf_preds, alpha=0.5, color='#00ff00')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             '--', color='#ff0000', lw=2)
    ax1.set_xlabel('Actual Price')
    ax1.set_ylabel('Predicted Price')
    ax1.set_title('Random Forest: Predicted vs Actual Prices')
    ax1.patch.set_alpha(0.0)
    
    # Plot 2: Prediction Error Distribution
    ax2 = plt.subplot(2, 2, 2)
    sns.histplot(rf_preds - y_test, kde=True, color='#00ff00', ax=ax2)
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Prediction Errors')
    ax2.patch.set_alpha(0.0)
    
    # Plot 3: Feature Importance
    ax3 = plt.subplot(2, 2, 3)
    features = [
        'Strike', 'Volatility', 'Time', 'moneyness', 
        'volatilityTimeValue', 'timeValue', 'delta', 
        'gamma', 'theta', 'vega'
    ]
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    sns.barplot(
        x='importance', 
        y='feature', 
        data=importance_df,
        hue='feature',  # Add hue parameter
        legend=False,   # Remove legend
        palette='viridis', 
        ax=ax3
    )
    ax3.set_title('Feature Importance')
    ax3.patch.set_alpha(0.0)
    
    # Plot 4: Residuals
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(rf_preds, rf_preds - y_test, alpha=0.5, color='#00ff00')
    ax4.axhline(y=0, color='#ff0000', linestyle='--')
    ax4.set_xlabel('Predicted Price')
    ax4.set_ylabel('Residual')
    ax4.set_title('Residual Plot')
    ax4.patch.set_alpha(0.0)
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Print metrics with highlighted colors
    print("\n\033[92mDetailed Model Performance Metrics:\033[0m")
    mape = np.mean(np.abs((y_test - rf_preds) / y_test)) * 100
    r2 = r2_score(y_test, rf_preds)
    rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    
    print(f"\033[96mMean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Root Mean Square Error (RMSE): ${rmse:.2f}\033[0m")
    
    # Save with transparent background
    plt.savefig('option_analysis.png', 
                dpi=300, 
                bbox_inches='tight', 
                transparent=True)
    plt.show()

def enhanced_rf_model(X_train, y_train, X_test, y_test):
    """
    Enhanced Random Forest model with hyperparameter tuning and feature importance analysis
    """
    # Create and train Random Forest with optimized parameters
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    rf.fit(X_train, y_train)
    
    # Make predictions
    rf_preds = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_preds)
    
    # Update feature importance with all features including Greeks
    feature_importance = pd.DataFrame({
        'feature': [
            'Strike', 'Volatility', 'Time', 'Moneyness', 
            'VolatilityTimeValue', 'TimeValue', 'Delta', 
            'Gamma', 'Theta', 'Vega'
        ],
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nRandom Forest Performance:")
    print(f"MSE: {rf_mse:.2f}")
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot the analysis
    plot_prediction_analysis(y_test, rf_preds, rf, X_test)
    
    return rf, rf_mse, rf_preds

def hybrid_lsm_rf(paths, strike_price, r, dt, rf_model):
    """
    Hybrid LSM-RF approach for American option pricing
    """
    num_steps, num_paths = paths.shape
    cashflows = np.maximum(strike_price - paths[-1], 0)
    
    for t in range(num_steps - 1, 0, -1):
        itm = paths[t] < strike_price
        if np.sum(itm) > 0:
            # Calculate all features for prediction
            S = paths[t, itm]
            T = t * dt
            sigma = VOLATILITY
            n_samples = np.sum(itm)
            
            # Calculate Greeks
            delta, gamma, theta, vega = zip(*[calculate_option_greeks(
                s, strike_price, T, r, sigma, option_type='put') for s in S])
            
            # Calculate additional features
            moneyness = S / strike_price
            volatility_time = np.full(n_samples, sigma * np.sqrt(T))
            time_value = np.maximum(strike_price - S, 0)
            
            X_current = np.column_stack([
                np.full(n_samples, strike_price),    # Strike
                np.full(n_samples, sigma),           # Volatility
                np.full(n_samples, T),               # Time
                moneyness,                           # Moneyness
                volatility_time,                     # VolatilityTimeValue
                time_value,                          # TimeValue
                delta,                               # Delta
                gamma,                               # Gamma
                theta,                               # Theta
                vega                                 # Vega
            ])
            
            continuation_values = rf_model.predict(X_current)
            exercise_values = strike_price - S
            cashflows[itm] = np.where(
                exercise_values > continuation_values,
                exercise_values,
                np.exp(-r * dt) * cashflows[itm]
            )
    
    option_price = np.mean(np.exp(-r * dt) * cashflows)
    return option_price

def display_pricing_comparison(current_price, traditional_price, hybrid_price):
    """
    Display pricing comparison with basic formatting
    """
    print("\n" + "="*40)
    print("        OPTIONS PRICING COMPARISON        ")
    print("="*40)
    
    print(f"\nSPY Price: ${current_price:.2f}")
    print(f"Traditional LSM Price: ${traditional_price:.2f}")
    print(f"Hybrid LSM-RF Price: ${hybrid_price:.2f}")
    print(f"Price Difference: ${abs(traditional_price - hybrid_price):.2f}")
    
    print("\n" + "="*40)

# Main execution
try:
    print("Starting Enhanced SPY Options Analysis with Greeks...")
    spy_options = fetch_historical_spy_options()
    data = process_spy_option_data(spy_options)
    
    if data.empty:
        raise ValueError("No valid data available after filtering")

    # Use enhanced feature set with Greeks
    X = data[[
        "Strike", 
        "Volatility", 
        "Time",
        "moneyness",
        "volatilityTimeValue",
        "timeValue",
        "delta",
        "gamma",
        "theta",
        "vega"
    ]].values
    y = data["BidPrice"].values

    # Train model and show performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model, rf_mse, rf_preds = enhanced_rf_model(X_train, y_train, X_test, y_test)
    
    # Get current SPY price
    spy = yf.Ticker("SPY")
    current_price = spy.history(period="1d")['Close'].iloc[-1]
    strike_price = current_price
    
    # Calculate paths and prices
    paths = simulate_paths(current_price, NUM_PATHS, NUM_STEPS, TIME_TO_MATURITY, RISK_FREE_RATE, VOLATILITY)
    
    # Calculate both pricing methods
    traditional_price = least_squares_monte_carlo(paths, strike_price, RISK_FREE_RATE, TIME_TO_MATURITY / NUM_STEPS)
    hybrid_price = hybrid_lsm_rf(paths, strike_price, RISK_FREE_RATE, TIME_TO_MATURITY / NUM_STEPS, rf_model)
    
    # Display the comparison
    display_pricing_comparison(current_price, traditional_price, hybrid_price)

except Exception as e:
    print(f"Error in analysis: {e}")
    import traceback
    print(traceback.format_exc())
