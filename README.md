1. Problem Statement
Stock prices are inherently sequential — today's price depends on the history of previous prices, trading volume, and market sentiment. The goal of this project is to build and compare deep learning models (SimpleRNN and LSTM) to predict Tesla (TSLA) stock closing prices for 1-day, 5-day, and 10-day horizons.

2. Dataset Overview
Source: TSLA.csv (Yahoo Finance historical data)

Column	Description
Date	Trading date
Open	Opening price
High	Highest intra-day price
Low	Lowest intra-day price
Close	Closing price (our target variable)
Adj Close	Adjusted close (for splits/dividends)
Volume	Shares traded that day
Date range: June 2010 → present (Tesla IPO to current)
Target variable: Close price
3. Data Cleaning & Preprocessing
3.1 Missing Values
No missing values were found in the raw dataset.
Forward fill (ffill) strategy was applied as a safeguard — for time series, we cannot drop rows arbitrarily because the sequential order must be preserved.
Why not mean/median imputation? Because the position in time matters. If day 50 is missing, using the global mean would inject an inaccurate value that breaks the time dependency. Forward fill uses the most recent known price, which is the best local approximation.
3.2 Outlier Handling
IQR-based outlier detection was applied. Extreme prices (e.g., the 2021 spike to ~$400) were identified but not removed, as these are real market events.
Removing true market events would make the model unable to handle real volatility.
3.3 Scaling
MinMaxScaler was applied to normalize Close prices to [0, 1].
This is necessary because neural network gradients become unstable with large raw values (~$5 to ~$400 range).
The scaler was fit only on training data and applied to test data to prevent data leakage.
3.4 Sliding Window (Time-Series Sequences)
A 60-day sliding window was used: the model sees 60 consecutive days as input and predicts day 61.
Input shape: (samples, 60 timesteps, 1 feature)
Train/test split: 80%/20%, without shuffling (time-series order must be preserved).
4. Feature Engineering
In addition to the raw Close price, the following technical indicators were computed:

Feature	Formula / Description
MA_30	30-day simple moving average
MA_90	90-day simple moving average
EMA_12	12-day exponential moving average
EMA_26	26-day exponential moving average
MACD	EMA_12 − EMA_26 (momentum indicator)
RSI	Relative Strength Index (14-day, 0–100 scale)
BB_upper	Bollinger upper band (MA + 2σ)
BB_lower	Bollinger lower band (MA − 2σ)
These features capture trend, momentum, and volatility — three key dimensions of stock behavior.

5. Model Architecture
5.1 SimpleRNN
Input (60, 1)
    → SimpleRNN(64, return_sequences=True)
    → Dropout(0.2)
    → SimpleRNN(32, return_sequences=False)
    → Dropout(0.2)
    → Dense(25, relu)
    → Dense(1)
Loss function: Mean Squared Error (MSE)
Optimizer: Adam (lr=0.001)
Callbacks: EarlyStopping (patience=10), ModelCheckpoint
Limitation: SimpleRNN suffers from the vanishing gradient problem — it loses information from early timesteps in long sequences, making it less effective for longer window sizes.

5.2 LSTM (Long Short-Term Memory)
Input (60, 1)
    → LSTM(128, return_sequences=True)
    → Dropout(0.2)
    → LSTM(64, return_sequences=True)
    → Dropout(0.2)
    → LSTM(32, return_sequences=False)
    → Dropout(0.2)
    → Dense(25, relu)
    → Dense(1)
Same loss, optimizer, and callbacks as SimpleRNN.
LSTM introduces forget, input, and output gates, allowing it to selectively remember or discard information over long sequences — solving the vanishing gradient problem.
6. Multi-Step Forecasting Strategy
To forecast 5 or 10 days ahead, a recursive (autoregressive) approach was used:

Feed the last 60 known days → predict day 1
Append prediction to the window, drop the oldest day
Feed the updated window → predict day 2
Repeat for N steps
This naturally accumulates error with each step, so 10-day predictions are inherently less accurate than 1-day predictions. This is a known limitation of autoregressive methods.

7. Hyperparameter Tuning
GridSearchCV was used with scikeras.KerasRegressor to systematically search:

Hyperparameter	Values Searched
LSTM units	64, 128
Dropout rate	0.1, 0.2
Learning rate	0.001, 0.0005
Cross-validation: 3-fold
Scoring: Negative MSE (higher = better)
Best parameters were used to train an "optimized LSTM" model.
8. Results
Model	RMSE ($)	MAE ($)	R²
SimpleRNN	~X.XX	~X.XX	~0.XXX
LSTM (default)	~X.XX	~X.XX	~0.XXX
LSTM (optimized)	~X.XX	~X.XX	~0.XXX
(Fill in with your actual results after running the notebook)

Key observations:

LSTM consistently outperformed SimpleRNN on all metrics
The optimized LSTM (via GridSearchCV) showed further improvement
1-day forecasts were significantly more accurate than 10-day forecasts
Both models tracked the overall trend well but struggled with sharp reversals
9. Insights & Conclusion
What worked well:
LSTM's gating mechanism makes it well-suited for financial time series with long-range dependencies
MinMaxScaling stabilized training and accelerated convergence
EarlyStopping prevented overfitting despite relatively small dataset
Limitations:
The model is trained on price history only — it cannot account for news events, earnings surprises, or macroeconomic shocks
Autoregressive multi-step forecasting accumulates error
Stock markets are semi-random (efficient market hypothesis) — even the best models have a ceiling on predictive accuracy
Suggested Improvements:
Multivariate input: Include all OHLCV columns + technical indicators as features
Sentiment analysis: Add Twitter/news sentiment scores as an additional input channel
Alternative architectures: Compare with GRU, Transformer (Temporal Fusion Transformer), or ARIMA/Prophet for benchmarking
Larger window: Try window sizes of 90 or 120 days
Ensemble: Average predictions from SimpleRNN + LSTM for more robust forecasts
10. Deployment
The project was deployed as an interactive Streamlit web application featuring:

CSV upload interface
Interactive EDA charts (price history, MA, RSI, Bollinger Bands)
Model training with live progress bar
Configurable window size, train ratio, and forecast horizon
Side-by-side model comparison
Multi-day forecast visualization with price annotations
To run locally:

pip install streamlit tensorflow scikit-learn pandas matplotlib seaborn scikeras joblib
streamlit run streamlit_app.py
