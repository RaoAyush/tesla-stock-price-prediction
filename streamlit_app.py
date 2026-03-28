"""
Tesla Stock Price Prediction — Streamlit App
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import io

# ── Try to load saved models (optional) ─────────────────────────
try:
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #E31937, #1f77b4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px; border-left: 4px solid #1f77b4;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Tesla_Motors.svg/320px-Tesla_Motors.svg.png", width=120)
    st.markdown("## ⚙️ Settings")

    st.markdown("### Model Configuration")
    window_size   = st.slider("Window Size (past days)", 30, 120, 60, 10)
    train_ratio   = st.slider("Train/Test Split", 0.7, 0.9, 0.8, 0.05)
    forecast_days = st.selectbox("Forecast Horizon", [1, 5, 10], index=0)

    st.markdown("### Model Architecture")
    model_choice = st.radio("Select Model", ["SimpleRNN", "LSTM", "Both (Compare)"])

    st.markdown("### Training")
    epochs     = st.slider("Max Epochs", 20, 150, 50, 10)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    st.markdown("---")
    st.markdown("**📌 About this App**")
    st.caption("Built with TensorFlow + Streamlit for Tesla stock price prediction using deep learning.")

# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean(uploaded_file):
    df = pd.read_csv(uploaded_file, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def add_features(df):
    df = df.copy()
    df['MA_30']  = df['Close'].rolling(30).mean()
    df['MA_90']  = df['Close'].rolling(90).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = df['EMA_12'] - df['EMA_26']
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_upper'] = df['MA_30'] + 2 * df['Close'].rolling(30).std()
    df['BB_lower'] = df['MA_30'] - 2 * df['Close'].rolling(30).std()
    df.dropna(inplace=True)
    return df

def create_sequences(data, win):
    X, y = [], []
    for i in range(win, len(data)):
        X.append(data[i-win:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_model(model_type, units, win, dropout=0.2, lr=0.001):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    model = Sequential(name=model_type)
    if model_type == 'SimpleRNN':
        from tensorflow.keras.layers import SimpleRNN as RNNLayer
        model.add(RNNLayer(units, return_sequences=True, input_shape=(win, 1)))
        model.add(Dropout(dropout))
        model.add(RNNLayer(units // 2))
        model.add(Dropout(dropout))
    else:
        from tensorflow.keras.layers import LSTM as LSTMLayer
        model.add(LSTMLayer(units, return_sequences=True, input_shape=(win, 1)))
        model.add(Dropout(dropout))
        model.add(LSTMLayer(units // 2))
        model.add(Dropout(dropout))

    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr), loss='mse', metrics=['mae'])
    return model

def predict_n_days(model, last_seq, n_days, scaler, win):
    seq  = last_seq.copy().reshape(1, win, 1)
    preds = []
    for _ in range(n_days):
        p = model.predict(seq, verbose=0)[0, 0]
        preds.append(p)
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, 0] = p
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

def get_metrics(actual, predicted):
    mse  = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(actual, predicted)
    r2   = r2_score(actual, predicted)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}

# ─────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📈 Tesla Stock Price Predictor</p>', unsafe_allow_html=True)
st.caption("Deep learning-based stock price prediction using SimpleRNN and LSTM networks")

tab1, tab2, tab3, tab4 = st.tabs(["📂 Data", "📊 EDA", "🤖 Models", "🔮 Forecast"])

# ─── Tab 1: Upload Data ───────────────────────────────────────────
with tab1:
    st.subheader("Upload TSLA Dataset")
    uploaded = st.file_uploader("Upload TSLA.csv", type=["csv"])

    if uploaded:
        df = load_and_clean(uploaded)
        df = add_features(df)
        st.success(f"✅ Dataset loaded: {len(df):,} trading days ({df.index.min().date()} → {df.index.max().date()})")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records",  f"{len(df):,}")
        col2.metric("Min Close",      f"${df['Close'].min():.2f}")
        col3.metric("Max Close",      f"${df['Close'].max():.2f}")
        col4.metric("Avg Close",      f"${df['Close'].mean():.2f}")

        st.markdown("### Preview")
        st.dataframe(df[['Open','High','Low','Close','Volume','RSI','MACD']].tail(15).style.format("{:.2f}"), use_container_width=True)

        st.markdown("### Missing Values")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("No missing values after cleaning.")
        else:
            st.warning(f"Missing values found: {missing[missing > 0].to_dict()}")

        # Store in session state
        st.session_state['df'] = df
    else:
        st.info("👆 Please upload TSLA.csv to begin. Download it from the project dataset link.")

# ─── Tab 2: EDA ───────────────────────────────────────────────────
with tab2:
    if 'df' not in st.session_state:
        st.warning("Please upload data in the Data tab first.")
    else:
        df = st.session_state['df']
        st.subheader("Exploratory Data Analysis")

        # Price history
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(df.index, df['Close'], color='steelblue', linewidth=1.2, label='Close')
        ax.plot(df.index, df['MA_30'], color='orange', linewidth=1.5, alpha=0.8, label='MA-30')
        ax.plot(df.index, df['MA_90'], color='red', linewidth=1.5, alpha=0.8, label='MA-90')
        ax.fill_between(df.index, df['Close'], alpha=0.05, color='steelblue')
        ax.set_title('TSLA Closing Price with Moving Averages')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        col1, col2 = st.columns(2)
        with col1:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            df['Daily_Return'] = df['Close'].pct_change()
            ax2.hist(df['Daily_Return'].dropna(), bins=80, color='steelblue', alpha=0.8, edgecolor='white')
            ax2.axvline(0, color='red', linestyle='--')
            ax2.set_title('Daily Returns Distribution')
            ax2.set_xlabel('Return')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with col2:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.plot(df.index, df['RSI'], color='purple', linewidth=1)
            ax3.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
            ax3.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
            ax3.set_title('RSI Indicator')
            ax3.set_ylabel('RSI')
            ax3.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        # Bollinger Bands
        fig4, ax4 = plt.subplots(figsize=(13, 4))
        recent = df.last('2Y')
        ax4.plot(recent.index, recent['Close'],    color='steelblue', linewidth=1.2, label='Close')
        ax4.plot(recent.index, recent['BB_upper'], color='red',       linewidth=1, linestyle='--', alpha=0.7, label='Upper BB')
        ax4.plot(recent.index, recent['BB_lower'], color='green',     linewidth=1, linestyle='--', alpha=0.7, label='Lower BB')
        ax4.fill_between(recent.index, recent['BB_lower'], recent['BB_upper'], alpha=0.07, color='gray')
        ax4.set_title('Bollinger Bands (Last 2 Years)')
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

# ─── Tab 3: Train Models ─────────────────────────────────────────
with tab3:
    if 'df' not in st.session_state:
        st.warning("Please upload data in the Data tab first.")
    else:
        df = st.session_state['df']
        st.subheader("Train Deep Learning Models")

        if not KERAS_AVAILABLE:
            st.error("TensorFlow not installed. Run: pip install tensorflow")
        else:
            if st.button("🚀 Train Model(s)", type="primary"):
                # Preprocessing
                scaler = MinMaxScaler((0, 1))
                scaled = scaler.fit_transform(df[['Close']].values)
                X, y   = create_sequences(scaled, window_size)
                X      = X.reshape(X.shape[0], X.shape[1], 1)

                split     = int(len(X) * train_ratio)
                X_tr, X_te = X[:split], X[split:]
                y_tr, y_te = y[:split], y[split:]

                from tensorflow.keras.callbacks import EarlyStopping
                es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

                test_dates    = df.index[window_size + split:]
                y_actual      = scaler.inverse_transform(y_te.reshape(-1, 1))
                results_store = {}
                history_store = {}

                models_to_train = []
                if model_choice == 'Both (Compare)':
                    models_to_train = ['SimpleRNN', 'LSTM']
                else:
                    models_to_train = [model_choice]

                for mtype in models_to_train:
                    st.markdown(f"**Training {mtype}...**")
                    prog = st.progress(0)
                    model = build_model(mtype, 64, window_size)

                    class ProgressCallback(
                        __import__('tensorflow').keras.callbacks.Callback
                    ):
                        def on_epoch_end(self, epoch, logs=None):
                            prog.progress(min(int((epoch+1)/epochs*100), 100))

                    h = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                                  validation_split=0.1, callbacks=[es, ProgressCallback()], verbose=0)

                    pred_sc = model.predict(X_te, verbose=0)
                    pred    = scaler.inverse_transform(pred_sc)
                    metrics = get_metrics(y_actual, pred)

                    results_store[mtype] = {'pred': pred, 'metrics': metrics, 'model': model}
                    history_store[mtype] = h.history
                    st.success(f"✅ {mtype} — RMSE: ${metrics['RMSE']:.2f} | R²: {metrics['R²']:.4f}")

                # Store results
                st.session_state.update({
                    'results': results_store,
                    'history': history_store,
                    'y_actual': y_actual,
                    'test_dates': test_dates,
                    'scaler': scaler,
                    'scaled': scaled,
                    'window_size': window_size
                })

                # ── Show metrics ─────────────────────────────────
                st.subheader("📊 Evaluation Metrics")
                rows = []
                for mname, res in results_store.items():
                    row = {'Model': mname}
                    row.update(res['metrics'])
                    rows.append(row)
                metrics_df = pd.DataFrame(rows).set_index('Model').round(4)
                st.dataframe(metrics_df.style.highlight_min(color='lightgreen').highlight_max(subset=['R²'], color='lightgreen'))

                # ── Plot predictions ─────────────────────────────
                st.subheader("📈 Actual vs Predicted")
                fig, ax = plt.subplots(figsize=(13, 5))
                ax.plot(test_dates[:len(y_actual)], y_actual, color='steelblue', lw=1.8, label='Actual', alpha=0.9)
                colors_map = {'SimpleRNN': 'coral', 'LSTM': 'green'}
                for mname, res in results_store.items():
                    ax.plot(test_dates[:len(res['pred'])], res['pred'], lw=1.5,
                            color=colors_map.get(mname, 'purple'), label=f'{mname} Predicted', alpha=0.85)
                ax.set_title('TSLA Close Price — Actual vs Predicted')
                ax.set_ylabel('Price (USD)')
                ax.legend()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── Training history ─────────────────────────────
                st.subheader("📉 Training Loss Curves")
                fig2, axes = plt.subplots(1, len(history_store), figsize=(13, 4))
                if len(history_store) == 1:
                    axes = [axes]
                for ax, (mname, hist) in zip(axes, history_store.items()):
                    ax.plot(hist['loss'], label='Train')
                    ax.plot(hist['val_loss'], label='Val')
                    ax.set_title(f'{mname} Loss')
                    ax.set_xlabel('Epoch')
                    ax.legend()
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()

# ─── Tab 4: Forecast ─────────────────────────────────────────────
with tab4:
    if 'results' not in st.session_state:
        st.warning("Please train models in the Models tab first.")
    else:
        st.subheader(f"🔮 {forecast_days}-Day Price Forecast")
        results_store = st.session_state['results']
        scaler        = st.session_state['scaler']
        scaled        = st.session_state['scaled']
        win           = st.session_state['window_size']
        df            = st.session_state['df']

        last_seq  = scaled[-win:]
        colors_fc = {'SimpleRNN': 'coral', 'LSTM': 'green'}

        forecasts = {}
        for mname, res in results_store.items():
            fc = predict_n_days(res['model'], last_seq, forecast_days, scaler, win)
            forecasts[mname] = fc

        # Show forecast table
        future_dates = pd.bdate_range(start=df.index[-1], periods=forecast_days+1)[1:]
        fc_df = pd.DataFrame(forecasts, index=future_dates)
        fc_df.index.name = 'Date'
        st.markdown("#### Predicted Prices")
        st.dataframe(fc_df.style.format("${:.2f}"), use_container_width=True)

        # Plot
        n_context = 90
        ctx_prices = df['Close'].values[-n_context:]
        ctx_dates  = df.index[-n_context:]

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(ctx_dates, ctx_prices, color='steelblue', lw=1.8, label='Historical', alpha=0.9)
        ax.axvline(x=df.index[-1], color='gray', linestyle=':', lw=1.5, alpha=0.7, label='Today')

        for mname, fc in forecasts.items():
            ax.plot(future_dates, fc, 'o--', lw=2, color=colors_fc.get(mname, 'purple'),
                    label=f'{mname} Forecast', markersize=7)
            for d, p in zip(future_dates, fc):
                ax.annotate(f'${p:.1f}', (d, p), textcoords='offset points',
                            xytext=(0, 10), fontsize=8, ha='center')

        ax.set_title(f'TSLA — {forecast_days}-Day Price Forecast')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("⚠️ Disclaimer: These predictions are for educational purposes only and should not be used for actual trading decisions.")
