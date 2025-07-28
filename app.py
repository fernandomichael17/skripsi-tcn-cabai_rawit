import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Activation, Input, Flatten, add
from tensorflow.keras.models import Model
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG & CSS ---
st.set_page_config(
    page_title="Prediksi Harga Cabai Rawit Bekasi - TCN Model",
    page_icon="🌶️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2E8B57; text-align: center; margin-bottom: 2rem; font-weight: bold; }
    .sub-header { font-size: 1.5rem; color: #1E6B3A; margin-top: 2rem; margin-bottom: 1rem; }
    .metric-container { background-color: #f0f8f0; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
    .sidebar-header { font-size: 1.2rem; color: #2E8B57; font-weight: bold; margin-bottom: 1rem; }
    .prediction-card { background-color: #f9f9f9; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df_raw = pd.read_csv("data_cabai_rawit.csv")
        if df_raw.shape[1] > df_raw.shape[0]:
            df_transposed = df_raw.T
            df_transposed.columns = df_transposed.iloc[0]
            df_transposed = df_transposed[1:]
        else:
            df_transposed = df_raw.copy()
        cabai_rawit = df_transposed.copy()
        if 'Date' in cabai_rawit.columns:
            cabai_rawit.index = pd.to_datetime(cabai_rawit['Date'])
        else:
            cabai_rawit.index = pd.to_datetime(cabai_rawit.index.str.replace(" ", ""), format="%d/%m/%Y", errors="coerce")
        price_column = 'Cabai Rawit' if 'Cabai Rawit' in cabai_rawit.columns else cabai_rawit.columns[0]
        cabai_rawit = cabai_rawit[price_column].replace('-', np.nan)
        if cabai_rawit.dtype == 'object':
            cabai_rawit = pd.to_numeric(cabai_rawit.str.replace(",", ""), errors='coerce')
        cabai_rawit = cabai_rawit.bfill().ffill()
        return cabai_rawit
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load('scaler.save')
    except:
        st.error("Scaler file not found. Please make sure 'scaler.save' exists.")
        return None

# --- TCN MODEL ---
def residual_block(x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.2):
    r = Conv1D(nb_filters, kernel_size, dilation_rate=dilation_rate, padding=padding)(x)
    r = Activation('relu')(r)
    r = Dropout(dropout_rate)(r)
    r = Conv1D(nb_filters, kernel_size, dilation_rate=dilation_rate, padding=padding)(r)
    r = Activation('relu')(r)
    r = Dropout(dropout_rate)(r)
    if x.shape[-1] != nb_filters:
        x = Conv1D(nb_filters, kernel_size=1, padding='same')(x)
    x = add([x, r])
    return Activation('relu')(x)

def load_trained_model(window_size):
    try:
        model_filename = f"saved_tcn_models/tcn_window_{window_size}.h5"
        return load_model(model_filename, custom_objects={'residual_block': residual_block})
    except Exception as e:
        st.error(f"Error loading model {model_filename}: {str(e)}")
        return None

def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def predict_future(model, last_sequence, scaler, days=30):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days):
        input_seq = current_sequence.reshape(1, len(current_sequence), 1)
        next_pred = model.predict(input_seq, verbose=0)[0, 0]
        predictions.append(next_pred)
        current_sequence = np.append(current_sequence[1:], next_pred)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return (
        np.sqrt(mse),                # RMSE
        mse,                         # MSE
        mean_absolute_error(y_true, y_pred),
        mean_absolute_percentage_error(y_true, y_pred) * 100,
        r2_score(y_true, y_pred)
    )

# --- MAIN APP ---
def main():
    st.markdown('<h1 class="main-header">🌶️ Prediksi Harga Cabai Rawit Bekasi - TCN Model</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Sistem Prediksi Harga Menggunakan Temporal Convolutional Network (TCN)</p>', unsafe_allow_html=True)

    cabai_rawit = load_data()
    scaler = load_scaler()
    if cabai_rawit is None or scaler is None:
        st.error("Failed to load required data or scaler. Please check your files.")
        return

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Pengaturan Model TCN</div>', unsafe_allow_html=True)
    window_options = [7, 14, 21, 30, 60]
    jangka_waktu_options = {"1 Minggu": 7, "1 Bulan": 30, "1 Tahun": 365}
    selected_window = st.sidebar.selectbox("Pilih Window Size:", window_options, index=3)
    selected_jangka_waktu = st.sidebar.selectbox("Pilih Jangka Waktu Prediksi:", list(jangka_waktu_options.keys()), index=1)
    prediction_days = jangka_waktu_options[selected_jangka_waktu]
    st.sidebar.markdown("**Informasi Model:**")
    st.sidebar.write("- Model: TCN (Temporal Convolutional Network)")
    st.sidebar.write("- Komoditas: Cabai Rawit (Bekasi)")
    st.sidebar.write(f"- Window Size: {selected_window} hari")
    st.sidebar.write(f"- Jangka Waktu Prediksi: {selected_jangka_waktu}")

    # Load model
    model = load_trained_model(selected_window)
    if model is None:
        st.error("Failed to load TCN model. Please check if the model file exists.")
        return

    # Data split & scaling
    test_size = 0.3
    scaled_data = scaler.fit_transform(cabai_rawit.values.reshape(-1, 1))
    split_idx = int(len(scaled_data) * (1 - test_size))
    train_data = scaled_data[:split_idx]
    test_data = scaled_data[split_idx:]

    # Test prediction
    X_test, y_test = create_dataset(test_data, selected_window)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    test_predictions = model.predict(X_test, verbose=0)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse, mse, mae, mape, r2 = calculate_metrics(y_test_actual, test_predictions)

    # Layout
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<h2 class="sub-header">📈 Grafik Data Historis dan Prediksi</h2>', unsafe_allow_html=True)
        last_sequence = scaled_data[-selected_window:]
        future_predictions = predict_future(model, last_sequence, scaler, days=prediction_days)
        last_date = cabai_rawit.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cabai_rawit.index, y=cabai_rawit.values, mode='lines', name='Data Historis', line=dict(color='blue', width=2)))
        test_dates = cabai_rawit.index[split_idx + selected_window:]
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_predictions.flatten(),
            mode='lines',
            name='Prediksi Test',
            line=dict(color='yellow', width=2)
        ))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name=f'Prediksi {selected_jangka_waktu}', line=dict(color='red', width=3), marker=dict(size=6)))
        fig.update_layout(title=f'Prediksi Harga Cabai Rawit Bekasi - TCN Model (Window: {selected_window} hari)', xaxis_title='Tanggal', yaxis_title='Harga (Rp)', height=500, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<h2 class="sub-header">📊 Evaluasi Model</h2>', unsafe_allow_html=True)
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MSE", f"{mse:.2f}")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("MAPE", f"{mape:.2f}%")
        st.metric("R² Score", f"{r2:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("**Informasi Data:**")
        st.write(f"- Total Data: {len(cabai_rawit)} hari")
        st.write(f"- Data Training: {len(train_data)} hari")
        st.write(f"- Data Testing: {len(test_data)} hari")
        st.write(f"- Periode: {cabai_rawit.index[0].strftime('%d/%m/%Y')} - {cabai_rawit.index[-1].strftime('%d/%m/%Y')}")

    # Tabel Prediksi
    st.markdown(f'<h2 class="sub-header">📋 Tabel Prediksi {selected_jangka_waktu}</h2>', unsafe_allow_html=True)
    prediction_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi Harga (Rp)': [f"{price:,.0f}" for price in future_predictions],
        'Harga Numerik': future_predictions
    })
    st.dataframe(prediction_df[['Tanggal', 'Prediksi Harga (Rp)']], use_container_width=True, hide_index=True)

    # Statistik Prediksi
    st.markdown('<h3 class="sub-header">📌 Statistik Prediksi</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Harga Rata-rata Prediksi", f"Rp {np.mean(future_predictions):,.0f}")
    with col2:
        st.metric("Harga Tertinggi", f"Rp {np.max(future_predictions):,.0f}", delta=f"Tanggal: {future_dates[np.argmax(future_predictions)].strftime('%d/%m/%Y')}")
    with col3:
        st.metric("Harga Terendah", f"Rp {np.min(future_predictions):,.0f}", delta=f"Tanggal: {future_dates[np.argmin(future_predictions)].strftime('%d/%m/%Y')}")

    # Perbandingan Window Size
    st.markdown('<h2 class="sub-header">🔍 Perbandingan Window Size</h2>', unsafe_allow_html=True)
    if st.button("Tampilkan Perbandingan Semua Window Size"):
        with st.spinner('Memproses perbandingan...'):
            comparison_results = []
            for window in window_options:
                try:
                    temp_model = load_trained_model(window)
                    if temp_model is not None:
                        X_temp, y_temp = create_dataset(test_data, window)
                        X_temp = X_temp.reshape(X_temp.shape[0], X_temp.shape[1], 1)
                        temp_pred = temp_model.predict(X_temp, verbose=0)
                        temp_pred = scaler.inverse_transform(temp_pred)
                        y_temp_actual = scaler.inverse_transform(y_temp.reshape(-1, 1))
                        temp_rmse, temp_mse, temp_mae, temp_mape, temp_r2 = calculate_metrics(y_temp_actual, temp_pred)
                        comparison_results.append({
                            'Window Size': window,
                            'RMSE': temp_rmse,
                            'MSE': temp_mse,
                            'MAE': temp_mae,
                            'MAPE (%)': temp_mape,
                            'R² Score': temp_r2
                        })
                except Exception as e:
                    st.warning(f"Could not load model for window size {window}: {str(e)}")
            if comparison_results:
                comparison_df = pd.DataFrame(comparison_results)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                best_model = comparison_df.loc[comparison_df['RMSE'].idxmin()]
                st.success(f"📊 **Model Terbaik**: Window Size {best_model['Window Size']} dengan nilai terbaik dari semua metrik")
                st.markdown("### Grafik Perbandingan Metrik")
                metrics = ['RMSE', 'MSE', 'MAE', 'MAPE (%)', 'R² Score']
                colors = ['#1E6B3A', '#8B0000', '#FFA500', '#FF6347', '#4682B4']
                for i, metric in enumerate(metrics):
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=comparison_df['Window Size'].astype(str),
                        y=comparison_df[metric] if metric in comparison_df.columns else comparison_df[metric.replace(' (%)', '')],
                        marker_color=colors[i]
                    ))
                    fig_bar.update_layout(
                        title=f'Perbandingan {metric} untuk Berbagai Window Size',
                        xaxis_title='Window Size (hari)',
                        yaxis_title=metric,
                        height=350,
                        xaxis=dict(type='category')
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

    # Download section
    st.markdown('<h2 class="sub-header">💾 Download Hasil</h2>', unsafe_allow_html=True)
    csv_data = prediction_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediksi CSV",
        data=csv_data,
        file_name=f"prediksi_cabai_rawit_bekasi_tcn_w{selected_window}_{selected_jangka_waktu.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">Sistem Prediksi Harga Cabai Rawit Bekasi menggunakan TCN (Temporal Convolutional Network)</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
