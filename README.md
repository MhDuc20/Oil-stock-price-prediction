# Hệ thống Giao dịch Dầu thô Tự động dựa trên Học máy và Học tăng cường

Dự án này xây dựng một hệ thống giao dịch dầu thô tự động sử dụng kết hợp mô hình học máy LSTM (Long Short-Term Memory) và học tăng cường (Reinforcement Learning). Hệ thống có khả năng dự đoán giá dầu thô và đưa ra quyết định giao dịch tối ưu nhằm tối đa hóa lợi nhuận.

## Tổng quan

Hệ thống gồm hai thành phần chính:
1. **Mô hình LSTM** dự đoán giá dầu thô dựa trên dữ liệu lịch sử
2. **Mô hình PPO (Proximal Policy Optimization)** đưa ra quyết định giao dịch (mua, giữ, bán) dựa trên trạng thái thị trường hiện tại và dự đoán giá

## Tính năng

- Xử lý và chuẩn hóa dữ liệu tài chính
- Tính toán các chỉ báo kỹ thuật (RSI, MA7, MA30, Volatility)
- Dự đoán giá dầu thô sử dụng mạng LSTM
- Đưa ra quyết định giao dịch tối ưu sử dụng học tăng cường (RL)
- Mô phỏng và đánh giá chiến lược giao dịch
- Trực quan hóa kết quả giao dịch và hiệu suất danh mục đầu tư
- Tính toán các chỉ số hiệu suất quan trọng (lợi nhuận, mức sụt giảm tối đa, số lần giao dịch)

## Yêu cầu

```
pandas>=1.3.0
numpy>=1.19.5
matplotlib>=3.4.0
tensorflow>=2.6.0
scikit-learn>=0.24.0
stable-baselines3>=1.5.0
gymnasium>=0.26.0
```

## Cài đặt

1. Clone repository:
```bash
git clone https://github.com/username/crude-oil-trading-system.git
cd crude-oil-trading-system
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
├── data/
│   └── Crude_Oil_Data.csv    # Dữ liệu lịch sử giá dầu thô
├── models/
│   ├── lstm_model/           # Lưu trữ mô hình LSTM đã huấn luyện
│   └── rl_model/             # Lưu trữ mô hình RL đã huấn luyện
├── notebooks/
│   ├── data_preparation.ipynb       # Tiền xử lý dữ liệu
│   ├── lstm_training.ipynb          # Huấn luyện mô hình LSTM
│   ├── rl_training.ipynb            # Huấn luyện mô hình RL
│   └── trading_evaluation.ipynb     # Đánh giá chiến lược giao dịch
├── src/
│   ├── data_processing.py    # Xử lý dữ liệu và tính chỉ báo kỹ thuật
│   ├── lstm_model.py         # Định nghĩa và huấn luyện mô hình LSTM
│   ├── trading_env.py        # Môi trường giao dịch cho mô hình RL
│   ├── rl_model.py           # Huấn luyện mô hình RL
│   └── evaluation.py         # Đánh giá hiệu suất và trực quan hóa
├── README.md
└── requirements.txt
```

## Quá trình

### 1. Chuẩn bị dữ liệu

```python
from src.data_processing import load_and_preprocess_data

# Tải và tiền xử lý dữ liệu
df = load_and_preprocess_data('data/Crude_Oil_Data.csv')
```

### 2. Huấn luyện mô hình LSTM

```python
from src.lstm_model import prepare_data, create_sequences, build_lstm_model

# Chuẩn bị dữ liệu cho LSTM
train_scaled, test_scaled, test_raw, scaler = prepare_data(df)
seq_length = 30
X_train, y_train = create_sequences(train_scaled, seq_length)

# Xây dựng và huấn luyện mô hình LSTM
lstm_model = build_lstm_model((seq_length, X_train.shape[2]))
lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
```

### 3. Huấn luyện mô hình RL

```python
from src.trading_env import CrudeOilTradingEnv
from src.rl_model import train_rl_model
from stable_baselines3.common.vec_env import DummyVecEnv

# Tạo môi trường giao dịch
env = DummyVecEnv([lambda: CrudeOilTradingEnv(test_scaled.fillna(0), test_raw)])

# Huấn luyện mô hình RL
rl_model = train_rl_model(env, total_timesteps=20000)
rl_model.save("models/rl_model/crude_oil_trading_model")
```

### 4. Đánh giá và trực quan hóa kết quả

```python
from src.evaluation import evaluate_model_with_trading

# Đánh giá mô hình và hiển thị kết quả
results_df, transactions_df, final_return = evaluate_model_with_trading(
    rl_model, enhanced_env, test_raw, test_scaled, seq_length, lstm_model, scaler
)
```

## Kết quả
Hệ thống giao dịch được đánh giá thông qua các chỉ số trong 4 năm(2020-2025)
Thống kê hiệu suất giao dịch:
Mức sụt giảm tối đa: -33.54%
Tổng kết chiến lược giao dịch:
Lợi nhuận cuối cùng: 133.42%
![image](https://github.com/user-attachments/assets/da272bb2-b7b1-4e1c-ad6b-d16b15f68819)

