import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

# 데이터셋 로드 및 처리
def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    data['Last seen'] = pd.to_datetime(data['Last seen'])
    data.sort_values('Last seen', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Normalized Usage'] = scaler.fit_transform(data[['Usage (MB)']])
    return data, scaler

# LSTM을 위한 데이터셋 생성
def create_dataset(dataset, look_back=7):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# LSTM 모델 구축 및 훈련
def build_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1)
    return model

# 메인 실행 함수
def main():
    file_path = 'C:\\Users\\Haemin\\OneDrive\\바탕 화면\\LSTM\\daily_network_sum_1206.csv'
    data, scaler = load_and_process_data(file_path)
    dataset = data['Normalized Usage'].values.reshape(-1, 1)
    X, y = create_dataset(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = build_and_train_model(X_train, y_train)

    # 예측 수행
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # 예측 결과를 원래 스케일로 변환
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_inv = scaler.inverse_transform([y_train])
    y_test_inv = scaler.inverse_transform([y_test])

    # 성능 평가 (예: MSE)
    train_mse = mean_squared_error(y_train_inv[0], train_predict[:,0])
    test_mse = mean_squared_error(y_test_inv[0], test_predict[:,0])
    print(f'Train MSE: {train_mse}, Test MSE: {test_mse}')

    # 미래 데이터 예측 (예: 마지막 7일 데이터로 다음 7일 예측)
    last_7_days = dataset[-7:].reshape(1, 7, 1)
    next_7_days = model.predict(last_7_days)
    next_7_days = scaler.inverse_transform(next_7_days)

    # 예측 결과 출력
    print("Next 7 days prediction:")
    print(next_7_days)


if __name__ == "__main__":
    main()
