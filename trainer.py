import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout



input_file = 'cleaned_data.csv'
df = pd.read_csv(input_file)

#정규화 (Date 제외한 모든 수치부분 정규화)
scaler = MinMaxScaler()

scale_cols = ['Open', 'High', 'Low', 'Close Close price adjusted for splits.', 'Adj Close', '3MA', '5MA', 'Volume']

scaled_df = scaler.fit_transform(df[scale_cols])
scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)
print(scaled_df)

# Train 
#
# 주가예측을 위해 3MA, 5MA, Adj Close 항목을 feature로,
# 정답으로 Adj Close을 label로 선정
# 시계열 데이터를 위한 window_size = 40 선정
#
# 입력 파라미터 feature, label => numpy type
def make_sequene_dataset(feature, label, window_size):

    feature_list = []      # 생성될 feature list
    label_list = []        # 생성될 label list

    for i in range(len(feature)-window_size):

        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])

    return np.array(feature_list), np.array(label_list)

# feature_df, label_df 생성
feature_cols = [ '3MA', '5MA', 'Adj Close','Volume' ]
label_cols = [ 'Adj Close' ]

feature_df = pd.DataFrame(scaled_df, columns=feature_cols)
label_df = pd.DataFrame(scaled_df, columns=label_cols)

# DataFrame => Numpy 변환
feature_np = feature_df.to_numpy()
label_np = label_df.to_numpy()

print(f'feature_np.shape: {feature_np.shape}, label_np.shape: {label_np.shape}')

#시계열 데이터 생성 (make_sequence_dataset)
window_size = 40

X, Y = make_sequene_dataset(feature_np, label_np, window_size)

print(f'X.shape: {X.shape}, Y.shape: {Y.shape}')

# Train, Test data split
split = int(len(X)*0.95)
x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(f'x_train.shape : {x_train.shape}, y_train.shape : {y_train.shape}')
print(f'x_test.shape : {x_test.shape}, y_test.shape : {y_test.shape}')

# model 생성
model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape=x_train[0].shape))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()

#모델 학습 (EarlyStopping 적용)
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.fit(x_train, y_train, 
          validation_data=(x_test, y_test),
          epochs=100, batch_size=16,
          callbacks=[early_stop])


#예측을 통한 정답과의 비교 (오차계산 MAPE 사용, 평균절대값백분율오차)
pred = model.predict(x_test)
#print(pred.shape)
#print(pred)

plt.figure(figsize=(20, 6))
plt.title('3MA + 5MA + Adj Close, window_size=40')
plt.ylabel('adj close')
plt.xlabel('period')
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')

plt.show()

# 평균절대값백분율오차계산 (MAPE)
print( np.sum(abs(y_test-pred)/y_test) / len(x_test) )