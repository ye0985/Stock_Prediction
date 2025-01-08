import pandas as pd

# 3 moving average, 5 moving average 계산 및 새로운 파일 생성
df = pd.read_csv('historical_data.csv')
df.head()

df.rename(columns={
    'Adj Close Adjusted close price adjusted for splits and dividend and/or capital gain distributions.': 'Adj Close'
}, inplace=True)

# 숫자만 남기고 문자열을 제거(숫자 형식으로 변환할 수 없는 값은 NaN으로 처리)
df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')

# NaN 값을 포함하는 행 삭제
df.dropna(subset=['Adj Close'], inplace=True)

# 이동평균 계산
df['3MA'] = df['Adj Close'].rolling(window=3).mean()
df['5MA'] = df['Adj Close'].rolling(window=5).mean()

# 3MA와 5MA가 추가된 데이터를 새로운 파일로 저장
output_file = 'historical_data_with_moving_averages.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')



