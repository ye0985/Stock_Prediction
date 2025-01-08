import pandas as pd
import matplotlib.pyplot as plt


input_file = 'historical_data_with_moving_averages.csv'

df = pd.read_csv(input_file)

#시각화
plt.title('Samsung Electronic Stock Price')
plt.ylabel('price')
plt.xlabel('period')
plt.grid()
plt.plot(df['Adj Close'],label = 'Adj Close')
#plt.show()

#정보 확인
#print(df.describe())

#missing data 확인 (0)
print(df.isnull().sum())

'''
# 최소값이 0 인 column 체크
for col in df.columns:
    if df[col].min() == 0:
        col_name = col
        print(col_name, type(col_name))
      
# 각 column에 0 몇개인지 확인
for col in df.columns:
    missing_rows = df.loc[df[col] == 0].shape[0]
    #print(col + ':' + str(missing_rows))
   
'''

for col in df.columns:
    zero_count = (df[col] == 0).sum()  # 값이 0인 행의 개수
    nan_count = df[col].isna().sum()  # NaN 값의 개수
    total_count = zero_count + nan_count
    print(f"{col}: 0 값={zero_count}, NaN 값={nan_count}, 총합={total_count}")


#print(df.isnull().sum())
#print(df.isnull().any())

# missing data(0) 삭제
df = df.dropna()

print(df.isnull().sum())

output_file = 'cleaned_data.csv'
df.to_csv(output_file, index = False)