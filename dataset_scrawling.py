import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import os

browser = webdriver.Chrome()
browser.maximize_window() 

url = 'https://finance.yahoo.com/quote/005930.KS/history/'
browser.get(url)

df = pd.read_html(browser.page_source)[0]
df.dropna(axis='index',how='all',inplace=True) # 모두 naan인 row 지우기
df.dropna(axis='columns',how='all',inplace=True)

f_name = 'historical_data.csv'
if os.path.exists(f_name): #파일이 있다면 헤더제외
    df.to_csv(f_name, encoding='utf-8-sig',index=False, mode='a', header=False)
else:
    df.to_csv(f_name, encoding='utf-8-sig',index=False)
    
browser.quit()

