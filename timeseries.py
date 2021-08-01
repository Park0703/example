import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import re
import time
path = 'C:/R/chromedriver.exe'
driver = webdriver.Chrome(path)

page_url = 'https://www.kamis.or.kr/customer/price/wholesale/period.do?action=monthly&yyyy=2021&period=3&countycode=&itemcategorycode=100&itemcode=111&kindcode=&productrankcode=0&convert_kg_yn=N'

name = []
# for page in p :
driver.get(page_url)
time.sleep(1.5)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
#부류
category = soup.find(name="h3", attrs = {'class':'s_tit5 fl'}).get_text()
category = category.replace('\t', '')
category = category.replace('\n', '')
# 품목, 품종, 등급
item = soup.find(name = "h3", attrs={'class':'s_tit6 fl'}).get_text()
item = item.replace('\t', '')
item = item.replace('\n', '')
# 단위
unit = soup.find(name = "h3", attrs={'class':'s_tit6 color fl'}).get_text()
unit = unit.replace('\t', '')
unit = unit.replace('\n', '')
name = category + item + unit
print(name)

# 테이블 목차 긁어오기
data = soup.find(name = "table", attrs = {'class':'wtable3'})
col_name = []
info = []
for c in data.find_all("tr") :
    datalist = []
for d in c.find_all('th') :
    data_row = d.get_text()
    datalist.append(data_row)
col_name.append(datalist)

# 빈리스트 정리
mokcha = list(filter(None, col_name))
for a in data.find_all("tr") :
    infolist = []
    for b in a.find_all("td") :
        info_raw = b.get_text()
        info_raw = info_raw.replace('-', '0')
        info_N = float(info_raw.replace(',', ''))
        infolist.append(info_N)
    info.append(infolist)
    # 빈 리스트 정
    table_data = list(filter(None, info))

df = pd.DataFrame(table_data, columns=mokcha)
df




