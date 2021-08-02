# 쿠팡 : 노트북을 사는데 평점, 리뷰가 많은 것으로
# 주소가 get방식으로 엄청 길어지면? 도메인 page를 찾고 페이지 5 

# get -- 누구나 볼수있게, 쉽게 페이지 요청, 큰 데이터 제한  
  # http://wwww.coupang.com/np/search?minPrice=1000&maxPrice=100000&page=1
  # 주소 뒤에 ? 그리고 나서 변수들 구분 minPrice & maxPrice & page
# post -- http body에 숨김, 보안, 파일업로드, 개인정보, 글 작성
  # https://www.coupang.com/np/search?id=nadocoing&pw=1234
  # ? 뒤에 id와 pw


import requests
import re 
from bs4 import BeautifulSoup
headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'}

# url = 'https://www.coupang.com/np/search?component=&q=%EB%85%B8%ED%8A%B8%EB%B6%81&channel=user'

# 페이지 넘기는 법
for i in range(1, 6) :
    print('페이지 :', i)  
    url = 'https://www.coupang.com/np/search?q=%EB%85%B8%ED%8A%B8%EB%B6%81&channel=user&component=&eventCategory=SRP&trcid=&traid=&sorter=scoreDesc&minPrice=&maxPrice=&priceRange=&filterType=&listSize=36&filter=&isPriceRange=false&brand=&offerCondition=&rating=0&page={}&rocketAll=false&searchIndexingToken=1=4&backgroundColor='.format(i)

    # html 정보가 느린 경우
    res = requests.get(url, headers = headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'lxml')

    # 광고 붙은 내용
    # items = soup.find_all('li', attrs={'class':re.compile('search-product')})
    # print(items[0].find('div', attrs={'class':'name'}).get_text())
    # 삼성전자 갤럭시북 플렉스2 미스틱브론즈 노트북 NT950QDZ-G58AZ (i5-1135G7 39.6cm MX450 WIN10 Home), 포함, 256GB, 8GB

# 평점
    items = soup.find_all('li', attrs={'class':re.compile('^search-product')})
    for item in items :
        # 광고를 제외하고 평점을 구하려면
        ad_badge = item.find('span', attrs = {'class':'ad-badge-text'})
        if ad_badge :
            # print('<광고상품 제외합니다.>')
            continue
      
        name = item.find('div', attrs={'class':'name'}).get_text()  # 제품명 
        # apple 제외하고 싶으면?
        if 'Apple' in name :
            # print('<Apple 상품 제외합니다.>')
            continue
        price = item.find('strong', attrs={'class':'price-value'}).get_text() # 가격
        
        #리뷰 100개 이상, 평점 4.5이상
        
        rate = item.find('em', attrs={'class':'rating'}) # 평점 # 평점이 없는 경우 get_text() 빼기
        if rate :
            rate = rate.get_text()
        else :
            # print('<평점없는 상품 제외합니다.>')
            continue    
          
        rate_cnt = item.find('span', attrs={'class':'rating-total-count'}) # 평점 수
        if rate_cnt :
              rate_cnt = rate_cnt.get_text()[1:-1] #(26), (635)이니 재가공
              # print('리뷰수', rate_cnt)
        else :
            # print('<평점 수 없는 상품 제외합니다.>')
            continue  
        
        link = item.find('a', attrs={'class':'search-product-link'})['href']
          
        #여기까지오면 최소 평점, 평점수는 있는 상품임
        if float(rate) >=4.5 and int(rate_cnt) >= 100 : # rate가 4.5 이상 and rate_cnt >=100

            # print(name, price, rate, rate_cnt)
            print(f'제품명 : {name}')
            print(f'가격 : {price}')
            print(f'평점 : {rate}, (평점  수 : {rate_cnt})')
            print('바로가기 : {}'.format('https://www.coupang.com' + link))
            print('--'*10)






