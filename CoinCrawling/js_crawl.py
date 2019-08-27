#-*-coding:utf-8-*-
from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import codecs
import re
import sys

URL='https://coinmarketcap.com/ko/all/views/all/'

class COIN:

	def __init__(self,date):
	
		self.tmp=[]
		self.f=''
		self.w=''
		self.date=date
		
	def crawl(self):
		
		column_list=['#','이름','종목','시가총액','가격','유통 공급량','24시간 거래량','1시간 변동률','24시간 변동률','7일 변동률']
		with open('coin_crawl_'+self.date+'.csv','w',-1,newline='') as f:
			w=csv.writer(f)
			w.writerow(column_list)
			
			# using webdriver for excuting javascript
			driver=webdriver.PhantomJS()
			print('\n[*] Get page source using PhantomJS ...\n')
			driver.get(URL)
			html=driver.page_source
			soup=BeautifulSoup(html,'html.parser')

			# get table in main page
			tr_list=soup.find_all('tr')
			
			#tr_list_split=tr_list[:5]
			# big loop for tr
			print('[*] Start Crawling ...')
			for i,v in enumerate(tr_list):
				if i==0:
					pass
				else:	
					tmp=[]
					# column 0: index
					index=v.find('td',{'class':'text-center'}).text
					index=index.strip()
					tmp.append(index)
					
					# column 1: coin name
					name=v.find('td',{'class':'no-wrap currency-name'}).text
					s=re.search('.+\n\n(.*)\n',name)
					s=s.groups()
					name=s[0]
					name=name.strip()
					tmp.append(name)
					
					# column 2: coin symbol
					symbol=v.find('td',{'class':'text-left col-symbol'}).text
					tmp.append(symbol)
					
					# column 3: marcat cap
					cap=v.find('td',{'class':'no-wrap market-cap text-right'})
					if cap is None:
						tmp.append('-')
					else:
						cap=cap.text
						if '₩' in cap:
							cap=cap.replace('₩','')
						tmp.append(cap)
					
					# column 4: price
					price=v.find('a',{'class':'price'})
					if price is None:
						tmp.append('-')
					else:
						price=price.text
						if '₩' in price:
							price=price.replace('₩','')					
						tmp.append(price)
					
					# column 5: supply
					supply=v.find('td',{'class':'no-wrap text-right circulating-supply'})
					if supply is None:
						tmp.append('-')
					else:
						supply=supply.text
						if '₩' in supply:
							supply=supply.replace('₩','')	
						supply=supply.replace('\n','')	
						tmp.append(supply)
	
					# column 6: volume		
					volume=v.find('a',{'class':'volume'})
					if volume is None:
						tmp.append('-')
					else:
						volume=volume.text
						if '₩' in volume:
							volume=volume.replace('₩','')
						tmp.append(volume)
						
					# column 7: percentage for 1 hour
					per_1=v.find('td',{'class':'percent-change','data-timespan':'1h'})
					if per_1 is None:
						tmp.append('-')
					else:
						per_1=per_1.text
						tmp.append(per_1)
					
					# column 8: percentage for 24 hour
					per_24=v.find('td',{'class':'percent-change','data-timespan':'24h'})
					if per_24 is None:
						tmp.append('-')
					else:
						per_24=per_24.text
						tmp.append(per_24)
					
					# column 9: percentage for 7 days
					per_7=v.find('td',{'class':'percent-change','data-timespan':'7d'})				
					if per_7 is None:
						tmp.append('-')
					else:
						per_7=per_7.text
						tmp.append(per_7)
					
					print(tmp)
					w.writerow(tmp)
		
def main():
	
	date=sys.argv[1]
	coin=COIN(date)
	coin.crawl()
	
if __name__ == '__main__':
	main()	
