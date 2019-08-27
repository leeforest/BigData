#-*-coding:utf-8-*-
# recommand tasty restaurant 
# usage: python tasty_rest.py [gu] [dong]

from multiprocessing import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import openpyxl
import requests
import json
import os
from pprint import pprint
import re
import sys
import place_select
import pandas as pd
from PIL import Image

####################################################
#class for crawling....
####################################################
class REST:
	
	def __init__(self):
		
		self.seoul_dict={}
		self.tmp_list=[]
		self.data=""
		self.tag=""
		self.uniq=""
		self.cat_list=[]
		self.cat_dict={}
	
	#get seoul code from seoul_code.xlsx
	def get_seoul_code(self):
		
		xl=openpyxl.load_workbook('seoul_code.xlsx')
		sh=xl.active
		tmp1_list=[]
		tmp2_list=[]

		for r in sh.rows:
			
			tmp1=r[0].value
			tmp2=r[1].value
			
			#self.seoul_dict: gu->key / dong->value (list type)
			if tmp1==None:
				self.seoul_dict[key]=tmp2_list
				tmp2_list=[]
				continue
			else:
				if tmp1 in tmp1_list:
					tmp2_list.append(tmp2)
				else:	
					tmp1_list.append(tmp1)
					tmp2_list.append(tmp2)
					key=tmp1
		
		'''
		for key in self.seoul_dict:
			print key
			for i in range(len(self.seoul_dict[key])):
				print self.seoul_dict[key][i]
			print '------------------------------'
		'''

	#parsing json data using items
	def get_json_data(self,tag,i):
		
		if tag in self.data["items"][i]:
			data=self.data["items"][i][tag]
			if ',' in data:
				data=data.replace(',','/')
			self.tmp+=data
			self.tmp+=','
		else:
			self.tmp+="N"
			self.tmp+=','

	#start crawling
	def crawling(self):
		
		print '[*] Start Crawling...'
		f=open('test2.csv','w')

		#loop for gu/dong
		for key in self.seoul_dict:
			gu=key
			print '-'+gu

			for i in range(len(self.seoul_dict[key])):
				
				dong=self.seoul_dict[key][i]
				print '  -'+dong
				gu_dong=gu+'+'+dong+'+'+u'맛집'
				
				start=[1,101,201]
				display=100
				
				#send query(start:1,display:100->start:101,display:100 ...)
				for j in range(len(start)):
					url='https://store.naver.com/sogum/api/businesses?start='+str(start[j])+'&display='+str(display)+'&query='+gu_dong+'&sortingOrder=reviewCount'
					
					#get json
					data=requests.get(url)
					self.data=json.loads(data.text,encoding='utf-8')
				
					#parsing data from json 
					for i in range(display):
						self.tmp=gu
						self.tmp+=','
						self.tmp+=dong
						self.tmp+=','
						
						self.get_json_data('name',i)
						self.get_json_data('category',i)
						self.get_json_data('desc',i)
						self.get_json_data('x',i)
						self.get_json_data('y',i)
						self.get_json_data('imageSrc',i)
						self.get_json_data('microReview',i)
						self.get_json_data('commonAddr',i)
						self.get_json_data('addr',i)

						if 'tags' in self.data["items"][i]:
							tags=self.data["items"][i]["tags"]
							tag_tmp='['
							for k in range(len(tags)):
								tag_tmp+=tags[k]
								tag_tmp+='/'
							tag_tmp+=']'
							self.tmp+=tag_tmp
						
						self.tmp+='\n'
						self.tmp=self.tmp.encode('utf-8')
						f.write(self.tmp)
					
			print '--------------------------------------'

	#find columns from csv that match with selected gu,dong
	def searching(self,gu,dong,n):

		print '\n'
		print '[*]'+gu+' '+dong+' 맛집을 검색하고 있습니다...'
		csv=pd.read_csv('crawl.csv',names=['gu','dong','name','category','desc','x','y','imageSrc','microReview','commonAddr','addr','tags'])

		self.uniq=csv.loc[(csv['gu']==gu) & (csv['dong']==dong)]
		self.uniq=self.uniq.head(n)
		
		return self.uniq

	#get category values from self.uniq
	def category(self):

		self.cat_list=self.uniq['category'].values.tolist()
		
		#for escape overlap(using dictionary)
		for cat in self.cat_list:
			self.cat_dict[cat]="1"

		self.cat_list=[]
		for key in self.cat_dict:
			self.cat_list.append(key)

		return self.cat_list


####################################################
#class for gui....
####################################################
class Window(QWidget):
	
	def __init__(self):
		
		QWidget.__init__(self)
		self.list = QListWidget(self)
		layout = QVBoxLayout(self)
		layout.addWidget(self.list)
		self.uniq=""
		self.cat_rest=""

	def add_list_item(self, text):
		
		item = QListWidgetItem(text)
		self.list.addItem(item)
		
		widget = QWidget(self.list)
		button = QToolButton(widget)
		
		layout = QHBoxLayout(widget)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addStretch()
		layout.addWidget(button)
		
		self.list.setItemWidget(item, widget)
		
		#connect button click to button_click
		button.clicked[()].connect(lambda: self.button_click(item))

	#when button(category) clicked, show image and information
	def button_click(self,item):
		
		reload(sys)
		sys.setdefaultencoding('utf-8')
		
		self.uniq=uniq
		self.cat=item.text()
		select_cat=str(self.cat).encode('utf-8')
	
		#send query with gu/dong/category
		self.new=self.uniq.loc[(self.uniq['gu']==gu) & (self.uniq['dong']==dong) & (self.uniq['category']==select_cat)]
		self.new=self.new.head(1)

		#get image
		imageurl=str(self.new['imageSrc'].values)

		imageurl=imageurl.replace('[','')
		imageurl=imageurl.replace(']','')
		imageurl=imageurl.replace('\'','')
		imageurl=imageurl.replace('\'','')
		
		#strooe image as png
		response=requests.get(imageurl)
		if response.status_code==200:
			with open("images/image.png",'wb') as f:
				f.write(response.content)
		
		#multi-processing
		manager=Manager()
		show_image=Process(target=self.show_image)
		show_info=Process(target=self.show_info)
		show_image.start()
		show_info.start()
		show_image.join()
		show_info.join()
	
	#show image about selected restaurant
	def show_image(self):

		img=Image.open('images/image.png')
		img=img.resize((500,500))
		img.save('images/image.png',quality=100)
		os.system('display images/image.png')

	def get_value(self,tag):
		
		for i in tag:
			new_tag=str(i)
		new_tag=new_tag.decode('utf-8')
		
		return new_tag

	#show information about selected restaurant
	def show_info(self):
		
		#get each values
		name=self.new['name'].values
		name=self.get_value(name)

		desc=self.new['desc'].values
		desc=self.get_value(desc)
		
		microReview=self.new['microReview'].values
		microReview=self.get_value(microReview)
		
		commonAddr=self.new['commonAddr'].values
		commonAddr=self.get_value(commonAddr)
		
		addr=self.new['addr'].values
		addr=self.get_value(addr)
		
		tags=self.new['tags'].values
		tags=self.get_value(tags)
	
		info='[*] '+name+'\n'
		info+='- 설명: '+desc+'\n'
		info+='- 리뷰: '+microReview+'\n'
		info+='- 주소: '+commonAddr+' '+addr+'\n'
		info+='- 키워드: '+tags
		
		#show information
		box=QMessageBox()
		box.about(window,self.cat,info)


if __name__=="__main__":
	
	global gu
	global dong
	gu=sys.argv[1]
	dong=sys.argv[2]
	
	rest=REST()
	
	####################################################
	#crawling....
	####################################################
	#rest.get_seoul_code()
	#rest.crawling()
	
	global uniq
	#searching...
	uniq=rest.searching(gu,dong,50)
	
	#searching category
	cat_list=rest.category()

	if os.path.isdir("images") is not True:
		os.system("mkdir images")
	else:
		pass

	#crate button for category
	title=gu+' '+dong+' 맛집 검색'
	title=QString(unicode(title,'utf-8'))
	app=QApplication(sys.argv)
	window=Window()
	
	for cat in cat_list:
		cat=QString(unicode(cat,'utf-8'))
		window.add_list_item(cat)
	window.setWindowTitle(title)
	window.setGeometry(500, 300, 300, 200)
	window.resize(300,400)
	
	#show gui
	window.show()
	sys.exit(app.exec_())


