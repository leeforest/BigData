#-*-coding:utf-8-*-
from PyQt4.QtGui import *
import matplotlib.pyplot as plt
import matplotlib.legend as legend
import csv
import sys
import os

#Initial Setting
#NUM=sys.argv[1]

class Window(QWidget):
	
	def __init__(self,coin_dic,num):
		QWidget.__init__(self)
		self.list = QListWidget(self)
		layout = QVBoxLayout(self)
		layout.addWidget(self.list)
		self.select_list=[]
		self.cap_list=[]
		self.dic={}
		self.coin_dic=coin_dic
		self.date_list=[]
		self.num=num

	def addListItem(self, text):
		item = QListWidgetItem(text)
		self.list.addItem(item)
		widget = QWidget(self.list)
		button = QToolButton(widget)
		layout = QHBoxLayout(widget)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addStretch()
		layout.addWidget(button)
		self.list.setItemWidget(item, widget)
		button.clicked[()].connect(lambda: self.handleButtonClicked(item))

	def handleButtonClicked(self, item):
		item=item.text()
		self.select_list.append(item)
		self.select_list = list(set(self.select_list))
		print(self.select_list)
		
		if len(self.select_list)==int(self.num):
			self.show_cap_graph()
			
	def show_cap_graph(self):
		cap=[]
		
		file_list=os.listdir('data')
		for file in file_list:
			tmp=file.split('_')
			tmp=tmp[2].replace('.csv','')
			date=tmp.replace(',','/')
			self.date_list.append(date)
		
		selected_index=[]
		for i,v in enumerate(self.select_list):
			index=v.split()[0]
			index=str(index)
			selected_index.append(str(index))
		selected_index.sort()
		
		for v in selected_index:
			self.dic[v]=''
			
		for i,file in enumerate(file_list):
			f=open('data\\'+file, 'r')
			line=csv.reader(f)
			
			for l in line:
				if l[0] in selected_index:
					if type(self.dic[l[0]]) != type(list()):
						self.dic[l[0]]=list()
					tmp=l[3].split()
					'''
					if tmp[1]=='조':
						cap=float(tmp[0])*1000000000000
					if tmp[1]=='십억':
						cap=float(tmp[0])*1000000000
					if tmp[1]=='백만':
						cap=float(tmp[0])*1000000
					'''	
					if tmp[1]=='조':
						cap=float(tmp[0])*1000000
					if tmp[1]=='십억':
						cap=float(tmp[0])*1000
					if tmp[1]=='백만':
						cap=float(tmp[0])*1
						
					self.dic[l[0]].append(cap)
		
		#get selected_coin_list's market cap
		#print(self.dic)
		
		#graph_x=['12/12','12/13','12/14','12/15']
		graph_x=self.date_list
		length=len(self.dic)
		for k,v in self.dic.items():
			coin_name=self.coin_dic[k]
			graph_y=v #lsit
			plt.plot(graph_x,graph_y,label=coin_name)
			plt.xlabel('date')
			plt.ylabel('Marcat Cap')
			plt.title('Coin Marcap Cap')
			plt.legend(bbox_to_anchor=(1.1,1.05))
		plt.show()
	
def get_coin_list():
	coin_list=[]
	f=open('coin_list.csv', 'r')
	line=csv.reader(f)
	
	count=0
	for i,v in enumerate(line):
		if i==0:
			pass
		else:
			str=v[0]+' '+v[1]+'('+v[2]+')'
			coin_list.append(str)
			count+=1
		
		if count>30:
			break
	
	return coin_list
			
def main():
	
	num=sys.argv[1]
	coin_list=get_coin_list()
	coin_dic={}
	for coin in coin_list:
		tmp=coin.split()
		coin_dic[tmp[0]]=tmp[1]
	
	app = QApplication(sys.argv)
	window = Window(coin_dic,num)
	for label in coin_list:
		window.addListItem(label)
	window.setGeometry(500, 300, 300, 200)
	window.show()
	sys.exit(app.exec_())
	
	
if __name__ == '__main__':
	main()