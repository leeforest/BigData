#-*-coding:utf-8-*-
from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import codecs
import re
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Window(QWidget):
	
	def __init__(self):
		
		QWidget.__init__(self)
		self.list = QListWidget(self)
		layout = QVBoxLayout(self)
		layout.addWidget(self.list)
		self.uniq=""
		self.cat_rest=""

	def add_list(self):
		
		text='a'
		item = QListWidgetItem(text)
		self.list.addItem(item)
		
		widget = QWidget(self.list)
		button = QToolButton(widget)
		
		layout = QHBoxLayout(widget)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addStretch()
		layout.addWidget(button)
		print('1')
		self.list.setItemWidget(item, widget)
		print('2')
		#connect button click to button_click
		button.clicked[()].connect(lambda: self.button_click(item))

	def button_click(self,item):
		pass
			
def main():
	
	app=QApplication(sys.argv)
	qt=Window()
	qt.add_list()
	
if __name__ == '__main__':
	main()	
