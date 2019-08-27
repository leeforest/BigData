import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import accuracy_score

class linear:

	def __init__(self):
		self.X_tr=[]
		self.Y_tr=[]
		self.X_te=[]
		self.Y_te=[]

	
	def get_data(self):
		with open('datasets/dataset2/tr_x.txt','r') as f:
			self.X_tr=f.read().split('\n')
			self.X_tr.pop()
			self.X_tr=np.array(self.X_tr).reshape(1,-1)
			self.X_tr=self.X_tr.astype(np.float).T

		with open('datasets/dataset2/tr_y.txt','r') as f:
			self.Y_tr=f.read().split('\n')
			self.Y_tr.pop()
			self.Y_tr=np.array(self.Y_tr)
			self.Y_tr=self.Y_tr.astype(np.float).T
		
		with open('datasets/dataset2/te_x.txt','r') as f:
			self.X_te=f.read().split('\n')
			self.X_te.pop()
			self.X_te=np.array(self.X_te).reshape(1,-1)
			self.X_te=self.X_te.astype(np.float).T

		with open('datasets/dataset2/te_y.txt','r') as f:
			self.Y_te=f.read().split('\n')
			self.Y_te.pop()
			self.Y_te=np.array(self.Y_te)
			self.Y_te=self.Y_te.astype(np.float).T
		
		#train data plot
		plt.figure(1)
		plt.title('Linear regression-train data')
		plt.scatter(self.X_tr,self.Y_tr)
		
		#test data plot
		plt.figure(2)
		plt.title('Linear regression-test data')
		plt.scatter(self.X_te,self.Y_te)
		

	def model(self,x):
		return 1/(1+np.exp(-x))

		
	def linear_regression(self):
		#train the model with X_tr and Y_tr
		line=linear_model.LinearRegression(normalize=True)
		line.fit(self.X_tr,self.Y_tr)

		#regression line in train data plot
		plt.figure(3)
		plt.title('Regression line in train data')
		X=np.linspace(0,2,100)
		plt.plot(X,line.coef_*X+line.intercept_,color='black')

		for i in range(len(self.X_tr)):
			if self.Y_tr[i]>=line.coef_*self.X_tr[i]+line.intercept_:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='red')
			if self.Y_tr[i]<line.coef_*self.X_tr[i]+line.intercept_:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='blue')
		
		#regression line in test data plot
		plt.figure(4)
		plt.title('Regression line in test data')
		X=np.linspace(0,2,100)
		plt.plot(X,line.coef_*X+line.intercept_,color='black')

		for i in range(len(self.X_te)):
			if self.Y_te[i]>=line.coef_*self.X_te[i]+line.intercept_:
				plt.scatter(self.X_te[i],self.Y_te[i],color='red')
			if self.Y_te[i]<line.coef_*self.X_te[i]+line.intercept_:
				plt.scatter(self.X_te[i],self.Y_te[i],color='blue')

		plt.show()


def main():
	
	line=linear()
	line.get_data()
	line.linear_regression()

if __name__=="__main__":
	
	main()
