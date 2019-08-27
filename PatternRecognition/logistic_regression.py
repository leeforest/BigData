import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import linear_model

class logistic:

	def __init__(self):
		self.X_tr=[]
		self.Y_tr=[]
		self.X_te=[]
		self.Y_te=[]


	def get_data(self):
		with open('datasets/dataset1/tr_x.txt','r') as f:
			self.X_tr=f.read().split('\n')
			self.X_tr.pop()
			self.X_tr=np.array(self.X_tr).reshape(1,-1)
			self.X_tr=self.X_tr.astype(np.float).T

		with open('datasets/dataset1/tr_t.txt','r') as f:
			self.Y_tr=f.read().split('\n')
			self.Y_tr.pop()
			self.Y_tr=np.array(self.Y_tr)
			self.Y_tr=self.Y_tr.astype(np.float).T

		with open('datasets/dataset1/te_x.txt','r') as f:
			self.X_te=f.read().split('\n')
			self.X_te.pop()
			self.X_te=np.array(self.X_te).reshape(1,-1)
			self.X_te=self.X_te.astype(np.float).T

		with open('datasets/dataset1/te_t.txt','r') as f:
			self.Y_te=f.read().split('\n')
			self.Y_te.pop()
			self.Y_te=np.array(self.Y_te)
			self.Y_te=self.Y_te.astype(np.float).T
		
		#train data plot
		plt.figure(1)
		plt.title('Logistic regression-train data')
		for i in range(len(self.X_tr)):
			if self.Y_tr[i]==0:
				plt.scatter(self.X_tr[i],0,color='blue')
		for i in range(len(self.X_tr)):
			if self.Y_tr[i]==1:
				plt.scatter(self.X_tr[i],0,color='red')
		
		#test data plot
		plt.figure(2)
		plt.title('Logistic regression-test data')
		for i in range(len(self.X_te)):
			if self.Y_te[i]==0:
				plt.scatter(self.X_te[i],0,color='blue')
		for i in range(len(self.X_te)):
			if self.Y_te[i]==1:
				plt.scatter(self.X_te[i],0,color='red')
	
	
	def model(self,x):
		return 1/(1+np.exp(-x))
		
	
	def logistic_regression(self):
		#train the model with X_tr and Y_tr
		lr=linear_model.LogisticRegression()
		lr.fit(self.X_tr,self.Y_tr)
		
		#train data classification plot
		predict=[]
		plt.figure(3)
		plt.title('Logistic regression-train data Classification')
		for i in range(len(self.X_tr)):
			if self.Y_tr[i]>self.model(self.X_tr[i]*lr.coef_+lr.intercept_):
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='blue')
				predict.append(1)
			if self.Y_tr[i]<self.model(self.X_tr[i]*lr.coef_+lr.intercept_):
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='red')
				predict.append(0)

		X=np.linspace(-5,5,100)
		line=self.model(X*lr.coef_+lr.intercept_).ravel()
		plt.plot(X,line,color='black')
		
		#accuracy,true positive,false positive,false negative,true negative of train data
		tp=0; fp=0; tn=0; fn=0
		for i in range(len(predict)):
			if predict[i]==self.Y_tr[i]:
				if predict[i]==1:
					tp+=1
				if predict[i]==0:
					tn+=1
			if predict[i]!=self.Y_tr[i]:
				if predict[i]==1:
					fn+=1
				if predict[i]==0:
					fp+=1
			
		print '[*] train data accuracy'
		print 'accuracy:',(float(tp+tn)/len(predict))*100,'%'
		print 'true positive:',(float(tp)/len(predict))*100,'%'
		print 'false positive:',(float(fp)/len(predict))*100,'%'
		print 'true negative:',(float(tn)/len(predict))*100,'%'
		print 'false negative:',(float(fn)/len(predict))*100,'%'
	
		#test data classification plot
		predict=[]
		plt.figure(4)
		plt.title('Logistic regression-train data Classification')
		for i in range(len(self.X_te)):
			if self.Y_te[i]>self.model(self.X_te[i]*lr.coef_+lr.intercept_):
				plt.scatter(self.X_te[i],self.Y_te[i],color='blue')
				predict.append(1)
			if self.Y_te[i]<self.model(self.X_te[i]*lr.coef_+lr.intercept_):
				plt.scatter(self.X_te[i],self.Y_te[i],color='red')
				predict.append(0)
		
		X=np.linspace(-5,5,100)
		line=self.model(X*lr.coef_+lr.intercept_).ravel()
		plt.plot(X,line,color='black')
		plt.show()

		#accuracy,true positive,false positive,false negative,true negative of test data
		tp=0; fp=0; tn=0; fn=0
		for i in range(len(predict)):
			if predict[i]==self.Y_te[i]:
				if predict[i]==1:
					tp+=1
				if predict[i]==0:
					tn+=1
			if predict[i]!=self.Y_te[i]:
				if predict[i]==1:
					fn+=1
				if predict[i]==0:
					fp+=1
			
		print '\n[*] test data accuracy'
		print 'accuracy:',(float(tp+tn)/len(predict))*100,'%'
		print 'true positive:',(float(tp)/len(predict))*100,'%'
		print 'false positive:',(float(fp)/len(predict))*100,'%'
		print 'true negative:',(float(tn)/len(predict))*100,'%'
		print 'false negative:',(float(fn)/len(predict))*100,'%'
		

def main():
	
	l=logistic()
	l.get_data()
	l.logistic_regression()

if __name__=="__main__":
	
	main()
