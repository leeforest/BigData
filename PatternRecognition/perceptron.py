import numpy as np
import scipy.special
import scipy.ndimage
import matplotlib.pyplot as plt

class perceptron:

	def __init__(self):

		#self.weight=np.random.rand(1,1)-0.3
		#self.weight=float(self.weight[0])
		#self.bias=np.random.rand(1,1)-0.3
		#self.bias=float(self.bias[0])
		self.weight=-0.1
		self.bias=-0.2
		self.lr=0.1


	def get_data(self):		
		with open('datasets/dataset1/tr_x.txt','r') as f:
			self.X_tr=f.read().split('\n')
			self.X_tr.pop()
			for i in range(len(self.X_tr)):
				self.X_tr[i]=float(self.X_tr[i])

		with open('datasets/dataset1/tr_t.txt','r') as f:
			self.Y_tr=f.read().split('\n')
			self.Y_tr.pop()
			for i in range(len(self.Y_tr)):
				self.Y_tr[i]=float(self.Y_tr[i])

		with open('datasets/dataset1/te_x.txt','r') as f:
			self.X_te=f.read().split('\n')
			self.X_te.pop()
			for i in range(len(self.X_te)):
				self.X_te[i]=float(self.X_te[i])

		with open('datasets/dataset1/te_t.txt','r') as f:
			self.Y_te=f.read().split('\n')
			self.Y_te.pop()
			for i in range(len(self.Y_te)):
				self.Y_te[i]=float(self.Y_te[i])
		
		#train data plot
		plt.figure(1)
		plt.title('Perceptron-train data')
		for i in range(len(self.X_tr)):
			if self.Y_tr[i]==0:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='blue')
		for i in range(len(self.X_tr)):
			if self.Y_tr[i]==1:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='red')
		
		#test data plot
		plt.figure(2)
		plt.title('Perceptron-test data')
		for i in range(len(self.X_te)):
			if self.Y_te[i]==0:
				plt.scatter(self.X_te[i],self.Y_te[i],color='blue')
		for i in range(len(self.X_te)):
			if self.Y_te[i]==1:
				plt.scatter(self.X_te[i],self.Y_te[i],color='red')
		

	#activation function: step function
	def activation_function(self,x):
		if x>=0:
			return 1
		if x<0:
			return -1


	def train(self):		
		while True:
			error_list=list()
			for i in range(len(self.X_tr)):
				y=self.activation_function(self.X_tr[i]*self.weight+self.bias)
				
				#calculate error
				error=self.Y_tr[i]-y
				
				#update weight,bias
				self.weight=self.weight+(self.lr*self.X_tr[i]*error)
				self.bias=self.bias+(self.lr*error)
				
				#least square
				error_list.append(pow(error,2))	
			error_sum=sum(error_list)
			if error_sum<=65:
				break

		print '[*] train result'
		print '- weight:',self.weight
		print '- bias:',self.bias
		print '- learning rate',self.lr
	
		#train data classification plot
		plt.figure(3)
		plt.title('Perceptron-train data Classification')
		
		predict=[]
		for i in range(len(self.X_tr)):
			if self.Y_tr[i]>self.X_tr[i]*self.weight+self.bias:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='red')
				predict.append(1)
			if self.Y_tr[i]<self.X_tr[i]*self.weight+self.bias:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='blue')
				predict.append(0)
		
		X=np.linspace(-4,4,100)
		plt.plot(X,X*self.weight+self.bias,color='black')
	
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
			
		print '\n[*] train data accuracy'
		print 'accuracy:',(float(tp+tn)/len(predict))*100,'%'
		print 'true positive:',(float(tp)/len(predict))*100,'%'
		print 'false positive:',(float(fp)/len(predict))*100,'%'
		print 'true negative:',(float(tn)/len(predict))*100,'%'
		print 'false negative:',(float(fn)/len(predict))*100,'%'
	
		
	def test(self):		
		#test data classification plot(using trained weight,bias)
		plt.figure(4)
		plt.title('Perceptron-test data Classification')
		
		predict=[]
		for i in range(len(self.X_te)):
			if self.Y_te[i]>self.X_te[i]*self.weight+self.bias:
				plt.scatter(self.X_te[i],self.Y_te[i],color='red')
				predict.append(1)
			if self.Y_te[i]<self.X_te[i]*self.weight+self.bias:
				plt.scatter(self.X_te[i],self.Y_te[i],color='blue')
				predict.append(0)
		
		X=np.linspace(-4,4,100)
		plt.plot(X,X*self.weight+self.bias,color='black')
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
	
	pc=perceptron()
	pc.get_data()
	pc.train()
	pc.test()

if __name__=='__main__':
		main()
