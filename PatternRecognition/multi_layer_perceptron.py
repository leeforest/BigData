import numpy as np
import scipy.special
import scipy.ndimage
import math
import matplotlib.pyplot as plt

class multi_layer_perceptron:

	def __init__(self,dataset):
		
		self.dataset=dataset
		#set hidden nodes and learning rate
		self.hnodes=3
		self.lr=0.2
		  
		#initiate weight,bias
		self.w_in_hid=np.random.rand(3,1)
		self.w_hid_out=np.random.rand(3,1)
		self.bias=0.1

		#activation function: sigmoid
		self.activation_function=lambda x:scipy.special.expit(x)
	
	def get_data(self):
		
		with open('datasets/dataset'+self.dataset+'/tr_x.txt','r') as f:
			tmp_list=f.read().split('\n')
			tmp_list.pop()
			self.X_tr=np.zeros((len(tmp_list),1))
			self.Y_tr=np.zeros((len(tmp_list),1))
			for i in range(len(tmp_list)):
				tmp=tmp_list[i].split(' ')
				self.X_tr[i]=float(tmp[0])
				self.Y_tr[i]=float(tmp[1])
		
		with open('datasets/dataset'+self.dataset+'/tr_t.txt','r') as f:
			self.L_tr=f.read().split('\n')
			self.L_tr.pop()
			self.L_tr=np.array(self.L_tr)
			self.L_tr=self.L_tr.astype(np.float).T
		
		with open('datasets/dataset'+self.dataset+'/te_x.txt','r') as f:
			tmp_list=f.read().split('\n')
			tmp_list.pop()
			self.X_te=np.zeros((len(tmp_list),1))
			self.Y_te=np.zeros((len(tmp_list),1))
			for i in range(len(tmp_list)):
				tmp=tmp_list[i].split(' ')
				self.X_te[i]=float(tmp[0])
				self.Y_te[i]=float(tmp[1])
		
		with open('datasets/dataset'+self.dataset+'/te_t.txt','r') as f:
			self.L_te=f.read().split('\n')
			self.L_te.pop()
			self.L_te=np.array(self.L_te)
			self.L_te=self.L_te.astype(np.float).T
		
		#train data plot
		plt.figure(1)
		plt.title('Multi Layer Perceptron-train data')
		for i in range(len(self.X_tr)):
			if self.L_tr[i]==0:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='blue')
		for i in range(len(self.X_tr)):
			if self.L_tr[i]==1:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='red')
		
		#test data plot
		plt.figure(2)
		plt.title('Multi Layer Perceptron-test data')
		for i in range(len(self.X_te)):
			if self.L_te[i]==0:
				plt.scatter(self.X_te[i],self.Y_te[i],color='blue')
		for i in range(len(self.X_te)):
			if self.L_te[i]==1:
				plt.scatter(self.X_te[i],self.Y_te[i],color='red')

	
	def train(self):
		while True:
			error_sum=[]
			for i in range(len(self.X_tr)):
				#calculate output:input layer->hidden layer->output layer
				hidden_input=self.X_tr[i]*self.w_in_hid
				hidden_output=self.activation_function(hidden_input+self.bias)
				final_input=np.dot(self.w_hid_out.T,hidden_output)
				final_output=self.activation_function(final_input+self.bias)

				#calculate error
				output_error=self.Y_tr[i]-final_output
				hidden_error=np.dot(self.w_hid_out,output_error) 

				#weight,bias update
				self.w_hid_out+=self.lr*output_error+hidden_output
				self.w_in_hid+=self.lr*hidden_error+self.X_tr[i]
				self.bias+=self.lr*output_error
					
				#least square
				error_sum.append(pow(output_error,2))
			error_sum=sum(error_sum)
			print error_sum
			if error_sum<402:
				break

		#train data classification plot
		plt.figure(3)
		plt.title('Multi Layer Perceptron-train data Classification')

		predict=[]
		for i in range(len(self.X_tr)):
			result=self.calculate(self.X_tr[i])
			if self.Y_tr[i]>result:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='red')
				predict.append(0)
			if self.Y_tr[i]<=result:
				plt.scatter(self.X_tr[i],self.Y_tr[i],color='blue')
				predict.append(1)

		#decision surface plot
		Y=[]
		X=np.linspace(-3,2,100)
		for i in range(len(X)):
			result=self.calculate(X[i])
			result=float(result[0][0])
			Y.append(result)
		plt.plot(X,Y,color='black')
		plt.show()
		
		#accuracy,true positive,false positive,false negative,true negative of train data
		tp=0; fp=0; tn=0; fn=0
		for i in range(len(predict)):
			if predict[i]==self.L_tr[i]:
				if predict[i]==1:
					tp+=1
				if predict[i]==0:
					tn+=1
			if predict[i]!=self.L_tr[i]:
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
		#train data classification plot
		plt.figure(4)
		plt.title('Multi Layer Perceptron-test data Classification')

		predict=[]
		for i in range(len(self.X_te)):
			result=self.calculate(self.X_te[i])
			if self.Y_te[i]>=result:
				plt.scatter(self.X_te[i],self.Y_te[i],color='red')
				predict.append(0)
			if self.Y_te[i]<result:
				plt.scatter(self.X_te[i],self.Y_te[i],color='blue')
				predict.append(1)

		#decision surface plot
		Y=[]
		X=np.linspace(-3,2,100)
		for i in range(len(X)):
			result=self.calculate(X[i])
			result=float(result[0][0])
			Y.append(result)
		plt.plot(X,Y,color='black')
		plt.show()
		
		#accuracy,true positive,false positive,false negative,true negative of train data
		tp=0; fp=0; tn=0; fn=0
		for i in range(len(predict)):
			if predict[i]==self.L_te[i]:
				if predict[i]==1:
					tp+=1
				if predict[i]==0:
					tn+=1
			if predict[i]!=self.L_te[i]:
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
	
	#final_output calculater
	def calculate(self,X):
		hidden_input=X*self.w_in_hid
		hidden_output=self.activation_function(hidden_input+self.bias)
		final_input=np.dot(self.w_hid_out.T,hidden_output)
		final_output=self.activation_function(final_input+self.bias)

		return final_output

   
def main():
   
	mlp=multi_layer_perceptron('3_1')
	mlp.get_data()
	mlp.train()
	mlp.test()
	'''
	mlp=multi_layer_perceptron('3_2')
	mlp.get_data()
	mlp.train()
	mlp.test()
	'''
if __name__=='__main__':
      main()
