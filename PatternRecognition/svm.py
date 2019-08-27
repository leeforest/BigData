import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import svm

class support_vector_machine:

	def __init__(self):
		self.X_tr=[]
		self.Y_tr=[]
		self.X_te=[]
		self.Y_te=[]


	def get_data(self):
		with open('datasets/dataset3_1/tr_x.txt','r') as f:
			tmp_list=f.read().split('\n')
			tmp_list.pop()
			self.X_tr=np.zeros((len(tmp_list),2))
			for i in range(len(tmp_list)):
				tmp=tmp_list[i].split(' ')
				self.X_tr[i][0]=float(tmp[0])
				self.X_tr[i][1]=float(tmp[1])
		
		with open('datasets/dataset3_1/tr_t.txt','r') as f:
			self.Y_tr=f.read().split('\n')
			self.Y_tr.pop()
			self.Y_tr=np.array(self.Y_tr)
			self.Y_tr=self.Y_tr.astype(np.float).T
		
		with open('datasets/dataset3_1/te_x.txt','r') as f:
			tmp_list=f.read().split('\n')
			tmp_list.pop()
			self.X_te=np.zeros((len(tmp_list),2))
			for i in range(len(tmp_list)):
				tmp=tmp_list[i].split(' ')
				self.X_te[i][0]=float(tmp[0])
				self.X_te[i][1]=float(tmp[1])
		
		with open('datasets/dataset3_1/te_t.txt','r') as f:
			self.Y_te=f.read().split('\n')
			self.Y_te.pop()
			self.Y_te=np.array(self.Y_te)
			self.Y_te=self.Y_te.astype(np.float).T

		#train data plot
		plt.figure(1)
		plt.title('SVM-train data')
		for i in range(len(self.Y_tr)):
			if self.Y_tr[i]==0:
				plt.scatter(self.X_tr[i][0],self.X_tr[i][1],color='blue')
			if self.Y_tr[i]==1:
				plt.scatter(self.X_tr[i][0],self.X_tr[i][1],color='red')
		
		#test data plot
		plt.figure(2)
		plt.title('SVM-test data')
		for i in range(len(self.Y_te)):
			if self.Y_te[i]==0:
				plt.scatter(self.X_te[i][0],self.X_te[i][1],color='blue')
			if self.Y_te[i]==1:
				plt.scatter(self.X_te[i][0],self.X_te[i][1],color='red')
		
		
	def SVM(self):
		sv=svm.SVC(kernel='linear',C=1000)
		result=sv.fit(self.X_tr,self.Y_tr)

		#train data classification plot(with dicision surface,support vector)
		plt.figure(3)
		plt.title('SVM-train data Classification')
		predict1=sv.predict(self.X_tr)
		predict1=predict1.astype(float)

		for i in range(100):
			if predict1[i]==0:
				plt.scatter(self.X_tr[:,0][i],self.X_tr[:,1][i],color='blue')
			if predict1[i]==1:
				plt.scatter(self.X_tr[:,0][i],self.X_tr[:,1][i],color='red')	
		
		#decision surface,support vector plot
		x=np.linspace(-3,2,100)
		y=np.linspace(-2,5,100)
		X,Y=np.meshgrid(x,y)
		xy=np.vstack([X.ravel(),Y.ravel()]).T
		Z=sv.decision_function(xy).reshape(X.shape)
		plt.contour(X,Y,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestypes=['--','-','--'])
		plt.scatter(sv.support_vectors_[:,0],sv.support_vectors_[:,1],facecolors='yellow',edgecolor='k')

		#test data classification plot(with dicision surface,support vector)
		plt.figure(4)
		plt.title('SVM-test data Classification')
		predict2=sv.predict(self.X_te)
		predict2=predict2.astype(float)
		for i in range(len(predict2)):
			if predict2[i]==0:
				plt.scatter(self.X_te[:,0][i],self.X_te[:,1][i],color='blue')
			if predict2[i]==1:
				plt.scatter(self.X_te[:,0][i],self.X_te[:,1][i],color='red')	
		
		#decision surface,support vector plot
		plt.contour(X,Y,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestypes=['--','-','--'])
		plt.scatter(sv.support_vectors_[:,0],sv.support_vectors_[:,1],facecolors='yellow',edgecolor='k')
		plt.show()
				
		#accuracy,true positive,false positive,false negative,true negative of train data
		tp=0; fp=0; tn=0; fn=0
		for i in range(len(predict1)):
			if predict1[i]==self.Y_tr[i]:
				if predict1[i]==1:
					tp+=1
				if predict1[i]==0:
					tn+=1
			if predict1[i]!=self.Y_tr[i]:
				if predict1[i]==1:
					fn+=1
				if predict1[i]==0:
					fp+=1
			
		print '\n[*] train data accuracy'
		print 'accuracy:',(float(tp+tn)/len(predict1))*100,'%'
		print 'true positive:',(float(tp)/len(predict1))*100,'%'
		print 'false positive:',(float(fp)/len(predict1))*100,'%'
		print 'true negative:',(float(tn)/len(predict1))*100,'%'
		print 'false negative:',(float(fn)/len(predict1))*100,'%'
		
		#accuracy,true positive,false positive,false negative,true negative of test data
		tp=0; fp=0; tn=0; fn=0
		for i in range(len(predict2)):
			if predict2[i]==self.Y_te[i]:
				if predict2[i]==1:
					tp+=1
				if predict2[i]==0:
					tn+=1
			if predict2[i]!=self.Y_te[i]:
				if predict2[i]==1:
					fn+=1
				if predict2[i]==0:
					fp+=1
			
		print '\n[*] train data accuracy'
		print 'accuracy:',(float(tp+tn)/len(predict2))*100,'%'
		print 'true positive:',(float(tp)/len(predict2))*100,'%'
		print 'false positive:',(float(fp)/len(predict2))*100,'%'
		print 'true negative:',(float(tn)/len(predict2))*100,'%'
		print 'false negative:',(float(fn)/len(predict2))*100,'%'
	

def main():
	
	s=support_vector_machine()
	s.get_data()
	s.SVM()

if __name__=="__main__":
	
	main()
