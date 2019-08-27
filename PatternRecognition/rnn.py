#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class RNN:
	def __init__(self,n_input,n_hidden,n_out,lr,seq_max_len):
	
		self.n_input=n_input
		self.n_hidden=n_hidden
		self.n_out=n_out
		self.lr=lr
		self.seq_max_len=seq_max_len

	def build(self):
		
		self.x=tf.placeholder(tf.float32,[None,self.seq_max_len,self.n_input])
		self.y_=tf.placeholder(tf.int32,[None])
		self.seq_len=tf.placeholder(tf.int32,[None])
		
		self.w = tf.Variable(tf.random_normal([self.n_hidden,self.n_out]))
		self.b = tf.Variable(tf.random_normal([self.n_out]))

		#generate LSTM cell
		cell=rnn.BasicLSTMCell(self.n_hidden)
		cell=rnn.DropoutWrapper(cell, output_keep_prob=0.7)
		outputs,states=tf.nn.dynamic_rnn(cell,self.x,dtype=tf.float32,sequence_length=self.seq_len)

		#calculate self.y
		outputs = tf.transpose(outputs, [1, 0, 2])
		outputs = outputs[-1]
		self.y=tf.matmul(outputs,self.w)+self.b
	
		#loss&train
		self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y,labels=self.y_))
		self.train=tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		
		#predict&accuracy
		self.correct_pred=tf.cast(tf.argmax(self.y,1),tf.int32)
		self.pred_check=tf.equal(self.correct_pred,self.y_)
		self.accuracy=tf.reduce_mean(tf.cast(self.pred_check,tf.float32))

def data_load():
	
	x=[]
	with open('text.txt','r') as f:

		tmp=f.read().split('\n')
		tmp.pop()

	############append
	for word in tmp:
		word=word.strip('\r')
		if len(word) !=1:
			x.append(word)
	return x

def make_input(seq_data,num_dic):
	
	input_list=[]
	input_list=[seq[0:-1] for seq in seq_data]	
	return input_list

def make_label(seq_data,num_dic):

	label_list=[]
	for seq in seq_data:
		label=num_dic[seq[-1]]
		label_list.append(label)
	return label_list

def input_encoding(seq_data,num_dic,dic_len):
	
	input_list=[]
	input_index=[]
	tmp_list=[]

	#get index of input
	for seq in seq_data:
		input_index=[num_dic[n] for n in seq]
		enc=np.eye(dic_len)[input_index]
		
		for i in range(len(input_index)):
			
			if enc[i][26]>0:
				enc[i][26]=float(0)
		
		input_list.append(enc)
		#input_list.append(np.eye(dic_len)[input_index])
	
	
	
	return input_list
	
def make_batch(num,tr_x,tr_y,seq_data,total_data):	
	data=[]
	idx=np.arange(0,len(tr_x))
	np.random.shuffle(idx)
	idx=idx[:num]
	x_shuffle=[tr_x[i] for i in idx]
	y_shuffle=[tr_y[i] for i in idx]
	seq_len=[len(seq_data[i]) for i in idx]	
	#x_data=[seq_data[i] for i in idx]
	data=[total_data[i] for i in idx]

	x_shuffle=np.asarray(x_shuffle)
	y_shuffle=np.asarray(y_shuffle)
	seq_len=np.asarray(seq_len)
	
	return x_shuffle,y_shuffle,seq_len,data

def padding(seq_data,seq_max_len):
	
	pad_seq_data=[]
	for i in range(len(seq_data)):
		seq=str(seq_data[i])
		if len(seq)<seq_max_len:
			pad_len=seq_max_len-len(seq)
			tmp=pad_len*"@"
			seq=seq+tmp
			pad_seq_data.append(seq)
		else:
			pad_seq_data.append(seq)
	
	return pad_seq_data

def main():
	char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
				'h', 'i', 'j', 'k', 'l', 'm', 'n',
				'o', 'p', 'q', 'r', 's', 't', 'u',
				'v', 'w', 'x', 'y', 'z', '@']

	#{'a':0,'b':1...}
	num_dic = {n: i for i, n in enumerate(char_arr)}
	dic_len = len(num_dic)

	#data load
	total_data=data_load()

	#get max len
	seq_max_len,seq_max_ele=max([(len(x),x) for x in total_data])
	seq_max_len+=-1

	#make label
	tr_y=make_label(total_data,num_dic)
	
	#remove last word for label
	input_data=make_input(total_data,num_dic)

	#padding
	pad_tr_x=padding(input_data,seq_max_len)

	#encoding
	tr_x=input_encoding(pad_tr_x,num_dic,dic_len)

	#parameter
	n_input=27
	n_hidden=256
	n_out=26
	lr=0.01

	with tf.Session() as sess:
		
		rnn=RNN(n_input=n_input,n_hidden=n_hidden,n_out=n_out,lr=lr,seq_max_len=seq_max_len)
		rnn.build()
		sess.run(tf.global_variables_initializer())
		
		#start train
		for i in range(5000):
			
			tr_x_batch,tr_y_batch,seq_len,data=make_batch(500,tr_x,tr_y,input_data,total_data)
			sess.run(rnn.train,feed_dict={rnn.x:tr_x_batch,rnn.y_:tr_y_batch,rnn.seq_len:seq_len})

			if i%50==0:
				predict,accuracy=sess.run([rnn.correct_pred,rnn.accuracy],feed_dict={rnn.x:tr_x_batch,rnn.y_:tr_y_batch,rnn.seq_len:seq_len})
				print i,"[train] accuracy: ",round(accuracy*100,2),'%'
				
				#test data
				te_x_batch,te_y_batch,te_seq_len,te_data=make_batch(100,tr_x,tr_y,input_data,total_data)
				predict_words = []
				te_predict,te_accuracy=sess.run([rnn.correct_pred,rnn.accuracy],feed_dict={rnn.x:te_x_batch,rnn.y_:te_y_batch,rnn.seq_len:te_seq_len})
				print '[test] accuracy:',round(te_accuracy*100,2),'%'

if __name__ == '__main__':
    main()
