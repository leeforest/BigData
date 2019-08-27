#-*-coding:utf-8-*-
'''
[*] Imagenet Data Classification using Deep Learning
image data from imagenet
-train data: 100,000
-val data: 10,000
-test data: random sample
-label: 200

'''

import tensorflow as tf
import numpy as np
import os
from skimage.transform import resize
import cv2

#load train and val data
def load_data():
	
	#load train data
	array_list=[]
	train_dir_list=[]
	root_dir="D:\\forest\\dataset\\tiny-imagenet-200\\train\\"
	train_dir_list=os.listdir(root_dir)
	
	for dir in train_dir_list:
		dir=root_dir+dir
		train_file_list=os.listdir(dir)
		
		for file in train_file_list:
			file=dir+"/"+file
			im=cv2.imread(file)
			tmp=np.asarray(im)
			tmp=cv2.resize(tmp,(32,32))
			tmp=tmp.flatten()
			if tmp.ndim==2:
				tmp=np.broadcast_to(tmp[...,np.newaxis],(64,64,3))
				tmp=cv2.resize(tmp,(32,32))
			tmp_flat=tmp.flatten()
			array_list.append(tmp_flat)
	
	total_train_array=np.asarray(array_list)
	
	#load val data
	array_list=[]
	root_dir='D:\\forest\\dataset\\tiny-imagenet-200\\val\\images\\'
	val_file_list=os.listdir(root_dir)
	for file in val_file_list:
		file=root_dir+file
		im=cv2.imread(file)
		tmp=np.asarray(im)
		tmp=cv2.resize(tmp,(32,32))
		if tmp.ndim==2:
			tmp=np.broadcast_to(tmp[...,np.newaxis],(64,64,3))
			tmp=cv2.resize(tmp,(32,32))
		tmp_flat=tmp.flatten()
		array_list.append(tmp_flat)
	
	total_val_array=np.asarray(array_list)
	
	return total_train_array,train_dir_list,total_val_array

#load label(words.txt->all label at imagenet/wnids.txt->matching)
def load_label():
	
	all_dirname_label_dict={}
	dirname_list=[]
	label_list=[]
	with open('D:\\forest\\dataset\\tiny-imagenet-200\\words.txt','r') as f:
		label=f.read().split('\n')
		label.pop()
		for i in range(len(label)):
			tmp=label[i].split('\t')
			dirname_list.append(tmp[0])
			label_list.append(tmp[1])
			all_dirname_label_dict[tmp[0]]=tmp[1]

	dirname_label_dict={}
	with open('D:\\forest\\dataset\\tiny-imagenet-200\\wnids.txt','r') as f:
		dirname=f.read().split('\n')
		dirname.pop()
		for i in range(len(dirname)):
			key=dirname[i]
			value=all_dirname_label_dict[key]
			dirname_label_dict[key]=value
	#print(dirname_label_dict)
	
	return dirname_label_dict


def load_val_label():
	val_image_list=[]
	val_label_list=[]
	val_image_label_dict={}
	with open('D:\\forest\\dataset\\tiny-imagenet-200\\val\\labels.txt','r') as f:
		label=f.read().split('\n')
		label.pop()
		for i in range(len(label)):
			tmp=label[i].split('\t')
			val_image_list.append(tmp[0])
			val_label_list.append(tmp[1])
			val_image_label_dict[tmp[0]]=tmp[1]
	return val_image_label_dict,val_label_list

#return next batch
def next_batch(num, data, labels):
	
	idx = np.arange(0,len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[i] for i in idx]
	labels_shuffle = [labels[i] for i in idx]
	
	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

	
def build_MLP_classifier(_X,_weights,_biases,keep_in_prob,keep_hidden_prob):
    
	layer1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), keep_in_prob)
	layer2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2'])), keep_hidden_prob)
	layer3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer2, _weights['h3']), _biases['b3'])), keep_hidden_prob)
	out = tf.nn.tanh(tf.add(tf.matmul(layer3,_weights['out']), _biases['out']))
	y_pred = tf.nn.softmax(out)
	return out,y_pred

#--------------------------------------------------------------------------------------

# Net parameters
n_input = 3072            
n_hidden_1 = 3000    
n_hidden_2 = 3000 
n_hidden_3 = 3000        
n_classes = 200

weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],stddev=0.1)),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],stddev=0.1)),
	'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],stddev=0.1)),
	'out': tf.Variable(tf.random_normal([n_hidden_3,n_classes],stddev=0.1)),
}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

X = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_in_prob = tf.placeholder(tf.float32)
keep_hidden_prob = tf.placeholder(tf.float32)

logits,y_pred=build_MLP_classifier(X, weights, biases, keep_prob)

#data load
y_train=[]
x_train,dir_list,x_val=load_data()
dir_array=np.asarray(dir_list)

print(x_train.shape)
int_dirname_dict={}
for i in range(len(dir_array)):
	int_dirname_dict[i]=dir_list[i]
	
	for j in range(500):
		y_train.append(i)

y_train=np.asarray(y_train)
y_train=y_train.reshape(100000,1)
y_train_one_hot=tf.squeeze(tf.one_hot(y_train,200),axis=1)
dirname_label_dict=load_label()

#loss function,Optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) # softmax loss
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

#accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#----------------------------------------------------------------------

config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.7

#start tensorflow session
saver=tf.train.Saver()
with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	
	#--------------------------------------------------------------------------------------------------
	#training start
	#--------------------------------------------------------------------------------------------------
	print('\n')
	print('---------------------------------------------')
	print('training start')
	print('---------------------------------------------')
	print('\n')
	
	for i in range(2500):
		batch = next_batch(500,x_train,y_train_one_hot.eval())
		if i%100==0:
			train_accuracy=accuracy.eval(feed_dict={X: batch[0], y: batch[1], keep_in_prob: 1.0,keep_hidden_prob=0.5})
			print("Epoch: ",i,"accuracy:",train_accuracy)
		
		sess.run(train_step,feed_dict={X: batch[0], y: batch[1], keep_in_prob: 0.8,keep_hidden_prob=0.5})
			
	print('\n')
	print('[*] trained_model save...')
	saver.save(sess,"./mlp_trained_model")

	
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess,tf.train.latest_checkpoint("./"))

	#--------------------------------------------------------------------------------------------------
	#val data test start
	#--------------------------------------------------------------------------------------------------
	print('\n')
	print('---------------------------------------------')
	print('val data test using trained model')
	print('---------------------------------------------')
		
	val_batch=next_batch(100,x_val,y_val_one_hot.eval())
	val_accuracy=accuracy.eval(feed_dict={x: val_batch[0], y: val_batch[1], keep_prob: 1.0})
	
	label=labeling.eval(feed_dict={x: val_batch[0], y: val_batch[1], keep_prob: 1.0})
	predict=prediction.eval(feed_dict={x: val_batch[0], y: val_batch[1], keep_prob: 1.0})
	print("val data accuracy: ",val_accuracy)
	
'''
[*] label information
y_train: int label to train(0~200)
int_dirname_dict: key(int)/value(dirname)
dirname_label_dict: key(dirname)/value(real label)
int--->dirname--->label(real class name)
'''

with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess,tf.train.latest_checkpoint("./"))
	#--------------------------------------------------------------------------------------------------
	#test data prediction
	#--------------------------------------------------------------------------------------------------
	print('\n')
	print('---------------------------------------------')
	print('test data prediction')
	print('---------------------------------------------')
	test_batch=select_test_sample(10,x_test)
	predict=sess.run(prediction,feed_dict={x: test_batch,keep_prob: 1.0})
	for i in range(len(predict)):
		key=int_dirname_dict[predict[i]]
		label=dirname_label_dict[key]
		print(test_file_list[i],": ",label)