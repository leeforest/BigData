import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
import os
from tensorflow.examples.tutorials.mnist import input_data
import cv2

def load_data():
	
	array_list=[]
	train_dir_list=[]
	root_dir="imagenet\\tiny-imagenet-200\\train\\n02085620\\" #dog...
	train_file_list=os.listdir(root_dir)
	
	for file in train_file_list:
		file=root_dir+"\\"+file
		im=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
		im=cv2.resize(im,(32,32)) #resize
		im=np.asarray(im)
		tmp=np.concatenate(im)
		array_list.append(tmp)	
	total_train_array=np.asarray(array_list)

	return total_train_array

def next_batch(num,data):

	batch=[]
	idx=np.arange(0,len(data))
	np.random.shuffle(idx)
	idx=idx[:num]
	data_shuffle=[data[ i] for i in idx]
	data_shuffle=np.asarray(data_shuffle)
	for data in data_shuffle:
		inputs=(np.asfarray(data)/255.0*0.99)+0.01 #scale
		batch.append(inputs)
	
	return np.asarray(batch)

def xavier_init(size):
	in_dim=size[0]
	xavier_stddev=1. / tf.sqrt(in_dim / 2.)
	return tf.random_normal(shape=size,stddev=xavier_stddev)

def Q(X,eps):
	inputs=tf.concat([X,eps],1)
	h=tf.nn.elu(tf.matmul(inputs,Q_W1) + Q_b1)
	z=tf.matmul(h,Q_W2) + Q_b2
	return z

def P(z):
	h=tf.nn.elu(tf.matmul(z,P_W1) + P_b1)
	logits=tf.matmul(h,P_W2) + P_b2
	prob=tf.nn.sigmoid(logits)
	return prob
	
def D(X,z):
	h=tf.nn.elu(tf.matmul(tf.concat([X,z],1),D_W1) + D_b1)
	out=tf.matmul(h,D_W2) + D_b2
	return out	

#paraeter&options
batch_size=50
lr=0.001
eps_dim=32*32
X_dim=32*32
y_dim=32*32
z_dim=64
h_dim=256

#Q(z|X,eps)
X=tf.placeholder(tf.float32,shape=[None,X_dim],name='X')
eps=tf.placeholder(tf.float32,shape=[None,eps_dim],name='eps')
X_eps=tf.placeholder(tf.float32,shape=[None,X_dim],name='X_eps')

Q_W1=tf.Variable(xavier_init([X_dim + eps_dim,h_dim]))
Q_b1=tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2=tf.Variable(xavier_init([h_dim,z_dim]))
Q_b2=tf.Variable(tf.zeros(shape=[z_dim]))

theta_Q=[Q_W1,Q_W2,Q_b1,Q_b2]

#P(X|z)
P_W1=tf.Variable(xavier_init([z_dim,h_dim]))
P_b1=tf.Variable(tf.zeros(shape=[h_dim]))

P_W2=tf.Variable(xavier_init([h_dim,X_dim]))
P_b2=tf.Variable(tf.zeros(shape=[X_dim]))

theta_P=[P_W1,P_W2,P_b1,P_b2]

#D(z)
D_W1=tf.Variable(xavier_init([z_dim + eps_dim,h_dim]))
D_b1=tf.Variable(tf.zeros(shape=[h_dim]))

D_W2=tf.Variable(xavier_init([h_dim,1]))
D_b2=tf.Variable(tf.zeros(shape=[1]))

theta_D=[D_W1,D_W2,D_b1,D_b2]

#Training
x=load_data()

z_sample=Q(X,eps)
z_sample_fake=Q(X_eps,eps)
p_prob=P(z_sample)
X_samples=P(z_sample_fake)

#Adversarial loss to approx. Q(z|X)
D_real=D(X,z_sample)
D_fake=D(X,z_sample_fake)

G_loss=-(tf.reduce_mean(D_real) + tf.reduce_mean(tf.log(p_prob)))
g_psi=tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(labels=D_real,logits=tf.ones_like(D_real)) +
	tf.nn.sigmoid_cross_entropy_with_logits(labels=D_fake,logits=tf.zeros_like(D_fake)))

opt=tf.train.AdamOptimizer(learning_rate=0.001)
G_solver=opt.minimize(G_loss,var_list=theta_P + theta_Q)
D_solver=opt.minimize(g_psi,var_list=theta_D)

sess=tf.Session()
i=0
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
		
	for it in range(100000):
		
		for batch in range(batch_size):
			X_mb=next_batch(batch_size,x)
	
			eps_mb=np.random.randn(batch_size,eps_dim)
			z_mb=np.random.randn(batch_size,eps_dim)

			_,G_loss_curr=sess.run([G_solver,G_loss],feed_dict={X: X_mb,eps: eps_mb})
			_,D_loss_curr=sess.run([D_solver,g_psi],feed_dict={X: X_mb,eps: eps_mb,X_eps: z_mb})

		if it % 100 == 0:
			print('Iter: {}; G_loss: {:.4}; D_loss: {:.4}'.format(it,G_loss_curr,D_loss_curr))
			eps_mb=np.random.randn(5,eps_dim)
			X_mb= next_batch(5,x)

			samples=sess.run(X_samples,feed_dict={X_eps: np.random.randn(16,eps_dim),eps: np.random.randn(16,eps_dim)})
			
			reconstructed,latent_rep=sess.run([p_prob,z_sample],feed_dict={X: X_mb,eps: eps_mb})
			n_examples=5
			fig,axs=plt.subplots(2,n_examples,figsize=(10,2))

			for example_i in range(n_examples):
				axs[0][example_i].imshow(np.reshape(X_mb[example_i,:],(32,32)))
				axs[0][example_i].axis('off')
				axs[1][example_i].imshow(np.reshape([reconstructed[example_i,:]],(32,32)))
				axs[1][example_i].axis('off')
			plt.draw()
			plt.savefig('avae_imagenet/{}.png'.format(str(i).zfill(3)))
			plt.close(fig)
		
			i += 1