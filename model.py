#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf 
import numpy as np 
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
rng = np.random.RandomState(23455)


class IR_quantum(object):
	def __init__(
		self, max_input_query,max_input_docu, vocab_size, embedding_size ,batch_size,
		embeddings,filter_sizes,num_filters,l2_reg_lambda = 0.0,trainable = True,
		pooling = 'max',overlap_needed = True,extend_feature_dim = 10):

		# self.dropout_keep_prob = dropout_keep_prob
		self.num_filters = num_filters
		self.embeddings = embeddings
		self.embedding_size = embedding_size
		self.vocab_size = vocab_size
		self.trainable = trainable
		self.filter_sizes = filter_sizes
		self.pooling = pooling
		self.total_embedding_dim = embedding_size
		self.batch_size = batch_size
		self.l2_reg_lambda = l2_reg_lambda
		self.para = []
		self.max_input_query = max_input_query
		self.max_input_docu = max_input_docu
		self.hidden_num = 128
		self.rng = 23455
		self.overlap_need = overlap_needed
		# if self.overlap_need:
		# 	self.total_embedding_dim = embedding_size + extend_feature_dim
		# else:
		# 	self.total_embedding_dim = embedding_size
		self.extend_feature_dim = extend_feature_dim
		self.conv1_kernel_num = 32
		# self.conv2_kernel_num = 32
		self.n_bins = 11
		self.stdv = 0.5
		print (self.max_input_query)
		print (self.max_input_docu)


	def weight_variable(self,shape):
		tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
		initial = tf.random_uniform(shape, minval=-tmp, maxval=tmp)
		return tf.Variable(initial)

	
	def creat_placeholder(self):
		self.query = tf.placeholder(tf.int32,[self.batch_size,self.max_input_query],name = "input_query")
		self.document = tf.placeholder(tf.int32,[self.batch_size,self.max_input_docu],name = "input_document")
		self.input_label = tf.placeholder(tf.float32,[self.batch_size,1],name = "input_label")

		# self.q_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_query],name = "q_overlap")
		# self.d_overlap = tf.placeholder(tf.int32,[self.batch_size,self.max_input_docu],name = "d_overlap")
		# self.tfidf_value = tf.placeholder(tf.float32,[self.batch_size,self.max_input_docu],name = 'tfidf_value')
		self.dropout_keep_prob = tf.placeholder(tf.float32,name ="dropout_keep_prob")

		self.input_mu = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_mu')
		self.input_sigma = tf.placeholder(tf.float32, shape=[self.n_bins], name='input_sigma')


		self.mu = tf.reshape(self.input_mu, shape=[1, 1, self.n_bins])
		self.sigma = tf.reshape(self.input_sigma, shape=[1, 1, self.n_bins])
	
		self.W1 = self.weight_variable([self.n_bins,1])
		self.b1 = tf.Variable(tf.zeros([1]))

	def load_embeddings(self):


		self.words_embeddings = tf.Variable(np.array(self.embeddings),name = "word_emb",dtype = "float32",trainable = False)
			
		self.embedded_chars_q = tf.nn.embedding_lookup(self.words_embeddings, self.query, name="q_emb")
		self.embedded_chars_d = tf.nn.embedding_lookup(self.words_embeddings, self.document, name="d_emb")

		self.norm_q = tf.sqrt(tf.reduce_sum(tf.square(self.embedded_chars_q), 2, keep_dims=True))
		self.normalized_q_embed = self.embedded_chars_q / self.norm_q
		self.norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.embedded_chars_d), 2, keep_dims=True))
		self.normalized_d_embed = self.embedded_chars_d / self.norm_d
		self.tmp = tf.transpose(self.normalized_d_embed, perm=[0, 2, 1])


	def model(self):
		# similarity matrix [n_batch, qlen, dlen]
		self.sim = tf.matmul(self.normalized_q_embed, self.tmp, name='similarity_matrix')

		# compute gaussian kernel
		rs_sim = tf.reshape(self.sim, [self.batch_size, self.max_input_query, self.max_input_docu, 1])

		tmp_model = tf.exp(-tf.square(tf.subtract(rs_sim, self.mu)) / (tf.multiply(tf.square(self.sigma), 2)))

		feats = []

		kde = tf.reduce_sum(tmp_model,[2])
		kde = tf.log(tf.maximum(kde,1e-10))*0.01

		# aggregated_kde = tf.reduce_sum(kde * q_weights, [1])

		aggregated_kde = tf.reduce_sum(kde, [1])

		feats.append(aggregated_kde)
		feats_tmp = tf.concat(feats,1)

		self.feats_flat = tf.reshape(feats_tmp,[-1,self.n_bins])

		self.lo = tf.matmul(self.feats_flat, self.W1)+self.b1 

		self.logits = tf.tanh(self.lo)
		self.scores = self.logits



	def create_loss(self):
		l2_loss = tf.constant(0.0)
		for p in self.para:
			l2_loss += tf.nn.l2_loss(p)
		with tf.name_scope("loss"):
			# p_pre = tf.nn.softmax(self.logits)
			self.p_label = tf.nn.softmax(self.input_label,dim = 0)
			cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.p_label, logits=self.logits, dim = 0))
			
			self.loss = tf.clip_by_value(cross_entropy, 1e-8, 3.0) 

	def build_graph(self):
		self.creat_placeholder()
		self.load_embeddings()
		self.model()
		self.create_loss()
		print ("end build graph")

