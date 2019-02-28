import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys

from tensorflow.python.util import nest
from tensorflow.python.ops import rnn, rnn_cell 

import random
#############################################################
class pep2peaks(object):
    def __init__(self, args):
        self.input_dim = args.input_dim
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_keep_prob=args.output_keep_prob
        self.output_dim=args.output_dim
        self.learning_rate=args.learning_rate
        self.l1_lamda=args.l1_lamda
        #self.l2_lamda=args.l2_lamda
        self.regular =True if args.ion_type=='regular' else False
        self.inference = tf.placeholder(tf.bool,name='inference')
        self.max_time=tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
        self.sequence_length = tf.placeholder(tf.int32, [None],name='sequence_length')
        self.encoder_inputs = tf.placeholder(tf.float32, shape=[None,None,self.input_dim], name="encoder_inputs") 
       
        self.decoder_targets = tf.placeholder(tf.float32, [None,None,self.output_dim],name='decoder_targets')
        self.pep2peaks()
    def attention(self,atten_inputs, attention_size):
        with tf.variable_scope("attention"):
            inputs_hidden_size=6*self.hidden_size
            # Attention mechanism
            attention_w = tf.get_variable(shape=[inputs_hidden_size, attention_size], name='attention_w')
            attention_b = tf.get_variable(shape=[attention_size], name='attention_b')

            
            u = tf.nn.tanh(tf.matmul(tf.reshape(atten_inputs, [-1, inputs_hidden_size]), attention_w) + attention_b)
            u_w =tf.Variable(tf.random_normal([attention_size,1], stddev=0.1))
            atten_score = tf.reshape(tf.matmul(u, u_w), [self.batch_size, self.max_time, 1])
            #mask
            mask = tf.cast(tf.sequence_mask(self.sequence_length,self.max_time), tf.float32)
            mask = tf.expand_dims(mask, 2)
            mask_atten_score=atten_score - (1 - mask) * 1e12

            alpha =tf.nn.softmax(mask_atten_score)
            atten_outputs = atten_inputs * alpha
            
            return atten_outputs
    def get_cell(self,hidden_size,num_layers):
        cells = []
        keep_prob=self.output_keep_prob
        for i in range(num_layers):
            with tf.variable_scope('cell_{}'.format(i)):
                cell=tf.contrib.rnn.LSTMCell(hidden_size,use_peepholes=True)
               
                #keep_prob+= self.output_keep_prob * ( i*1.0 / float(self.num_layers))
                #cell=rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=self.output_keep_prob)
                cells.append(cell)
        muti_cells = rnn_cell.MultiRNNCell(cells)
        return muti_cells
    def encoder(self):
        with tf.variable_scope("encoder"):
           
            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.get_cell(self.hidden_size,self.num_layers),
                                                cell_bw=self.get_cell(self.hidden_size,self.num_layers),
                                                inputs=self.encoder_inputs,
                                                sequence_length=self.sequence_length,
                                                dtype=tf.float32, time_major=False)
            encoder_outputs=tf.concat([encoder_fw_outputs, encoder_bw_outputs],axis=2)

            encoder_final_state_c = tf.concat((encoder_fw_final_state[self.num_layers-1].c, encoder_bw_final_state[self.num_layers-1].c), 1)
            encoder_final_state_h = tf.concat((encoder_fw_final_state[self.num_layers-1].h, encoder_bw_final_state[self.num_layers-1].h), 1)
            encoder_final_state = LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h)
            hidden_state=tf.concat(encoder_final_state,axis=1)
           
            return encoder_outputs,hidden_state
    def decoder(self,encoder_outputs,hidden_state):
        with tf.variable_scope("decoder"):
            decoder_hidden_size=self.hidden_size//4
            
            hidden_state=tf.tile(tf.expand_dims(hidden_state,1),multiples=[1,self.max_time,1])
            atten_inputs=tf.concat([encoder_outputs,hidden_state],axis=2)
            atten_outputs=self.attention(atten_inputs,self.hidden_size)

            decoder_initial_inputs=tf.concat([atten_outputs,hidden_state],axis=2)
          
            decoder_outputs,_=tf.nn.dynamic_rnn(cell=self.get_cell(decoder_hidden_size,self.num_layers),
                                                inputs=decoder_initial_inputs,
                                                sequence_length=self.sequence_length,
                                                dtype=tf.float32, time_major=False)

           
            outputs=tf.reshape(decoder_outputs,[self.batch_size*self.max_time,decoder_hidden_size])
            finally_outputs=tf.layers.dense(inputs=outputs,units=self.output_dim,kernel_regularizer=tf.contrib.layers.l1_regularizer(self.l1_lamda))
            return  finally_outputs
   
    def pep2peaks(self):
       
        with tf.variable_scope('pep2peaks'): 
            encoder_outputs,hidden_state=self.encoder()
            self.decoder_prediction=self.decoder(encoder_outputs,hidden_state)

        with tf.variable_scope('loss'):
            mask=tf.to_float(tf.reshape(tf.sequence_mask(self.sequence_length,self.max_time),[self.batch_size*self.max_time,1]))
            labels=tf.reshape(self.decoder_targets,[self.batch_size*self.max_time,self.output_dim])
            
            self.loss=tf.cond(tf.cast(self.regular,tf.bool),
                              lambda:tf.losses.absolute_difference(labels,self.decoder_prediction,weights=mask),
                              lambda:tf.losses.mean_squared_error(labels,self.decoder_prediction,weights=mask)
                                                )
            
           
            tf.summary.scalar('loss', self.loss)
            tf.add_to_collection('loss', self.loss)
        #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops):
        with tf.variable_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)
            tf.add_to_collection('train_op', self.train_op)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables()) 
   
    def train(self, sess, max_time,encoder_inputs,decoder_targets,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_targets: decoder_targets,
                    self.sequence_length:sequence_length,
                    self.inference:False
                   
                    }
       
        _, loss, summary= sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        
        return loss, summary

    def eval(self, sess, max_time,encoder_inputs,decoder_targets,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_targets: decoder_targets,
                    self.sequence_length:sequence_length,
                    self.inference:True
                    }
        loss,prediction = sess.run([self.loss,self.decoder_prediction], feed_dict=feed_dict)
        
        return loss,prediction
    def predict(self, sess, max_time,encoder_inputs,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.sequence_length:sequence_length,
                    self.inference:True
                    }
        prediction = sess.run(self.decoder_prediction, feed_dict=feed_dict)
        
        return prediction

 