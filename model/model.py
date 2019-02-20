import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple,DropoutWrapper
import sys

from tensorflow.python.util import nest
from tensorflow.python.ops import rnn, rnn_cell 

import random
#############################################################
class seq2seq(object):
    def __init__(self, args):
        self.input_dim = args.input_dim
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_keep_prob=args.output_keep_prob
        self.output_dim=args.output_dim
        self.learning_rate=args.learning_rate
        self.is_inference = tf.placeholder(tf.bool,name='is_inference')
        self.max_time=tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
        #self.max_time=40
        #self.batch_size=50
        self.sequence_length = tf.placeholder(tf.int32, [None],name='sequence_length')
        self.encoder_inputs = tf.placeholder(tf.float32, shape=[None,None,self.input_dim], name="encoder_inputs") 
        self.decoder_inputs = tf.placeholder(tf.float32, shape=[None,None,self.output_dim], name="decoder_inputs") 
        self.decoder_targets = tf.placeholder(tf.float32, [None,None,self.output_dim],name='decoder_targets')
        self.seq2seq_model()

    def encoder(self):
        with tf.variable_scope("encoder",reuse=tf.AUTO_REUSE) as encoder_scope:
            encoder_inputs_2d=tf.reshape(self.encoder_inputs,[self.batch_size*self.max_time,self.input_dim])
            encoder_cell_inputs=tf.layers.dense(inputs=encoder_inputs_2d,units=self.hidden_size,activation=tf.nn.relu)
            encoder_cell_inputs_3d=tf.reshape(encoder_cell_inputs,[self.batch_size,self.max_time,self.hidden_size])

            
            encoder_fw_cells = []
            encoder_bw_cells = []
            keep_prob=self.output_keep_prob
            for i in range(self.num_layers):
                with tf.variable_scope('encoder_lstm_{}'.format(i)):
                    cell=tf.contrib.rnn.GLSTMCell(self.hidden_size)
                    #keep_prob+= self.output_keep_prob * ( i*1.0 / float(self.num_layers))
                    #cell=rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=self.output_keep_prob)
                    encoder_fw_cells.append(cell)
                    encoder_bw_cells.append(cell)
            encoder_muti_fw_cell = rnn_cell.MultiRNNCell(encoder_fw_cells)
            encoder_muti_bw_cell = rnn_cell.MultiRNNCell(encoder_bw_cells)

            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_muti_fw_cell,
                                                cell_bw=encoder_muti_bw_cell,
                                                inputs=encoder_cell_inputs_3d,
                                                sequence_length=self.sequence_length,
                                                dtype=tf.float32, time_major=False)

            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

            self.encoder_final_state=[]
            for i in range(self.num_layers):
                encoder_final_state_c = tf.concat(
                    (encoder_fw_final_state[i].c, encoder_bw_final_state[i].c), 1)

                encoder_final_state_h = tf.concat(
                    (encoder_fw_final_state[i].h, encoder_bw_final_state[i].h), 1)

                encoder_final_state = LSTMStateTuple(
                    c=encoder_final_state_c,
                    h=encoder_final_state_h
                )
                self.encoder_final_state.append(encoder_final_state) 
            return encoder_outputs,encoder_bw_final_state
    

    
    def decoder(self,encoder_outputs,encoder_states):
       
        batch_size,max_time,_=tf.unstack(tf.shape(encoder_outputs))
        decoder_sequence_length=self.sequence_length
        decoder_max_time=max_time
        ##decder cell
        decoder_cells=[]
        for i in range(self.num_layers):
            with tf.variable_scope('decoder_lstm_{}'.format(i)):
                cell=tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size)
                #cell=rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=self.output_keep_prob) 
                decoder_cells.append(cell)
                
        decoder_cell = rnn_cell.MultiRNNCell(decoder_cells)
        decoder_out_layer = tf.layers.Dense(units = self.output_dim,
                        
                             )
        ### attention
        #attention_mechanism=tf.contrib.seq2seq.BahdanauAttention(
        #    num_units=self.hidden_size, 
        #    memory=encoder_outputs,
        #    memory_sequence_length=self.sequence_length,
        #    normalize=True,
        #    #score_mask_value=0.0
        #    
        #    )
        #attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
        #                        attention_mechanism=attention_mechanism,
        #                      
        #                        alignment_history = True) 

        #decoder_initial_state = attn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            #############################################
        with tf.variable_scope("decoder"):
            
            
            #trainning
          
            #decoder_inputs =tf.concat([tf.fill([self.batch_size,1,self.output_dim], -1.0),self.decoder_targets],axis=1) 
       
            training_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(self.decoder_targets, 
                                               sequence_length=decoder_sequence_length,
                                               sampling_probability=tf.constant(0.5)
                                               )
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
                                  helper=training_helper,
                                  initial_state=self.encoder_final_state,
                                   output_layer=decoder_out_layer
                                  )
         
            self.training_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, maximum_iterations=decoder_max_time,impute_finished=True) 
            #training_final_outputs=tf.slice(self.training_outputs.rnn_output,[0,1,0],[self.batch_size,self.max_time,self.output_dim]) 
        #inference
        with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
            def _sample_fn(decoder_outputs):
                 return decoder_outputs
            def _end_fn(_):
                 return tf.tile([False], [self.batch_size]) 
            inference_helper = tf.contrib.seq2seq.InferenceHelper(
                                sample_fn=lambda outputs:outputs,
                                sample_shape=[self.output_dim], 
                                sample_dtype=tf.float32,
                                start_inputs=tf.fill([self.batch_size,self.output_dim], 0.0),
                                end_fn=lambda sample_ids:False)
            inference_helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(tf.zeros_like(self.decoder_targets), 
                                               sequence_length=self.sequence_length, 
                                               sampling_probability=tf.constant(0.5),
                                               seed=2018 )
           
           
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                  helper=inference_helper,
                                  initial_state=self.encoder_final_state,
                                   output_layer=decoder_out_layer
                                  )
         
            self.inference_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, maximum_iterations=decoder_max_time,impute_finished=True)
            #inference_final_outputs=tf.slice(self.inference_outputs.rnn_output,[0,1,0],[self.batch_size,self.max_time,self.output_dim])
        #return training_final_outputs,inference_final_outputs
        return self.training_outputs.rnn_output,self.inference_outputs.rnn_output
    def seq2seq_model(self):
        with tf.variable_scope('pep2inten'): 
            
            encoder_outputs,encoder_states=self.encoder()
           
            training_outputs,inference_outputs=self.decoder(encoder_outputs,encoder_states)

            self.decoder_prediction=tf.reshape(tf.cond(self.is_inference,lambda:inference_outputs,lambda:training_outputs),[self.batch_size*self.max_time,self.output_dim])
            
            #self.decoder_prediction=tf.where(tf.logical_not(self.decoder_prediction<tf.constant(0.0)),self.decoder_prediction,tf.zeros([self.batch_size*self.max_time,self.output_dim]))
            #self.decoder_prediction=tf.where(tf.logical_not(self.decoder_prediction>tf.constant(1.0)),self.decoder_prediction,tf.ones([self.batch_size*self.max_time,self.output_dim]))


            mask=tf.to_float(tf.reshape(tf.sequence_mask(self.sequence_length,self.max_time),[self.batch_size*self.max_time,1]))
            labels=tf.reshape(self.decoder_targets,[self.batch_size*self.max_time,self.output_dim])
            self.loss=tf.losses.absolute_difference(labels,self.decoder_prediction,mask)
            tf.summary.scalar('loss', self.loss)
        
            with tf.variable_scope('tain_op'):
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())
        
    def train(self, sess, max_time,encoder_inputs,decoder_targets,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_targets: decoder_targets,
                    self.sequence_length:sequence_length,
                    self.is_inference:False
                   
                    }
       
        _, loss, summary,aa = sess.run([self.train_op, self.loss, self.summary_op,self.training_outputs], feed_dict=feed_dict)
        #if i==0:
        #    with open('data/train_temp.txt','a') as f:
        #        f.write(str(max_time)+','+str(sequence_length[0])+'\n') 
        #        for line in aa[0][0]:
        #            f.write(str(line)+'\n')
        #        f.write('\n\n')
        #        for line in decoder_targets[0]:
        #            f.write(str(line)+'\n')
        #        f.write('#########################################\n')
        return loss, summary

    def eval(self, sess, max_time,encoder_inputs,decoder_targets,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_targets: decoder_targets,
                    self.sequence_length:sequence_length,
                    self.is_inference:True
                    }
        loss,prediction,aa = sess.run([self.loss,self.decoder_prediction,self.inference_outputs], feed_dict=feed_dict)
        #if i==0:
        #    with open('data/temp.txt','a') as f:
        #        f.write(str(max_time)+','+str(sequence_length[0])+'\n') 
        #        for line in aa[0][0]:
        #            f.write(str(line)+'\n')
        #        f.write('\n\n')
        #        for line in decoder_targets[0]:
        #            f.write(str(line)+'\n')
        #        f.write('#########################################\n')
        return loss,prediction

    def predict(self, sess, max_time,encoder_inputs,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.sequence_length:sequence_length,
                    self.is_inference:True
                    }
        prediction = sess.run(self.decoder_prediction, feed_dict=feed_dict)
        return prediction

class pep2inten(object):
    def __init__(self, args):
        self.input_dim = args.input_dim
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.output_keep_prob=args.output_keep_prob
        self.output_dim=args.output_dim
        self.learning_rate=args.learning_rate
        self.l1_lamda=args.l1_lamda
        #self.l2_lamda=args.l2_lamda
        self.is_inference = tf.placeholder(tf.bool,name='is_inference')
        self.max_time=tf.placeholder(shape=None,dtype=tf.int32,name='max_time')
        self.batch_size=tf.placeholder(shape=None,dtype=tf.int32,name='batch_size')
        self.sequence_length = tf.placeholder(tf.int32, [None],name='sequence_length')
        self.encoder_inputs = tf.placeholder(tf.float32, shape=[None,None,self.input_dim], name="encoder_inputs") 
        self.decoder_targets = tf.placeholder(tf.float32, [None,None,self.output_dim],name='decoder_targets')
        self.pep2inten()
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
   
    def pep2inten(self):
       
        with tf.variable_scope('pep2inten'): 
            encoder_outputs,hidden_state=self.encoder()
            self.decoder_prediction=self.decoder(encoder_outputs,hidden_state)

        with tf.variable_scope('loss'):
            mask=tf.to_float(tf.reshape(tf.sequence_mask(self.sequence_length,self.max_time),[self.batch_size*self.max_time,1]))
            labels=tf.reshape(self.decoder_targets,[self.batch_size*self.max_time,self.output_dim])
           
            self.loss=tf.losses.mean_squared_error(labels,self.decoder_prediction,weights=mask)
           
            tf.summary.scalar('loss', self.loss)
            tf.add_to_collection('loss', self.loss)

        with tf.variable_scope('tain_op'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss=self.loss)
            tf.add_to_collection('tain_op', self.train_op)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables()) 
   
    def train(self, sess, max_time,encoder_inputs,decoder_targets,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.decoder_targets: decoder_targets,
                    self.sequence_length:sequence_length,
                    self.is_inference:False
                   
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
                    self.is_inference:True
                    }
        loss,prediction = sess.run([self.loss,self.decoder_prediction], feed_dict=feed_dict)
        
        return loss,prediction
    def predict(self, sess, max_time,encoder_inputs,sequence_length):
        feed_dict={
                    self.max_time:max_time,
                    self.batch_size:len(encoder_inputs),
                    self.encoder_inputs: encoder_inputs,
                    self.sequence_length:sequence_length,
                    self.is_inference:True
                    }
        prediction = sess.run(self.decoder_prediction, feed_dict=feed_dict)
        
        return prediction

 