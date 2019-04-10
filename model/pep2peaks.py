
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys 
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
dirs = os.path.join( os.path.dirname(__file__),'..')
os.sys.path.append(os.path.join( os.path.dirname(__file__), '..'))
from tools.get_data import *
from tools.pearson import *
from model import *
from progressbar import *
seed = 64
np.random.seed(seed)
tf.set_random_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-train', type=int, default=2,help='1=train, 2=test, 0=train and test')
    parser.add_argument('--is_transfer', type=bool, default=True,help='')
    parser.add_argument('--num-iter', type=int, default=200,help='')
    parser.add_argument('--batch_size', type=int, default=256,help='')
    parser.add_argument('--hidden_size', type=int, default=256,help='')
    parser.add_argument('--num_layers', type=int, default=2,help='')
    parser.add_argument('--l1_lamda', type=float, default=0.1,help='')
    parser.add_argument('--l2_lamda', type=float, default=0.003,help='')
    parser.add_argument('--output_keep_prob', type=float, default=0.5,help='')
    parser.add_argument('--learning-rate', type=float, default=1e-4,help='')
    parser.add_argument('--is_mobile', type=str, default='all', help="mobile,non_mobile,partial_mobile,all,") 

    parser.add_argument('--ion_type', type=str, default='regular', help="regular or internal") 
    parser.add_argument('--model', type=str, default='models/regular_humanbody/', help="") 
    parser.add_argument('--input_dim', type=int, default=173,help='173 if regular else 198')
    parser.add_argument('--output_dim', type=int, default=4,help='4 if regular else 2')
    parser.add_argument('--min_internal_ion_len', type=int, default=1,help='')
    parser.add_argument('--max_internal_ion_len', type=int, default=2,help='')
    return parser.parse_args()

def get_batch_peptide(merge_list,_batch_size,is_train):
    number_of_peptide=len(merge_list[0])
    peplens=[len(_list) for _list in merge_list[0]]
    if is_train:
        #sorted data can accelerate training speed
        sorted_merge_list=[]
        data_sorted=[(peplen,ions_list ) for peplen,ions_list  in zip(peplens,merge_list[0])] 
        data_sorted.sort() 
        sorted_peplens=[peplen for peplen,ions_list  in data_sorted] 
        sorted_ions_list=[ions_list for peplen,ions_list  in data_sorted]
        sorted_merge_list.append(sorted_ions_list)
        pep_lists=sorted_merge_list
    else:
        pep_lists= merge_list

    batch_peptide=[]
    seq_length=[]
    _batch_number=int(number_of_peptide/_batch_size)
    for i in range(_batch_number):
        batch_peptide.append(pep_lists[0][i*_batch_size:(i+1)*_batch_size])
        seq_length.append([])
        for j in range(len(batch_peptide[i])):
            seq_length[len(batch_peptide)-1].append(len(pep_lists[0][i*_batch_size+j]))
    if _batch_number*_batch_size < number_of_peptide:
        seq_length.append([])
        batch_peptide.append(pep_lists[0][_batch_number*_batch_size:])
        for k in range(len(batch_peptide[-1])):
            seq_length[len(batch_peptide)-1].append(len(pep_lists[0][_batch_number*_batch_size+k]))
        _batch_number+=1
    return batch_peptide,_batch_number,seq_length

def padding_data(data,max_ions_number,is_target):
    _ydim=data.shape[1]
    dv=max_ions_number-data.shape[0]
    data=data.tolist()
    #if is_target:
    #    data.extend(np.ones((1,_ydim)).astype('float32').tolist())
    if dv > 0:
        data.extend(np.zeros((dv,_ydim)).astype('int32').tolist())
    return data

def train(args):
    model = pep2peaks(args)
    
    with tf.Session() as sess:
        if args.is_transfer:
            model.saver.restore(sess, tf.train.get_checkpoint_state('models/regular_proteometools_NCE35_mobile/').model_checkpoint_path)
        writer = tf.summary.FileWriter("model/logs", sess.graph)
        sess.run(tf.global_variables_initializer())
        #train data
        _,_,train_X,train_y,merge_train_list,_=datam.get_data('data/data_proteometools/NCE35_b_y_train.txt',args.min_internal_ion_len,args.max_internal_ion_len,args.is_mobile)
        print(str(len(merge_train_list[0]))+' train peptides ,DataShape:('+str(np.array(train_X).shape)+str(np.array(train_y).shape)+')')
        batch_peptide,_batch_number,seq_length=get_batch_peptide(merge_train_list,args.batch_size,True)
        
        #val data
        _,_,val_X,val_y,merge_val_list,_=datam.get_data('data/data_proteometools/NCE35_b_y_train.txt',args.min_internal_ion_len,args.max_internal_ion_len,args.is_mobile)
        print(str(len(merge_val_list[0]))+' val peptides ,DataShape:('+str(np.array(val_X).shape)+str(np.array(val_y).shape)+')')
        val_batch_peptide,val_batch_number,val_seq_length=get_batch_peptide(merge_val_list,args.batch_size,False)
        
        print('..trainning')
        best_loss=1000.0
        
        for Iter in range(args.num_iter):
            train_loss=0
            per_80=int(len(batch_peptide)*0.8)
            permutation_batch = np.random.permutation(len(batch_peptide))[:per_80]
            suffled_batch_peptide=np.array(batch_peptide)[permutation_batch].tolist()
            suffled_seq_length=np.array(seq_length)[permutation_batch].tolist()
            #pbar = ProgressBar(widgets=['Training',Percentage(), ' ', Bar('#'),' ',' ', ETA(), ' ']).start()
            for i,(train_piptide_index) in enumerate(suffled_batch_peptide):
                #pbar.update(int((i / (len(suffled_batch_peptide) - 1)) * 100))
                encoder_inputs=[];decoder_inputs=[];decoder_targets=[]
                max_ions_number=max(suffled_seq_length[i])
                permutation_peptide = np.random.permutation(len(train_piptide_index))
                suffled_seq=np.array(suffled_seq_length[i])[permutation_peptide].tolist()
                suffled_train_piptide_index=np.array(train_piptide_index)[permutation_peptide].tolist()
                for j in range(len(suffled_train_piptide_index)):
                    train_ion_index=datam.get_split_list(suffled_train_piptide_index[j])
                    encoder_inputs.append(padding_data(train_X[np.array(train_ion_index)],max_ions_number,False))
                    decoder_targets.append(padding_data(train_y[np.array(train_ion_index)],max_ions_number,True))
                loss,train_summary=model.train(sess, max_ions_number,np.array(encoder_inputs),np.array(decoder_targets),suffled_seq)
                train_loss+=loss
            #pbar.finish()
            val_loss=0
            #pbar = ProgressBar(widgets=['Validating',Percentage(), ' ', Bar('#'),' ',' ', ETA(), ' ']).start()
            for i, val_piptide_index in enumerate(val_batch_peptide):
                #pbar.update(int((i / (len(val_batch_peptide) - 1)) * 100))
                encoder_inputs=[];decoder_inputs=[];decoder_targets=[]
                max_ions_number=max(val_seq_length[i])
                for j in range(len(val_piptide_index)):
                    train_ion_index=datam.get_split_list(val_piptide_index[j])
                    encoder_inputs.append(padding_data(val_X[np.array(train_ion_index)],max_ions_number,False))
                    decoder_targets.append(padding_data(val_y[np.array(train_ion_index)],max_ions_number,True))
                loss_val,_=model.eval(sess, max_ions_number,np.array(encoder_inputs),np.array(decoder_targets),val_seq_length[i])
                val_loss+=loss_val
            #pbar.finish() 
            print('Iter:{0}/{1}  train loss:{2:.4} val loss:{3:.4}'.format(Iter+1,args.num_iter,(train_loss/_batch_number),(val_loss/val_batch_number)))
            if best_loss > val_loss/val_batch_number:
                best_loss=val_loss/val_batch_number
                print("Saved best Model:",model.saver.save(sess, 'models/regular_proteometools_NCE35_mobile_transfer_non_mobile/model.ckpt'))
            writer.add_summary(train_summary, Iter)
    tf.reset_default_graph()
  
def model_predict(args):
    model=pep2peaks(args)
    print('predicting..')
    print('pep type:'+args.is_mobile)
    idx,peptide,test_X,test_y,merge_test_list,ions=datam.get_data('data/data_proteometools/NCE30_b_y_test.txt',args.min_internal_ion_len,args.max_internal_ion_len,args.is_mobile)
    
    print(str(len(merge_test_list[0]))+' test peptides ,DataShape:('+str(np.array(test_X).shape)+str(np.array(test_y).shape)+')')
    test_batch_peptide,_batch_number,seq_length=get_batch_peptide(merge_test_list,args.batch_size,False)
    test_loss=[];pred=[]
    with tf.Session() as session:
        model.saver.restore(session, tf.train.get_checkpoint_state(args.model).model_checkpoint_path)
        for i, test_piptide_index in enumerate(test_batch_peptide):
            encoder_inputs=[];decoder_inputs=[];decoder_targets=[]
            max_ions_number=max(seq_length[i])
            for j in range(len(test_piptide_index)):
                test_ion_index=datam.get_split_list(test_piptide_index[j])
                encoder_inputs.append(padding_data(test_X[np.array(test_ion_index)],max_ions_number,False))
                decoder_targets.append(padding_data(test_y[np.array(test_ion_index)],max_ions_number,True))
            #pred_=model.predict(session,max_ions_number,np.array(encoder_inputs),seq_length[i])
           
            loss,pred_=model.eval(session,max_ions_number,np.array(encoder_inputs),np.array(decoder_targets),seq_length[i])
            
            test_loss.append(loss)
            #print('batch '+str(i+1)+' test loss'+str(loss))
            #pred_[pred_>1]=1
            #pred_[pred_<0]=0 
            for k in range(len(decoder_targets)): 
                for pred_s in pred_[k*max_ions_number:k*max_ions_number+seq_length[i][k]]:
                    pred.append(pred_s)
        
        print('avg test loss '+str(np.mean(np.array(test_loss))))
    return idx,peptide,pred,merge_test_list,test_y,ions

def get_merge_pred(merge_list,pred,args):
    print('get predict spectrum intensity list...')
    
    if args.ion_type=='internal':
        merge_list.append(datam.merge_list_2label(pred))
    elif args.ion_type=='regular':
        merge_list.append(datam.merge_list_4label(pred))
    return merge_list

   
def calc_pear(test_idx,peptide,pred,merge_list,args,test_y,ions):
    pred_pd=pear.write_pred(test_idx,peptide,pred,ions)
    merge_list=get_merge_pred(merge_list,pred_pd,args)
    pear.get_pearson(merge_list,test_idx,peptide,pred,test_y)
   
def test(args):
    test_idx,peptide,pred,merge_test_list,test_y,ions=model_predict(args)
    
    person_mean=calc_pear(test_idx,peptide,pred,merge_test_list,args,test_y,ions)

def main(args):
    
    if(args.is_train==1):
        train(args)
    elif(args.is_train==2):
        test(args)
    else:
        train(args)
        test(args)
if __name__ == '__main__':
    #with tf.device('/cpu:0'):
        args=parse_args()
        
        datam=GetData(args.ion_type)
        pear=CalcPerson(args.ion_type)
       
        main(args)


