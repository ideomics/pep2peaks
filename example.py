from model import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse
from tools.get_data import *
from tools.pearson import *
import tensorflow as tf
from data_preprocessing.tools import *
class example(object):
    def __init__(self):
        self.regular_model='seq2seq-WGAN/model/regular_proteometools_NCE35_all/'
        self.internal_model='seq2seq-WGAN/model/internal_proteometools_NCE35_all/'
        self.modi=''
        self.charge=3
        self.internal_ion_min_length=1
        self.internal_ion_max_length=2
        self.peptide='RAEYWENYPPAH'
        self.datam=GetData('example')

        self.PROTON= 1.007276466583
        self.H = 1.0078250322
        self.O = 15.9949146221
        
        self.N = 14.0030740052
        self.C = 12.00
        self.isotope = 1.003

        self.CO = self.C + self.O
        self.H2O= self.H * 2 + self.O

        self.predict()
    def parse_args(self,input_dim,output_dim):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_dim', type=int, default=input_dim,help='')
        parser.add_argument('--hidden_size', type=int, default=256,help='')
        parser.add_argument('--num_layers', type=int, default=2,help='')
        parser.add_argument('--l1_lamda', type=float, default=0.1,help='')
        parser.add_argument('--output_keep_prob', type=float, default=0.5,help='')
        parser.add_argument('--output_dim', type=int, default=output_dim,help='')
        parser.add_argument('--learning-rate', type=float, default=5e-4,help='')
        return parser.parse_args()

    def plot(self,regular_ion_type,regular_mzs,regular_pred_inten,internal_ion_type,internal_mzs,internal_pred_inten):
        regular_h=len(regular_ion_type)//2
        internal_h=len(internal_ion_type)//2
           
        name_y=[0,20,40,60,80,100]

        bar_width=5
        fon_size=20
        y_color='red'
        b_color='blue'
        ay_color='purple'
        by_color='green'
        font_Arial_Black=FontProperties(fname=r"C:\\Windows\\Fonts\\ariblk.ttf",size=fon_size)
        plt.figtext(.45,.90, self.peptide+'('+str(self.charge)+'+)',fontproperties=FontProperties(fname=r"C:\\Windows\\Fonts\\ariblk.ttf",size=24))
        #plt.figtext(.90,.05, 'Predicted',fontproperties=font_Arial_Black)
       # plt.figtext(.96,.54, 'm/z',fontsize=fon_size)
        
        plt.bar(regular_mzs[:regular_h],regular_pred_inten[:regular_h],width=bar_width,color=b_color)
        plt.bar(regular_mzs[regular_h:],regular_pred_inten[regular_h:],width=bar_width,color=y_color)

        plt.bar(internal_mzs[:internal_h],internal_pred_inten[:internal_h],width=bar_width,color=by_color)
        plt.bar(internal_mzs[internal_h:],internal_pred_inten[internal_h:],width=bar_width,color=ay_color)
        #annotate start
        #annotate end
        plt.ylabel('Relative Abundance',fontsize=fon_size)
        plt.xlabel('m/z',fontsize=fon_size)

        plt.ylim([0,1.0])
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0],name_y)
        plt.xlim(50,900)
        #plt.tick_params(labelsize=fon_size)

        #plt.gca().spines['top'].set_visible(False)
        #plt.gca().spines['right'].set_visible(False)
        plt.show()     
    def predict(self):
        
        print('predicting..')
        ions_number=len(self.peptide)-1 
        #regular
        model=pep2inten(self.parse_args(188,4))
        regular_mzs=[];regular_ion_type=[]
        bs,bs_mass,b_name,ys,ys_mass,y_name=get_y_and_b(self.peptide)
        bs1_mz=np.array(bs_mass)+self.PROTON
        ys1_mz=np.array(ys_mass)+self.H2O+self.PROTON

        bs2_mz=(np.array(bs_mass)+2*self.PROTON)/2
        ys2_mz=(np.array(ys_mass)+self.H2O+2*self.PROTON)/2
        
        regular_mzs.extend(bs1_mz)
        regular_mzs.extend(bs2_mz)
        regular_mzs.extend(ys1_mz)
        regular_mzs.extend(ys2_mz)
        regular_ion_type=[x+'+' for x in b_name]
        regular_ion_type.extend([x+'++' for x in b_name])
        regular_ion_type.extend([x+'+' for x in y_name])
        regular_ion_type.extend([x+'++' for x in y_name])
        encoder_inputs=[]
        for i in range(ions_number):
            line=self.peptide+'\t'+str(self.charge)+'\t'+bs[i]+','+ys[i]+'\t'+self.modi+'\t\t'+b_name[i]+'+,'+b_name[i]+'++,'+y_name[i]+'+,'+y_name[i]+'++'+'\t0,0,0,0'
            _,vector,_,_=self.datam.ion_b_y_featurize_4label(line)
            encoder_inputs.append(vector)
        regular_pred=[]
        with tf.Session() as session:
            model.saver.restore(session, tf.train.get_checkpoint_state(self.regular_model).model_checkpoint_path)
            pred=model.predict(session,ions_number,self.datam.vertor_normalize(np.array(encoder_inputs))[np.newaxis,:,:],[ions_number])
            pred[pred<0]=0
            pred[pred>1]=1
            for k in range(4):
                regular_pred.extend(np.array(pred)[:,k].tolist())
        tf.reset_default_graph()        
        #internal
        model=pep2inten(self.parse_args(217,2))
        internal_mzs=[];internal_ion_type=[]
        bys,bys_mass,bys_name=get_all_by(self.peptide,self.internal_ion_min_length,self.internal_ion_max_length)
        bys_mz=np.array(bys_mass)+self.PROTON
        ays_mz=bys_mz-self.CO
        internal_mzs.extend(bys_mz)
        internal_mzs.extend(ays_mz)
        internal_ion_type=[x+'+' for x in bys_name]
        internal_ion_type.extend([x.replace('b','a')+'+' for x in bys_name])
        encoder_inputs=[]
        for i in range(len(internal_ion_type)//2):
            line=self.peptide+'\t'+str(self.charge)+'\t'+bys[i]+'\t'+self.modi+'\t\t'+bys_name[i]+'+,'+bys_name[i].replace('b','a')+'\t0,0'
            _,vector,_,_,_=self.datam.ion_ay_by_featurize_2label(line)
            encoder_inputs.append(vector)
        internal_pred=[]
        with tf.Session() as session:
            model.saver.restore(session, tf.train.get_checkpoint_state(self.internal_model).model_checkpoint_path)
            pred=model.predict(session,len(internal_ion_type)//2,self.datam.vertor_normalize(np.array(encoder_inputs))[np.newaxis,:,:],[len(internal_ion_type)//2])
            pred[pred<0]=0
            pred[pred>1]=1
            for k in range(2):
                internal_pred.extend(np.array(pred)[:,k].tolist())
        print(regular_ion_type)
        print(regular_pred)
        print(internal_ion_type)
        print(internal_pred)
        self.plot(regular_ion_type,regular_mzs,regular_pred,internal_ion_type,internal_mzs,internal_pred)
if __name__=='__main__':
    example()