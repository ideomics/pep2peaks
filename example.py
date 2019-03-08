from model import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import argparse
from tools.get_data import *
from tools.pearson import *
import tensorflow as tf
from data_preprocessing.tools import *
from model.model import *
class example(object):
    def __init__(self):
        self.regular_model='models/regular_proteometools_NCE25_no_inten_threshold/'
        self.internal_model='models/internal_proteometools_NCE25_no_inten_threshold/'
        self.modi='6,Oxidation[M],15,Oxidation[M]'
        self.charge=2
        self.internal_ion_min_length=1
        self.internal_ion_max_length=2
        self.peptide='VLDDTmAVADILTSmVVDVSDLLDQAR'
        self.datam=GetData('example')

        self.PROTON= 1.007276466583
        self.H = 1.0078250322
        self.O = 15.9949146221
        
        self.N = 14.0030740052
        self.C = 12.00
        self.isotope = 1.003

        self.CO = self.C + self.O
        self.H2O= self.H * 2 + self.O
        self.data_tools=data_tools()
        self.predict()
    def parse_args(self,input_dim,output_dim,ion_type):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_dim', type=int, default=input_dim,help='')
        parser.add_argument('--hidden_size', type=int, default=256,help='')
        parser.add_argument('--num_layers', type=int, default=2,help='')
        parser.add_argument('--l1_lamda', type=float, default=0.1,help='')
        parser.add_argument('--ion_type', type=str, default=ion_type, help="regular or internal") 
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
    def merge_same_ions(self,_ions,pred):
        pred_by=[];pred_ay=[];ions=[]
        _pred_by=pred[:,0]
        _pred_ay=pred[:,1]
        d = defaultdict(list)
        for k,va in [(v,i) for i,v in enumerate(_ions)]:
            d[k].append(va) 
        for ion,indexs in d.items():
            pred_by.append(np.sum(np.array(_pred_by)[indexs]))
            pred_ay.append(np.sum(np.array(_pred_ay)[indexs]))
            ions.append(ion) 
        return ions,pred_by,pred_ay
    def predict(self):
        
        print('predicting..')
        ions_b_y,_,ions_ay_by=self.data_tools.get_ions_list(self.peptide,self.internal_ion_min_length,self.internal_ion_max_length)
        #regular
        model=pep2peaks(self.parse_args(173,4,'regular'))

        

       
        encoder_inputs=[]
        max_time=len(self.peptide)-1
        for i in range(max_time):
            line=self.peptide+'\t'+str(self.charge)+'\t'+ions_b_y[0][0][i]+','+ions_b_y[2][0][i]+'\t'+self.modi+'\t'+ions_b_y[0][2][i]+','+ions_b_y[1][2][i]+','+ions_b_y[2][2][i]+','+ions_b_y[3][2][i]+'\t0,0,0,0'
            
            _,vector,_,_=self.datam.ion2vec_4label(line)
          
            encoder_inputs.append(vector)
        with tf.Session() as session:
            model.saver.restore(session, tf.train.get_checkpoint_state(self.regular_model).model_checkpoint_path)
            regular_pred=model.predict(session,max_time,np.array(encoder_inputs)[np.newaxis,:,:],[max_time])
            regular_pred[regular_pred<0]=0
            regular_pred[regular_pred>1]=1
            #for k in range(4):
            #    regular_pred.extend(np.array(pred)[:,k].tolist())
        tf.reset_default_graph()        
        #internal
        model=pep2peaks(self.parse_args(198,2,'internal'))
        
        encoder_inputs=[]
        max_time=0
        for i in range(self.internal_ion_max_length):
            max_time+=(len(self.peptide)-(i+2))
        internal_ions=[]
        for i in range(max_time):
            internal_ions.append((self.peptide+'\t'+str(self.charge)+'\t'+ions_ay_by[0][0][i]+'\t'+self.modi+'\t'+ions_ay_by[0][2][i]+','+ions_ay_by[1][2][i]+'\t0,0').split('\t'))
        self.data_tools.avg_same_ion_inten(ions_ay_by[0][0],internal_ions)
        for internal_ion in internal_ions:
            _,vector,_,_,_=self.datam.ion2vec_2label('\t'.join(internal_ion))
            encoder_inputs.append(vector)
        with tf.Session() as session:
            model.saver.restore(session, tf.train.get_checkpoint_state(self.internal_model).model_checkpoint_path)
            internal_pred=model.predict(session,max_time,np.array(encoder_inputs)[np.newaxis,:,:],[max_time])
            internal_pred[internal_pred<0]=0
            internal_pred[internal_pred>1]=1

        ions,pred_by,pred_ay=self.merge_same_ions(ions_ay_by[0][0],internal_pred) 

        ##################
        print('regular ions:')
        for i in range(len(ions_b_y[0][0])):
            print(ions_b_y[0][0][i]+": "+ions_b_y[0][2][i]+":"+str(round(regular_pred[i][0],4))+","+ions_b_y[1][2][i]+":"+str(round(regular_pred[i][1],4)))
        for i in range(len(ions_b_y[0][0])):
            print(ions_b_y[2][0][i]+": "+ions_b_y[2][2][i]+":"+str(round(regular_pred[i][2],4))+", "+ions_b_y[3][2][i]+":"+str(round(regular_pred[i][3],4)))
        print('\ninternal ions:')
        for i in range(len(ions)):
            print(ions[i]+": by:"+str(round(pred_by[i],4))+", ay:"+str(round(pred_ay[i],4))) 
        
        #self.plot(regular_ion_type,regular_mzs,regular_pred,internal_ion_type,internal_mzs,internal_pred)
if __name__=='__main__':
    example()