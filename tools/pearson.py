import pandas as pd
from math import sqrt
import math
import numpy as np
from scipy.stats.stats import pearsonr,spearmanr
from collections import defaultdict
class CalcPerson(object):
    def __init__(self,ion_type):
        self.ion_type=ion_type


    def multipl(self,a,b):
        multipl_of_a_b = 0.0
        for i in range(len(a)):
            temp = a[i] * b[i]
            multipl_of_a_b+=temp
        return multipl_of_a_b
    # https://wikimedia.org/api/rest_v1/media/math/render/svg/832f0c5c22a0d6f2596c150de811247438a503de
    def pearson_r(self,x,y):
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_of_xy = multipl(x,y)
        sum_of_x2 = sum([pow(i,2) for i in x])
        sum_of_y2 = sum([pow(j,2) for j in y])
        numerator = sum_of_xy - (float(sum_x) * float(sum_y) / n)
        denominator = sqrt((sum_of_x2 - float(sum_x ** 2) / n) * (sum_of_y2 - float(sum_y ** 2) / n))
        
        pcc=numerator / denominator
        return pcc
    def get_regular_pearson_x(self,merge_list,pos,m):
        if self.ion_type=='internal':
            num_label=2
        elif self.ion_type=='regular':
            num_label=4
        x=[]
        for j in range(len(merge_list[0][pos])):
            for k in range(1,num_label+1):
                x.append(merge_list[m][pos][j][k])
        return x
    def get_internal_pearson_x(self,merge_list,pos,m,ion_len):
      
        x=[]
        for j in range(len(merge_list[0][pos])):
            if merge_list[m][pos][j][3]==ion_len: 
                for k in range(1,3):
                    x.append(merge_list[m][pos][j][k])
        return x
    def get_pearson_x(self,merge_list,pos,m):
      
        x=[]
        for j in range(len(merge_list[0][pos])):
           
            for k in range(1,3):
                x.append(merge_list[m][pos][j][k])
        return x
    def get_labels_pred(self,merge_list,i):
        lable=[];pred=[]
        _label=self.get_pearson_x(merge_list,i,0)
        _pred=self.get_pearson_x(merge_list,i,1)
        for j in range(len(_label)):
            if _label[j]>=0.01:
                lable.append(_label[j])
                pred.append(_pred[j])
        return lable,pred
    def get_internal_pearson(self,merge_list,test_idx,peptide,pred,test_y):
        person_list_dic=defaultdict(list)
        pearson_list=[]
        ion_len_list=[1,2,3,4,5]
       
        print('calculate person coefficient..')
        sum_person = 0.0
        for i in range(len(merge_list[0])):
            s_pear=pearsonr(self.get_pearson_x(merge_list,i,0),self.get_pearson_x(merge_list,i,1))[0] 
            if np.isnan(s_pear):
                continue
            else:
                pearson_list.append(s_pear)
            for j in ion_len_list:
                _s_pear=pearsonr(self.get_internal_pearson_x(merge_list,i,0,j),self.get_internal_pearson_x(merge_list,i,1,j))[0]
                if np.isnan(_s_pear):
                    continue
                else:
                    _key='ion_len_'+str(j)
                    if _key in person_list_dic:
                        person_list_dic[_key].append(_s_pear)
                    else:
                        person_list_dic[_key]=[]
                        person_list_dic[_key].append(_s_pear)
     
        print(np.mean(pearson_list))
        _list=[]
        for key,value in person_list_dic.items():
            _list.extend(value)
            print(len(value))
            print(key+': '+str(np.mean(value)))
            with open('data/ay_by_'+key+'.txt','w') as f:
                _str=''
                for pers_s  in value:
                    _str+=str(pers_s)+','
                f.write(_str.strip(','))
        print('all: '+str(np.mean(_list)))
    def get_regular_pearson(self,merge_list,test_idx,peptide,pred,test_y):
        person_list_dic=defaultdict(list)
        peplen_list_dic=defaultdict(list)
        pccs_for_plot_dic=defaultdict(list)
        pccs=[];peplens=[] 
        cunt0_9=0;cunt_all=0
        print('calculate person coefficient..')
        sum_person = 0.0
        for i in range(len(merge_list[0])):
            pccs.extend([pearsonr(self.get_regular_pearson_x(merge_list,i,0),self.get_regular_pearson_x(merge_list,i,1))[0]]*len(merge_list[0][i]))
            pep_len=len(merge_list[0][i])+1
            peplens.append(pep_len)
            for _len in range(50):
                if pep_len==_len:
                    _s_pear=pearsonr(self.get_regular_pearson_x(merge_list,i,0),self.get_regular_pearson_x(merge_list,i,1))[0]
                    cunt_all+=1
                    if np.isnan(_s_pear):
                        continue
                    
                    else:
                        if _s_pear>=0.9:
                            cunt0_9+=1
                        _key=_len
                        if _key in person_list_dic:
                            person_list_dic[_key].append(_s_pear)
                            peplen_list_dic[_key].append(pep_len)
                            #pccs_for_plot_dic[_key].extend([_s_pear]*len(merge_list[0][i]))
                        else:
                            person_list_dic[_key]=[]
                            person_list_dic[_key].append(_s_pear)
                            peplen_list_dic[_key]=[]
                            peplen_list_dic[_key].append(pep_len)
                            #pccs_for_plot_dic[_key]=[_s_pear]*len(merge_list[0][i])
                    break
           
       
        #sum_person=np.sum(person_list[i])
        #print(np.mean(eds))
        #_contrast = pd.DataFrame({"Number":test_idx,"Peptide":peptide,"predB1":np.array(pred)[:,0].tolist(),"predB2":np.array(pred)[:,1].tolist(),"predY1":np.array(pred)[:,2].tolist(),"predY2":np.array(pred)[:,3].tolist(),"realB1":test_y[:,0].tolist(),"realB2":test_y[:,1].tolist(),"realY1":test_y[:,2].tolist(),"realY2":test_y[:,3].tolist(),"PCC":pccs,"LEN":peplens})
        _contrast = pd.DataFrame({"Number":test_idx,"Peptide":peptide,"pred1":np.array(pred)[:,0].tolist(),"pred22":np.array(pred)[:,1].tolist(),"real1":test_y[:,0].tolist(),"real2":test_y[:,1].tolist(),"PCC":pccs,"LEN":peplens})
        _contrast.to_csv('data//_contrast.csv',index=False)

        _list=[]
        with open('data/mm_b_y_peplen_pep2inten.txt','w') as f:
            f.write(str(cunt0_9)+','+str(cunt_all)+'\n')
            for key,value in person_list_dic.items():
                _list.extend(value)
                
                print('pep len '+str(key)+': '+str(np.mean(value)))
                
                f.write(str(key)+','+str(np.mean(value))+'\n')
            
            print('all: '+str(np.mean(_list)))
    def get_len_regular_pearson(self,merge_list,test_idx,peptide,pred,test_y):
        person_list_dic=defaultdict(list)
        pccs=[]
    
        pep_len_list=[5,10,15,20,25,3000]
        print('calculate person coefficient..')
        sum_person = 0.0
        for i in range(len(merge_list[0])):
            pccs.extend([pearsonr(self.get_regular_pearson_x(merge_list,i,0),self.get_regular_pearson_x(merge_list,i,1))[0]]*len(merge_list[0][i]))
            pep_len=len(merge_list[0][i])+1
            for j in range(len(pep_len_list)):
                if pep_len>pep_len_list[j] and pep_len<=pep_len_list[j+1]:
                    _s_pear=pearsonr(self.get_regular_pearson_x(merge_list,i,0),self.get_regular_pearson_x(merge_list,i,1))[0]
                   
                    if np.isnan(_s_pear):
                        continue
                    
                    else:
                       
                        _key=str(pep_len_list[j]+1)+'-'+str(pep_len_list[j+1])
                        if _key in person_list_dic:
                            person_list_dic[_key].append(_s_pear)
                          
                          
                        else:
                            person_list_dic[_key]=[]
                            person_list_dic[_key].append(_s_pear)
                         
                    break
           
       
       
        _list=[]
     
        _contrast = pd.DataFrame({"Number":test_idx,"Peptide":peptide,"pred1":np.array(pred)[:,0].tolist(),"pred2":np.array(pred)[:,1].tolist(),"real1":test_y[:,0].tolist(),"real2":test_y[:,1].tolist(),"PCC":pccs})
        _contrast.to_csv('data//internal_contrast.csv',index=False)
        for key,value in person_list_dic.items():
            _list.extend(value)
            
            print('pep len '+str(key)+': '+str(np.mean(value)))
            
           
        
        print('all: '+str(np.mean(_list)))

    def write_pred(self,test_idx,peptide,dtrain_predictions,ions_len):
        print('write predict data in file')
        
      
        if self.ion_type == 'internal':
            pred = pd.DataFrame({"Number":test_idx,"Peptide":peptide,"IonLen":ions_len,"IntensityBy":np.array(dtrain_predictions)[:,0].tolist(),"IntensityAy":np.array(dtrain_predictions)[:,1].tolist()})
        elif self.ion_type == 'regular':
            pred = pd.DataFrame({"Number":test_idx,"Peptide":peptide,"IntensityB1":np.array(dtrain_predictions)[:,0].tolist(),"IntensityB2":np.array(dtrain_predictions)[:,1].tolist(),"IntensityY1":np.array(dtrain_predictions)[:,2].tolist(),"IntensityY2":np.array(dtrain_predictions)[:,3].tolist()})

        pred.to_csv('data//pred.csv',index=False)
        return pred
   
