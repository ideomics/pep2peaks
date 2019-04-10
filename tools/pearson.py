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
    def get_pearson_x(self,merge_list,pos):
        merged_same_ions=[];_real=[];_pred=[];ions=[]
        if self.ion_type=='regular':
            for j in range(len(merge_list[0][pos])):
                for k in range(1,5):
                    _real.append(merge_list[0][pos][j][k])
                    _pred.append(merge_list[1][pos][j][k])
        elif self.ion_type=='internal':
            
            _real_by=np.array(merge_list[0][pos])[:,1].astype(np.float)
            _real_ay=np.array(merge_list[0][pos])[:,2].astype(np.float)

            _pred_by=np.array(merge_list[1][pos])[:,1].astype(np.float)
            _pred_ay=np.array(merge_list[1][pos])[:,2].astype(np.float)
            
            ions=np.array(merge_list[0][pos])[:,3].tolist()

            d = defaultdict(list)
            for k,va in [(v,i) for i,v in enumerate(ions)]:
                d[k].append(va) 
            for _,indexs in d.items():
                _real.append(np.sum(np.array(_real_by)[indexs]))
                _real.append(np.sum(np.array(_real_ay)[indexs]))

                _pred.append(np.sum(np.array(_pred_by)[indexs]))
                _pred.append(np.sum(np.array(_pred_ay)[indexs]))
        
        return _real,_pred
    
   
    def get_pearson(self,merge_list,test_idx,peptide,pred,test_y):
        person_list_dic=defaultdict(list)
        pccs=[];peplens=[]
    
        pep_len_list=[5,10,15,20,25,3000]
        print('calculate person coefficient..')
        sum_person = 0.0
        for i in range(len(merge_list[0])):
            _real,_pred=self.get_pearson_x(merge_list,i)
            pccs.extend([pearsonr(_real,_pred)[0]]*len(merge_list[0][i]))
            if self.ion_type=='internal':
               
                pep_len=(len(merge_list[0][i])+5)/2
            else:
                pep_len=len(merge_list[0][i])+1
            
            peplens.extend([pep_len]*len(merge_list[0][i]))
            for j in range(len(pep_len_list)):
                if pep_len>pep_len_list[j] and pep_len<=pep_len_list[j+1]:
                   
                    _s_pear=pearsonr(_real,_pred)[0]
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
        if self.ion_type=='internal': 
            _contrast = pd.DataFrame({"Number":test_idx,"Peptide":peptide,"pred1":np.array(pred)[:,0].tolist(),"pred2":np.array(pred)[:,1].tolist(),"real1":test_y[:,0].tolist(),"real2":test_y[:,1].tolist(),"PCC":pccs,"LEN":peplens})
            _contrast.to_csv('data/pred/internal_pred.csv',index=False)
        elif self.ion_type=='regular': 
            _contrast = pd.DataFrame({"Number":test_idx,"Peptide":peptide,"predB1":np.array(pred)[:,0].tolist(),"predB2":np.array(pred)[:,1].tolist(),"predY1":np.array(pred)[:,2].tolist(),"predY2":np.array(pred)[:,3].tolist(),"realB1":test_y[:,0].tolist(),"realB2":test_y[:,1].tolist(),"realY1":test_y[:,2].tolist(),"realY2":test_y[:,3].tolist(),"PCC":pccs,"LEN":peplens})
            _contrast.to_csv('data/pred/regular_pred.csv',index=False)
        _list=[]
       
        for key,value in person_list_dic.items():
            _list.extend(value)
            print('pep len '+str(key)+': '+str(np.mean(value)))
            with open('data/pred/'+self.ion_type+'_'+key+'.txt','w') as f:
                _str=''
                for pers_s  in value:
                    _str+=str(pers_s)+','
                f.write(_str.strip(','))
        print('all: '+str(np.mean(_list)))

    def write_pred(self,test_idx,peptide,dtrain_predictions,ions):
        print('write predict data in file')
        
      
        if self.ion_type == 'internal':
            pred = pd.DataFrame({"Number":test_idx,"Peptide":peptide,"ion":ions,"IntensityBy":np.array(dtrain_predictions)[:,0].tolist(),"IntensityAy":np.array(dtrain_predictions)[:,1].tolist()})
        elif self.ion_type == 'regular':
            pred = pd.DataFrame({"Number":test_idx,"Peptide":peptide,"IntensityB1":np.array(dtrain_predictions)[:,0].tolist(),"IntensityB2":np.array(dtrain_predictions)[:,1].tolist(),"IntensityY1":np.array(dtrain_predictions)[:,2].tolist(),"IntensityY2":np.array(dtrain_predictions)[:,3].tolist()})

        #pred.to_csv('data//pred.csv',index=False)
        return pred
   
