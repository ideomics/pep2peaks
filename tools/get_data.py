import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
class GetData(object):
    def __init__(self,ion_type):
        self.dicB={'A':206.4,'C':206.2,'D':208.6,'E':215.6,'F':212.1,\
      'G':202.7,'H':223.7,'I':210.8,'K':221.8,'L':209.6,\
      'M':213.3,'N':212.8,'P':214.4,'Q':214.2,'R':237.0,\
      'S':207.6,'T':211.7,'V':208.7,'W':216.1,'Y':213.1,\
       'c':206.2,'m':213.3,'n':212.8,'MissV':0}#碱性
        self.dicM={'A':71.037114,'C':103.009185,'D':115.026943,'E':129.042593,'F':147.068414,\
      'G':57.021464,'H':137.058912,'I':113.084064,'K':128.094963,'L':113.084064,\
      'M':131.040485,'N':114.042927,'P':97.052764,'Q':128.058578,'R':156.101111,\
      'S':87.032028,'T':101.047678,'V':99.068414,'W':186.079313,'Y':163.063329,\
       'c':160.0306486796,'m':147.035399708,'n':115.026943025,'MissV':0
      }
        self.dicS={'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,\
      'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,\
      'S':15,'T':16,'V':17,'W':18,'Y':19,\
       'c':20,'m':21,'n':22,'MissV':-1}
        self.dicHe={'A':1.24,'C':0.79,'D':0.89,'E':0.85,'F':1.26,'G':1.15,'H':0.97,\
       'I':1.29,'K':0.88,'L':1.28,'M':1.22,'N':0.94,'P':0.57,'Q':0.96,\
       'R':0.95,'S':1.00,'T':1.09,'V':1.27,'W':1.07,'Y':1.11,\
       'c':0.79,'m':1.22,'n':0.94,'MissV':0}#螺旋性
        self.dicHy={'A':0.16,'C':2.50,'D':-2.49,'E':-1.50,'F':5.00,'G':-3.31,'H':-4.63,'I':4.41,
      'K':-5.00,'L':4.76,'M':3.23,'N':-3.79,'P':-4.92,'Q':-2.76,'R':-2.77,'S':-2.85,
      'T':-1.08,'V':3.02,'W':4.88,'Y':2.00,\
       'c':2.50,'m':3.23,'n':-3.79,'MissV':0}#疏水性
        self.dicP={'A':6.02,'C':5.02,'D':2.97,'E':3.22,'F':5.48,'G':5.97,'H':7.59,'I':6.02,
      'K':9.74,'L':5.98,'M':5.75,'N':5.41,'P':6.30,'Q':5.65,'R':10.76,'S':5.68,
      'T':6.53,'V':5.97,'W':5.89,'Y':5.66,\
       'c':5.02,'m':5.75,'n':5.41,'MissV':0}#等电点
        self.PROTON= 1.007276466583
        self.H = 1.0078250322
        self.O = 15.9949146221
        self.C = 12.00
        self.CO = self.C + self.O
        self.H2O=self.H * 2 + self.O
      
       
        self.ion_type=ion_type
    def get_split_list(self,array_list):
       
        list=[]
        for n in array_list:
            list.append(int(int(n[0])-1))
        return list
   
    def get_split_list2(self,array_list):
        list=[]
        for n in array_list:
            for m in n:
                list.append(int(m[0]-1))
        return list

  

    def ion2vec_2label(self,line,fragmentation_window_size=1):
        vector=[]
        f_line = line.split('\t')
        peptide_for_fragment=[]
        for i in range(fragmentation_window_size-1):
            peptide_for_fragment.append('MissV')
        peptide = list(f_line[0])
        peptide_for_fragment.extend(peptide)
        for i in range(fragmentation_window_size-1):
            peptide_for_fragment.append('MissV')
        charge=int(f_line[1])

        ion = list(f_line[2])
        fragment_pos1=len(peptide)-int(f_line[4].split(',')[0][1:-1].split('b')[0])
        fragment_pos2=int(f_line[4].split(',')[0][1:-1].split('b')[1])

        window_aa_list1=peptide_for_fragment[fragment_pos1-1:fragment_pos1+2*fragmentation_window_size-1]  
        window_aa_list2=peptide_for_fragment[fragment_pos2-1:fragment_pos2+2*fragmentation_window_size-1]  

        ### featurize start ###
        # number of each AA in a peptide 23
        v_t=[0]*23
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vector.extend(v_t)

        
        # number of each AA in a fragment 23
        v_t=[0]*23
        for key in self.dicS.keys():
            if key in ion:
                v_t[self.dicS[key]] = ion.count(key)
        vector.extend(v_t)


        # fragmentation point 1， 46*fragmentation_window_size
        j=0;v_t=[0]*46*fragmentation_window_size
        for i in range(len(window_aa_list1)):
            aa_pos=self.dicS[window_aa_list1[i]]
            if aa_pos!=-1:
                v_t[j*23+aa_pos]=1
            j+=1
        vector.extend(v_t)

        # fragmentation point 2， 46*fragmentation_window_size
        j=0;v_t=[0]*46*fragmentation_window_size
        for i in range(len(window_aa_list2)):
            aa_pos=self.dicS[window_aa_list2[i]]
            if aa_pos!=-1:
                v_t[j*23+aa_pos]=1
            j+=1
        vector.extend(v_t)

        # AA of C/N-terminal 46
        v_t=[0]*46
        v_t[self.dicS[peptide[0]]]=1
        v_t[23+self.dicS[peptide[-1]]]=1
        vector.extend(v_t) 

      

        # distance from the fragmentation point 1 to the C/N-terminal
        vector.extend([fragment_pos1])
        vector.extend([len(peptide) - fragment_pos1])

        # distance from the fragmentation point 2 to the C/N-terminal
        vector.extend([fragment_pos2])
        vector.extend([len(peptide) - fragment_pos2])


        # number of basic AA in peptide or fragment

        vector.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vector.extend([ion.count('K') + ion.count('R') + ion.count('H')])
       
        #length of fragment
        vector.extend([len(ion)])
        # length of peptide
        vector.extend([len(peptide)])
        # charge
        v_t = [0]*5
        v_t[charge-1] = 1
        vector.extend(v_t)
        #intensity contribution
        vector.extend([float(f_line[6])])

        labels=list(map(eval,f_line[5].split(',')))

        return f_line[0],vector,labels,charge,f_line[2]

  

    def ion2vec_4label(self,line,fragmentation_window_size=1):
        vector=[]
        f_line = line.split('\t')
        peptide_for_fragment=[]
        for i in range(fragmentation_window_size-1):
            peptide_for_fragment.append('MissV')
        peptide = list(f_line[0])
        peptide_for_fragment.extend(peptide)
        for i in range(fragmentation_window_size-1):
            peptide_for_fragment.append('MissV')
        charge=int(f_line[1])


        ion_b = list(f_line[2].split(',')[0])
        ion_y = list(f_line[2].split(',')[1])
        fragment_pos=len(ion_b)
       
        window_aa_list=peptide_for_fragment[fragment_pos-1:fragment_pos+2*fragmentation_window_size-1]  

        ## featurize start ###
        # number of each AA in a peptide 23
        v_t=[0]*23
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vector.extend(v_t)

        v_t=[0]*46
        # number of each AA in a b-ion 23
        for key in self.dicS.keys():
            if key in ion_b:
                v_t[self.dicS[key]] = ion_b.count(key)

        # number of each AA in a y-ion 23
        for key in self.dicS.keys():
            if key in ion_y:
                v_t[23 + self.dicS[key]] = ion_y.count(key)
        vector.extend(v_t)


        # fragmentation point，46*fragmentation_window_size
        j=0;v_t=[0]*46*fragmentation_window_size
        for i in range(len(window_aa_list)):
            aa_pos=self.dicS[window_aa_list[i]]
            if aa_pos!=-1:
                v_t[j*23+aa_pos]=1
            j+=1
        vector.extend(v_t)

        # AA of C/N-terminal 46
        v_t=[0]*46;
        v_t[self.dicS[peptide[0]]]=1
        v_t[23+self.dicS[peptide[-1]]]=1
        vector.extend(v_t) 

        # is the fragmentation point at the end or begin
        if len(ion_b) == 1:
            vector.extend([1])
        else:
            vector.extend([0])
        
        
       # num=peptide.count('K') + peptide.count("R") + peptide.count('H')
       # if num < charge:
       #     vector.extend([1,0,0])
       # elif peptide.count('R') >= charge:
       #     vector.extend([0,1,0])
       # else:
       #     vector.extend([0,0,1])
        # number of basic AA in peptide or fragment
        vector.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vector.extend([ion_b.count('K') + ion_b.count('R') + ion_b.count('H')])
        vector.extend([ion_y.count('K') + ion_y.count('R') + ion_y.count('H')])
       
        # distance from the fragmentation point  to the C/N-terminal
        vector.extend([len(ion_b)])
        vector.extend([len(peptide) - len(ion_b)])
        # length of peptide
        vector.extend([len(peptide)])
       

        # charge
        v_t = [0]*5
        v_t[charge-1] = 1
        vector.extend(v_t)

       
        

        labels=list(map(eval,f_line[5].split(',')))
      
        
        return f_line[0],vector,labels,charge

   

    def discretization(self,label,num_classes):
        data=np.array(label.copy())

        #0-1离散
       
        data[data>0]=1
        #data[data>0]=1


        ##聚类离散
        #kmodel = KMeans(n_clusters = num_classes)
        #kmodel.fit(data.reshape((len(data), 1)))
        #c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
        #w=c.rolling(window=2, center=False).mean().iloc[1:]
        ##w = pd.rolling_mean(c, 2).iloc[1:]
        #w = [0] + list(w[0]) + [data.max()]
        #d = pd.cut(data, w, labels = range(num_classes))

        #p=np.array(d)
        #where_are_nan = np.isnan(p)
        #p[where_are_nan]=0
        
        ##等宽离散
        #d = pd.cut(data, num_classes, labels = range(num_classes))
        #p=np.array(d)

        ##等频离散
        #w = [1.0*i/num_classes for i in range(num_classes+1)]
        #is_zero=data.isin([0])
        #is_one=data.isin([1])
        #data=data[(True ^ data.isin([0,1]))]
        #data=data.append(pd.Series([0,1-1e-10]))
        #c = data.describe(percentiles = w)[4:4+num_classes+1] 
        #d = pd.cut(label, c, labels = range(1,num_classes+1))
        #p=np.array(d)
    
        #p[is_zero]=0
        #p[is_one]=num_classes+1

        ##label_encoder = LabelEncoder()
        ##integer_encoded = label_encoder.fit_transform(p.astype('int32'))
        ##print(integer_encoded)
        ##print(integer_encoded.shape)

        ##onehot_encoder = OneHotEncoder(sparse=False)
        ##integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        ##onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        ##print(onehot_encoded.shape)
        ##return onehot_encoded
        
        return data.astype('int32')

    def merge_list_2label(self,data):
        #Number,Intensity
        temp_peptide = ''
        temp_list = []
        intensity_list = []
        #print(data)
        for row in data.itertuples():
            peptide = row.Peptide
            if peptide != temp_peptide:
                if temp_peptide == '':
                    temp_list.append([row.Number,row.IntensityBy,row.IntensityAy,row.ion])
                else:
                    intensity_list.append(temp_list)
                    temp_list = []
                    temp_list.append([row.Number,row.IntensityBy,row.IntensityAy,row.ion])
                temp_peptide = peptide
            else:
                temp_list.append([row.Number,row.IntensityBy,row.IntensityAy,row.ion])
        intensity_list.append(temp_list)
        return intensity_list
    def merge_list_4label(self,data):
        #Number,Intensity
        temp_peptide = ''
        temp_list = []
        intensity_list = []
        #print(data)
        for row in data.itertuples():
            peptide = row.Peptide
            if peptide != temp_peptide:
                if temp_peptide == '':
                    temp_list.append([row.Number,row.IntensityB1,row.IntensityB2,row.IntensityY1,row.IntensityY2])
                else:
                    intensity_list.append(temp_list)
                    temp_list = []
                    temp_list.append([row.Number,row.IntensityB1,row.IntensityB2,row.IntensityY1,row.IntensityY2])
                temp_peptide = peptide
            else:
                temp_list.append([row.Number,row.IntensityB1,row.IntensityB2,row.IntensityY1,row.IntensityY2])
        intensity_list.append(temp_list)
        return intensity_list
    def merge_list_1label(self,data):
        #Number,Intensity
        temp_peptide = ''
        temp_list = []
        intensity_list = []
        #print(data)
        for row in data.itertuples():
            peptide = row.Peptide
            if peptide != temp_peptide:
                if temp_peptide == '':
                    temp_list.append([row.Number,row.Intensity])
                else:
                    intensity_list.append(temp_list)
                    temp_list = []
                    temp_list.append([row.Number,row.Intensity])
                temp_peptide = peptide
            else:
                temp_list.append([row.Number,row.Intensity])
        intensity_list.append(temp_list)
        return intensity_list

    def merge_list_ap(self,data,is_discre=False):
        merge_list=[]
        if is_discre:
            merge_list.append(self.merge_list_1label(data))
        else:
            if self.ion_type == 'internal':
                merge_list.append(self.merge_list_2label(data))
            elif self.ion_type == 'regular':
                merge_list.append(self.merge_list_4label(data))
          
        return merge_list
    def vector_normalize(self,data):
        num=data.shape[1]
        for i in range(114,num):
            _max=max(data[:,i])
            _min=min(data[:,i])
            if _max != _min:
                data[:,i]=(data[:,i]-_min)/(_max-_min)
        return data
    
    def get_data(self,path,min_internal_len,max_internal_len,is_mobile):
        print('loading data...')
        X=[];y=[];idx=[];cunt=0;peptides=[];ions=[]
       
        all_lines=[];mobile_lines=[];non_mobile_lines=[];partial_mobiles_lines=[]
        with open(path,'r') as rf:
                while True:
                    line=rf.readline()
                    if not line:
                        break
                    all_lines.append(line)
                    peptide = line.split('\t')[0]
                    charge = int(line.split('\t')[1])

                    num = peptide.count('R')+peptide.count('K')+peptide.count('H')
                    if num < charge:
                        mobile_lines.append(line)
                    elif peptide.count('R') >= charge:
                        non_mobile_lines.append(line)
                    else:
                        partial_mobiles_lines.append(line)

        if is_mobile=='mobile':
            lines=mobile_lines
        elif is_mobile=='non_mobile':
            lines=non_mobile_lines
        elif is_mobile=='partial_mobile':
            lines=partial_mobiles_lines
        else:
            lines=all_lines

        if self.ion_type == 'internal': 
            for line in lines:
                if len(line.split('\t')[2])>=min_internal_len and len(line.split('\t')[2])<=max_internal_len:

                    pep,vector,label,charge,ion=self.ion2vec_2label(line)
                    ions.append(ion)
                    cunt+=1
                    X.append(vector)
                    y.append(label)
                    peptides.append(pep+'#'+str(charge))
                    idx.append(cunt)
                   
            merge_dataframe = pd.DataFrame({"Number":idx,"Peptide":peptides,"ion":ions,"IntensityBy":np.array(y)[:,0].tolist(),"IntensityAy":np.array(y)[:,1].tolist()}) 
        elif self.ion_type == 'regular':
            for line in lines:
                
                if len(line.split('\t')[0])<=20:
                    pep,vector,label,charge=self.ion2vec_4label(line)
                    cunt+=1
                    X.append(vector)
                    y.append(label)
                    peptides.append(pep+'#'+str(charge))
                    idx.append(cunt) 
            merge_dataframe = pd.DataFrame({"Number":idx,"Peptide":peptides,"IntensityB1":np.array(y)[:,0].tolist(),"IntensityB2":np.array(y)[:,1].tolist(),"IntensityY1":np.array(y)[:,2].tolist(),"IntensityY2":np.array(y)[:,3].tolist()})
       
        merge_list=self.merge_list_ap(merge_dataframe)
        return np.array(idx),np.array(peptides), np.array(X,dtype=np.float32),np.array(y),merge_list,ions
   
  