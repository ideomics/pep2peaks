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
       'c':1,'m':10,'n':11,'MissV':-1}
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
        self.prev=1
        self.next=1
        self.aa2vector = self.AAVectorDict()
        self.AA_idx = dict(zip("ACDEFGHIKLMNPQRSTVWY",range(0,len(self.aa2vector))))
        self.ion_type=ion_type
    def AAVectorDict(self):
        aa2vector_map = {}
        s = "ACDEFGHIKLMNPQRSTVWY"
        v = [0]*len(s)
        v[0] = 1
        for i in range(len(s)):
            aa2vector_map[s[i]] = list(v)
            v[i],v[(i+1) % 20] = 0,1
        return aa2vector_map

    def get_split_list(self,array_list):
        list=[]
        for n in array_list:
            list.append(int(n[0]-1))
        return list
    def gbm_get_split_list(self,array_list):
        list=[]
        for n in array_list:
            list.append(n)
        return list
    def get_split_list2(self,array_list):
        list=[]
        for n in array_list:
            for m in n:
                list.append(int(m[0]-1))
        return list

    def ion_featurize_1label(self,line,fragmentation_window_size=1):
        vertor=[]
        f_line = line.split('\t')
        peptide_for_fragment=[]
        for i in range(fragmentation_window_size-1):
            peptide_for_fragment.append('MissV')
        peptide = list(f_line[0])
        peptide_for_fragment.extend(peptide)
        for i in range(fragmentation_window_size-1):
            peptide_for_fragment.append('MissV')
        charge=int(f_line[1])

        ion_type=f_line[5][0]

        ion = list(f_line[2])
        fragment_pos=0
        if ion_type=='y':
            fragment_pos=len(peptide)-len(ion)
        else:
            fragment_pos=len(ion)
        #获取碎裂点左右各fragmentation_window_size个氨基酸
        window_aa_list=peptide_for_fragment[fragment_pos-1:fragment_pos+2*fragmentation_window_size-1]  

        #### featurize start ###
        #肽中每种氨基酸出现的次数 20
        v_t=[0]*20
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vertor.extend(v_t)

        v_t=[0]*40
        #b,y碎片离子中每种氨基酸出的次数 40

        for key in self.dicS.keys():
            if key in ion:
                v_t[self.dicS[key]] = ion.count(key)

        if ion_type == 'y':        
            other_ion = peptide[:len(peptide) - len(ion)]
        else:
            other_ion = peptide[len(ion):]

        for key in self.dicS.keys():
            if key in other_ion:
                v_t[20 + self.dicS[key]] = other_ion.count(key)

              

        vertor.extend(v_t)


        #碎裂窗口，size=fragmentation_window_size 40*fragmentation_window_size
        j=0;v_t=[0]*40*fragmentation_window_size
        for i in range(len(window_aa_list)):
            aa_pos=self.dicS[window_aa_list[i]]
            if aa_pos!=-1:
                v_t[j*20+aa_pos]=1
            j+=1
        vertor.extend(v_t)

        #C/N端肽的身份 40
        v_t=[0]*40
        v_t[self.dicS[peptide[0]]]=1
        v_t[20+self.dicS[peptide[-1]]]=1
        vertor.extend(v_t) 

        #碎裂点是否在肽的一端 1
        if len(ion) == 1:
            vertor.extend([1])
        else:
            vertor.extend([0])


        #计算肽的质量、碱性、疏水性、螺旋性
        pep_mass = self.H2O+2*self.PROTON ;pep_bias = 0.0;pep = peptide;pep_he = 0;pep_hy=0;pep_p=0
        for i in range(len(peptide)):
            pep_mass += self.dicM[peptide[i]]
            pep_bias += self.dicB[peptide[i]]
            pep_he += self.dicHe[peptide[i]]
            pep_hy += self.dicHy[peptide[i]]
            pep_p += self.dicP[peptide[i]]
            

        #计算碎片离子的质量、碱性、疏水性、螺旋
        ion_mass = self.H2O + self.PROTON ;ion_bias = 0.0;ion_he=0;ion_hy=0;ion_p=0
        for i in range(len(ion)):
            ion_mass += self.dicM[ion[i]]
            ion_bias += self.dicB[ion[i]]
            ion_he += self.dicHe[ion[i]]
            ion_hy += self.dicHy[ion[i]]
            ion_p += self.dicP[ion[i]]

        #b,y离子的质量与肽质量的比
        t=ion_mass / pep_mass
        vertor.extend([t])
        vertor.extend([1-t])

        #碎裂点距肽N,C端的距离
        if ion_type == 'y':
            vertor.extend([len(peptide) - len(ion)])
            vertor.extend([len(ion)])
            
        else:
            vertor.extend([len(ion)])
            vertor.extend([len(peptide) - len(ion)])



        #碎裂点两端2*fragmentation_window_size个氨基酸的碱性、螺旋性、疏水性、等电点

        v_t=[0]*4*2*fragmentation_window_size
        for i in range(len(window_aa_list)):
            v_t[i]=self.dicB[window_aa_list[i]]
            v_t[1*2*fragmentation_window_size+i]=self.dicHe[window_aa_list[i]]
            v_t[2*2*fragmentation_window_size+i]=self.dicHy[window_aa_list[i]]
            v_t[3*2*fragmentation_window_size+i]=self.dicP[window_aa_list[i]]
        vertor.extend(v_t)

        #b,y离子的碱性、螺旋性、疏水性、等电点 
        vertor.extend([ion_bias])
        vertor.extend([pep_bias-ion_bias])
        vertor.extend([ion_he])
        vertor.extend([pep_he-ion_he])
        vertor.extend([ion_hy])
        vertor.extend([pep_hy-ion_hy])
        vertor.extend([ion_p])
        vertor.extend([pep_p-ion_p])
        
        #两个碎片离子的质量 
        vertor.extend([ion_mass])
        vertor.extend([pep_mass-ion_mass])
        

         #碎裂点两个氨基酸的碱性、螺旋性、疏水性的差值和均值
        v_t_2=[0]*8
        v_t_2[0] = (v_t[fragmentation_window_size-1] + v_t[fragmentation_window_size]) / 2.0
        v_t_2[1] = abs(v_t[fragmentation_window_size-1] - v_t[fragmentation_window_size])

        v_t_2[2] = (v_t[3*fragmentation_window_size-1] + v_t[3*fragmentation_window_size]) / 2.0
        v_t_2[3] = abs(v_t[3*fragmentation_window_size-1] - v_t[3*fragmentation_window_size])

        v_t_2[4] = (v_t[5*fragmentation_window_size-1] + v_t[5*fragmentation_window_size]) / 2.0
        v_t_2[5] = abs(v_t[5*fragmentation_window_size-1] - v_t[5*fragmentation_window_size])

        v_t_2[6] = (v_t[7*fragmentation_window_size-1] + v_t[7*fragmentation_window_size]) / 2.0
        v_t_2[7] = abs(v_t[7*fragmentation_window_size-1] - v_t[7*fragmentation_window_size])
        vertor.extend(v_t_2)

        #Y离子的质荷比减去肽的质荷比
        if ion_type== 'y':
            vertor.extend([ion_mass- pep_mass/charge])
        else:
            vertor.extend([pep_mass- ion_mass- pep_mass/charge])
           


        #肽,两个碎片中碱性氨基酸的个数
        pep_bias_count=peptide.count('K') + peptide.count("R") + peptide.count('H')
        ion_bias_count=ion.count('K') + ion.count('R') + ion.count('H')
        vertor.extend([pep_bias_count])
        vertor.extend([ion_bias_count])
        vertor.extend([pep_bias_count-ion_bias_count])
       

        #肽序列的长度
        vertor.extend([len(peptide)])

        #肽的碱性、疏水性、螺旋性
        vertor.extend([pep_bias])
        vertor.extend([pep_he])
        vertor.extend([pep_hy])
        vertor.extend([pep_p])
        

        #肽的质荷比
        vertor.extend([pep_mass / charge])
     
        
        #两个碎片离子长度跟肽长度的比值
        vertor.extend([len(ion) / float(len(peptide))])
        vertor.extend([1-(len(ion) / float(len(peptide)))])

       

        if ion_type == 'y':
            vertor.extend([0,1])
        else:
            vertor.extend([1,0])
        labels=float(f_line[6].replace('\n',''))
        
        return f_line[0],vertor,labels

    def ion_ay_by_featurize_2label(self,line,fragmentation_window_size=1):
        vertor=[]
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
        fragment_pos1=len(peptide)-int(f_line[5].split(',')[0][1:-1].split('b')[0])
        fragment_pos2=int(f_line[5].split(',')[0][1:-1].split('b')[1])

        

        #获取两个碎裂点左右各fragmentation_window_size个氨基酸
        window_aa_list1=peptide_for_fragment[fragment_pos1-1:fragment_pos1+2*fragmentation_window_size-1]  
        window_aa_list2=peptide_for_fragment[fragment_pos2-1:fragment_pos2+2*fragmentation_window_size-1]  

        ### featurize start ###
        #肽中每种氨基酸出现的次数 20
        v_t=[0]*20
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vertor.extend(v_t)

        v_t=[0]*20
        #碎片离子中每种氨基酸出的次数 20
        for key in self.dicS.keys():
            if key in ion:
                v_t[self.dicS[key]] = ion.count(key)

        vertor.extend(v_t)


        #碎裂窗口1，size=fragmentation_window_size 40*fragmentation_window_size
        j=0;v_t=[0]*40*fragmentation_window_size
        for i in range(len(window_aa_list1)):
            aa_pos=self.dicS[window_aa_list1[i]]
            if aa_pos!=-1:
                v_t[j*20+aa_pos]=1
            j+=1
        vertor.extend(v_t)

        #碎裂窗口2，size=fragmentation_window_size 40*fragmentation_window_size
        j=0;v_t=[0]*40*fragmentation_window_size
        for i in range(len(window_aa_list2)):
            aa_pos=self.dicS[window_aa_list2[i]]
            if aa_pos!=-1:
                v_t[j*20+aa_pos]=1
            j+=1
        vertor.extend(v_t)

        #C/N端肽的身份 40
        v_t=[0]*40;
        v_t[self.dicS[peptide[0]]]=1
        v_t[20+self.dicS[peptide[-1]]]=1
        vertor.extend(v_t) 

      
        #计算肽的质量、碱性、疏水性、螺旋性
        pep_mass = self.H2O+2*self.PROTON ;pep_bias = 0.0;pep = peptide;pep_he = 0;pep_hy=0;pep_p=0
        for i in range(len(peptide)):
            pep_mass += self.dicM[peptide[i]]
            pep_bias += self.dicB[peptide[i]]
            pep_he += self.dicHe[peptide[i]]
            pep_hy += self.dicHy[peptide[i]]
            pep_p += self.dicP[peptide[i]]
            

        #计算碎片离子的质量,碱性、疏水性、螺旋性
        ion_mass=self.PROTON;ion_bias = 0.0;ion_he=0;ion_hy=0;ion_p=0
        for i in range(len(ion)):
            ion_mass+=self.dicM[ion[i]]
            ion_bias += self.dicB[ion[i]]
            ion_he += self.dicHe[ion[i]]
            ion_hy += self.dicHy[ion[i]]
            ion_p += self.dicP[ion[i]]
      

        #by,ay离子的质量与肽质量的比
        vertor.extend([ion_mass / pep_mass])
        vertor.extend([(ion_mass-self.CO) / pep_mass])

        #碎裂点1距肽N,C端的距离
        vertor.extend([fragment_pos1])
        vertor.extend([len(peptide) - fragment_pos1])

        #碎裂点2距肽N,C端的距离
        vertor.extend([fragment_pos2])
        vertor.extend([len(peptide) - fragment_pos2])


        #碎裂点1两端2*fragmentation_window_size个氨基酸的碱性、螺旋性、疏水性、等电点 
        v_t1=[0]*4*2*fragmentation_window_size
        for i in range(len(window_aa_list1)):
            v_t1[i]=self.dicB[window_aa_list1[i]]
            v_t1[1*2*fragmentation_window_size+i]=self.dicHe[window_aa_list1[i]]
            v_t1[2*2*fragmentation_window_size+i]=self.dicHy[window_aa_list1[i]]
            v_t1[3*2*fragmentation_window_size+i]=self.dicP[window_aa_list1[i]]
        vertor.extend(v_t1)

        #碎裂点2两端2*fragmentation_window_size个氨基酸的碱性、螺旋性、疏水性、等电点 
        v_t2=[0]*4*2*fragmentation_window_size
        for i in range(len(window_aa_list2)):
            v_t2[i]=self.dicB[window_aa_list2[i]]
            v_t2[1*2*fragmentation_window_size+i]=self.dicHe[window_aa_list2[i]]
            v_t2[2*2*fragmentation_window_size+i]=self.dicHy[window_aa_list2[i]]
            v_t2[3*2*fragmentation_window_size+i]=self.dicP[window_aa_list2[i]]
        vertor.extend(v_t2)

        #碎片离子的质量,碱性、疏水性、螺旋性、等电点
        vertor.extend([ion_mass])
        vertor.extend([ion_bias])
        vertor.extend([ion_he])
        vertor.extend([ion_hy])
        vertor.extend([ion_p])
        
                

        #碎裂点1两个氨基酸的碱性、螺旋性、疏水性的差值和均值
        v_t_2=[0]*8
        v_t_2[0] = (v_t1[fragmentation_window_size-1] + v_t1[fragmentation_window_size]) / 2.0
        v_t_2[1] = abs(v_t1[fragmentation_window_size-1] - v_t1[fragmentation_window_size])

        v_t_2[2] = (v_t1[3*fragmentation_window_size-1] + v_t1[3*fragmentation_window_size]) / 2.0
        v_t_2[3] = abs(v_t1[3*fragmentation_window_size-1] - v_t1[3*fragmentation_window_size])

        v_t_2[4] = (v_t1[5*fragmentation_window_size-1] + v_t1[5*fragmentation_window_size]) / 2.0
        v_t_2[5] = abs(v_t1[5*fragmentation_window_size-1] - v_t1[5*fragmentation_window_size])

        v_t_2[6] = (v_t1[7*fragmentation_window_size-1] + v_t1[7*fragmentation_window_size]) / 2.0
        v_t_2[7] = abs(v_t1[7*fragmentation_window_size-1] - v_t1[7*fragmentation_window_size])
        vertor.extend(v_t_2)

        #碎裂点2两个氨基酸的碱性、螺旋性、疏水性的差值和均值
        v_t_2=[0]*8
        v_t_2[0] = (v_t2[fragmentation_window_size-1] + v_t2[fragmentation_window_size]) / 2.0
        v_t_2[1] = abs(v_t2[fragmentation_window_size-1] - v_t2[fragmentation_window_size])

        v_t_2[2] = (v_t2[3*fragmentation_window_size-1] + v_t2[3*fragmentation_window_size]) / 2.0
        v_t_2[3] = abs(v_t2[3*fragmentation_window_size-1] - v_t2[3*fragmentation_window_size])

        v_t_2[4] = (v_t2[5*fragmentation_window_size-1] + v_t2[5*fragmentation_window_size]) / 2.0
        v_t_2[5] = abs(v_t2[5*fragmentation_window_size-1] - v_t2[5*fragmentation_window_size])

        v_t_2[6] = (v_t2[7*fragmentation_window_size-1] + v_t2[7*fragmentation_window_size]) / 2.0
        v_t_2[7] = abs(v_t2[7*fragmentation_window_size-1] - v_t2[7*fragmentation_window_size])
        vertor.extend(v_t_2)

       


        #肽,碎片中碱性氨基酸的个数
        vertor.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vertor.extend([ion.count('K') + ion.count('R') + ion.count('H')])
       

        #肽序列的长度
        vertor.extend([len(peptide)])

        #肽的碱性、疏水性、螺旋性
        vertor.extend([pep_bias])
        vertor.extend([pep_he])
        vertor.extend([pep_hy])
        vertor.extend([pep_p])
        

        ##肽的质荷比
        vertor.extend([pep_mass / charge])
     
        
        #碎片离子长度跟肽长度的比值
        vertor.extend([len(ion) / float(len(peptide))])

        #肽带电量
        vchg = [0]*5
        vchg[charge-1] = 1
        vertor.extend(vchg)

        labels=[float(f_line[6].split(',')[0]),float(f_line[6].split(',')[1].replace('\n',''))]

        
        
        return f_line[0],vertor,labels,charge,len(ion)

    def ion_b_y_gbm_featurize_4label(self,line,fragmentation_window_size=1):
        
        vertor_b1=[];vertor_b2=[];vertor_y1=[];vertor_y2=[]
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
       

        

        #获取碎裂点左右各fragmentation_window_size个氨基酸
        window_aa_list=peptide_for_fragment[fragment_pos-1:fragment_pos+2*fragmentation_window_size-1]  

        ### featurize start ###
        #肽中每种氨基酸出现的次数 20
        v_t=[0]*20
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vertor_b1.extend(v_t)
        vertor_b2.extend(v_t)
        vertor_y1.extend(v_t)
        vertor_y2.extend(v_t)

        v_t=[0]*40
        #b碎片离子中每种氨基酸出的次数 20
        for key in self.dicS.keys():
            if key in ion_b:
                v_t[self.dicS[key]] = ion_b.count(key)

        #y碎片离子中每种氨基酸出的次数 20
        for key in self.dicS.keys():
            if key in ion_y:
                v_t[20 + self.dicS[key]] = ion_y.count(key)

       

        vertor_b1.extend(v_t)
        vertor_b2.extend(v_t)
        vertor_y1.extend(v_t)
        vertor_y2.extend(v_t)


        #碎裂窗口，size=fragmentation_window_size 40*fragmentation_window_size
        j=0;v_t=[0]*40*fragmentation_window_size
        for i in range(len(window_aa_list)):
            aa_pos=self.dicS[window_aa_list[i]]
            if aa_pos!=-1:
                v_t[j*20+aa_pos]=1
            j+=1
        vertor_b1.extend(v_t)
        vertor_b2.extend(v_t)
        vertor_y1.extend(v_t)
        vertor_y2.extend(v_t)

        #C/N端肽的身份 40
        v_t=[0]*40;
        v_t[self.dicS[peptide[0]]]=1
        v_t[20+self.dicS[peptide[-1]]]=1
        vertor_b1.extend(v_t)
        vertor_b2.extend(v_t)
        vertor_y1.extend(v_t)
        vertor_y2.extend(v_t)

        #碎裂点是否在肽的一端 1
        if len(ion_b) == 1:
            vertor_b1.extend([1])
            vertor_b2.extend([1])
            vertor_y1.extend([1])
            vertor_y2.extend([1])
        else:
            vertor_b1.extend([0])
            vertor_b2.extend([0])
            vertor_y1.extend([0])
            vertor_y2.extend([0]) 


        #计算肽的质量、碱性、疏水性、螺旋性
        pep_mass = self.H2O+2*self.PROTON ;pep_bias = 0.0;pep = peptide;pep_he = 0;pep_hy=0;pep_p=0
        for i in range(len(peptide)):
            pep_mass += self.dicM[peptide[i]]
            pep_bias += self.dicB[peptide[i]]
            pep_he += self.dicHe[peptide[i]]
            pep_hy += self.dicHy[peptide[i]]
            pep_p += self.dicP[peptide[i]]
            

        #计算b离子的质量、碱性、疏水性、螺旋性
        ion_b_mass = self.PROTON;ion_b_bias = 0.0;ion_b_he=0;ion_b_hy=0;ion_b_p=0
        for i in range(len(ion_b)):
            ion_b_mass += self.dicM[ion_b[i]]
            ion_b_bias += self.dicB[ion_b[i]]
            ion_b_he += self.dicHe[ion_b[i]]
            ion_b_hy += self.dicHy[ion_b[i]]
            ion_b_p += self.dicP[ion_b[i]]
        #计算y离子的质量、碱性、疏水性、螺旋性
        ion_y_mass = self.H2O + self.PROTON ;ion_y_bias = 0.0;ion_y_he=0;ion_y_hy=0;ion_y_p=0
        for i in range(len(ion_y)):
            ion_y_mass += self.dicM[ion_y[i]]
            ion_y_bias += self.dicB[ion_y[i]]
            ion_y_he += self.dicHe[ion_y[i]]
            ion_y_hy += self.dicHy[ion_y[i]]
            ion_y_p += self.dicP[ion_y[i]]

        #离子的质量与肽质量的比
        vertor_b1.extend([ion_b_mass / pep_mass])
        vertor_b2.extend([(ion_b_mass+self.PROTON) / pep_mass]) 
        vertor_y1.extend([ion_y_mass / pep_mass])
        vertor_y2.extend([(ion_y_mass+self.PROTON) / pep_mass])

        #碎片离子的质量 
        vertor_b1.extend([ion_b_mass])
        vertor_b2.extend([ion_b_mass+self.PROTON])
        vertor_y1.extend([ion_y_mass])
        vertor_y2.extend([ion_y_mass+self.PROTON])

        #离子质荷比
        vertor_b1.extend([ion_b_mass])
        vertor_b2.extend([(ion_b_mass+self.PROTON) / 2]) 
        vertor_y1.extend([ion_y_mass])
        vertor_y2.extend([(ion_y_mass+self.PROTON) / 2])
        
        #碎片离子电荷数
        vertor_b1.extend([1])
        vertor_b2.extend([2])
        vertor_y1.extend([1])
        vertor_y2.extend([2])

        #碎裂点距肽N,C端的距离
        vertor_b1.extend([len(ion_b)])
        vertor_b2.extend([len(ion_b)]) 
        vertor_y1.extend([len(peptide) - len(ion_b)])
        vertor_y2.extend([len(peptide) - len(ion_b)])
     


        #碎裂点两端2*fragmentation_window_size个氨基酸的碱性、螺旋性、疏水性、等电点 
        v_t=[0]*4*2*fragmentation_window_size
        for i in range(len(window_aa_list)):
            v_t[i]=self.dicB[window_aa_list[i]]
            v_t[1*2*fragmentation_window_size+i]=self.dicHe[window_aa_list[i]]
            v_t[2*2*fragmentation_window_size+i]=self.dicHy[window_aa_list[i]]
            v_t[3*2*fragmentation_window_size+i]=self.dicP[window_aa_list[i]]
        #vertor.extend(v_t)

        #b,y离子的碱性、螺旋性、疏水性、等电点 
        #vertor.extend([ion_b_bias])
        #vertor.extend([ion_y_bias])
        #vertor.extend([ion_b_he])
        #vertor.extend([ion_y_he])
        #vertor.extend([ion_b_hy])
        #vertor.extend([ion_y_hy])
        #vertor.extend([ion_b_p])
        #vertor.extend([ion_y_p])
        
        
        

         #碎裂点两个氨基酸的碱性、螺旋性、疏水性的差值和均值
        #v_t_2=[0]*8
        #v_t_2[0] = (v_t[fragmentation_window_size-1] + v_t[fragmentation_window_size]) / 2.0
        #v_t_2[1] = abs(v_t[fragmentation_window_size-1] - v_t[fragmentation_window_size])

        #v_t_2[2] = (v_t[3*fragmentation_window_size-1] + v_t[3*fragmentation_window_size]) / 2.0
        #v_t_2[3] = abs(v_t[3*fragmentation_window_size-1] - v_t[3*fragmentation_window_size])

        #v_t_2[4] = (v_t[5*fragmentation_window_size-1] + v_t[5*fragmentation_window_size]) / 2.0
        #v_t_2[5] = abs(v_t[5*fragmentation_window_size-1] - v_t[5*fragmentation_window_size])

        #v_t_2[6] = (v_t[7*fragmentation_window_size-1] + v_t[7*fragmentation_window_size]) / 2.0
        #v_t_2[7] = abs(v_t[7*fragmentation_window_size-1] - v_t[7*fragmentation_window_size])
        #vertor.extend(v_t_2)

        ##Y离子的质荷比减去肽的质荷比
        #vertor.extend([ion_y_mass- pep_mass/charge])


        #肽,两个碎片中碱性氨基酸的个数
        #vertor.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        #vertor.extend([ion_b.count('K') + ion_b.count('R') + ion_b.count('H')])
        #vertor.extend([ion_y.count('K') + ion_y.count('R') + ion_y.count('H')])
       

        #肽序列的长度
        vertor_b1.extend([len(peptide)])
        vertor_b2.extend([len(peptide)])
        vertor_y1.extend([len(peptide)])
        vertor_y2.extend([len(peptide)])

        #肽的碱性、疏水性、螺旋性
        #vertor.extend([pep_bias])
        #vertor.extend([pep_he])
        #vertor.extend([pep_hy])
        #vertor.extend([pep_p])
        

        ##肽的质荷比
        vertor_b1.extend([pep_mass / charge])
        vertor_b2.extend([pep_mass / charge])
        vertor_y1.extend([pep_mass / charge])
        vertor_y2.extend([pep_mass / charge])
     
        
        #碎片离子长度跟肽长度的比值
        vertor_b1.extend([len(ion_b) / float(len(peptide))])
        vertor_b2.extend([len(ion_b) / float(len(peptide))])
        vertor_y1.extend([len(ion_y) / float(len(peptide))])
        vertor_y2.extend([len(ion_y) / float(len(peptide))])

        #肽带电量
        vchg = [0]*5
        vchg[charge-1] = 1
        vertor_b1.extend(vchg)
        vertor_b2.extend(vchg)
        vertor_y1.extend(vchg)
        vertor_y2.extend(vchg)

     

        
        labels=[float(f_line[6].split(',')[0]),float(f_line[6].split(',')[1]),float(f_line[6].split(',')[2]),float(f_line[6].split(',')[3].replace('\n',''))]

        
        
        return f_line[0],vertor_b1,vertor_b2,vertor_y1,vertor_y2,labels

    def ion_b_y_featurize_1label_for_attention(self,line,fragmentation_window_size=1):
        vector_b1=[];vector_b2=[];vector_y1=[];vector_y2=[];vector=[]
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
       

        

        #获取碎裂点左右各fragmentation_window_size个氨基酸
        window_aa_list=peptide_for_fragment[fragment_pos-1:fragment_pos+2*fragmentation_window_size-1]  

        ### featurize start ###
        #1,肽中每种氨基酸出现的次数 20
        v_t=[0]*20
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vector_b1.extend(v_t)
        vector_b2.extend(v_t)
        vector_y1.extend(v_t)
        vector_y2.extend(v_t)

        v_t=[0]*20
        #b碎片离子中每种氨基酸出的次数 20
        for key in self.dicS.keys():
            if key in ion_b:
                v_t[self.dicS[key]] = ion_b.count(key)

        vector_b1.extend(v_t)
        vector_b2.extend(v_t)

        v_t=[0]*20
        #y碎片离子中每种氨基酸出的次数 20
        for key in self.dicS.keys():
            if key in ion_y:
                v_t[self.dicS[key]] = ion_y.count(key)

        vector_y1.extend(v_t)
        vector_y2.extend(v_t)


        #碎裂窗口，size=fragmentation_window_size 40*fragmentation_window_size
        j=0;v_t=[0]*40*fragmentation_window_size
        for i in range(len(window_aa_list)):
            aa_pos=self.dicS[window_aa_list[i]]
            if aa_pos!=-1:
                v_t[j*20+aa_pos]=1
            j+=1
        vector_b1.extend(v_t)
        vector_b2.extend(v_t)
        vector_y1.extend(v_t)
        vector_y2.extend(v_t)

        #C/N端肽的身份 40
        v_t=[0]*40;
        v_t[self.dicS[peptide[0]]]=1
        v_t[20+self.dicS[peptide[-1]]]=1

        vector_b1.extend(v_t)
        vector_b2.extend(v_t)
        vector_y1.extend(v_t)
        vector_y2.extend(v_t)

        #碎裂点是否在肽的一端 1
        if len(ion_b) == 1:
            vector_b1.extend([1])
            vector_b2.extend([1])
            vector_y1.extend([1])
            vector_y2.extend([1])
        else:
            vector_b1.extend([0])
            vector_b2.extend([0])
            vector_y1.extend([0])
            vector_y2.extend([0])


        #计算肽的质量、碱性、疏水性、螺旋性
        pep_mass = self.H2O+2*self.PROTON ;pep_bias = 0.0;pep = peptide;pep_he = 0;pep_hy=0;pep_p=0
        for i in range(len(peptide)):
            pep_mass += self.dicM[peptide[i]]
            pep_bias += self.dicB[peptide[i]]
            pep_he += self.dicHe[peptide[i]]
            pep_hy += self.dicHy[peptide[i]]
            pep_p += self.dicP[peptide[i]]
            

        #计算b离子的质量、碱性、疏水性、螺旋性
        ion_b_mass = self.PROTON;ion_b_bias = 0.0;ion_b_he=0;ion_b_hy=0;ion_b_p=0
        for i in range(len(ion_b)):
            ion_b_mass += self.dicM[ion_b[i]]
            ion_b_bias += self.dicB[ion_b[i]]
            ion_b_he += self.dicHe[ion_b[i]]
            ion_b_hy += self.dicHy[ion_b[i]]
            ion_b_p += self.dicP[ion_b[i]]
        #计算y离子的质量、碱性、疏水性、螺旋性
        ion_y_mass = self.H2O + self.PROTON ;ion_y_bias = 0.0;ion_y_he=0;ion_y_hy=0;ion_y_p=0
        for i in range(len(ion_y)):
            ion_y_mass += self.dicM[ion_y[i]]
            ion_y_bias += self.dicB[ion_y[i]]
            ion_y_he += self.dicHe[ion_y[i]]
            ion_y_hy += self.dicHy[ion_y[i]]
            ion_y_p += self.dicP[ion_y[i]]

        #b,y离子的质量与肽质量的比
        vector_b1.extend([ion_b_mass / pep_mass])
        vector_b2.extend([(ion_b_mass+self.PROTON) / pep_mass])

        vector_y1.extend([ion_y_mass / pep_mass])
        vector_y2.extend([(ion_y_mass+self.PROTON) / pep_mass])
        


        #碎裂点距肽N,C端的距离
        vector_b1.extend([len(ion_b)])
        vector_b2.extend([len(ion_b)])

        vector_y1.extend([len(peptide) - len(ion_b)])
        vector_y2.extend([len(peptide) - len(ion_b)])


        #碎裂点两端2*fragmentation_window_size个氨基酸的碱性、螺旋性、疏水性、等电点 
        v_t=[0]*4*2*fragmentation_window_size
        for i in range(len(window_aa_list)):
            v_t[i]=self.dicB[window_aa_list[i]]
            v_t[1*2*fragmentation_window_size+i]=self.dicHe[window_aa_list[i]]
            v_t[2*2*fragmentation_window_size+i]=self.dicHy[window_aa_list[i]]
            v_t[3*2*fragmentation_window_size+i]=self.dicP[window_aa_list[i]]
        vector_b1.extend(v_t)
        vector_b2.extend(v_t)
        vector_y1.extend(v_t)
        vector_y2.extend(v_t)

        #b,y离子的碱性、螺旋性、疏水性、等电点 
        vector_b1.extend([ion_b_bias])
        vector_b2.extend([ion_b_bias])
        vector_b1.extend([ion_b_he])
        vector_b2.extend([ion_b_he])
        vector_b1.extend([ion_b_hy])
        vector_b2.extend([ion_b_hy])
        vector_b1.extend([ion_b_p])
        vector_b2.extend([ion_b_p])

        vector_y1.extend([ion_y_bias])
        vector_y2.extend([ion_y_bias])
        vector_y1.extend([ion_y_he])
        vector_y2.extend([ion_y_he])
        vector_y1.extend([ion_y_hy])
        vector_y2.extend([ion_y_hy])
        vector_y1.extend([ion_y_p])
        vector_y2.extend([ion_y_p])
        
        #碎片离子的质量
        vector_b1.extend([ion_b_mass])
        vector_b2.extend([(ion_b_mass+self.PROTON)])
        vector_y1.extend([ion_y_mass])
        vector_y2.extend([(ion_y_mass+self.PROTON)])
       

         #碎裂点两个氨基酸的碱性、螺旋性、疏水性的差值和均值
        v_t_2=[0]*8
        v_t_2[0] = (v_t[fragmentation_window_size-1] + v_t[fragmentation_window_size]) / 2.0
        v_t_2[1] = abs(v_t[fragmentation_window_size-1] - v_t[fragmentation_window_size])

        v_t_2[2] = (v_t[3*fragmentation_window_size-1] + v_t[3*fragmentation_window_size]) / 2.0
        v_t_2[3] = abs(v_t[3*fragmentation_window_size-1] - v_t[3*fragmentation_window_size])

        v_t_2[4] = (v_t[5*fragmentation_window_size-1] + v_t[5*fragmentation_window_size]) / 2.0
        v_t_2[5] = abs(v_t[5*fragmentation_window_size-1] - v_t[5*fragmentation_window_size])

        v_t_2[6] = (v_t[7*fragmentation_window_size-1] + v_t[7*fragmentation_window_size]) / 2.0
        v_t_2[7] = abs(v_t[7*fragmentation_window_size-1] - v_t[7*fragmentation_window_size])
        vector_b1.extend(v_t_2)
        vector_b2.extend(v_t_2)
        vector_y1.extend(v_t_2)
        vector_y2.extend(v_t_2)

        ##Y离子的质荷比减去肽的质荷比
        vector_b1.extend([ion_y_mass- pep_mass/charge])
        vector_b2.extend([ion_y_mass- pep_mass/charge])
        vector_y1.extend([ion_y_mass- pep_mass/charge])
        vector_y2.extend([ion_y_mass- pep_mass/charge])


        #肽,两个碎片中碱性氨基酸的个数
        vector_b1.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vector_b2.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vector_y1.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vector_y2.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])

        vector_b1.extend([ion_b.count('K') + ion_b.count('R') + ion_b.count('H')])
        vector_b2.extend([ion_b.count('K') + ion_b.count('R') + ion_b.count('H')])

        vector_y1.extend([ion_y.count('K') + ion_y.count('R') + ion_y.count('H')])
        vector_y2.extend([ion_y.count('K') + ion_y.count('R') + ion_y.count('H')])

       
       

        #肽序列的长度

        vector_b1.extend([len(peptide)])
        vector_b2.extend([len(peptide)])
        vector_y1.extend([len(peptide)])
        vector_y2.extend([len(peptide)])

        #肽的碱性、疏水性、螺旋性
        
        vector_b1.extend([pep_bias])
        vector_b2.extend([pep_bias])
        vector_y1.extend([pep_bias])
        vector_y2.extend([pep_bias])

        vector_b1.extend([pep_he])
        vector_b2.extend([pep_he])
        vector_y1.extend([pep_he])
        vector_y2.extend([pep_he])
    
        vector_b1.extend([pep_hy])
        vector_b2.extend([pep_hy])
        vector_y1.extend([pep_hy])
        vector_y2.extend([pep_hy])

        vector_b1.extend([pep_p])
        vector_b2.extend([pep_p])
        vector_y1.extend([pep_p])
        vector_y2.extend([pep_p]) 
        
        ##肽的质荷比
     
        vector_b1.extend([pep_mass / charge])
        vector_b2.extend([pep_mass / charge])
        vector_y1.extend([pep_mass / charge])
        vector_y2.extend([pep_mass / charge])

        #两个碎片离子长度跟肽长度的比值
        

        vector_b1.extend([len(ion_b) / float(len(peptide))])
        vector_b2.extend([len(ion_b) / float(len(peptide))])
        vector_y1.extend([len(ion_y) / float(len(peptide))])
        vector_y2.extend([len(ion_y) / float(len(peptide))])
        #肽带电量
        vchg = [0]*5
        vchg[charge-1] = 1

        vector_b1.extend(vchg)
        vector_b2.extend(vchg)
        vector_y1.extend(vchg)
        vector_y2.extend(vchg)

         
        vector_b1.extend([0,1,0])
        vector_b2.extend([0,0,1])
        vector_y1.extend([0,1,0])
        vector_y2.extend([0,0,1])

        labels=[float(f_line[6].split(',')[0]),float(f_line[6].split(',')[1]),float(f_line[6].split(',')[2]),float(f_line[6].split(',')[3].replace('\n',''))]
        
        vector.append(vector_b1) 
        vector.append(vector_b2) 
        vector.append(vector_y1) 
        vector.append(vector_y2) 
        
        return f_line[0],vector,labels,charge

    def ion_b_y_featurize_4label(self,line,fragmentation_window_size=1):
        vertor=[]
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
       

        

        #获取碎裂点左右各fragmentation_window_size个氨基酸
        window_aa_list=peptide_for_fragment[fragment_pos-1:fragment_pos+2*fragmentation_window_size-1]  

        ### featurize start ###
        #肽中每种氨基酸出现的次数 20
        v_t=[0]*20
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vertor.extend(v_t)

        v_t=[0]*40
        #b碎片离子中每种氨基酸出的次数 20
        for key in self.dicS.keys():
            if key in ion_b:
                v_t[self.dicS[key]] = ion_b.count(key)

        #y碎片离子中每种氨基酸出的次数 20
        for key in self.dicS.keys():
            if key in ion_y:
                v_t[20 + self.dicS[key]] = ion_y.count(key)

       

        vertor.extend(v_t)


        #碎裂窗口，size=fragmentation_window_size 40*fragmentation_window_size
        j=0;v_t=[0]*40*fragmentation_window_size
        for i in range(len(window_aa_list)):
            aa_pos=self.dicS[window_aa_list[i]]
            if aa_pos!=-1:
                v_t[j*20+aa_pos]=1
            j+=1
        vertor.extend(v_t)

        #C/N端肽的身份 40
        v_t=[0]*40;
        v_t[self.dicS[peptide[0]]]=1
        v_t[20+self.dicS[peptide[-1]]]=1
        vertor.extend(v_t) 

        #碎裂点是否在肽的一端 1
        if len(ion_b) == 1:
            vertor.extend([1])
        else:
            vertor.extend([0])


        #计算肽的质量、碱性、疏水性、螺旋性
        pep_mass = self.H2O+2*self.PROTON ;pep_bias = 0.0;pep = peptide;pep_he = 0;pep_hy=0;pep_p=0
        for i in range(len(peptide)):
            pep_mass += self.dicM[peptide[i]]
            pep_bias += self.dicB[peptide[i]]
            pep_he += self.dicHe[peptide[i]]
            pep_hy += self.dicHy[peptide[i]]
            pep_p += self.dicP[peptide[i]]
            

        #计算b离子的质量、碱性、疏水性、螺旋性
        ion_b_mass = self.PROTON;ion_b_bias = 0.0;ion_b_he=0;ion_b_hy=0;ion_b_p=0
        for i in range(len(ion_b)):
            ion_b_mass += self.dicM[ion_b[i]]
            ion_b_bias += self.dicB[ion_b[i]]
            ion_b_he += self.dicHe[ion_b[i]]
            ion_b_hy += self.dicHy[ion_b[i]]
            ion_b_p += self.dicP[ion_b[i]]
        #计算y离子的质量、碱性、疏水性、螺旋性
        ion_y_mass = self.H2O + self.PROTON ;ion_y_bias = 0.0;ion_y_he=0;ion_y_hy=0;ion_y_p=0
        for i in range(len(ion_y)):
            ion_y_mass += self.dicM[ion_y[i]]
            ion_y_bias += self.dicB[ion_y[i]]
            ion_y_he += self.dicHe[ion_y[i]]
            ion_y_hy += self.dicHy[ion_y[i]]
            ion_y_p += self.dicP[ion_y[i]]

        #b,y离子的质量与肽质量的比
        vertor.extend([ion_b_mass / pep_mass])
        vertor.extend([ion_y_mass / pep_mass])

        #碎裂点距肽N,C端的距离
        vertor.extend([len(ion_b)])
        vertor.extend([len(peptide) - len(ion_b)])


        #碎裂点两端2*fragmentation_window_size个氨基酸的碱性、螺旋性、疏水性、等电点 
        v_t=[0]*4*2*fragmentation_window_size
        for i in range(len(window_aa_list)):
            v_t[i]=self.dicB[window_aa_list[i]]
            v_t[1*2*fragmentation_window_size+i]=self.dicHe[window_aa_list[i]]
            v_t[2*2*fragmentation_window_size+i]=self.dicHy[window_aa_list[i]]
            v_t[3*2*fragmentation_window_size+i]=self.dicP[window_aa_list[i]]
        vertor.extend(v_t)

        #b,y离子的碱性、螺旋性、疏水性、等电点 
        vertor.extend([ion_b_bias])
        vertor.extend([ion_y_bias])
        vertor.extend([ion_b_he])
        vertor.extend([ion_y_he])
        vertor.extend([ion_b_hy])
        vertor.extend([ion_y_hy])
        vertor.extend([ion_b_p])
        vertor.extend([ion_y_p])
        
        #两个碎片离子的质量 
        vertor.extend([ion_b_mass])
        vertor.extend([ion_y_mass])
        

         #碎裂点两个氨基酸的碱性、螺旋性、疏水性的差值和均值
        v_t_2=[0]*8
        v_t_2[0] = (v_t[fragmentation_window_size-1] + v_t[fragmentation_window_size]) / 2.0
        v_t_2[1] = abs(v_t[fragmentation_window_size-1] - v_t[fragmentation_window_size])

        v_t_2[2] = (v_t[3*fragmentation_window_size-1] + v_t[3*fragmentation_window_size]) / 2.0
        v_t_2[3] = abs(v_t[3*fragmentation_window_size-1] - v_t[3*fragmentation_window_size])

        v_t_2[4] = (v_t[5*fragmentation_window_size-1] + v_t[5*fragmentation_window_size]) / 2.0
        v_t_2[5] = abs(v_t[5*fragmentation_window_size-1] - v_t[5*fragmentation_window_size])

        v_t_2[6] = (v_t[7*fragmentation_window_size-1] + v_t[7*fragmentation_window_size]) / 2.0
        v_t_2[7] = abs(v_t[7*fragmentation_window_size-1] - v_t[7*fragmentation_window_size])
        vertor.extend(v_t_2)

        ##Y离子的质荷比减去肽的质荷比
        vertor.extend([ion_y_mass- pep_mass/charge])


        #肽,两个碎片中碱性氨基酸的个数
        vertor.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vertor.extend([ion_b.count('K') + ion_b.count('R') + ion_b.count('H')])
        vertor.extend([ion_y.count('K') + ion_y.count('R') + ion_y.count('H')])
       

        #肽序列的长度
        vertor.extend([len(peptide)])

        #肽的碱性、疏水性、螺旋性
        vertor.extend([pep_bias])
        vertor.extend([pep_he])
        vertor.extend([pep_hy])
        vertor.extend([pep_p])
        

        ##肽的质荷比
        vertor.extend([pep_mass / charge])
     
        
        #两个碎片离子长度跟肽长度的比值
        vertor.extend([len(ion_b) / float(len(peptide))])
        vertor.extend([len(ion_y) / float(len(peptide))])

        #肽带电量
        v_t = [0]*5
        v_t[charge-1] = 1
        vertor.extend(v_t)

        labels=[float(f_line[6].split(',')[0]),float(f_line[6].split(',')[1]),float(f_line[6].split(',')[2]),float(f_line[6].split(',')[3].replace('\n',''))]
        
        #if is_train:
        #    v_t=[0]*4
         #   for k in range(self.number_label):
          #      if labels[k]>0.0:
           #         v_t[k]=1
        #    vertor.extend(v_t)
     #   else:
      #      v_t=classes.split('\t')[1].replace('\n','').split(',')
    #        vertor.extend([int(float(_v)) for _v in v_t])

        
        

        
        
        return f_line[0],vertor,labels,charge
    def ion_b_y_featurize_4label_for_classfication(self,line,fragmentation_window_size=1):
        vertor=[]
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
       

        

        #获取碎裂点左右各fragmentation_window_size个氨基酸
        window_aa_list=peptide_for_fragment[fragment_pos-1:fragment_pos+2*fragmentation_window_size-1]  

        ### featurize start ###
        #肽中每种氨基酸出现的次数 23
        v_t=[0]*23
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vertor.extend(v_t)

        v_t=[0]*40
        #b碎片离子中每种氨基酸出的次数 23
        for key in self.dicS.keys():
            if key in ion_b:
                v_t[self.dicS[key]] = ion_b.count(key)

        #y碎片离子中每种氨基酸出的次数 23
        for key in self.dicS.keys():
            if key in ion_y:
                v_t[23 + self.dicS[key]] = ion_y.count(key)

       

        vertor.extend(v_t)
        
        
        
        #碎裂窗口，size=fragmentation_window_size 40*fragmentation_window_size
        j=0;v_t=[0]*46*fragmentation_window_size
        for i in range(len(window_aa_list)):
            aa_pos=self.dicS[window_aa_list[i]]
            if aa_pos!=-1:
                v_t[j*23+aa_pos]=1
            j+=1
        vertor.extend(v_t)

        #C/N端肽的身份 40
        v_t=[0]*46;
        v_t[self.dicS[peptide[0]]]=1
        v_t[23+self.dicS[peptide[-1]]]=1
        vertor.extend(v_t) 

        #碎裂点是否在肽的一端 1
        if len(ion_b) == 1:
            vertor.extend([1])
        else:
            vertor.extend([0])


        #计算肽的质量、碱性、疏水性、螺旋性
        #pep_mass = self.H2O+2*self.PROTON ;pep_bias = 0.0;pep = peptide;pep_he = 0;pep_hy=0;pep_p=0
        #for i in range(len(peptide)):
        #    pep_mass += self.dicM[peptide[i]]
        #    pep_bias += self.dicB[peptide[i]]
        #    pep_he += self.dicHe[peptide[i]]
        #    pep_hy += self.dicHy[peptide[i]]
        #    pep_p += self.dicP[peptide[i]]
        #    

        ##计算b离子的质量、碱性、疏水性、螺旋性
        #ion_b_mass = self.PROTON;ion_b_bias = 0.0;ion_b_he=0;ion_b_hy=0;ion_b_p=0
        #for i in range(len(ion_b)):
        #    ion_b_mass += self.dicM[ion_b[i]]
        #    ion_b_bias += self.dicB[ion_b[i]]
        #    ion_b_he += self.dicHe[ion_b[i]]
        #    ion_b_hy += self.dicHy[ion_b[i]]
        #    ion_b_p += self.dicP[ion_b[i]]
        ##计算y离子的质量、碱性、疏水性、螺旋性
        #ion_y_mass = self.H2O + self.PROTON ;ion_y_bias = 0.0;ion_y_he=0;ion_y_hy=0;ion_y_p=0
        #for i in range(len(ion_y)):
        #    ion_y_mass += self.dicM[ion_y[i]]
        #    ion_y_bias += self.dicB[ion_y[i]]
        #    ion_y_he += self.dicHe[ion_y[i]]
        #    ion_y_hy += self.dicHy[ion_y[i]]
        #    ion_y_p += self.dicP[ion_y[i]]

        #b,y离子的质量与肽质量的比
        #vertor.extend([ion_b_mass / pep_mass])
        #vertor.extend([ion_y_mass / pep_mass])

        #碎裂点距肽N,C端的距离
        vertor.extend([len(ion_b)])
        vertor.extend([len(peptide) - len(ion_b)])


        #碎裂点两端2*fragmentation_window_size个氨基酸的碱性、螺旋性、疏水性、等电点 
        #v_t=[0]*4*2*fragmentation_window_size
        #for i in range(len(window_aa_list)):
        #    v_t[i]=self.dicB[window_aa_list[i]]
        #    v_t[1*2*fragmentation_window_size+i]=self.dicHe[window_aa_list[i]]
        #    v_t[2*2*fragmentation_window_size+i]=self.dicHy[window_aa_list[i]]
        #    v_t[3*2*fragmentation_window_size+i]=self.dicP[window_aa_list[i]]
        #vertor.extend(v_t)

        #b,y离子的碱性、螺旋性、疏水性、等电点 
       # vertor.extend([ion_b_bias])
       # vertor.extend([ion_y_bias])
       # vertor.extend([ion_b_he])
       # vertor.extend([ion_y_he])
       # vertor.extend([ion_b_hy])
       # vertor.extend([ion_y_hy])
       # vertor.extend([ion_b_p])
       # vertor.extend([ion_y_p])
        
        #两个碎片离子的质量 
        #vertor.extend([ion_b_mass])
        #vertor.extend([ion_y_mass])
        

         #碎裂点两个氨基酸的碱性、螺旋性、疏水性的差值和均值
        #v_t_2=[0]*8
        #v_t_2[0] = (v_t[fragmentation_window_size-1] + v_t[fragmentation_window_size]) / 2.0
        #v_t_2[1] = abs(v_t[fragmentation_window_size-1] - v_t[fragmentation_window_size])

        #v_t_2[2] = (v_t[3*fragmentation_window_size-1] + v_t[3*fragmentation_window_size]) / 2.0
        #v_t_2[3] = abs(v_t[3*fragmentation_window_size-1] - v_t[3*fragmentation_window_size])

        #v_t_2[4] = (v_t[5*fragmentation_window_size-1] + v_t[5*fragmentation_window_size]) / 2.0
        #v_t_2[5] = abs(v_t[5*fragmentation_window_size-1] - v_t[5*fragmentation_window_size])

        #v_t_2[6] = (v_t[7*fragmentation_window_size-1] + v_t[7*fragmentation_window_size]) / 2.0
        #v_t_2[7] = abs(v_t[7*fragmentation_window_size-1] - v_t[7*fragmentation_window_size])
        #vertor.extend(v_t_2)

        ##Y离子的质荷比减去肽的质荷比
        #vertor.extend([ion_y_mass- pep_mass/charge])


        #肽,两个碎片中碱性氨基酸的个数
        vertor.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vertor.extend([ion_b.count('K') + ion_b.count('R') + ion_b.count('H')])
        vertor.extend([ion_y.count('K') + ion_y.count('R') + ion_y.count('H')])
       

        #肽序列的长度
        vertor.extend([len(peptide)])

        #肽的碱性、疏水性、螺旋性
        #vertor.extend([pep_bias])
        #vertor.extend([pep_he])
        #vertor.extend([pep_hy])
        #vertor.extend([pep_p])
        

        ##肽的质荷比
        #vertor.extend([pep_mass / charge])
     
        
        #两个碎片离子长度跟肽长度的比值
        vertor.extend([len(ion_b) / float(len(peptide))])
        vertor.extend([len(ion_y) / float(len(peptide))])

        #肽带电量
        vchg = [0]*5
        vchg[charge-1] = 1
        vertor.extend(vchg)

        labels=[float(f_line[6].split(',')[0]),float(f_line[6].split(',')[1]),float(f_line[6].split(',')[2]),float(f_line[6].split(',')[3].replace('\n',''))]
       

        
        

        
        
        return f_line[0],vertor,labels

    def ion_b_y_featurize_4label_only_seq(self,line,fragmentation_window_size=1):
        vertor=[]
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
       

        

        #获取碎裂点左右各fragmentation_window_size个氨基酸
        window_aa_list=peptide_for_fragment[fragment_pos-1:fragment_pos+2*fragmentation_window_size-1]  

        ### featurize start ###
        #肽中每种氨基酸出现的次数 23
        v_t=[0]*23
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vertor.extend(v_t)

        v_t=[0]*46
        #b碎片离子中每种氨基酸出的次数 23
        for key in self.dicS.keys():
            if key in ion_b:
                v_t[self.dicS[key]] = ion_b.count(key)

        #y碎片离子中每种氨基酸出的次数 23
        for key in self.dicS.keys():
            if key in ion_y:
                v_t[23 + self.dicS[key]] = ion_y.count(key)

       

        vertor.extend(v_t)
        
        
        
        #碎裂窗口，size=fragmentation_window_size 40*fragmentation_window_size
        j=0;v_t=[0]*46*fragmentation_window_size
        for i in range(len(window_aa_list)):
            aa_pos=self.dicS[window_aa_list[i]]
            if aa_pos!=-1:
                v_t[j*23+aa_pos]=1
            j+=1
        vertor.extend(v_t)

        #C/N端肽的身份 40
        v_t=[0]*46;
        v_t[self.dicS[peptide[0]]]=1
        v_t[23+self.dicS[peptide[-1]]]=1
        vertor.extend(v_t) 

        #碎裂点是否在肽的一端 1
        if len(ion_b) == 1:
            vertor.extend([1])
        else:
            vertor.extend([0])


        #计算肽的质量、碱性、疏水性、螺旋性
        #pep_mass = self.H2O+2*self.PROTON ;pep_bias = 0.0;pep = peptide;pep_he = 0;pep_hy=0;pep_p=0
        #for i in range(len(peptide)):
        #    pep_mass += self.dicM[peptide[i]]
        #    pep_bias += self.dicB[peptide[i]]
        #    pep_he += self.dicHe[peptide[i]]
        #    pep_hy += self.dicHy[peptide[i]]
        #    pep_p += self.dicP[peptide[i]]
        #    

        ##计算b离子的质量、碱性、疏水性、螺旋性
        #ion_b_mass = self.PROTON;ion_b_bias = 0.0;ion_b_he=0;ion_b_hy=0;ion_b_p=0
        #for i in range(len(ion_b)):
        #    ion_b_mass += self.dicM[ion_b[i]]
        #    ion_b_bias += self.dicB[ion_b[i]]
        #    ion_b_he += self.dicHe[ion_b[i]]
        #    ion_b_hy += self.dicHy[ion_b[i]]
        #    ion_b_p += self.dicP[ion_b[i]]
        ##计算y离子的质量、碱性、疏水性、螺旋性
        #ion_y_mass = self.H2O + self.PROTON ;ion_y_bias = 0.0;ion_y_he=0;ion_y_hy=0;ion_y_p=0
        #for i in range(len(ion_y)):
        #    ion_y_mass += self.dicM[ion_y[i]]
        #    ion_y_bias += self.dicB[ion_y[i]]
        #    ion_y_he += self.dicHe[ion_y[i]]
        #    ion_y_hy += self.dicHy[ion_y[i]]
        #    ion_y_p += self.dicP[ion_y[i]]

        #b,y离子的质量与肽质量的比
        #vertor.extend([ion_b_mass / pep_mass])
        #vertor.extend([ion_y_mass / pep_mass])

        #碎裂点距肽N,C端的距离
        vertor.extend([len(ion_b)])
        vertor.extend([len(peptide) - len(ion_b)])


        #碎裂点两端2*fragmentation_window_size个氨基酸的碱性、螺旋性、疏水性、等电点 
        #v_t=[0]*4*2*fragmentation_window_size
        #for i in range(len(window_aa_list)):
        #    v_t[i]=self.dicB[window_aa_list[i]]
        #    v_t[1*2*fragmentation_window_size+i]=self.dicHe[window_aa_list[i]]
        #    v_t[2*2*fragmentation_window_size+i]=self.dicHy[window_aa_list[i]]
        #    v_t[3*2*fragmentation_window_size+i]=self.dicP[window_aa_list[i]]
        #vertor.extend(v_t)

        #b,y离子的碱性、螺旋性、疏水性、等电点 
       # vertor.extend([ion_b_bias])
       # vertor.extend([ion_y_bias])
       # vertor.extend([ion_b_he])
       # vertor.extend([ion_y_he])
       # vertor.extend([ion_b_hy])
       # vertor.extend([ion_y_hy])
       # vertor.extend([ion_b_p])
       # vertor.extend([ion_y_p])
        
        #两个碎片离子的质量 
        #vertor.extend([ion_b_mass])
        #vertor.extend([ion_y_mass])
        

         #碎裂点两个氨基酸的碱性、螺旋性、疏水性的差值和均值
        #v_t_2=[0]*8
        #v_t_2[0] = (v_t[fragmentation_window_size-1] + v_t[fragmentation_window_size]) / 2.0
        #v_t_2[1] = abs(v_t[fragmentation_window_size-1] - v_t[fragmentation_window_size])

        #v_t_2[2] = (v_t[3*fragmentation_window_size-1] + v_t[3*fragmentation_window_size]) / 2.0
        #v_t_2[3] = abs(v_t[3*fragmentation_window_size-1] - v_t[3*fragmentation_window_size])

        #v_t_2[4] = (v_t[5*fragmentation_window_size-1] + v_t[5*fragmentation_window_size]) / 2.0
        #v_t_2[5] = abs(v_t[5*fragmentation_window_size-1] - v_t[5*fragmentation_window_size])

        #v_t_2[6] = (v_t[7*fragmentation_window_size-1] + v_t[7*fragmentation_window_size]) / 2.0
        #v_t_2[7] = abs(v_t[7*fragmentation_window_size-1] - v_t[7*fragmentation_window_size])
        #vertor.extend(v_t_2)

        ##Y离子的质荷比减去肽的质荷比
        #vertor.extend([ion_y_mass- pep_mass/charge])


        #肽,两个碎片中碱性氨基酸的个数
        vertor.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vertor.extend([ion_b.count('K') + ion_b.count('R') + ion_b.count('H')])
        vertor.extend([ion_y.count('K') + ion_y.count('R') + ion_y.count('H')])
       

        #肽序列的长度
        vertor.extend([len(peptide)])

        #肽的碱性、疏水性、螺旋性
        #vertor.extend([pep_bias])
        #vertor.extend([pep_he])
        #vertor.extend([pep_hy])
        #vertor.extend([pep_p])
        

        ##肽的质荷比
        #vertor.extend([pep_mass / charge])
     
        
        #两个碎片离子长度跟肽长度的比值
        vertor.extend([len(ion_b) / float(len(peptide))])
        vertor.extend([len(ion_y) / float(len(peptide))])

        #肽带电量
        vchg = [0]*5
        vchg[charge-1] = 1
        vertor.extend(vchg)

        labels=[float(f_line[6].split(',')[0]),float(f_line[6].split(',')[1]),float(f_line[6].split(',')[2]),float(f_line[6].split(',')[3].replace('\n',''))]
       

        
        

        
        
        return f_line[0],vertor,labels

    def ion_b_y_featurize_2label(self,line,fragmentation_window_size=1):
        vertor=[]
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
       

        

        #获取碎裂点左右各fragmentation_window_size个氨基酸
        window_aa_list=peptide_for_fragment[fragment_pos-1:fragment_pos+2*fragmentation_window_size-1]  

        ### featurize start ###
        #肽中每种氨基酸出现的次数 20
        v_t=[0]*20
        for key in self.dicS.keys():
            if key in peptide:
                v_t[self.dicS[key]] = peptide.count(key)
        vertor.extend(v_t)

        v_t=[0]*40
        #b碎片离子中每种氨基酸出的次数 20
        for key in self.dicS.keys():
            if key in ion_b:
                v_t[self.dicS[key]] = ion_b.count(key)

        #y碎片离子中每种氨基酸出的次数 20
        for key in self.dicS.keys():
            if key in ion_y:
                v_t[20 + self.dicS[key]] = ion_y.count(key)

       

        vertor.extend(v_t)


        #碎裂窗口，size=fragmentation_window_size 40*fragmentation_window_size
        j=0;v_t=[0]*40*fragmentation_window_size
        for i in range(len(window_aa_list)):
            aa_pos=self.dicS[window_aa_list[i]]
            if aa_pos!=-1:
                v_t[j*20+aa_pos]=1
            j+=1
        vertor.extend(v_t)

        #C/N端肽的身份 40
        v_t=[0]*40;
        v_t[self.dicS[peptide[0]]]=1
        v_t[20+self.dicS[peptide[-1]]]=1
        vertor.extend(v_t) 

        #碎裂点是否在肽的一端 1
        if len(ion_b) == 1:
            vertor.extend([1])
        else:
            vertor.extend([0])


        #计算肽的质量、碱性、疏水性、螺旋性
        pep_mass = self.H2O+2*self.PROTON ;pep_bias = 0.0;pep = peptide;pep_he = 0;pep_hy=0;pep_p=0
        for i in range(len(peptide)):
            pep_mass += self.dicM[peptide[i]]
            pep_bias += self.dicB[peptide[i]]
            pep_he += self.dicHe[peptide[i]]
            pep_hy += self.dicHy[peptide[i]]
            pep_p += self.dicP[peptide[i]]
            

        #计算b离子的质量、碱性、疏水性、螺旋性
        ion_b_mass = self.PROTON;ion_b_bias = 0.0;ion_b_he=0;ion_b_hy=0;ion_b_p=0
        for i in range(len(ion_b)):
            ion_b_mass += self.dicM[ion_b[i]]
            ion_b_bias += self.dicB[ion_b[i]]
            ion_b_he += self.dicHe[ion_b[i]]
            ion_b_hy += self.dicHy[ion_b[i]]
            ion_b_p += self.dicP[ion_b[i]]
        #计算y离子的质量、碱性、疏水性、螺旋性
        ion_y_mass = self.H2O + self.PROTON ;ion_y_bias = 0.0;ion_y_he=0;ion_y_hy=0;ion_y_p=0
        for i in range(len(ion_y)):
            ion_y_mass += self.dicM[ion_y[i]]
            ion_y_bias += self.dicB[ion_y[i]]
            ion_y_he += self.dicHe[ion_y[i]]
            ion_y_hy += self.dicHy[ion_y[i]]
            ion_y_p += self.dicP[ion_y[i]]

        #b,y离子的质量与肽质量的比
        vertor.extend([ion_b_mass / pep_mass])
        vertor.extend([ion_y_mass / pep_mass])

        #碎裂点距肽N,C端的距离
        vertor.extend([len(ion_b)])
        vertor.extend([len(peptide) - len(ion_b)])


        #碎裂点两端2*fragmentation_window_size个氨基酸的碱性、螺旋性、疏水性、等电点 
        v_t=[0]*4*2*fragmentation_window_size
        for i in range(len(window_aa_list)):
            v_t[i]=self.dicB[window_aa_list[i]]
            v_t[1*2*fragmentation_window_size+i]=self.dicHe[window_aa_list[i]]
            v_t[2*2*fragmentation_window_size+i]=self.dicHy[window_aa_list[i]]
            v_t[3*2*fragmentation_window_size+i]=self.dicP[window_aa_list[i]]
        vertor.extend(v_t)

        #b,y离子的碱性、螺旋性、疏水性、等电点 
        vertor.extend([ion_b_bias])
        vertor.extend([ion_y_bias])
        vertor.extend([ion_b_he])
        vertor.extend([ion_y_he])
        vertor.extend([ion_b_hy])
        vertor.extend([ion_y_hy])
        vertor.extend([ion_b_p])
        vertor.extend([ion_y_p])
        
        #两个碎片离子的质量 
        vertor.extend([ion_b_mass])
        vertor.extend([ion_y_mass])
        

         #碎裂点两个氨基酸的碱性、螺旋性、疏水性的差值和均值
        v_t_2=[0]*8
        v_t_2[0] = (v_t[fragmentation_window_size-1] + v_t[fragmentation_window_size]) / 2.0
        v_t_2[1] = abs(v_t[fragmentation_window_size-1] - v_t[fragmentation_window_size])

        v_t_2[2] = (v_t[3*fragmentation_window_size-1] + v_t[3*fragmentation_window_size]) / 2.0
        v_t_2[3] = abs(v_t[3*fragmentation_window_size-1] - v_t[3*fragmentation_window_size])

        v_t_2[4] = (v_t[5*fragmentation_window_size-1] + v_t[5*fragmentation_window_size]) / 2.0
        v_t_2[5] = abs(v_t[5*fragmentation_window_size-1] - v_t[5*fragmentation_window_size])

        v_t_2[6] = (v_t[7*fragmentation_window_size-1] + v_t[7*fragmentation_window_size]) / 2.0
        v_t_2[7] = abs(v_t[7*fragmentation_window_size-1] - v_t[7*fragmentation_window_size])
        vertor.extend(v_t_2)

        ##Y离子的质荷比减去肽的质荷比
        vertor.extend([ion_y_mass- pep_mass/charge])


        #肽,两个碎片中碱性氨基酸的个数
        vertor.extend([peptide.count('K') + peptide.count("R") + peptide.count('H')])
        vertor.extend([ion_b.count('K') + ion_b.count('R') + ion_b.count('H')])
        vertor.extend([ion_y.count('K') + ion_y.count('R') + ion_y.count('H')])
       

        #肽序列的长度
        vertor.extend([len(peptide)])

        #肽的碱性、疏水性、螺旋性
        vertor.extend([pep_bias])
        vertor.extend([pep_he])
        vertor.extend([pep_hy])
        vertor.extend([pep_p])
        

        ##肽的质荷比
        vertor.extend([pep_mass / charge])
     
        
        #两个碎片离子长度跟肽长度的比值
        vertor.extend([len(ion_b) / float(len(peptide))])
        vertor.extend([len(ion_y) / float(len(peptide))])

        #肽带电量
        vchg = [0]*5
        vchg[charge-1] = 1
        vertor.extend(vchg)
        
        labels=[float(f_line[6].split(',')[0]),float(f_line[6].split(',')[2]).replace('\n','')]

        
        
        return f_line[0],vertor,labels

    def ion_featurize_4label_pdeep(self,line,fragmentation_window_size=3):

        f_line = line.split('\t')
        peptide = f_line[0].upper()
        charge=int(f_line[1])
        ion_b = list(f_line[2].split(',')[0])
        ionidx = len(ion_b)
        seqidx=ionidx
        ion_type=f_line[5][0]
        # look at the ion's previous "prev" N_term AA
        v = []
        for i in range(seqidx - self.prev, seqidx):
            if i < 0:
                v.extend([0]*len(self.aa2vector))
            else:
                v.extend(self.aa2vector[peptide[i]])
        # look at the ion's next "next" C_term AAs
        for i in range(seqidx, seqidx + self.next):
            if i >= len(peptide):
                v.extend([0]*len(self.aa2vector))
            else:
                v.extend(self.aa2vector[peptide[i]])
        
        #the number of each AA before "prev" in NTerm
        NTerm_AA_Count = [0]*len(self.aa2vector)
        for i in range(seqidx - self.prev):
            NTerm_AA_Count[self.AA_idx[peptide[i]]] += 1
        v.extend(NTerm_AA_Count)
        
        #the number of each AA after "next" in CTerm
        CTerm_AA_Count = [0]*len(self.aa2vector)
        for i in range(seqidx + self.next, len(peptide)):
            CTerm_AA_Count[self.AA_idx[peptide[i]]] += 1
        v.extend(CTerm_AA_Count)
        
        if ionidx == 1: CTerm = 1
        else: CTerm = 0
        if ionidx == len(peptide)-1: NTerm = 1
        else: NTerm = 0
        v.extend([NTerm,CTerm])
        
        

        vchg = [0]*6
        vchg[charge-1] = 1
        v.extend(vchg)

        labels=[float(f_line[6].split(',')[0]),float(f_line[6].split(',')[1]),float(f_line[6].split(',')[2]),float(f_line[6].split(',')[3].replace('\n',''))]
        #labels=[float(f_line[6].split(',')[1]),float(f_line[6].split(',')[0].replace('\n',''))]
        #labels=float(f_line[6].replace('\n',''))
        return f_line[0],v,labels


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
                    temp_list.append([row.Number,row.IntensityBy,row.IntensityAy,row.IonLen])
                else:
                    intensity_list.append(temp_list)
                    temp_list = []
                    temp_list.append([row.Number,row.IntensityBy,row.IntensityAy,row.IonLen])
                temp_peptide = peptide
            else:
                temp_list.append([row.Number,row.IntensityBy,row.IntensityAy,row.IonLen])
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
    def vertor_normalize(self,data):
        num=data.shape[1]
        for i in range(num):
            _max=max(data[:,i])
            _min=min(data[:,i])
            if _max != _min:
                data[:,i]=(data[:,i]-_min)/(_max-_min)
        return data
    def plot(self,data):
        num=5
        inten_space=np.linspace(0,1,num=num)
        inten_space=np.hstack((inten_space,1.1))
        plt.hist(data[:,0], bins=1000, alpha=0.7)
        #for i in range(4):
        #    i_data=data[:,i]
        #    cunt=[];intens=[]
        #    print(len(i_data))
        #    print(np.sum(i_data==0.0))
        #    print(np.sum(i_data>0.1))
        #    for j in range(num):
        #        cunt.append(np.sum((i_data>=inten_space[j])&(i_data<inten_space[j+1])))
        #    #dict_i=dict(zip(*np.unique(i_data, return_counts=True)))
        #    #i_inten=list(dict_i.keys())
        #    #i_inten_cunt=list(dict_i.values())
        #    plt.plot(np.linspace(0,1,num=num),cunt,label=str(i))
        #plt.legend()
        plt.show()
    def get_data(self,path,internal_len):
        print('loading data...')
        X=[];y=[];idx=[];cunt=0;peptides=[];ions_len=[]
        with open(path,'r') as rf:
            while True:
                line=rf.readline()
                if not line:
                    break
                if len(line.split('\t')[2])==1 or len(line.split('\t')[2])==2:
                    if self.ion_type == 'internal':
                        pep,vertor,label,charge,ion_len=self.ion_ay_by_featurize_2label(line)
                        ions_len.append(ion_len)
                    elif self.ion_type == 'regular':
                        pep,vertor,label,charge=self.ion_b_y_featurize_4label(line)
                    
                    cunt+=1
                    X.append(vertor)
                    y.append(label)
                    peptides.append(pep+'#'+str(charge))
                    
                    idx.append(cunt)
    
       
        if self.ion_type == 'internal':
            merge_dataframe = pd.DataFrame({"Number":idx,"Peptide":peptides,"IonLen":ions_len,"IntensityBy":np.array(y)[:,0].tolist(),"IntensityAy":np.array(y)[:,1].tolist()})
       
        elif self.ion_type == 'regular':
            merge_dataframe = pd.DataFrame({"Number":idx,"Peptide":peptides,"IntensityB1":np.array(y)[:,0].tolist(),"IntensityB2":np.array(y)[:,1].tolist(),"IntensityY1":np.array(y)[:,2].tolist(),"IntensityY2":np.array(y)[:,3].tolist()})
       
        merge_list=self.merge_list_ap(merge_dataframe)
        return np.array(idx),np.array(peptides),self.vertor_normalize(np.array(X,dtype=np.float32)),np.array(y),merge_list,ions_len
    def get_features(self,path):
        print('loading data...')
        X=[];y=[];idx=[];cunt=0;peptides=[]
        with open(path,'r') as rf:
            while True:
                line=rf.readline()
                if not line:
                    break
                if len(line.split('\t')[0])<=20:
                   
                    pep,vector,label,charge=self.ion_b_y_featurize_1label_for_attention(line)
                    
                    for i in range(4):
                        cunt+=1
                        X.append(vector[i])
                        y.append(label[i])
                        peptides.append(pep+'#'+str(charge))
                        idx.append(cunt)
        
        
        merge_dataframe = pd.DataFrame({"Number":idx,"Peptide":peptides,"Intensity":y})
        merge_list=self.merge_list_ap(merge_dataframe)
        return np.array(idx),np.array(peptides),self.vertor_normalize(np.array(X,dtype=np.float32)),np.array(y),merge_list
 
    def get_discretization_data(self,path,num_classes):
        print('loading data...')
        X=[];y=[];idx=[];cunt=0;peptides=[]
        with open(path,'r') as rf:
            while True:
                line=rf.readline()
                
                if not line:
                    break
        
                if self.number_label == 2:
                    pep,vertor,label=self.ion_ay_by_featurize_2label(line)
                elif self.number_label == 1:
                    pep,vertor,label=self.ion_featurize_1label(line)
                elif self.number_label == 4:
                    pep,vertor,label=self.ion_featurize_4label_pdeep(line)
                cunt+=1 
                X.append(vertor)
                y.append(label) 
                peptides.append(pep)
                idx.append(cunt) 
             
                #for k in range(self.number_label):
                #    vertor.extend(k+1)
                #    cunt+=1
                #    y.append(label[k])
                #    peptides.append(pep)
                #    idx.append(cunt)
            
        merge_dataframe = pd.DataFrame({"Number":idx,"Peptide":peptides,"Intensity":y})
        merge_dataframe.to_csv('temp.csv')
        merge_list=self.merge_list_ap(merge_dataframe,True)

        return idx,peptides,self.vertor_normalize(np.array(X,dtype=np.float32)),np.array(self.discretization(y,num_classes)),merge_list