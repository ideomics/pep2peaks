import os
from collections import defaultdict
import numpy as np

dicM={'A':71.037114,'C':103.009185,'D':115.026943,'E':129.042593,'F':147.068414,\
      'G':57.021464,'H':137.058912,'I':113.084064,'K':128.094963,'L':113.084064,\
      'M':131.040485,'N':114.042927,'P':97.052764,'Q':128.058578,'R':156.101111,\
      'S':87.032028,'T':101.047678,'V':99.068414,'W':186.079313,'Y':163.063329,\
       'c':160.0306486796,'m':147.035399708,'n':115.026943025
      }

PROTON= 1.007276466583
H = 1.0078250322
O = 15.9949146221


C = 12.00

H2O= H * 2 + O
CO = C + O
def delete_all(filePath):
            if os.path.exists(filePath):
                for fileList in os.walk(filePath):
                    for name in fileList[2]:
                        os.chmod(os.path.join(fileList[0],name), stat.S_IWRITE)
                        os.remove(os.path.join(fileList[0],name))
                shutil.rmtree(filePath)
                return "delete ok"
            else:
                return "no filepath"
def makedir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder
def get_files(rootdir):
    files=[]
    names=[]
    _list=os.listdir(rootdir)
    for i in range(0,len(_list)):
        path=os.path.join(rootdir,_list[i])
        if os.path.isfile(path):
            files.append(path)
            names.append(_list[i])
    return files,names

def get_spectrums(file_name,_type):
        dic=defaultdict(list)
        with open(file_name,'r') as rf:
            while True:
                line = rf.readline()
                if not line:
                    break
                if 'BEGIN IONS' in line:
                    
                    line=rf.readline()
                    _spec_t=line.split('=')[1].replace("\n", "")
                    
                    for step in range(2):
                        rf.__next__()
                    if _type=='proteometools':
                        rf.__next__()
                    if _type=='mm':
                        _nce=0
                    else:
                        _nce=int(rf.readline().split('=')[1].replace("\n", ""))
                    _spec=_spec_t+'#'+str(_nce)
                    dic[_spec]=[]
                    if _type=='proteometools':
                        for step in range(2):
                            rf.__next__()
                    _pepmass=float(rf.readline().split('=')[1].replace("\n", ""))
                    rf.__next__()
                    line=rf.readline()
                    temp_list=[]
                    while 'END IONS' not in line:
                        #if abs(float(line.split()[0]) - _pepmass)> 0.02:
                        temp_=[]
                        temp_.append(float(line.split()[0]))
                        temp_.append(float(line.split()[1]))
                        temp_list.append(temp_)
                        line=rf.readline()
                    max_=max(np.array(temp_list)[:,1])
                    for i in range(len(temp_list)):
                        temp_list[i].append(float(temp_list[i][1]/max_))
                        temp_list[i].append(max_)
                        dic[_spec].append(temp_list[i])
                    i=0
        return dic
def get_psms(file_name):
    psms=[]
    with open(file_name,'r') as rf:
        while True:
            line=rf.readline()
            if not line:
                break
            psm=[] #spectrum,charge,peptide,modification,nce
            psm.append(line.split('\t')[2])
            psm.append(line.split('\t')[3])
            peptide_list_=list(line.split('\t')[4])
            modification_=line.split('\t')[5]
            if 'NULL' not in modification_:
                modification_list_=modification_.split(',')
                i=0
                while i < len(modification_list_):
                    if modification_list_[i+1] == 'Oxidation[M]':
                        peptide_list_[int(modification_list_[i])-1] = 'm'
                    elif modification_list_[i+1] == 'Deamidated[N]':
                        peptide_list_[int(modification_list_[i])-1] = 'n'
                    elif modification_list_[i+1] == 'Carbamidomethyl[C]':
                        peptide_list_[int(modification_list_[i])-1] = 'c'
                    i = i+2
            peptide_=''.join(peptide_list_)
            
            psm.append(peptide_)
            psm.append(modification_)
            psm.append(line.split('\t')[0])
            psms.append(psm)
    return psms
def pep_mass(aa):
    return dicM[aa];

def get_y_and_b(peptide):
    len_=len(peptide)
    bs=[];ys=[];bs_mass=[];ys_mass=[];b_name=[];y_name=[];
    pep_to_list=list(peptide)
    pep_mass_list=list(map(pep_mass,pep_to_list))
    sum_mass=np.cumsum(np.array(pep_mass_list))
    for i in range(1,len_):
        bs.append(peptide[0:i])
        bs_mass.append(sum_mass[i-1])
        b_name.append('b'+str(i))
        ys.append(peptide[i:len_])
        ys_mass.append(sum_mass[len_-1]-sum_mass[i-1])
        y_name.append('y'+str(len_-i))
   
    bs1_mz=np.array(bs_mass)+PROTON
    ys1_mz=np.array(ys_mass)+H2O+PROTON
   
    ##b++/y++
    bs2_mz=(np.array(bs_mass)+2*PROTON)/2
    ys2_mz=(np.array(ys_mass)+H2O+2*PROTON)/2
    #print(bs1_mz.tolist())
    #print(bs2_mz.tolist())
    #print(ys1_mz.tolist())
    #print(ys2_mz.tolist())
    return bs,bs_mass,b_name,ys,ys_mass,y_name

def get_all_by(peptide,internal_ion_min_length,internal_ion_max_length):
    len_= len(peptide)
    bys=[];bys_mass=[];_name=[];
    pep_to_list=list(peptide) 
    pep_mass_list=list(map(pep_mass,pep_to_list)) 
    sum_mass=np.cumsum(np.array(pep_mass_list))
    for i in range(internal_ion_min_length+1,internal_ion_max_length+2):
        if len_> i:
            for j in range(1,len_-(i-1)):
                bys.append(peptide[j:j+i-1])
                _name.append(str('y'+str(len_-j)+'b'+str(j+i-1)))
                bys_mass.append(sum_mass[j+i-2]-sum_mass[j-1])
    bys_mz=np.array(bys_mass)+PROTON
    ays_mz=bys_mz-CO
    #print(bys_mz.tolist())
    #print(ays_mz.tolist())
    #print(_name)
    return bys,bys_mass,_name

def closest_mz(mzs,val):
    temp=abs(mzs-val)
    _min=temp.min()
    index=np.where(temp==_min)
   
    if _min<=0.012:
        return index[0][0],mzs[index[0][0]],val
    else:
        return -1,-1,-1
if __name__=='__main__':
    #ESFADVLPEAAALVK
    #YKFILFGLNDAK
    get_all_by('VLDDTmAVADILTSmVVDVSDLLDQAR',1,2)