import os
from collections import defaultdict
import numpy as np
import random
import linecache

class data_tools(object):
    def __init__(self, **kwargs):
        
        self.dicM={'A':71.037114,'C':103.009185,'D':115.026943,'E':129.042593,'F':147.068414,\
      'G':57.021464,'H':137.058912,'I':113.084064,'K':128.094963,'L':113.084064,\
      'M':131.040485,'N':114.042927,'P':97.052764,'Q':128.058578,'R':156.101111,\
      'S':87.032028,'T':101.047678,'V':99.068414,'W':186.079313,'Y':163.063329,\
       'c':160.0306486796,'m':147.035399708,'n':115.026943025
      }
        self.PROTON= 1.007276466583
        self.H = 1.0078250322
        self.O = 15.9949146221
        self.CO = 27.9949146200
        self.N = 14.0030740052
        self.C = 12.00
        self.isotope = 1.003

        self.CO = self.C + self.O
        self.CO2 = self.C + self.O * 2
        self.NH = self.N + self.H
        self.NH3 = self.N + self.H * 3
        self.HO = self.H + self.O
        self.H2O= self.H * 2 + self.O

    def delete_all(self,filePath):
                if os.path.exists(filePath):
                    for fileList in os.walk(filePath):
                        for name in fileList[2]:
                            os.chmod(os.path.join(fileList[0],name), stat.S_IWRITE)
                            os.remove(os.path.join(fileList[0],name))
                    shutil.rmtree(filePath)
                    return "delete ok"
                else:
                    return "no filepath"
    def makedir(self,folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        return folder
    def get_files(self,rootdir):
        files=[]
        names=[]
        _list=os.listdir(rootdir)
        for i in range(0,len(_list)):
            path=os.path.join(rootdir,_list[i])
            if os.path.isfile(path):
                files.append(path)
                names.append(_list[i])
        return files,names

    def get_spectrums(self,file_name,_type,is_calc_fdr=False):
            dic=defaultdict(list)
            with open(file_name,'r') as rf:
                while True:
                    line = rf.readline()
                    if not line:
                        break
                    if 'BEGIN IONS' in line:
                        
                        line=rf.readline()
                        _spec_t=line.split('=')[1].replace("\n", "")
                        peptide=rf.readline().split('=')[1].replace("\n", "")
                        if is_calc_fdr and len(peptide)!=15:
                            continue
                        rf.__next__()
                        if _type=='proteometools':
                            rf.__next__()
                        if _type=='mm':
                            _nce=0
                        else:
                            _nce=int(rf.readline().split('=')[1].replace("\n", ""))
                        _spec=_spec_t+'#'+str(_nce)
                        if is_calc_fdr:
                            _spec=_spec_t+'#'+str(_nce)+'#'+peptide
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
    def get_psms(self,file_name):
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
    def pep_mass(self,aa):
        return self.dicM[aa];

    def get_y_and_b(self,peptide):
        len_=len(peptide)
        bs=[];ys=[];bs_mass=[];ys_mass=[];b_name=[];y_name=[];
        pep_to_list=list(peptide)
        pep_mass_list=list(map(self.pep_mass,pep_to_list))
        sum_mass=np.cumsum(np.array(pep_mass_list))
        for i in range(1,len_):
            bs.append(peptide[0:i])
            bs_mass.append(sum_mass[i-1])
            b_name.append('b'+str(i))
            ys.append(peptide[i:len_])
            ys_mass.append(sum_mass[len_-1]-sum_mass[i-1])
            y_name.append('y'+str(len_-i))
       
        bs1_mz=np.array(bs_mass)+self.PROTON
        ys1_mz=np.array(ys_mass)+self.H2O+self.PROTON
       
        ##b++/y++
        bs2_mz=(np.array(bs_mass)+2*self.PROTON)/2
        ys2_mz=(np.array(ys_mass)+self.H2O+2*self.PROTON)/2
        #print(bs1_mz.tolist())
        #print(bs2_mz.tolist())
        #print(ys1_mz.tolist())
        #print(ys2_mz.tolist())
        return bs,bs_mass,b_name,ys,ys_mass,y_name

    def get_all_by(self,peptide,internal_ion_min_length,internal_ion_max_length):
        len_= len(peptide)
        bys=[];bys_mass=[];_name=[];
        pep_to_list=list(peptide) 
        pep_mass_list=list(map(self.pep_mass,pep_to_list)) 
        sum_mass=np.cumsum(np.array(pep_mass_list))
        for i in range(internal_ion_min_length+1,internal_ion_max_length+2):
            if len_> i:
                for j in range(1,len_-(i-1)):
                    bys.append(peptide[j:j+i-1])
                    _name.append(str('y'+str(len_-j)+'b'+str(j+i-1)))
                    bys_mass.append(sum_mass[j+i-2]-sum_mass[j-1])
        bys_mz=np.array(bys_mass)+self.PROTON
        #ays_mz=bys_mz-CO
        #print(bys_mz.tolist())
        #print(ays_mz.tolist())
        #print(_name)
        return bys,bys_mass,_name
    def closest_mz(self,mzs,val):
       
        mz=mzs[:,0]
        de_val=((mz-val)/val)*10**6
        index=np.where((de_val>=-20)&(de_val<=20))
       
        if len(index[0])>0:
           
            intens=mzs[:,1][index[0]]
            max_intens=max(intens)
            
            intens_index=np.where(intens==max_intens)[0][0]
            peak_index=index[0][intens_index]
            _ppm=de_val[peak_index]
            return peak_index,_ppm
        else:
            return -1,-1
    def avg_same_ion_inten(self,ions,match_result):
        d = defaultdict(list)
        for k,va in [(v,i) for i,v in enumerate(ions)]:
            d[k].append(va)
        for _,value in d.items():
            repeter_ion_count=len(value)
            if repeter_ion_count > 1:
                r_inten=match_result[min(value)][5]
              
                new_by_inten=float(r_inten.split(',')[0])/repeter_ion_count
                new_ay_inten=float(r_inten.split(',')[1])/repeter_ion_count
                new_inten=str(new_by_inten)+','+str(new_ay_inten)
                for posi in value:
                    match_result[posi][5]= new_inten
                    match_result[posi].extend([str(float(1/repeter_ion_count))])
               
            else:
                match_result[value[0]].extend(['1'])
        return match_result
    def get_ions_list(self,peptide,internal_ion_min_length,internal_ion_max_length):
        ions_b_y=[];ions_a=[]
        #b+/y+
        bs,bs_mass,b_name,ys,ys_mass,y_name=self.get_y_and_b(peptide)
        bs1_mz=np.array(bs_mass)+self.PROTON
        ys1_mz=np.array(ys_mass)+self.H2O+self.PROTON
        as1_mz=np.array(bs_mass)+self.PROTON-self.CO
        #b++/y++
        bs2_mz=(np.array(bs_mass)+2*self.PROTON)/2
        ys2_mz=(np.array(ys_mass)+self.H2O+2*self.PROTON)/2
        as2_mz=(np.array(bs_mass)+2*self.PROTON-self.CO)/2

        ions_b_y.append([bs,bs1_mz,[_name+'+' for _name in b_name]])
        ions_b_y.append([bs,bs2_mz,[_name+'++' for _name in b_name]])
        ions_b_y.append([ys,ys1_mz,[_name+'+' for _name in y_name]])
        ions_b_y.append([ys,ys2_mz,[_name+'++' for _name in y_name]])

        ions_a.append([bs,as1_mz,[_name.replace('b','a')+'+' for _name in b_name]])
        ions_a.append([bs,as2_mz,[_name.replace('b','a')+'++' for _name in b_name]])

        ions_ay_by=[]
        bys,bys_mass,bys_name=self.get_all_by(peptide,internal_ion_min_length,internal_ion_max_length)
        #by+/ay+
        bys_mz=np.array(bys_mass)+self.PROTON
        ays_mz=bys_mz-self.CO
        ions_ay_by.append([bys,bys_mz,[_name+'+' for _name in bys_name]])
        ions_ay_by.append([bys,ays_mz,[_name.replace('b','a')+'+' for _name in bys_name]])
        return ions_b_y,ions_a,ions_ay_by

    def noise_spectrum(self,annotated_peaks,spectrum):
        peaks_count= int(annotated_peaks[-1])
        #random_peaks=random.sample(list(range(1,peaks_count)))
       
        #for i in range(len(spectrum)):
        #    if spectrum[i][1]<inten_threshold:
        #        spectrum = np.delete(spectrum, i, 0)
        spectrum=np.insert(spectrum, 4, values=1, axis=1)          
        
        selected_peaks=0
        while selected_peaks<10:
            random_line_num=random.randint(1,peaks_count-1)
            #print(random_line_num)
            random_peak_line=annotated_peaks[random_line_num].strip('\n').split('\t')
            peak_mz=float(random_peak_line[3])
            peak_inten=float(random_peak_line[5])
            spectrum_mz=spectrum[:,0]
            ppm_error=((spectrum_mz-peak_mz)/peak_mz)*10**6
            index=np.where((ppm_error>=-20)&(ppm_error<=20))
            if len(index[0])>0:
                continue
            else:
                random_peak=np.array(list(map(float,random_peak_line[3:])))
                spectrum=np.insert(spectrum, -1, values=random_peak, axis=0)
              
                selected_peaks+=1
        return spectrum
    
if __name__=='__main__':
    #ESFADVLPEAAALVK
    #YKFILFGLNDAK
    get_all_by('VLDDTmAVADILTSmVVDVSDLLDQAR',1,2)