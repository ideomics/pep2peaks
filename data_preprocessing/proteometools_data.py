import os
from collections import defaultdict
import shutil
import zipfile
import re
import numpy as np
import stat  
import subprocess
from tools import *
import random
import  time
from progressbar import *
class proteome_tools_data():
    def __init__(self,_dis):
        '''
        The preprocessing data of ProteomeTools is all in the 'data/pre_data/ProteomeToolsData'.
        to use this script you should prepare .raw files , .zip files , SpectrumAnalyzer and Pparse.

        At first, use SpectrumAnalyzer to analyzer .raw files,it will produce *-analysis.txt files. 
        And then use Pparse to analyzer .raw files, it will produce .mgf,.xtract and so on.
        put all these files into 'data/pre_data/ProteomeToolsData/InitialFile/raw' .

        Second,put all your .zip files which are identificatied into 'data/pre_data/ProteomeToolsData/InitialFile/identification_results'

        finally, you can change NCE and Ion Type which you want to produce

        The final result of this script is stored in 'data/pre_data/ProteomeToolsData/result/mgf_result/NCE*/ay_by.txt' and
        'data/pre_data/ProteomeToolsData/result/mgf_result/NCE*/b_y.txt'
       
        '''
        
        self.driver=""
        self.data_tools=data_tools()
        #self.data_tools.delete_all(self.driver+'data/pre_data/ProteomeToolsData/result/')
        self.dissociation=_dis
        self.mgf_folder=self.driver+'data/pre_data/ProteomeToolsData/raw_to_mgf/'+self.dissociation
        self.mgf_result_folder=self.data_tools.makedir(self.driver+'data/pre_data/ProteomeToolsData/result/mgf_result/'+self.dissociation)
        self.intetity_folder=self.driver+'data/pre_data/ProteomeToolsData/InitialFile/identification_results/'+self.dissociation
        self.raw_folder=self.driver+'data/pre_data/ProteomeToolsData/InitialFile/raw/'+self.dissociation
        self.raw_to_mgf_folder=self.data_tools.makedir('data/pre_data/ProteomeToolsData/raw_to_mgf/'+self.dissociation)
        
        self.ion_type=['b+','b++','y+','y++','ay+','by+'] 
        self.charge=[2,3,4,5]
        self.nce=[25,30,35]
        
        self.psms_folder=self.driver+'data/pre_data/ProteomeToolsData/result/mgf_result/'+self.dissociation
        self.spectrum_folder=self.driver+'data/pre_data/ProteomeToolsData/raw_to_mgf/'+self.dissociation
        

        #print('################ Phase 1 #################')
        #self.get_nce()
        #self.new_extracted_mgf()
        #print('################ Phase 2 #################')
        #self.read_mgf(self.mgf_folder)
        #print('################ Phase 3 #################')
        #self.ion_run()
        print('################ calc spectrum fdr #################')
        self.spectrum_fdr()

    
    

    ################################################
    def getEvalue(self,list):
        return float(list[0])
    def remove(self):
        for i in os.listdir(self.raw_folder):
            extention = os.path.splitext(i)
            if extention[1] == '.csv' or extention[1] == '.ms1' or extention[1] =='.ms2' or extention[1] =='.xtract':
                os.remove(self.raw_folder + '\\' + i)
    def modified(self,ms):
        modi_index=[]
        if '(ox)' in ms:
            index=[i.start()-2 for i in re.finditer('(ox)', ms)]
            for j in index:
                modi_index.append([j,'Oxidation[M]'])
        if '(de)' in ms: 
            index=[i.start()-2 for i in re.finditer('(de)', ms)]
            for j in index:
                modi_index.append([j,'Deamidated[N]'])
        if '(ca)' in ms: 
            index=[i.start()-2 for i in re.finditer('(ca)', ms)]
            for j in index:
                modi_index.append([j,'Carbamidomethyl[C]'])
        modi_index.sort(key=self.getEvalue)
        for j in range(1,len(modi_index)):
            modi_index[j][0]-=j*4
        if len(modi_index):
            str_=''
            for modi in modi_index:
                str_+=','.join(str(j) for j in modi)+','
            return str_.strip(',')
        else:
            return 'NULL'
    def un_zip(self,file_name,mk):
        zip_file = zipfile.ZipFile(file_name)
        for names in zip_file.namelist():
            zip_file.extract(names, mk)
        zip_file.close()
    def get_nce(self):
        ll = os.listdir(self.raw_folder)
        self.ce_dic=defaultdict(list)
        for l in ll:
            ex = os.path.splitext(l)
            if ex[1] == '.txt':
               with open(self.raw_folder+'\\'+l,'rb') as c: 
                   c.__next__()
                   while True:
                       line= c.readline().decode()
                       if not line:
                           break
                       self.ce_dic[ex[0]+'#'+str(line.split('\t')[0])]=line.split('\t')
        
    def filte_pif_and_score(self,filename):
        dic=defaultdict(list)
        with open(filename,'r') as file_to_read:
            file_to_read.__next__()
            while True:
                line=file_to_read.readline()
                if not line:
                    break
                _pif=line.split('\t')[29]
                _score=line.split('\t')[24]
                if float(_pif) >=0.7 and float(_score)>=100:
                    _key=line.split('\t')[0]+'-analysis#'+line.split('\t')[1]
                    if _key in self.ce_dic.keys():
                        _nce=self.ce_dic[_key][6]
                    else:
                        _nce='0'
                    _modi = self.modified(str(line.split('\t')[7]))
                    _charge=line.split('\t')[12]
                    _peptide=line.split('\t')[3]
                    l=[line.split('\t')[1],line.split('\t')[0],_peptide,_charge,line.split('\t')[18],_pif,_score,line.split('\t')[23],_modi,_nce]
                    _w_key=_peptide+'#'+_nce+'#'+str(_charge)
                    if _w_key in dic.keys():
                        if float(_score) > float(dic[_w_key][6]):
                            dic[_w_key]=l
                    else:
                        dic[_w_key]=l
        with open(self.intetity_folder+'\msms.txt','a') as wf:
            for index,value in dic.items():
                wf.write('\t'.join(value)+'\n')
    def search_and_write_mgf(self,mgf_file,dic,raw):
        with open(mgf_file,'r') as mgf:
            i = 0
            mgf_content=[]
            slines = (slines for slines in mgf.readlines())
            for sline in slines:
               
                if sline.strip() == '':
                    break
                if sline.strip() == 'END IONS':
                    
                    mgf_content.append('END IONS' + '\n')
                    l=[]
                    with open(self.raw_to_mgf_folder+'/'+raw+'.mgf', 'a') as result_mgf: 
                        r_pep_mz=round(float(mgf_content[4].split('=')[1]),3)
                        aa=mgf_content[1].split('.')[1]
                        _key=raw+'#'+str(mgf_content[1].split('.')[1])+'#'+str(mgf_content[1].split('.')[3])+'#'+str(r_pep_mz)
                        
                        l=dic[_key]
                        if  len(l):
                            for d in range(len(mgf_content)):
                                if d == 2:
                                    result_mgf.write('SQE=' + l[2] + '\n')
                                elif d == 3:
                                    result_mgf.write(
                                            'Modifications=' + l[8] +'\nNCE=' + l[9].replace('\n','') + '\nScore=' + l[6] + '\nPIF=' + l[5] + '\nPEPMASS='+l[4]+'\n')
                                if d!= 4:
                                    result_mgf.write(mgf_content[d])
                    mgf_content = [] 
                if sline.strip() == 'BEGIN IONS':
                    mgf_content.append('BEGIN IONS\n')
                if sline.strip() != 'BEGIN IONS' and sline.strip() != 'END IONS':
                    
                    if 'CHARGE=' in sline:
                        sline = sline[:-2]+'\n'
                    mgf_content.append(sline)
    def Pparse_mgf(self,rawfilename):
        for i in os.listdir(self.raw_folder):
            extention = os.path.splitext(i)
            if extention[1] == '.raw' and rawfilename in extention[0]:
                subprocess.call(r'"C:\\pParseStandAlone_X64\\pParse.exe" -D '+self.raw_folder+'\\'+i+' -p 0', shell=True)
            self.remove()              
    def _match(self,dic):
        files,names=self.data_tools.get_files(self.raw_folder)
        raw_list=[]
        for index in range(len(files)):
            extention=os.path.splitext(names[index])
            if extention[1] == '.raw':
                raw_list.append(extention[0])
        for raw in raw_list:
            self.Pparse_mgf(raw)
            dic_this_raw=defaultdict(list)
            for key,value in dic.items():
                if raw in key:
                    dic_this_raw[key]=value
            print('############################\nuse\t'+raw+'.raw,number of msms:'+str(len(dic_this_raw)))
            mgf_list=[]
            for index in range(len(files)):
                extention=os.path.splitext(names[index])
                if extention[1] == '.mgf' and raw in extention[0] :
                    mgf_list.append(files[index])
            f=open(self.raw_to_mgf_folder+'/'+raw+'.mgf', 'w')
            f.close()
            print('wirte in '+self.raw_to_mgf_folder+'/'+raw+'.mgf')
            for mgf in mgf_list:
                print('---use\t'+mgf)
                self.search_and_write_mgf(mgf,dic_this_raw,raw)
            print('done!\n############################')
              
    def new_extracted_mgf(self):
        files,names = self.data_tools.get_files(self.intetity_folder)
        print('filte with pif>=0.7 and score>=100 ,write in \t'+self.intetity_folder+'\msms.txt')
        #--------------------------------------------------------------#
        for file_index in range(len(files)):
            extention = names[file_index].split('.')
            if extention[1]=='zip':
                self.data_tools.makedir(self.intetity_folder+'/'+extention[0])
                print('use\t'+files[file_index])
                self.un_zip(files[file_index],self.intetity_folder+'/'+extention[0])
        mkfile = open(self.intetity_folder+'\\msms.txt','w')
        mkfile.write('Scan number\tRaw File\tSequence\tCharge\tPEPMZ\tPIF\tScore\tPEP\tModifications\tNCE\n')
        mkfile.close()
        raw_names=[]
        for f in os.listdir(self.raw_folder):
            _exten=os.path.splitext(f)
            if _exten[1]=='.raw':
                raw_names.append(_exten[0][11:].replace('-','_'))
        for ff in os.listdir(self.intetity_folder):
            extention = os.path.splitext(ff)
            if extention[1]=='' and extention[0].replace('-tryptic','').replace('-unspecific','').replace('-','_') in raw_names:
                self.filte_pif_and_score(self.intetity_folder+'/'+extention[0]+'\\msms.txt')
        print('all done!')
        #--------------------------------------------------------------#
        with open(self.intetity_folder+'/msms.txt','r') as msms:
            dic=defaultdict(list)
            msms.__next__()
            while True:
                line = msms.readline()
                if not line:
                    break
               
                _key=line.split('\t')[1]+'#'+line.split('\t')[0]+'#'+line.split('\t')[3]+'#'+str(round(float(line.split('\t')[4]),3))
                if _key in dic.keys():
                    print('error in '+ _key)
                dic[_key]=line.split('\t')
            print('Total number of msms '+str(len(dic)))
            self._match(dic)
        print('all done')
    ################################

    def write_result(self,dic,mgf_name):
        for charge_and_collision_energy,values in dic.items():
            print(charge_and_collision_energy)
            _folder=self.data_tools.makedir(self.mgf_result_folder+'/NCE'+str(charge_and_collision_energy.split('|')[1].split('_')[1])+'/'+os.path.splitext(mgf_name)[0])
            _charge=re.sub("\D", "", charge_and_collision_energy.split('|')[0])
            if int(_charge) in self.charge:
                f=open(_folder+'/'+charge_and_collision_energy.split('|')[0]+'.txt','w')
                f.close()
                with open(_folder+'/'+charge_and_collision_energy.split('|')[0]+'.txt','a') as wf:
                    for _value in values:
                        wf.write('\t'.join(_value)+'\n')
    def get_line_value(self,_line):
        return _line.split('=')[1].replace('\n','')
    def read_mgf(self,mgf_folder):
        files,names=self.data_tools.get_files(mgf_folder)
        for file_index in range(len(files)):
            dic=defaultdict(list)
            print('use\t'+files[file_index])
            with open(files[file_index],'r') as rf:
                while True:
                    line= rf.readline()
                    if not line:
                        break
                    if 'BEGIN IONS' in line:
                        _spec_name=self.get_line_value(rf.readline())
                        _seq=self.get_line_value(rf.readline())
                        _charge = int(self.get_line_value(rf.readline()))
                        _modification=self.get_line_value(rf.readline())
                        _nce=int(self.get_line_value(rf.readline()))
                        _score=self.get_line_value(rf.readline())
                        rf.__next__();
                        _pepmass=self.get_line_value(rf.readline())
                        rf.__next__(); 
                        dic['charge'+str(_charge)+'|NCE_'+str(_nce)].append([str(_nce),_score,_spec_name,str(_charge),_seq,_modification,_pepmass])
            self.write_result(dic,'/mgf_'+str(names[file_index]))
            print('done!\n############################')
        print('all done!')
     ######################################
    def merge_psms_charge_file(self,read_folder,_nce,mgf_name):
        filelist=[]
        files,_=self.data_tools.get_files(read_folder)
        for file in files:
            filelist.append(file) 
        fr_folder=self.data_tools.makedir(read_folder+'/all_charge')
        fr_file_name=fr_folder+'/psms_all_charge.txt'
        with open(fr_file_name,'w') as newfile:
            for item in filelist:
                for txt in open(item,'r'):
                    newfile.write(txt)

        
    def ion_run(self):
        spectrum_files,spectrum_names=self.data_tools.get_files(self.spectrum_folder)
        for _nce in self.nce:
            print(' NCE '+str(_nce))
            for spectrum_file_index in range(len(spectrum_files)):
                print('   ######################################\n   --use ' + spectrum_files[spectrum_file_index])
                _folder=self.data_tools.makedir(self.psms_folder+'/NCE'+str(_nce)+'/mgf_'+os.path.splitext(spectrum_names[spectrum_file_index])[0])
                self.merge_psms_charge_file(_folder,_nce,str(spectrum_names[spectrum_file_index]))
                psms_file,_=self.data_tools.get_files(_folder+'/all_charge')
                psms=self.data_tools.get_psms(psms_file[0])
                print('   --use ' + psms_file[0])
                spectrums=self.data_tools.get_spectrums(spectrum_files[spectrum_file_index],'proteometools')
                self.data_tools.match_psms_with_spectra(self.data_tools.makedir(_folder+'/all_charge/match_ion'),psms,spectrums,'ProteomeToolsData')
                
                print('   ######################################')
       
    
    #def match_psms_with_spectra2(self,ion_folder):
    #    sum_of_intensity=0.0
    #    for psm in self.psms:
    #        spec_name=psm[0]+'#'+psm[4]
    #        if spec_name in self.spectrums.keys():
    #            _spectrum=np.array(self.spectrums[spec_name])
    #            sum_of_intensity+=np.sum(_spectrum[:,1])
    #    with open(ion_folder+'/sum_of_intensity.txt','w') as wf:
    #        wf.write(str(sum_of_intensity)+'\n')
    def match_psms_with_spectra(self,ion_folder):
        regular_annotated_peaks=open('data/pre_data/ProteomeToolsData/regular_annotated_peaks.txt','a') 
        internal_annotated_peaks=open('data/pre_data/ProteomeToolsData/internal_annotated_peaks.txt','a') 
        match_b_y_result=[];match_ay_by_result=[];spectrums_name=[]
        match_b_y_result_with_thres=[];match_ay_by_result_with_thres=[]
        for psm in self.psms:
            spec_name=psm[0]+'#'+psm[4]
            if spec_name in self.spectrums.keys():
                spectrums_name.append(spec_name)
                peptide=psm[2]
                _spectrum=np.array(self.spectrums[spec_name])
                _spectrum_with_inten_threshold=_spectrum
                charge=psm[1]
                ions_b_y,ions_a,ions_ay_by=self.data_tools.get_ions_list(peptide,self.internal_ion_min_length,self.internal_ion_max_length)
                #ions match
                ########### no intensity threshold###################
                 #b+,b++,y+,y++
                _str=''
                match_pep_result=[];b_y_result=[]
                for i in range(len(peptide)-1):
                    match_pep_result.append(peptide)
                    temp_ion='';temp_mz='';temp_ion_name='';temp_inten_r='';temp_inten=''
                    for j in range(4):
                        if j%2==0:
                            temp_ion+=ions_b_y[j][0][i]+','
                        index,_ppm=self.data_tools.closest_mz(_spectrum[:,[0,2]],ions_b_y[j][1][i])
                        if index!=-1:
                          
                            _str+= str(_spectrum[index][0])+','+str(_ppm)+','+str(_spectrum[index][2])+'\n'
                            annotated_str=peptide+'\t'+ions_b_y[j][0][i]+'\t'+ions_b_y[j][2][i]+'\t'+str(_spectrum[index][0])+'\t'+str(_spectrum[index][1])+'\t'+str(_spectrum[index][2])+'\t'+str(_spectrum[index][3])+'\t0\n'
                            regular_annotated_peaks.write(annotated_str)
                            temp_inten_r+=str(_spectrum[index][2])+','
                            _spectrum = np.delete(_spectrum, index, 0)
                        else:
                            temp_inten_r+='0.0,'
                        temp_ion_name+=ions_b_y[j][2][i]+','
                    b_y_result.append([peptide,charge,temp_ion[:-1],psm[3],temp_ion_name[:-1],temp_inten_r[:-1]])
                match_b_y_result.append(b_y_result)
                #with open('data/pre_data/ProteomeToolsData/proteo_regular_ions_error_distribution_20_ppm.txt','a') as f:
                #    f.write(_str)

                 #a+,a++
             
                for i in range(len(peptide)-1):
                    for j in range(2):
                        index,_=self.data_tools.closest_mz(_spectrum[:,[0,2]],ions_a[j][1][i])
                        if index!=-1:
                            _spectrum = np.delete(_spectrum, index, 0)

                #ay+ by+
                _str=''
                match_pep_result=[];ay_by_result=[]
               
                for i in range(len(ions_ay_by[0][1])):
                    match_pep_result.append(peptide)
                    temp_mz='';temp_ion_name='';temp_inten_r='';temp_inten=''
                    for j in range(2):
                        temp_ion=ions_ay_by[j][0][i]
                        index,_ppm_i=self.data_tools.closest_mz(_spectrum[:,[0,2]],ions_ay_by[j][1][i])
                        if index!=-1:
                            _str+= str(_spectrum[index][0])+','+str(_ppm_i)+','+str(_spectrum[index][2])+'\n'
                            annotated_str=peptide+'\t'+ions_ay_by[j][0][i]+'\t'+ions_ay_by[j][2][i]+'\t'+str(_spectrum[index][0])+'\t'+str(_spectrum[index][1])+'\t'+str(_spectrum[index][2])+'\t'+str(_spectrum[index][3])+'\t0\n'
                            internal_annotated_peaks.write(annotated_str)
                            temp_inten_r+=str(_spectrum[index][2])+','
                            _spectrum = np.delete(_spectrum, index, 0)
                        else:
                            temp_inten_r+='0.0,'
                        temp_ion_name+=ions_ay_by[j][2][i]+','
                    ay_by_result.append([peptide,charge,temp_ion,psm[3],temp_ion_name[:-1],temp_inten_r[:-1]])
               
                match_ay_by_result.append(self.data_tools.avg_same_ion_inten(ions_ay_by[0][0],ay_by_result))
                #with open('data/pre_data/ProteomeToolsData/proteo_internal_ions_error_distribution_20_ppm.txt','a') as f:
                #    f.write(_str) 
                ########### no intensity threshold###################


                ############ with intensity threshold###################
                # #b+,b++,y+,y++
                #_str=''
                #match_pep_result_with_thres=[];b_y_result_with_thres=[]
                #for i in range(len(peptide)-1):
                #    match_pep_result_with_thres.append(peptide)
                #    temp_ion='';temp_mz='';temp_ion_name='';temp_inten_r='';temp_inten=''
                #    for j in range(4):
                #        if j%2==0:
                #            temp_ion+=ions_b_y[j][0][i]+','
                #        index,_ppm=self.data_tools.closest_mz(_spectrum_with_inten_threshold[:,[0,2]],ions_b_y[j][1][i],True)
                #        if index!=-1:
                #          
                #            temp_inten_r+=str(_spectrum_with_inten_threshold[index][2])+','
                #            _spectrum_with_inten_threshold = np.delete(_spectrum_with_inten_threshold, index, 0)
                #        else:
                #            temp_inten_r+='0.0,'
                #        temp_ion_name+=ions_b_y[j][2][i]+','
                #    b_y_result_with_thres.append([peptide,charge,temp_ion[:-1],psm[3],temp_ion_name[:-1],temp_inten_r[:-1]])
                #match_b_y_result_with_thres.append(b_y_result_with_thres)
                ##with open('data/pre_data/ProteomeToolsData/proteo_regular_ions_error_distribution_20_ppm.txt','a') as f:
                ##    f.write(_str)

                # #a+,a++
             
                #for i in range(len(peptide)-1):
                #    for j in range(2):
                #        index,_=self.data_tools.closest_mz(_spectrum_with_inten_threshold[:,[0,2]],ions_a[j][1][i],True)
                #        if index!=-1:
                #            _spectrum_with_inten_threshold = np.delete(_spectrum_with_inten_threshold, index, 0)

                ##ay+ by+
                #_str=''
                #match_pep_result_with_thres=[];ay_by_result_with_thres=[]
               
                #for i in range(len(ions_ay_by[0][1])):
                #    match_pep_result_with_thres.append(peptide)
                #    temp_mz='';temp_ion_name='';temp_inten_r='';temp_inten=''
                #    for j in range(2):
                #        temp_ion=ions_ay_by[j][0][i]
                #        index,_ppm_i=self.data_tools.closest_mz(_spectrum_with_inten_threshold[:,[0,2]],ions_ay_by[j][1][i],True)
                #        if index!=-1:
                #            
                #            temp_inten_r+=str(_spectrum_with_inten_threshold[index][2])+','
                #            _spectrum_with_inten_threshold = np.delete(_spectrum_with_inten_threshold, index, 0)
                #        else:
                #            temp_inten_r+='0.0,'
                #        temp_ion_name+=ions_ay_by[j][2][i]+','
                #    ay_by_result_with_thres.append([peptide,charge,temp_ion,psm[3],temp_ion_name[:-1],temp_inten_r[:-1]])
               
                #match_ay_by_result_with_thres.append(self.data_tools.avg_same_ion_inten(ions_ay_by[0][0],ay_by_result_with_thres))
                ##with open('data/pre_data/ProteomeToolsData/proteo_internal_ions_error_distribution_20_ppm.txt','a') as f:
                ##    f.write(_str) 
                ############ with intensity threshold###################
        

        b_y_train_wf=open(ion_folder+'/b_y_train.txt','w')      
        b_y_test_wf=open(ion_folder+'/b_y_test.txt','w')

        ay_by_train_wf=open(ion_folder+'/ay_by_train.txt','w')      
        ay_by_test_wf=open(ion_folder+'/ay_by_test.txt','w')

        b_y_train_wf_inten_threshold=open(ion_folder+'/b_y_train_with_inten_threshold.txt','w')      
        b_y_test_wf_inten_threshold=open(ion_folder+'/b_y_test_inten_threshold.txt','w')

        ay_by_train_wf_inten_threshold=open(ion_folder+'/ay_by_train_inten_threshold.txt','w')      
        ay_by_test_wf_inten_threshold=open(ion_folder+'/ay_by_test_inten_threshold.txt','w')

        print(len(match_b_y_result))
        print(len(match_ay_by_result))

        for k in range(len(match_b_y_result)):
            a= random.random()
            if a<0.9:
                for _list in match_b_y_result[k]:
                    b_y_train_wf.write('\t'.join(_list)+'\t'+spectrums_name[k]+'\n')
                    b_y_train_wf_inten_threshold.write('\t'.join(_list)+'\t'+spectrums_name[k]+'\n')
                for _list in match_ay_by_result[k]:
                    ay_by_train_wf.write('\t'.join(_list)+'\t'+spectrums_name[k]+'\n')
                    ay_by_train_wf_inten_threshold.write('\t'.join(_list)+'\t'+spectrums_name[k]+'\n')
            else:
                for _list in match_b_y_result[k]:
                    b_y_test_wf.write('\t'.join(_list)+'\t'+spectrums_name[k]+'\n') 
                    b_y_test_wf_inten_threshold.write('\t'.join(_list)+'\t'+spectrums_name[k]+'\n') 
                for _list in match_ay_by_result[k]:
                    ay_by_test_wf.write('\t'.join(_list)+'\t'+spectrums_name[k]+'\n') 
                    ay_by_test_wf_inten_threshold.write('\t'.join(_list)+'\t'+spectrums_name[k]+'\n') 

        b_y_train_wf.close()
        b_y_test_wf.close()
        ay_by_train_wf.close()
        ay_by_test_wf.close()

        b_y_train_wf_inten_threshold.close()
        b_y_test_wf_inten_threshold.close()
        ay_by_train_wf_inten_threshold.close()
        ay_by_test_wf_inten_threshold.close()
    
    def spectrum_fdr(self):
        spectrum_files,spectrum_names=self.data_tools.get_files(self.spectrum_folder)
        #spectrum_files,spectrum_names=self.data_tools.get_files('data/pre_data/ProteomeToolsData/Results_temp')
        
        f_=open('data/pre_data/ProteomeToolsData/regular_annotated_peaks.txt','r') 
        regular_annotated_peaks=f_.readlines()
        f_2=open('data/pre_data/ProteomeToolsData/regular_annotated_peaks.txt','r') 
        internal_annotated_peaks=f_2.readlines()
        self.spectrums=defaultdict(list)
        pbar = ProgressBar(widgets=['Read Spectrum files',Percentage(), ' ', Bar('#'),' ',' ', ETA(), ' ']).start()
        for spectrum_file_index in range(len(spectrum_files)):
            pbar.update(int((spectrum_file_index / (len(spectrum_files) - 1)) * 100))
            spectrum=self.data_tools.get_spectrums(spectrum_files[spectrum_file_index],'proteometools',True)
            self.data_tools.makedir('data\pre_data\ProteomeToolsData/spectrum_fdr')
            if len(spectrum)>0:
                self.spectrums.update(spectrum)
        pbar.finish()
        f_regular=open('data\pre_data\ProteomeToolsData/spectrum_fdr/regular_fdrs.txt','a')
        f_internal=open('data\pre_data\ProteomeToolsData/spectrum_fdr/internal_fdrs.txt','a')
       
        print(len(self.spectrums))
        random_keys=random.sample(self.spectrums.keys(), 2000)

        
        #num_of_spectrum=len(self.spectrums)
        #print(num_of_spectrum)
        #while cunt<1000:
        #    random_key_index=random.randint(0,num_of_spectrum-1)
        #    if random_key_index not in random_key_index_set:
         #       random_key_index_set.append(random_key_index) 
        #        random_key=list(self.spectrums.keys())[random_key_index]
        pbar = ProgressBar(widgets = ['Processed Spectrum:',Percentage(), ' ', Bar('#'),' ',' ', ETA(), ' ']).start()
        #for i in range(len(self.inten_thresholds)):
            
        for i in range(len(random_keys)):       
            pbar.update(int((i / (len(random_keys) - 1)) * 100))
            self.data_tools.calc_fdr(f_regular,f_internal,regular_annotated_peaks,internal_annotated_peaks,np.array(self.spectrums[random_keys[i]]),random_keys[i])
        #    else:
        #        continue
        pbar.finish() 
        f_regular.close()
        f_internal.close()
        print('   ######################################')
  
  

if __name__=='__main__':
    dissociation=['HCD']
    for dis in dissociation:
        proteome_tools_data(dis)