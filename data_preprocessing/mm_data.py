from numpy import *
import os
from tools import *
class data_preprocessing(object):
    def __init__(self,mgf_name):
        _sqt='data/pre_data/MMdata/comet/'+mgf_name+'/'
        _dat='data/pre_data/MMdata/mascot/'+mgf_name+'/'
        _txt='data/pre_data/MMdata/pFind/'+mgf_name+'/'
        self.mgf_name=mgf_name
        self.comet_sqt=makedir(_sqt)
        self.mascot_dat=makedir(_dat)
        self.pfind_txt=makedir(_txt)
        makedir('data/pre_data/MMdata/3result')
        self.mgf_folder='data/pre_data/MMdata/mgf/'+mgf_name+'/'
        self.dicM={'A':71.037114,'C':103.009185,'D':115.026943,'E':129.042593,'F':147.068414,\
      'G':57.021464,'H':137.058912,'I':113.084064,'K':128.094963,'L':113.084064,\
      'M':131.040485,'N':114.042927,'P':97.052764,'Q':128.058578,'R':156.101111,\
      'S':87.032028,'T':101.047678,'V':99.068414,'W':186.079313,'Y':163.063329,\
       'c':160.0306486796,'m':147.035399708,'n':115.026943025
      }
        self.PROTON= 1.007276466583
        self.H = 1.0078250322
        self.O = 15.9949146221
        
        self.N = 14.0030740052
        self.C = 12.00
        self.isotope = 1.003

        self.CO = self.C + self.O
        self.H2O= self.H * 2 + self.O
        self.internal_ion_max_length=8
        self.internal_ion_min_length=1
        self.charge=[2,3]
        self.first_folder='first_result'
        self.mod_folder='mod_result'
        self.fdr_folder='FDR0_01'
        self.spectrum_sequence_folder='spectrum_sequence'
        self.spectrum_sequence_file_folder_list=[]
        #####################
        #self.data_run()
        #self.merge_3_result()
        #self.fdr_merge()
        #print('remove decoy redundant')
        #self.remove_decoy_redundant()
        #self.psms_mpc_classify()
        ###########################

        self.ion_run()

    def pep2_proteins_to_psms(self,_proteins_files,_psms_files,write_files,charge):
        proteins_files,proteins_names=get_files(_proteins_files)
        psms_files,psms_names=get_files(_psms_files)
        for file_index in range(len(proteins_files)):
            f_proteins = open(proteins_files[file_index])
            if psms_names[file_index].split('_')[2] in proteins_names[file_index]:
                print('right')
            else:
                print('false')
            f_psms = open(psms_files[file_index])
            fr_folder=makedir(write_files+'/proteins_to_psms_charge'+str(charge))
            fr = open(fr_folder+'/proteins_to_psms_'+psms_names[file_index].split('_')[2]+'_charge'+str(charge)+'.txt','w')

            sequence = []
            for line in f_proteins:
                seq = line.split('\t')[1:]
                for i in range(len(seq)):
                    if not seq[i] in sequence:
                        sequence.append(seq[i])
            f_proteins.close()

            for line in f_psms:
                seq = line.split('\t')[3]
                if seq in sequence:
                    for i in range(6):
                        if i == 5:
                            fr.write(line.split('\t')[i]+'\n')
                        else:
                            fr.write(line.split('\t')[i]+'\t')
            f_psms.close()
            fr.close()
    def pep2_proteins(self,read_files,write_files,charge):
        files,names=get_files(read_files)
        for file_index in range(len(files)):
            f = open(files[file_index])
            fr_folder=makedir(write_files+'/FDR0.01_all_pep2_proteins_charge'+str(charge))
            fr = open(fr_folder+'/FDR0.01_pep2_proteins_'+names[file_index].split('_')[2]+'_charge'+str(charge)+'.txt','w')

            for line in f:
                if len(line.split('\t')) <= 3:
                    continue
                else:
                    for i in range(len(line.split('\t'))):
                        if i < len(line.split('\t')) - 1:
                            fr.write(line.split('\t')[i]+'\t')
                        else:
                            fr.write(line.split('\t')[i])
            f.close()
            fr.close()
        self.pep2_proteins_to_psms(write_files+'/FDR0.01_all_pep2_proteins_charge'+str(charge),write_files+'/FDR0.01_all_charge'+str(charge),write_files,charge)
    def proteins(self,read_files,write_files,charge):
        files,names=get_files(read_files)
        for file_index in range(len(files)):
            f = open(files[file_index])
            fr_folder=makedir(write_files+'/FDR0.01_all_proteins_charge'+str(charge))
            fr = open(fr_folder+'/FDR0.01_proteins_'+names[file_index].split('_')[2]+'_charge'+str(charge)+'.txt','w')

            proteins = []
            peptides = []
            for line in f:
                pro = line.split('\t')[5]
                pep = line.split('\t')[3]
                proteinlen = len(pro.split(','))
                for i in range(proteinlen):
                    if i == proteinlen-1:
                        if not pro.split(',')[i].split('\n')[0] in proteins:
                            peplist = []
                            peplist.append(pep)
                            proteins.append(pro.split(',')[i].split('\n')[0])
                            peptides.append(peplist)
                        else:
                            idx = proteins.index(pro.split(',')[i].split('\n')[0])
                            if not pep in peptides[idx]:
                                peptides[idx].append(pep)
                    else:
                        '''if i == 0:
                            continue'''
                        if not pro.split(',')[i] in proteins:
                            peplist = []
                            peplist.append(pep)
                            proteins.append(pro.split(',')[i])
                            peptides.append(peplist)
                        else:
                            idx = proteins.index(pro.split(',')[i])
                            if not pep in peptides[idx]:
                                peptides[idx].append(pep)
                        

            f.close()

            for i in range(len(proteins)):
                fr.write(proteins[i]+'\t')
                for j in range(len(peptides[i])):
                    fr.write(peptides[i][j]+'\t')
                fr.write('\n')

            fr.close()
            print('filter less than 2 peptide in one proteins')
        self.pep2_proteins(write_files+'/FDR0.01_all_proteins_charge'+str(charge),write_files,charge)
    def merge_spectrum_sequence_file(self,read_folder_list,write_foler,charge):
        filelist=[]
        for _folder in read_folder_list:
            files,_=get_files(_folder)
            for file in files:
                if 'charge'+str(charge) in file:
                    filelist.append(file) 
        fr_folder=makedir(write_foler+'/charge'+str(charge))
        fr_file_name=fr_folder+'/spectrum_sequence_charge'+str(charge)+'.txt'
        newfile=open(fr_file_name,'w')
        for item in filelist:
            for txt in open(item,'r'):
                newfile.write(txt)

        newfile.close()
    def fdr_filter(self,read_files,names,write_files):
        for file_index in range(len(read_files)):
            f_name =  read_files[file_index]
            print('filter fdr\t'+f_name)
            _folder='/FDR0.01_all_charge'
            for charge in self.charge:
                if 'charge'+str(charge) in f_name:
                    _folder+=str(charge)
            fr_folder=makedir(write_files+_folder)
            fr_name = fr_folder+'/FDR0.01_'+names[file_index]
            f = open(f_name)
            fr = open(fr_name,'w')

            psms = [];peptides = []
            for line in f:
                psm = []
                fdr = float(line.split('\t')[1])
                if fdr > 0.01:
                    continue
                psm.append(line.split('\t')[0])
                psm.append(line.split('\t')[3])
                psm.append(line.split('\t')[4])
                psm.append(line.split('\t')[5])
                psm.append(line.split('\t')[6])
                psm.append(line.split('\t')[7].split('\n')[0])
                psms.append(psm)
            f.close()

            for i in range(len(psms)):
                for j in range(6):
                    fr.write(psms[i][j]+'\t')
                fr.write('\n')

            fr.close()
        for charge in self.charge:

            self.proteins(write_files+'/FDR0.01_all_charge'+str(charge),write_files,charge)
        #for charge in self.charge:
        #    print('merge all charge'+str(charge)+' files to folder FDR0.01_all_charge'+str(charge))
        #    self.merge_file(write_files,makedir(write_files+'/FDR0.01_all_charge'+str(charge)),charge)

    def extract_spectrum_sequence(self,psms_file_list,names,write_folder,software,charge):
        for index in range(len(psms_file_list)):
            f = open(psms_file_list[index])
            _write_folder=makedir(write_folder+'/'+self.spectrum_sequence_folder)
            
            fr = open(_write_folder+'/'+software+'_spectrum_sequence_'+names[index].split('_')[3]+'_charge'+str(charge)+'.txt','w')

            for line in f:
                spectrum = line.split('\t')[1]
                sequence = line.split('\t')[3]
                fr.write(spectrum+'\t'+sequence+'\t'+software+'\n')
            f.close()
            fr.close()
    def merge_3_result(self):
        print('merge 3 result')
        for charge in self.charge:
            write_folder='data/pre_data/MMdata/3result/'+self.mgf_name+'/merge_result/spectrum_result/charge'+str(charge)
            makedir(write_folder)
            f = open('data/pre_data/MMdata/3result/'+self.mgf_name+'/spectrum_sequence/charge'+str(charge)+'/spectrum_sequence_charge'+str(charge)+'.txt')
            f_mpc = open(write_folder+"/result_mpc.txt",'w')
            f_mp = open(write_folder+"/result_mp.txt",'w')
            f_pc = open(write_folder+"/result_pc.txt",'w')
            f_mc = open(write_folder+"/result_mc.txt",'w')
            f_p = open(write_folder+"/result_p.txt",'w')
            f_c = open(write_folder+"/result_c.txt",'w')
            f_m = open(write_folder+"/result_m.txt",'w')
            f_wrong = open(write_folder+"/result_wrong.txt",'w')

            spectra = [];sequence = [];label = []
            for line in f:
                spectra.append(line.split('\t')[0].strip())
                sequence.append(line.split('\t')[1].strip())
                label.append(line.split('\t')[2].split('\n')[0])
            f.close()
            reduplicative = []
            for i in range(len(spectra)):
                if spectra[i] in reduplicative:
                    continue
                else:
                    reduplicative.append(spectra[i])
                    count = spectra.count(spectra[i])
                    if count == 3:
                        idx2 = spectra.index(spectra[i],i+1)
                        idx3 = spectra.index(spectra[i],idx2+1)
                        if sequence[i] != sequence[idx2] or sequence[i] != sequence[idx3] or sequence[idx2] != sequence[idx3]:
                            f_wrong.write(spectra[i]+'\t'+sequence[i]+'\t'+label[i]+'\n')
                            f_wrong.write(spectra[idx2]+'\t'+sequence[idx2]+'\t'+label[idx2]+'\n')
                            f_wrong.write(spectra[idx3]+'\t'+sequence[idx3]+'\t'+label[idx3]+'\n')
                        else:
                            f_mpc.write(spectra[i]+'\t'+sequence[i]+'\n')
                    elif count == 2:
                        idx = spectra.index(spectra[i],i+1)
                        flag = label[idx]
                        if sequence[i] != sequence[idx]:
                            f_wrong.write(spectra[i]+'\t'+sequence[i]+'\t'+label[i]+'\n')
                            f_wrong.write(spectra[idx]+'\t'+sequence[idx]+'\t'+label[idx]+'\n')
                        else:
                            if label[i] == 'pFind' and flag == 'Comet' or flag == 'pFind' and label[i] == 'Comet':
                                f_pc.write(spectra[i]+'\t'+sequence[i]+'\n')
                            elif label[i] == 'pFind' and  flag== 'Mascot' or flag == 'pFind' and label[i] == 'Mascot':
                                f_mp.write(spectra[i]+'\t'+sequence[i]+'\n')
                            elif label[i] == 'Comet' and flag == 'Mascot' or flag == 'Comet' and label[i] == 'Mascot':
                                f_mc.write(spectra[i]+'\t'+sequence[i]+'\n')
                    elif count == 1:
                        if label[i] == 'pFind':
                            f_p.write(spectra[i]+'\t'+sequence[i]+'\n')
                        elif label[i] == 'Comet':
                            f_c.write(spectra[i]+'\t'+sequence[i]+'\n')
                        elif label[i] == 'Mascot':
                            f_m.write(spectra[i]+'\t'+sequence[i]+'\n')
                    elif count > 3:
                        f_wrong.write(spectra[i]+'\t'+sequence[i]+'\n')
            f_mpc.close();f_mp.close();f_pc.close();f_mc.close();f_p.close();f_m.close();f_c.close()
    def fdr_merge(self):
        for charge in self.charge:
            f_ss = open('data/pre_data/MMdata/3result/'+self.mgf_name+'/merge_result/spectrum_result/charge'+str(charge)+'/result_mpc.txt')
            spectra = []
            for line in f_ss:
                spectra.append(line.split('\t')[0])
            f_ss.close()
            write_folder=makedir('data/pre_data/MMdata/3result/'+self.mgf_name+'/merge_result/psms_result/charge'+str(charge))
            fr = open(write_folder+'/psms_mpc.txt',"w")
            files,names=get_files('data/pre_data/MMdata/mascot/'+self.mgf_name+'/FDR0_01/proteins_to_psms_charge'+str(charge))
            for file in files:
                f_psms = open(file)
                decoy = 0;target = 0
                for line in f_psms:
                    spectrum = line.split('\t')[1]
                    if spectrum in spectra:
                        for i in range(5):
                            fr.write(line.split('\t')[i]+'\t')
                        fr.write(line.split('\t')[5])
                        proteins = line.split('\t')[5].split(',')
                        count = 0
                        for i in range(len(proteins)):
                            if proteins[i].__contains__('REVERSE_'):
                                count += 1
                        if count == len(proteins):
                            decoy += 1
                        else:
                            target += 1
                f_psms.close()
            fr.close()
    def remove_decoy_redundant(self):
        for charge in self.charge:
            f = open('data/pre_data/MMdata/3result/'+self.mgf_name+'/merge_result/psms_result/charge'+str(charge)+'/psms_mpc.txt')
            write_folder=makedir('data/pre_data/MMdata/3result/'+self.mgf_name+'/merge_result/no_redundant_decoy_psms_result/charge'+str(charge))
            fr = open(write_folder+'/psms_mpc_no_redundant_decoy.txt','w')

            psms = [];peptides = []
            for line in f:
                psm = []
                proteins = line.split('\t')[5].split('\n')[0]
                psm.append(line.split('\t')[0])
                psm.append(line.split('\t')[1])
                psm.append(line.split('\t')[2])
                psm.append(line.split('\t')[3])
                psm.append(line.split('\t')[4])
                psm.append(line.split('\t')[5].split('\n')[0])
                charge = line.split('\t')[2]
                sequence = line.split('\t')[3]
                modification = line.split('\t')[4]
                if [charge,sequence,modification] in peptides:
                    idx = peptides.index([charge,sequence,modification])
                    if float(psms[idx][0]) <= float(psm[0]):
                        continue
                    else:
                        psms[idx] = psm
                        continue
                    
                count = 0
                for i in range(len(proteins)):
                    if proteins[i].__contains__('REVERSE_'):
                        count += 1
                if count == len(proteins):
                    continue
                else:
                    peptides.append([charge,sequence,modification])
                    psms.append(psm)
                    continue
            f.close()

            for i in range(len(peptides)):
                for j in range(6):
                    fr.write(psms[i][j]+'\t')
                fr.write('\n')

            fr.close()
    def psms_mpc_classify(self):
        for charge in self.charge:
            
            write_folder=makedir('data/pre_data/MMdata/3result/'+self.mgf_name+'/merge_result/finally_result/charge'+str(charge))
            for _mgf in range(self.num_mgf): 
                f = open('data/pre_data/MMdata/3result/'+self.mgf_name+'/merge_result/no_redundant_decoy_psms_result/charge'+str(charge)+'/psms_mpc_no_redundant_decoy.txt')
                fr = open(write_folder+'/psms_'+self.mgf_name+str(_mgf+1)+'.txt','w')
                for line in f:
                    spectrum = line.split('\t')[1]
                    name_s = spectrum.split('.')[0]
                    if name_s[-1] == str(_mgf+1):
                        fr.write('0\t')
                        for i in range(5):
                            fr.write(line.split('\t')[i]+'\t')
                        fr.write(line.split('\t')[5]+'\n')
                fr.close();f.close()
    def data_run(self):
        spectrum_sequence_file_folder_list=[]
        #comet
        files,names=get_files(self.comet_sqt+'sqt')
        self.num_mgf=len(files)
        comet=_comet(files,names,makedir(self.comet_sqt+self.first_folder))
        comet._comet_file()
        files,names=get_files(self.comet_sqt+self.first_folder)
        comet.comet_Sequence_modification(files,names,makedir(self.comet_sqt+self.mod_folder))
        files,names=get_files(self.comet_sqt+self.mod_folder)
        self.fdr_filter(files,names,makedir(self.comet_sqt+self.fdr_folder))
        spectrum_sequence_file_folder_list.append(self.comet_sqt+'/'+self.spectrum_sequence_folder)
        for charge in self.charge:
            files,names=get_files(self.comet_sqt+self.fdr_folder+'/proteins_to_psms_charge'+str(charge))
            self.extract_spectrum_sequence(files,names,self.comet_sqt,'Comet',charge)
        print('------------------end comet')
        #mascot
        files,names=get_files(self.mascot_dat+'dat')
        mascot=_mascot(files,names,makedir(self.mascot_dat+self.first_folder))
        mascot._mascot_file()
        files,names=get_files(self.mascot_dat+self.first_folder)
        mascot.mascot_change_format(files,names,makedir(self.mascot_dat+self.mod_folder))
        files,names=get_files(self.mascot_dat+self.mod_folder)
        self.fdr_filter(files,names,makedir(self.mascot_dat+self.fdr_folder))
        spectrum_sequence_file_folder_list.append(self.mascot_dat+'/'+self.spectrum_sequence_folder)
        for charge in self.charge:
            files,names=get_files(self.mascot_dat+self.fdr_folder+'/proteins_to_psms_charge'+str(charge))
            self.extract_spectrum_sequence(files,names,self.mascot_dat,'Mascot',charge)
        print('------------------end mascot')
        #pfind
        files,names=get_files(self.pfind_txt+'txt')
        pfind=_pFind(files,names,makedir(self.pfind_txt+self.first_folder))
        pfind._pfind_file()
        files,names=get_files(self.pfind_txt+self.first_folder)
        pfind.pFind_modification(files,names,makedir(self.pfind_txt+self.mod_folder))
        files,names=get_files(self.pfind_txt+self.mod_folder)
        self.fdr_filter(files,names,makedir(self.pfind_txt+self.fdr_folder))
        spectrum_sequence_file_folder_list.append(self.pfind_txt+'/'+self.spectrum_sequence_folder)
        for charge in self.charge:
            files,names=get_files(self.pfind_txt+self.fdr_folder+'/proteins_to_psms_charge'+str(charge))
            self.extract_spectrum_sequence(files,names,self.pfind_txt,'pFind',charge)
        print('------------------end pfind')

        for charge in self.charge:
            self.merge_spectrum_sequence_file(spectrum_sequence_file_folder_list,'data/pre_data/MMdata/3result/'+self.mgf_name+'/spectrum_sequence',charge)

    def merge_psms_charge_file(self,spec_name):
        filelist=[]
        _folder='data/pre_data/MMdata/3result/'+self.mgf_name+'/merge_result/finally_result/charge'
        for charge in self.charge:
            filelist.append(_folder+str(charge)+'/psms_'+spec_name+'.txt') 
        fr_folder=makedir(_folder+'_all/'+spec_name)
        fr_file_name=fr_folder+'/psms_all_charge.txt'
        with open(fr_file_name,'w') as newfile:
            for item in filelist:
                for txt in open(item,'r'):
                    newfile.write(txt)

    def ion_run(self):
        spectrum_files,spectrum_names=get_files(self.mgf_folder)
        for spectrum_file_index in range(len(spectrum_files)):
            print('   ######################################\n   --use ' + spectrum_files[spectrum_file_index])
            _folder=makedir('data/pre_data/MMdata/matched_ion/'+os.path.splitext(spectrum_names[spectrum_file_index])[0])
            self.merge_psms_charge_file(os.path.splitext(spectrum_names[spectrum_file_index])[0])
            psms_file,_=get_files('data/pre_data/MMdata/3result/'+self.mgf_name+'/merge_result/finally_result/charge_all/'+os.path.splitext(spectrum_names[spectrum_file_index])[0])
            self.psms=get_psms(psms_file[0])
            print('   --use ' + psms_file[0])
            self.spectrums=get_spectrums(spectrum_files[spectrum_file_index],'mm')
            self.match_psms_with_spectra(makedir(_folder))
            print('   ######################################')
    def match_psms_with_spectra2(self,ion_folder):
        sum_of_intensity=0.0
        for psm in self.psms:
            spec_name=psm[0]+'#'+psm[4]
            if spec_name in self.spectrums.keys():
                _spectrum=np.array(self.spectrums[spec_name])
                sum_of_intensity+=np.sum(_spectrum[:,1])
        with open(ion_folder+'/sum_of_intensity.txt','w') as wf:
            wf.write(str(sum_of_intensity)+'\n')
    def match_psms_with_spectra(self,ion_folder):
       
        match_b_y_result=[];spectrums=[]
        match_ay_by_result=[]
        for psm in self.psms:
            spec_name=psm[0]+'#'+psm[4]
            if spec_name in self.spectrums.keys():
                spectrums.append(spec_name)
                peptide=psm[2]
                _spectrum=np.array(self.spectrums[spec_name])
                charge=psm[1]
               

                ions_b_y=[];ions_a=[]
                #b+/y+
                bs,bs_mass,b_name,ys,ys_mass,y_name=get_y_and_b(peptide)
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
                bys,bys_mass,bys_name=get_all_by(peptide,self.internal_ion_min_length,self.internal_ion_max_length)
                #by+/ay+
                bys_mz=np.array(bys_mass)+self.PROTON
                ays_mz=bys_mz-self.CO
                ions_ay_by.append([bys,bys_mz,[_name+'+' for _name in bys_name]])
                ions_ay_by.append([bys,ays_mz,[_name.replace('b','a')+'+' for _name in bys_name]])
               

                #ions match
                
                #b+,b++,y+,y++
                _str=''
                match_pep_result=[];b_y_result=[]
                for i in range(len(peptide)-1):
                    match_pep_result.append(peptide)
                    temp_ion='';temp_mz='';temp_ion_name='';temp_inten_r='';temp_inten=''
                    for j in range(4):
                        if j%2==0:
                            temp_ion+=ions_b_y[j][0][i]+','
                        index,_mz,_val=closest_mz(_spectrum[:,0],ions_b_y[j][1][i])
                        if index!=-1:
                          
                            temp_mz+=str(_spectrum[index][0])+','
                            _str+= str(_spectrum[index][0])+','+str(_mz-_val)+','+str(_spectrum[index][2])+'\n'
                            temp_inten_r+=str(_spectrum[index][2])+','
                            temp_inten+=str(_spectrum[index][1])+','
                            _spectrum = np.delete(_spectrum, index, 0)
                        else:
                            temp_mz+='0.0,'
                            temp_inten_r+='0.0,'
                            temp_inten+='0.0,'
                        temp_ion_name+=ions_b_y[j][2][i]+','
                    b_y_result.append([peptide,charge,temp_ion[:-1],psm[3],temp_mz[:-1],temp_ion_name[:-1],temp_inten_r[:-1],temp_inten[:-1]])
                match_b_y_result.append(b_y_result)
                #with open('data/pre_data/MMdata/regular_ions_error_distribution_0.05_Da.txt','a') as f:
                #    f.write(_str)

                 #a+,a++
                for i in range(len(peptide)-1):
                    for j in range(2):
                        index=closest_mz(_spectrum[:,0],ions_a[j][1][i])
                        if index!=-1:
                            _spectrum = np.delete(_spectrum, index, 0)

                #ay+ by+
                _str=''
                match_pep_result=[]
                ay_by_result=[]
                for i in range(len(bys_mz)):
                    match_pep_result.append(peptide)
                    temp_mz='';temp_ion_name='';temp_inten_r='';temp_inten=''
                    for j in range(2):
                        temp_ion=ions_ay_by[j][0][i]
                        index,_mz,_val=closest_mz(_spectrum[:,0],ions_ay_by[j][1][i])
                        if index!=-1:
                            
                            temp_mz+=str(_spectrum[index][0])+','
                            _str+= str(_spectrum[index][0])+','+str(_mz-_val)+','+str(_spectrum[index][2])+'\n'
                            temp_inten_r+=str(_spectrum[index][2])+','
                            temp_inten+=str(_spectrum[index][1])+','
                            _spectrum = np.delete(_spectrum, index, 0)
                        else:
                            temp_mz+='0.0,'
                            temp_inten_r+='0.0,'
                            temp_inten+='0.0,'
                        temp_ion_name+=ions_ay_by[j][2][i]+','
                    ay_by_result.append([peptide,charge,temp_ion,psm[3],temp_mz[:-1],temp_ion_name[:-1],temp_inten_r[:-1],temp_inten[:-1]])
                    
                match_ay_by_result.append(ay_by_result)
             
                #with open('data/pre_data/MMdata/internal_ions_error_distribution_0.05_Da.txt','a') as f:
                #    f.write(_str) 

        b_y_train_wf=open(ion_folder+'/b_y_train.txt','w')      
        b_y_test_wf=open(ion_folder+'/b_y_test.txt','w')
        pep_temp=''
        for k in range(len(match_b_y_result)):
            a= random.random()
            if a<0.9:
                for _list in match_b_y_result[k]:
                    b_y_train_wf.write('\t'.join(_list)+'\t'+spectrums[k]+'\n')
            else:
                for _list in match_b_y_result[k]:
                    b_y_test_wf.write('\t'.join(_list)+'\t'+spectrums[k]+'\n') 
        b_y_train_wf.close()
        b_y_test_wf.close()

       
        ay_by_train_wf=open(ion_folder+'/ay_by_train.txt','w')      
        ay_by_test_wf=open(ion_folder+'/ay_by_test.txt','w')
        pep_temp=''
        for k in range(len(match_ay_by_result)):
            a= random.random()
            if a<0.9:
                for _list in match_ay_by_result[k]:
                    ay_by_train_wf.write('\t'.join(_list)+'\t'+spectrums[k]+'\n')
            else:
                for _list in match_ay_by_result[k]:
                    ay_by_test_wf.write('\t'.join(_list)+'\t'+spectrums[k]+'\n') 
        ay_by_train_wf.close()
        ay_by_test_wf.close()



class _comet():
    def __init__(self,read_files,file_names,write_files):
        self.read_files=read_files
        self.write_files=write_files
        self.file_names=file_names
        
    def _comet_file(self):
        for file_index in range(len(self.read_files)):
            fname =  self.read_files[file_index]
            print(fname)
            frname2 = self.write_files+'/'+self.file_names[file_index]+'_charge2.txt'
            frname3 = self.write_files+'/'+self.file_names[file_index]+'_charge3.txt'
            f = open(fname)
            fr2 = open(frname2,'w')
            fr3 = open(frname3,'w')
            psms = []
            spectrum1 = [];spectrum2 = [];spectrum3 = []
            flag = 0;sure = 0;pureRev = 0
            for line in f:
                if line.split('\t')[0] == 'S':
                    spectrum1 = []
                    spectrum2 = []
                    flag = 0;sure = 0;pureRev = 0
                    spectrum1.append(line.split('\t')[10].split('\n')[0])
                    spectrum2.append(line.split('\t')[10].split('\n')[0])
                    spectrum3.append(line.split('\t')[10].split('\n')[0])
                    spectrum1.append(int(line.split('\t')[3]))
                    spectrum2.append(int(line.split('\t')[3]))
                    spectrum3.append(int(line.split('\t')[3]))
                    continue
                if pureRev == 0 and line.split('\t')[0] == 'M' and line.split('\t')[1] == '1' and sure == 0:
                    spectrum1.append(line.split('\t')[1])
                    spectrum1.insert(0,float(line.split('\t')[6]))
                    spectrum1.append(float(line.split('\t')[5]))
                    spectrum1.append(line.split('\t')[9].split('.')[1])
                    continue
                if pureRev == 0 and line.split('\t')[0] == 'L' and sure == 0:
                    i = 0;numRev = 0;length = len(line.split('\t')[1].split(';'));spectrum = []
                    while i < length:
                        spectrum.append(line.split('\t')[1].split(';')[i].split('\n')[0])
                        if line.split('\t')[1].split(';')[i].__contains__('REVERSE_'):
                            numRev += 1
                        i += 1
                    if numRev == length:
                        pureRev = 1
                        spectrum1.append('1')
                        spectrum1.append(','.join(spectrum))
                    else:
                        spectrum1.append('0')
                        spectrum1.append(','.join(spectrum))
                        psms.append(spectrum1)
                        sure = 1
                    continue
                if pureRev == 1 and line.split('\t')[1] == '1' and line.split('\t')[0] == 'M' and sure == 0:
                    spectrum2.append(line.split('\t')[1])
                    spectrum2.insert(0,float(line.split('\t')[6]))
                    spectrum2.append(float(line.split('\t')[5]))
                    spectrum2.append(line.split('\t')[9].split('.')[1])
                    continue
                elif pureRev == 1 and line.split('\t')[1] == '2' and line.split('\t')[0] == 'M' and sure == 0:
                    psms.append(spectrum1)
                    sure = 1
                    continue
                if pureRev == 1 and line.split('\t')[0] == 'L' and sure == 0:
                    i = 0;numRev = 0;length = len(line.split('\t')[1].split(';'));spectrum = []
                    while i < length:
                        spectrum.append(line.split('\t')[1].split(';')[i].split('\n')[0])
                        if line.split('\t')[1].split(';')[i].__contains__('REVERSE_'):
                            numRev += 1
                        i += 1
                    if numRev == length:
                        flag = 1
                        spectrum2.append('1')
                        spectrum2.append(','.join(spectrum))
                    else:
                        spectrum2.append('0')
                        spectrum2.append(','.join(spectrum))
                        psms.append(spectrum2)
                        sure = 1
                    continue
                if flag == 1 and line.split('\t')[1] == '1' and line.split('\t')[0] == 'M' and sure == 0:
                    spectrum3.append(line.split('\t')[1])
                    spectrum3.insert(0,float(line.split('\t')[6]))
                    spectrum3.append(float(line.split('\t')[5]))
                    spectrum3.append(line.split('\t')[9].split('.')[1])
                    continue
                elif flag == 1 and line.split('\t')[1] == '2' and line.split('\t')[0] == 'M' and sure == 0:
                    psms.append(spectrum1)
                    sure = 1
                    continue
                if flag == 1 and line.split('\t')[0] == 'L' and sure == 0:
                    i = 1;numRev = 0;length = len(line.split('\t')[1].split(';'));spectrum = []
                    while i < length:
                        spectrum.append(line.split('\t')[1].split(';')[i].split('\n')[0])
                        if line.split('\t')[1].split(';')[i].__contains__('REVERSE_'):
                            numRev += 1
                        i += 1
                    if numRev == length - 1:
                        flag = 1
                        spectrum3.append('1')
                        spectrum3.append(','.join(spectrum))
                        psms.append(spectrum1)
                        sure = 1
                    else:
                        spectrum3.append('0')
                        spectrum3.append(','.join(spectrum))
                        psms.append(spectrum3)
                        sure = 1
                    continue
            f.close()

            psms = sorted(psms,key=lambda x:x[0])
            decoy = 0
            target = 0
            psm2 = [];psm3 = [];psm4 = [];psm5 = []
            for i in range(len(psms)):
                if psms[i][2] == 2:
                    psm2.append(psms[i])
                elif psms[i][2] == 3:
                    psm3.append(psms[i])
                elif psms[i][2] == 4:
                    psm4.append(psms[i])
                elif psms[i][2] == 5:
                    psm5.append(psms[i])

            def RPM(psms):
                no_mobile = [];mobile = [];part_mobile = []
                for i in range(len(psms)):
                    peptide = psms[i][5]
                    charge = psms[i][2]
                    num = peptide.count('R')+peptide.count('K')+peptide.count('H')
                    if num < charge:
                        mobile.append(psms[i])
                    elif peptide.count('R') >= charge:
                        no_mobile.append(psms[i])
                    else:
                        part_mobile.append(psms[i])
                return mobile,no_mobile,part_mobile

            psm2_mobile,psm2_no_mobile,psm2_part_mobile = RPM(psm2)
            psm3_mobile,psm3_no_mobile,psm3_part_mobile = RPM(psm3)

            def FDR(ps,fr,charge):
                print('charge'+str(charge)+':'+str(len(ps))+',',)
                decoy = 0;target = 0;i = 0;Fdr = 0
                while i < len(ps):
                    if ps[i][6] == '1':
                        decoy += 1
                    else:
                        target += 1
                    if target == 0:
                        Fdr = 0
                    else:
                        Fdr = float(decoy)/target
                    fr.write(str(ps[i][0])+'\t')
                    fr.write(str(Fdr)+'\t'+str(target)+'\t')
                    fr.write(str(ps[i][1])+'\t'+str(ps[i][2])+'\t'+str(ps[i][5])+'\t'+str(ps[i][7])+'\n')
                    i += 1

            FDR(psm2_mobile,fr2,'2_mobile');FDR(psm2_no_mobile,fr2,'2_no_mobile');FDR(psm2_part_mobile,fr2,'2_part_mobile')
            fr2.close()
            FDR(psm3_mobile,fr3,'3_mobile');FDR(psm3_no_mobile,fr3,'3_no_mobile');FDR(psm3_part_mobile,fr3,'3_part_mobile')
            fr3.close()           
    def comet_Sequence_modification(self,read_files,names,write_files):
        for file_index in range(len(read_files)):
            f_name =  read_files[file_index]
            print('mod\t'+f_name)
            fr_name = write_files+'/mod_'+names[file_index]
            f =  open(f_name,'r')
            fr = open(fr_name,'w')

            for line in f:
                sequence = list(line.split('\t')[5])
                modif = [];count1 = 0;count2 = 0
                if '^' in sequence:
                    modif.append('0')
                    modif.append('Delta_H(2)C(2)[AnyN-term]')
                    del sequence[0]
                    del sequence[0]
                if '#' in sequence:
                    modif.append('0')
                    modif.append('Acetyl[ProteinN-term]')
                    del sequence[0]
                    del sequence[0]
                if '@' in sequence:
                    count1 = sequence.count('@')
                if '*' in sequence:
                    count2 = sequence.count('*')
                for i in range(len(sequence)-count1-count2):
                    if sequence[i] == '@':
                        modif.append(str(i))
                        modif.append('Deamidated[N]')
                        del sequence[i]
                    if sequence[i] == '*':
                        modif.append(str(i))
                        modif.append('Oxidation[M]')
                        del sequence[i]
                sequence = ''.join(sequence)
                if modif:
                    modif = ','.join(modif)
                else:
                    modif = 'NULL'
                protein = line.split('\t')[6].split('\n')[0]
                fr.write(line.split('\t')[0]+'\t'+line.split('\t')[1]+'\t'+line.split('\t')[2]+'\t'+line.split('\t')[3]+'\t'+line.split('\t')[4]+'\t')
                fr.write(sequence+'\t'+modif+'\t'+protein+'\n')

            f.close()
            fr.close()
            print('mod complete')

class _mascot():
    def __init__(self,read_files,file_names,write_files):
        self.read_files=read_files
        self.write_files=write_files
        self.file_names=file_names
        
    def _mascot_file(self):
        for file_index in range(len(self.read_files)):
            fname =  self.read_files[file_index]
            print(fname)
            frname2 = self.write_files+'/'+self.file_names[file_index]+'_charge2.txt'
            frname3 = self.write_files+'/'+self.file_names[file_index]+'_charge3.txt'
            f = open(fname)
            fr2 = open(frname2,'w')
            fr3 = open(frname3,'w')

            psms = [];k = 1;spectrum = [];psm = []
            for line in f:
                sequence1 = 'q'+str(k)+'_p1='
                if line.__contains__(sequence1):
                    if len(line.split(',')) > 3 and len(line.split(',')[4]) > 5:
                        psm = []
                        psm.append(line.split(',')[4])
                        psm.append(line.split(',')[6])
                        psm.insert(0,float(line.split(',')[7]))
                        psm.append(line.split(';')[1].split('\n')[0])
                        psm.append(k)
                        if line.split(';')[1].__contains__('REVERSE_'):
                            if 2 * (line.split(';')[1].count('REVERSE_')) == line.split(';')[1].count('|'):
                                psm.append('1')
                                psms.append(psm)
                                k = k+1
                            else:
                                psm.append('0')
                                psms.append(psm)
                                k = k+1
                        else:
                            psm.append('0')
                            psms.append(psm)
                            k = k+1
                        #print psm
                    elif not line.__contains__('_terms'):
                        k = k+1
                    continue
                
                sequence2 = 'q'+str(k-1)+'_p2='
                if line.__contains__(sequence2) and len(line.split(',')[4]) > 5 and len(psm) > 6 and psm[5] == '1' and float(line.split(',')[7]) == psm[0] and k-1 == psm[4]:
                    if len(line.split(',')) > 3 and len(line.split(',')[4]) > 5:
                        psm2 = []
                        psm2.append(line.split(',')[4])
                        psm2.append(line.split(',')[6])
                        psm2.insert(0,float(line.split(',')[7]))
                        psm2.append(line.split(';')[1].split('\n')[0])
                        psm2.append(k-1)
                        if line.split(';')[1].__contains__('REVERSE_'):
                            if 2 * (line.split(';')[1].count('REVERSE_')) == line.split(';')[1].count('|'):
                                psm2.append('1')
                            else:
                                psm2.append('0')
                                psms[-1] = psm2
                        else:
                            psm2.append('0')
                            psms[-1] = psm2
                    continue
                
                sequence3 = 'q'+str(k-1)+'_p3='
                if line.__contains__(sequence3) and len(line.split(',')[4]) > 5 and len(psm) > 6 \
                   and psm[5] == '1' and float(line.split(',')[7]) == psm[0] and k-1 == psm[4]:
                    if len(line.split(',')) > 3 and len(line.split(',')[4]) > 5:
                        psm3 = []
                        psm3.append(line.split(',')[4])
                        psm3.append(line.split(',')[6])
                        psm3.insert(0,float(line.split(',')[7]))
                        psm3.append(line.split(';')[1].split('\n')[0])
                        psm3.append(k-1)
                        if line.split(';')[1].__contains__('REVERSE_'):
                            if 2 * (line.split(';')[1].count('REVERSE_')) == line.split(';')[1].count('|'):
                                psm3.append('1')
                            else:
                                psm3.append('0')
                                psms[-1] = psm3
                        else:
                            psm3.append('0')
                            psms[-1] = psm3
                    continue 

                if line.__contains__('Content-Type: application/x-Mascot; name="que'):
                    spectrum = []
                    spectrum.append(int(line.split('"')[1].split('ry')[1]))
                    continue
                if line.__contains__('title=2009'):
                    spectrum.append(line.split('=')[1].split('\n')[0])
                    continue
                if line.__contains__('charge='):
                    spectrum.append(int(line.split('+')[0].split('=')[1]))
                    if len(spectrum) < 2:
                        for j in range(len(psms)):
                           if psms[j][4] == spectrum[0]:
                               del(psms[j])
                               break
                    else:
                        for j in range(len(psms)):
                            if psms[j][4] == spectrum[0]:
                                psms[j].append(spectrum[1])
                                psms[j].append(spectrum[2])
                                break
                    continue
            psms = sorted(psms,reverse = True)

            for j in range(len(psms)):
                if len(psms[j])<7:
                    print(psms[j])
            
            decoy = 0
            target = 0
            psm2 = [];psm3 = [];psm4 = [];psm5 = []

            for i in range(len(psms)):
                if psms[i][7] == 2:
                    psm2.append(psms[i])
                elif psms[i][7] == 3:
                    psm3.append(psms[i])
                elif psms[i][7] == 4:
                    psm4.append(psms[i])
                elif psms[i][7] == 5:
                    psm5.append(psms[i])
            
            def FDR(ps,fr,charge):
                print('charge'+str(charge)+':'+str(len(ps))+',',)
                decoy = 0;target = 0;i = 0;Fdr = 0
                while i < len(ps):
                    if ps[i][5] == '1':
                        decoy += 1
                    else:
                        target += 1
                    if target == 0:
                        Fdr = 0
                    else:
                        Fdr = float(decoy)/target
                    fr.write(str(ps[i][0])+'\t')
                    fr.write(str(Fdr)+'\t'+str(target)+'\t')
                    fr.write(str(ps[i][6])+'\t'+str(ps[i][7])+'\t'+str(ps[i][1])+'\t'+str(ps[i][2])+'\t'+str(ps[i][3])+'\n')
                    i += 1

            FDR(psm2,fr2,2)
            fr2.close()
            FDR(psm3,fr3,3)
            fr3.close()
    def mascot_change_format(self,read_files,names,write_files):
        for file_index in range(len(read_files)):
            f_name =  read_files[file_index]
            print('mod\t'+f_name)
            fr_name = write_files+'/mod_'+names[file_index]
            f =  open(f_name,'r')
            fr = open(fr_name,'w')
            for line in f:
                spectrum = line.split('\t')[3]
                modification = line.split('\t')[6]
                proteins = line.split('\t')[7]
                spectrum = spectrum.split('%2e')
                modif_site = []
                modification = list(modification)
                if '1' in modification:
                    modif_site = ['0','Acetyl[ProteinN-term]']
                elif '4' in modification:
                    modif_site = ['0','Delta_H(2)C(2)[AnyN-term]']
                if '2' in modification:
                    count = modification.count('2')
                    for i in range(count):
                        idx = modification.index('2')
                        modif_site.append(str(idx))
                        modif_site.append('Deamidated[N]')
                        modification[idx] = '0'
                if '3' in modification:
                    count = modification.count('3')
                    for i in range(count):
                        idx = modification.index('3')
                        modif_site.append(str(idx))
                        modif_site.append('Oxidation[M]')
                        modification[idx] = '0'
                if len(modif_site) == 0:
                    modif_site.append('NULL')
                count = proteins.count('"')
                protein = []
                for i in range(count):
                    if proteins.split('"')[i].__contains__(':'):
                        continue
                    elif i > 0:
                        protein.append(proteins.split('"')[i])
                fr.write(line.split('\t')[0]+'\t'+line.split('\t')[1]+'\t'+line.split('\t')[2]+'\t'\
                         +'.'.join(spectrum)+'\t'+line.split('\t')[4]+'\t'+line.split('\t')[5]+'\t'\
                         +','.join(modif_site)+'\t'+','.join(protein)+'\n')

            f.close()
            fr.close()
            print('mod complete')

class _pFind():
    def __init__(self,read_files,file_names,write_files):
        self.read_files=read_files
        self.write_files=write_files
        self.file_names=file_names
        
    def _pfind_file(self):
        for file_index in range(len(self.read_files)):
            fname =  self.read_files[file_index]
            print(fname)
            frname2 = self.write_files+'/'+self.file_names[file_index]+'_charge2.txt'
            frname3 = self.write_files+'/'+self.file_names[file_index]+'_charge3.txt'
            #frname4 = self.write_files+'/'+self.file_names[file_index]+'_charge4.txt'
            f = open(fname)
            fr2 = open(frname2,'w')
            fr3 = open(frname3,'w')
            #fr4 = open(frname4,'w')
            
         

            psms = []
            spectra = [];spectra1 = [];spectra3 = []
            flag = 0;sure = 0;no3 = 0;no_3 = 0;no = 0
            for line in f:
                if line.__contains__('Input='):
                    spectra = [];spectra1 = [];spectra3= []
                    flag = 0;sure = 0;no3 = 0;no_3 = 0;no = 0
                    spectra.append(line.split('=')[1].split('\n')[0])
                    spectra1.append(line.split('=')[1].split('\n')[0])
                    spectra3.append(line.split('=')[1].split('\n')[0])
                    continue
                if line.__contains__('Charge='):
                    spectra.append(int(line.split('=')[1]))
                    spectra1.append(int(line.split('=')[1]))
                    spectra3.append(int(line.split('=')[1]))
                    continue
                if line.__contains__('NO1_Score='):
                    spectra.append(float(line.split('=')[1]))
                    continue
                if line.__contains__('NO1_EValue='):
                    spectra.insert(0,float(line.split('=')[1]))
                    continue
                if line.__contains__('NO1_SQ='):
                    spectra.append(line.split('=')[1].split('\n')[0])
                    continue
                if line.__contains__('NO1_Proteins='):
                    spectra.append(line.split('=')[1])
                    proteins = line.split('=')[1]
                    if proteins.__contains__('REVERSE_'):
                        prolen = len(proteins.split(','))
                        numRev = 0
                        for i in range(prolen):
                            if proteins.split(',')[i].__contains__('REVERSE_'):
                                numRev += 1
                        if numRev == prolen - 1:
                            spectra.append('1')
                        else:
                            spectra.append('0')
                    else:
                        spectra.append('0')
                    continue
                if line.__contains__('NO1_Modify_Pos='):
                    modif = line.split('=')[1].split('\n')[0]
                    if len(modif) == 1:
                        modif = 'NULL'
                        spectra.append(modif)
                        continue
                    spectra.append(','.join(line.split('=')[1].split('\n')[0].split(',')[1:]))
                    continue
                if line.__contains__('NO1_Modify_Name='):
                    modif = line.split('=')[1].split('\n')[0]
                    if len(modif) == 1:
                        modif = 'NULL'
                        spectra.append(modif)
                        continue
                    spectra.append(','.join(line.split('=')[1].split('\n')[0].split(',')[1:]))
                    continue
                if line.__contains__('NO1_deltaMass='):
                    flag = 1
                    continue
                if not line.__contains__('NO2_') and not line.__contains__('=') and flag == 1 and sure == 0:
                    psms.append(spectra)
                    continue
                if line.__contains__('NO2_Score='):
                    spectra1.append(float(line.split('=')[1]))
                    continue
                if line.__contains__('NO2_EValue='):
                    spectra1.insert(0,float(line.split('=')[1]))
                    continue
                if line.__contains__('NO2_SQ='):
                    spectra1.append(line.split('=')[1].split('\n')[0])
                    continue
                if line.__contains__('NO2_Proteins='):
                    spectra1.append(line.split('=')[1])
                    proteins = line.split('=')[1]
                    if proteins.__contains__('REVERSE_'):
                        prolen = len(proteins.split(','))
                        numRev = 0
                        for i in range(prolen):
                            if proteins.split(',')[i].__contains__('REVERSE_'):
                                numRev += 1
                        if numRev == prolen - 1:
                            spectra1.append('1')
                        else:
                            spectra1.append('0')
                    else:
                        spectra1.append('0')
                    continue
                if line.__contains__('NO2_Modify_Pos='):
                    modif = line.split('=')[1].split('\n')[0]
                    if len(modif) == 1:
                        modif = 'NULL'
                        spectra.append(modif)
                        continue
                    spectra1.append(','.join(line.split('=')[1].split('\n')[0].split(',')[1:]))
                    continue
                if line.__contains__('NO2_Modify_Name='):
                    modif = line.split('=')[1].split('\n')[0]
                    if len(modif) == 1:
                        modif = 'NULL'
                        spectra.append(modif)
                        continue
                    spectra1.append(','.join(line.split('=')[1].split('\n')[0].split(',')[1:]))
                    if spectra[3] == spectra1[3]:
                        if spectra[5].__contains__('REVERSE_') and not spectra1[5].__contains__('REVERSE_') or spectra[6] == '1':
                            if spectra1[6] == '1':
                                no3 = 1
                                sure = 1
                                continue
                            else:
                                psms.append(spectra1)
                                sure = 1
                        else:
                            psms.append(spectra)
                            sure = 1
                    else:
                        psms.append(spectra)
                        sure = 1
                        continue
                if no3 == 1 and line.__contains__('NO3_Score='):
                    score = float(line.split('=')[1])
                    spectra3.append(score)
                    if score == spectra[3]:
                        no_3 = 1
                        continue
                    else:
                        psms.append(spectra)
                        sure = 1
                        continue
                if no_3 == 1 and line.__contains__('NO3_EValue='):
                    spectra3.insert(0,float(line.split('=')[1]))
                    continue
                if no_3 == 1 and line.__contains__('NO3_SQ='):
                    spectra3.append(line.split('=')[1].split('\n')[0])
                    continue
                if no_3 == 1 and line.__contains__('NO3_Proteins='):
                    spectra3.append(line.split('=')[1])
                    proteins = line.split('=')[1]
                    if proteins.__contains__('REVERSE_'):
                        prolen = len(proteins.split(','))
                        numRev = 0
                        for i in range(prolen):
                            if proteins.split(',')[i].__contains__('REVERSE_'):
                                numRev += 1
                        if numRev == prolen - 1:
                            psms.append(spectra)
                        else:
                            spectra3.append('0')
                            no = 1
                    else:
                        spectra3.append('0')
                        no = 1
                    continue
                if no == 1 and line.__contains__('NO3_Modify_Pos='):
                    modif = line.split('=')[1].split('\n')[0]
                    if len(modif) == 1:
                        modif = 'NULL'
                        spectra.append(modif)
                        continue
                    spectra3.append(','.join(line.split('=')[1].split('\n')[0].split(',')[1:]))
                    continue
                if no == 1 and line.__contains__('NO3_Modify_Name='):
                    modif = line.split('=')[1].split('\n')[0]
                    if len(modif) == 1:
                        modif = 'NULL'
                        spectra.append(modif)
                        continue
                    spectra3.append(','.join(line.split('=')[1].split('\n')[0].split(',')[1:]))
                    psms.append(spectra3)
                    continue
            f.close()

            psms = sorted(psms)
            decoy = 0
            target = 0
            psm2 = [];psm3 = [];psm4 = [];psm5 = []

            for i in range(len(psms)):
                if psms[i][2] == 2:
                    psm2.append(psms[i])
                elif psms[i][2] == 3:
                    psm3.append(psms[i])
                elif psms[i][2] == 4:
                    psm4.append(psms[i])
                elif psms[i][2] == 5:
                    psm5.append(psms[i])

            def FDR(ps,fr,charge):
                print('charge'+str(charge)+':'+str(len(ps))+',',)
                decoy = 0;target = 0;i = 0;Fdr = 0
                while i < len(ps):
                    if ps[i][6] == '1':
                        decoy += 1
                    else:
                        target += 1
                    if target == 0:
                        Fdr = 0
                    else:
                        Fdr = float(decoy)/target
                    fr.write(str(ps[i][0])+'\t')
                    fr.write(str(Fdr)+'\t'+str(target)+'\t')
                    fr.write(ps[i][1]+'\t'+str(ps[i][2])+'\t'+ps[i][4]+'\t'+ps[i][7]+','+ps[i][8]+'\t'+ps[i][5])
                    i += 1
                fr.close()

            FDR(psm2,fr2,2);FDR(psm3,fr3,3)#;FDR(psm4,fr4,4)          
    def pFind_modification(self,read_files,names,write_files):
        for file_index in range(len(read_files)):
            f_name =  read_files[file_index]
            print('mod\t'+f_name)
            fr_name = write_files+'/mod_'+names[file_index]
            f =  open(f_name,'r')
            fr = open(fr_name,'w')
            psms = []
            for line in f:
                psm = []
                psm.append(line.split('\t')[0])
                psm.append(line.split('\t')[1])
                psm.append(line.split('\t')[2])
                psm.append(line.split('\t')[3])
                psm.append(line.split('\t')[4])
                psm.append(line.split('\t')[5])
                psm.append(line.split('\t')[6])
                proteins = line.split('\t')[7].split('\n')[0].split(',')[1:]
                psm.append(','.join(proteins))
                psms.append(psm)
            for i in range(len(psms)):
                for j in range(6):
                    fr.write(psms[i][j]+'\t')
                if psms[i][6].__contains__('NULL'):
                    fr.write('NULL'+'\t'+psms[i][7]+'\n')
                else:
                    length = len(psms[i][6].split(','))
                    if length == 2:
                        fr.write(psms[i][6].split(',')[0]+','+psms[i][6].split(',')[1]+'\t')
                    elif length == 4:
                        fr.write(psms[i][6].split(',')[0]+','+psms[i][6].split(',')[2]+','+psms[i][6].split(',')[1]+','+psms[i][6].split(',')[3]+'\t')
                    elif length == 6:
                        fr.write(psms[i][6].split(',')[0]+','+psms[i][6].split(',')[3]+','+psms[i][6].split(',')[1]+','+psms[i][6].split(',')[4]+','+\
                                 psms[i][6].split(',')[2]+','+psms[i][6].split(',')[5]+'\t')
                    elif length == 8:
                        fr.write(psms[i][6].split(',')[0]+','+psms[i][6].split(',')[4]+','+psms[i][6].split(',')[1]+','+psms[i][6].split(',')[5]+','+\
                                 psms[i][6].split(',')[2]+','+psms[i][6].split(',')[6]+','+psms[i][6].split(',')[3]+','+psms[i][6].split(',')[7]+'\t')
                    elif length == 10:
                        fr.write(psms[i][6].split(',')[0]+','+psms[i][6].split(',')[5]+','+psms[i][6].split(',')[1]+','+psms[i][6].split(',')[6]+','+\
                                 psms[i][6].split(',')[2]+','+psms[i][6].split(',')[7]+','+psms[i][6].split(',')[3]+','+psms[i][6].split(',')[8]+','+\
                                 psms[i][6].split(',')[4]+','+psms[i][6].split(',')[9]+'\t')
                    fr.write(psms[i][7]+'\n')

            fr.close()
            print('mod complete')

if __name__=='__main__':
    for mgf_name in ['Ecoli','Hela','Yeast']:
        data_preprocessing(mgf_name)
    