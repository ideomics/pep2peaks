import os
import random
 
def is_file_contain_word(file_list, query_word):
    c_file_list=[]
    for _file in file_list:
        if query_word == os.path.basename(_file):
           c_file_list.append(_file) 
    return c_file_list
 
 
def get_all_file(floder_path):
    file_list = []
    if floder_path is None:
        raise Exception("floder_path is None")
    for dirpath, dirnames, filenames in os.walk(floder_path):
        for name in filenames:
            file_list.append(dirpath + '\\' + name)
    return file_list



def merge_ions_proteometools(base_dir='D:\\xiaoluolin\\gan_\\gan_\\data\\pre_data\\ProteomeToolsData\\result\\mgf_result\\HCD\\NCE'):
    NCE=[25,30,35]
    for file in ['b_y_train.txt','b_y_test.txt','ay_by_train.txt','ay_by_test.txt']:
        for _nce in NCE:
            f=open('data/data_proteometools/NCE'+str(_nce)+'_'+file,'w')
            filelist=is_file_contain_word(get_all_file(base_dir+str(_nce)), file)
            
            for item in filelist:
                for txt in open(item,'r'):
                    f.write(txt)
            f.close()
            print('NCE'+str(_nce)+'_'+file+' compelete')
def merge_ions_mm(base_dir='D:\\xiaoluolin\\gan_\\gan_\\data\\pre_data\\MMdata\\matched_ion'):
  
    for file in ['b_y_train.txt','b_y_test.txt','ay_by_train.txt','ay_by_test.txt']:
      
        f=open('data/data_proteometools/'+file,'w')
        filelist=is_file_contain_word(get_all_file(base_dir), file)
        
        for item in filelist:
            for txt in open(item,'r'):
                f.write(txt)
            
        f.close()
        print(file+' compelete') 
def main(data_type):
    if data_type=='proteometools':
        merge_ions_proteometools()
    elif data_type=='mm': 
        merge_ions_mm()
if __name__=='__main__':
    main('proteometools')