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

base_dir='D:\\xiaoluolin\\gan_\\gan_\\data\\pre_data\\ProteomeToolsData\\result\\mgf_result\\HCD\\NCE25'
#base_dir='D:\\xiaoluolin\\gan_\\gan_\\data\\pre_data\\MMdata\\matched_ion'
#f_all=open('data/data_proteometools/ay_by_all_NCE25.txt','w')
#f_train=open('data/data_mm/train_new.txt','w')
#f_test=open('data/data_mm/test_new.txt','w')

f=open('data/data_proteometools/NCE25_ay_by_test.txt','w')
for file in ['ay_by_test.txt']:
    filelist=is_file_contain_word(get_all_file(base_dir), file)
    
    for item in filelist:
        for txt in open(item,'r'):
            f.write(txt)
        #a= random.random()
        #if a<0.9:
        #    for txt in open(item,'r'):
        #        f_train.write(txt)
        #else:
        #    for txt in open(item,'r'):
        #        f_test.write(txt)
    #f_train.close()
    #f_test.close()
    #f_all.close()
    f.close()
