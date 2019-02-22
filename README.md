# pep2peaks

pep2peaks:Intensity prediction model of internal ions based on seq2seq

1, Environmental requirements: python 3.5.2, tendorflow GPU 1.8.0

2,Data.
   The experimental data is a text document in .txt format. The content format is: 
   peptide(tab)charge[tab]ion[tab]modification[tab]ion-mass[tab]ion-type[tab]ion-relative-intensity[tab]ion-absolute-intensity 
	
   In the regular ion data, ions with 1 and 2 charges, while internal ions are only ions with 1 charge.

	
	example：
	The content of a regular ion of a peptide LLDEGR:
	LLDEGR	2	L,LDEGR	NULL	0.0,0.0,589.29501,295.15133	b1+,b1++,y5+,y5++	0.0,0.0,0.688788517033,0.00819747961622	0.0,0.0,797391.8,9490.0
	LLDEGR	2	LL,DEGR	NULL	227.17565,0.0,476.21061,0.0	b2+,b2++,y4+,y4++	0.0380382057833,0.0,1.0,0.0	44035.8,0.0,1157672.9,0.0
	LLDEGR	2	LLD,EGR	NULL	0.0,0.0,361.1835,0.0	b3+,b3++,y3+,y3++	0.0,0.0,0.518362829431,0.0	0.0,0.0,600094.6,0.0
	LLDEGR	2	LLDE,GR	NULL	0.0,0.0,232.14079,0.0	b4+,b4++,y2+,y2++	0.0,0.0,0.273431122038,0.0	0.0,0.0,316543.8,0.0
	LLDEGR	2	LLDEG,R	NULL	0.0,0.0,175.11924,0.0	b5+,b5++,y1+,y1++	0.0,0.0,0.126727506535,0.0	0.0,0.0,146709.0,0.0
	
	The content of a internal ion of a peptide AELFLR:
	AELFLR	2	E	NULL	0.0,102.05501	y5b2+,y5a2+	0.0,0.338476117442	0.0,47933.6
	AELFLR	2	L	NULL	0.0,86.09643	y4b3+,y4a3+	0.0,0.242746753719	0.0,34376.8
	AELFLR	2	F	NULL	0.0,120.08088	y3b4+,y3a4+	0.0,0.258950442712	0.0,36671.5
	AELFLR	2	L	NULL	0.0,0.0	y2b5+,y2a5+	0.0,0.0	0.0,0.0
	AELFLR	2	EL	NULL	243.13436,215.1393	y5b3+,y5a3+	0.0392583036227,0.0840166958654	5559.6,11898.1
	AELFLR	2	LF	NULL	261.16006,233.16513	y4b4+,y4a4+	0.176993543804,0.220219622232	25065.1,31186.6
	AELFLR	2	FL	NULL	0.0,0.0	y3b5+,y3a5+	0.0,0.0	0.0,0.0	
	AELFLR	2	ELF	NULL	390.2018,0.0	y5b4+,y5a4+	0.0128269495163,0.0	1816.5,0.0	
	AELFLR	2	LFL	NULL	0.0,0.0	y4b5+,y4a5+	0.0,0.0	0.0,0.0	
	AELFLR	2	ELFL	NULL	0.0,0.0	y5b5+,y5a5+	0.0,0.0	0.0,0.0
	
3,use pep2peaks.
    The models that have been trained in this project are stored in the models/.
    For the prediction of a single peptide, this project provides a sample interface located in example.py. For the prediction of multiple peptides, the peptide needs to be written into the file as shown above. You can also use scipt data_preprocessing/mm_data.py or data_preprocessing/proteometools_data.py which can generate the data needed for the experiment,the input files of the scripts are original spectrum files(.raw) and identification result files.
	
   The prediction script is located in model/pep2peaks.py, and the parameter is_train is set to 2, indicating that only the test is performed, and the parameter model needs to be set. Indicates the location of the model. Set the parameter is_train to 1 if retraining is required.
	
   In addition, whether it is training or prediction, different parameters need to be set for different ion types.
   For training or prediction of regular ions, you need to set:
	ion_type=regular
	input_dim=188
	output_dim=4
   For training or prediction of internal ions, you need to set:		
	ion_type=internal
	input_dim=217
	output_dim=2
	min_internal_ion_len=？
	max_internal_ion_len=？
   Where ion_type represents the ion type, input_dim and output_dim represent the input dimension and output dimension of the data respectively. In this project, the input dimension of the regular ion is 188, and the input dimension of the internal ion is 217. For details, please refer to the file tools/get_data.py. The output dimension of a regular ion is 4, which stands for b+/b++/y+/y++, and the output dimension of the internal ion is 2, which stands for by+/ay+. Parameter min_internal_ion_len and max_internal_ion_len represent the minimum fragment length and the maximum fragment length of the internal ions that need to be trained or predicted in the experiments of internal ions (both sides of the threshold are closed intervals).
	
contact:
   hpwang@sdut.edu.cn
   xzachariah0604@gmail.com
