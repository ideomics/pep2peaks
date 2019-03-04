# pep2peaks

## [pep2peaks: prediction of regular and internal fragment ion peaks in peptide MS/MS spectra based on seq2seq](http://)

#### 1, Environmental requirements<br/>
		python 3.5.2, tendorflow GPU 1.8.0

#### 2,Data.<br/>
   The experimental data is a text document in .txt format. 
   The content format of regular ions is: 
   peptide[tab]charge[tab]ion[tab]modification[tab]ion-type[tab]ion-relative-intensity[tab]spectrum
	
   In the regular ion data, ions with 1 and 2 charges, while internal ions are only with 1 charge.

	
	example：
	The content of a regular ion of a peptide HYLEAAAR:
	HYLEAAAR	2	H,YLEAAAR	NULL	b1+,b1++,y7+,y7++	0.055801678502188666,0.0,0.34054747771956806,0.022027830716458356	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	HY,LEAAAR	NULL	b2+,b2++,y6+,y6++	0.2768934594496414,0.0,0.4428575446171102,0.0	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	HYL,EAAAR	NULL	b3+,b3++,y5+,y5++	0.011067781570827382,0.0,0.1445644306575231,0.0018195846107667126	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	HYLE,AAAR	NULL	b4+,b4++,y4+,y4++	0.010873951004725246,0.0,0.20736317446973368,0.0	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	HYLEA,AAR	NULL	b5+,b5++,y3+,y3++	0.012034780575952114,0.0,0.024729345088921436,0.0	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	HYLEAA,AR	NULL	b6+,b6++,y2+,y2++	0.008291075370718621,0.0,0.022299075528757272,0.0	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	HYLEAAA,R	NULL	b7+,b7++,y1+,y1++	0.0016220911696451649,0.0,0.24635664659539197,0.0	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	
	The content of a internal ion of a peptide HYLEAAAR:
	HYLEAAAR	2	Y	NULL	y7b2+,y7a2+	0.0,0.09076689353627398	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	L	NULL	y6b3+,y6a3+	0.0,0.0	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	E	NULL	y5b4+,y5a4+	0.0,0.0	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	A	NULL	y4b5+,y4a5+	0.0,0.0	0.3333333333333333	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	A	NULL	y3b6+,y3a6+	0.0,0.0	0.3333333333333333	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	A	NULL	y2b7+,y2a7+	0.0,0.0	0.3333333333333333	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	YL	NULL	y7b3+,y7a3+	0.04892274434120106,0.06699506787705252	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	LE	NULL	y6b4+,y6a4+	0.060810968760477604,0.014570875671321308	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	EA	NULL	y5b5+,y5a5+	0.013852321110397133,0.002953854908444587	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	AA	NULL	y4b6+,y4a6+	0.011079209192142626,0.008861509850801634	0.5	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	AA	NULL	y3b7+,y3a7+	0.011079209192142626,0.008861509850801634	0.5	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	YLE	NULL	y7b4+,y7a4+	0.04399630090778939,0.0	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	LEA	NULL	y6b5+,y6a5+	0.08481754766813421,0.0	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	EAA	NULL	y5b6+,y5a6+	0.03907976232742645,0.0	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	AAA	NULL	y4b7+,y4a7+	0.02222595521469935,0.01906431789035959	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	YLEA	NULL	y7b5+,y7a5+	0.024426231892090332,0.00207158218092243	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	LEAA	NULL	y6b6+,y6a6+	0.036779421940703674,0.015000337478372575	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	EAAA	NULL	y5b7+,y5a7+	0.021541313114629056,0.013034018368700885	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	YLEAA	NULL	y7b6+,y7a6+	0.010990545666860007,0.008214072683512782	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	LEAAA	NULL	y6b7+,y6a7+	0.01034114673504765,0.007269654548357907	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	HYLEAAAR	2	YLEAAA	NULL	y7b7+,y7a7+	0.004034553944134434,0.003024313807694397	1	01625b_GA4-TUM_first_pool_25_01_01-3xHCD-1h-R1.8323.8323.2.0.dta#35
	
#### 3,use pep2peaks.<br/>
The models that have been trained in this project are stored in the models/.<br/>
For the prediction of a single peptide, this project provides a sample interface located in example.py. For the prediction of multiple peptides, the peptide needs to be written into the file as shown above. You can also use scipt data_preprocessing/mm_data.py or data_preprocessing/proteometools_data.py which can generate the data needed for the experiment,the input files of the scripts are original spectrum files(.raw) and identification result files.<br/>
	
The prediction script is located in model/pep2peaks.py, and the parameter is_train is set to 2, indicating that only the test is performed, and the parameter model needs to be set. Indicates the location of the model. Set the parameter is_train to 1 if retraining is required.<br/>
	
In addition, whether it is training or prediction, different parameters need to be set for different ion types.<br/><br/>
For regular ions, you need to set:<br/>

	ion_type=regular
	input_dim=188
	output_dim=4
For internal ions, you need to set:<br/>	

	ion_type=internal
	input_dim=217
	output_dim=2
	min_internal_ion_len=？
	max_internal_ion_len=？
Where ion_type represents the ion type, input_dim and output_dim represent the input dimension and output dimension of the data respectively. In this project, the input dimension of the regular ion is 188, and the input dimension of the internal ion is 217. For details, please refer to the file tools/get_data.py. The output dimension of a regular ion is 4, which stands for b+/b++/y+/y++, and the output dimension of the internal ion is 2, which stands for by+/ay+. Parameter min_internal_ion_len and max_internal_ion_len represent the minimum fragment length and the maximum fragment length of the internal ions that need to be trained or predicted in the experiments of internal ions (both sides of the threshold are closed intervals).<br/>
	
#### contact:<br/>
		hpwang@sdut.edu.cn
		xzachariah0604@gmail.com
