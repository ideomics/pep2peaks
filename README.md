# pep2peaks

pep2peaks:基于seq2seq的内部离子的强度预测模型

１,环境需求：python 3.5.2，tendorflow GPU 1.8.0

2,数据
	实验数据为.txt格式的文本文档。
	其内容格式为：
	peptide[tab]charge[tab]ion[tab]modification[tab]ion-mass[tab]ion-type[tab]ion-relative-intensity[tab]ion-absolute-intensity
	且在该模型中，常规离子数据为带1个和2个电荷的离子，而内部离子仅为带1个电荷的离子。
	
	例：
	某条肽序列的常规离子的内容为
	LLDEGR	2	L,LDEGR	NULL	0.0,0.0,589.29501,295.15133	b1+,b1++,y5+,y5++	0.0,0.0,0.688788517033,0.00819747961622	0.0,0.0,797391.8,9490.0
	LLDEGR	2	LL,DEGR	NULL	227.17565,0.0,476.21061,0.0	b2+,b2++,y4+,y4++	0.0380382057833,0.0,1.0,0.0	44035.8,0.0,1157672.9,0.0
	LLDEGR	2	LLD,EGR	NULL	0.0,0.0,361.1835,0.0	b3+,b3++,y3+,y3++	0.0,0.0,0.518362829431,0.0	0.0,0.0,600094.6,0.0
	LLDEGR	2	LLDE,GR	NULL	0.0,0.0,232.14079,0.0	b4+,b4++,y2+,y2++	0.0,0.0,0.273431122038,0.0	0.0,0.0,316543.8,0.0
	LLDEGR	2	LLDEG,R	NULL	0.0,0.0,175.11924,0.0	b5+,b5++,y1+,y1++	0.0,0.0,0.126727506535,0.0	0.0,0.0,146709.0,0.0
	
	某条肽序列的内部离子的内容为
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
	
	
	在该项目中
3,使用pep2peaks
	本实验中已经训练好的模型存放于models/目录下面。\n
	对于单条肽序列的预测，本实验提供了示例接口位于example.py。对于多条肽的预测，需要将肽序列写入文件中，文件内容格式如上所示，预测脚本位于model/pep2peaks.py，并将参数is_train设置为2，表示仅测试，同时需要设置参数model，表示已经训练好的模型的位置。如果需要重新训练，则将参数is_train设置为1。
	另外，无论是训练还是预测，对于不同的离子类型，需要设置不同的参数。对于常规离子的训练或预测，需要设置：
		ion_type=regular
		input_dim=188
		output_dim=4
	而对于内部离子的预测，需要设置		
		ion_type=internal
		input_dim=217
		output_dim=2
		min_internal_ion_len=？
		max_internal_ion_len=？
	其中ion_type代表离子类型，input_dim和output_dim分别代表数据的输入维度和输出维度，在本项目中，常规离子的输入维度为188，内部离子的输入维度为217，详情请参考文件tools/get_data.py。常规离子的输出维度为4，分别代表b+/b++/y+/y++，内部离子的输出维度为2，分别代表by+/ay+。min_internal_ion_len和max_internal_ion_len代表内部离子的训练或预测实验中需要训练或预测的内部离子的最小碎片长度和最大碎片长度（两侧临界值均为闭区间）。
	
	
	
