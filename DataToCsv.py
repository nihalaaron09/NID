
fname='/cfarhomes/nihal09/Documents/KDDlabels/KDD_Train_Full.txt'


# obtained specific features and labels 
wanted=[41,0,1,4,5,22,23]
with open(fname) as fin:
	lines=fin.readlines()

newd = []

for x in lines:
	sp=x.split(',')
	newd.append([sp[i] for i in wanted])   

# replaced all protocol names 
for x in range(len(newd)):
	newd[x]=[w.replace('tcp','0').replace('udp','100').replace('icmp','200') for w in newd[x]]
	
#replace labels
for x in range(len(newd)):
	if newd[x][0]!='normal':
		newd[x][0]='1'
	else:
		newd[x][0]='0'
		 	


file='Py_Extracted_train_full.txt';
fout=open(file,'w');
fout.write('LABEL,FEAT_1,FEAT_2,FEAT_3,FEAT_4,FEAT_5,FEAT_6')

#for idx,d in enumerate(newd):
	
for d in newd:
	fout.write('\n' + ','.join(d))


fout.close()


