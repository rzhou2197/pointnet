import numpy as np
import os
Dir='n/'
filenames=os.listdir(Dir)
for file in filenames:
	infile=open(Dir+file)
	record=[]
	for line in infile:
		line=line.strip().split()
		record.append([float(line[0]),float(line[1]),float(line[2])])
	record=np.array(record)
	portion=os.path.splitext(file)
	np.save(Dir+portion[0]+'.npy',record)

