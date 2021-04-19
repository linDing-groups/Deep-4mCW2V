import pandas as pd
import numpy as np
import os
import sys
import math
import random

def npyToTXT(npyfile, svmfile, pos_num):
    dataDataVecs = np.load(npyfile)
    g = open(svmfile,'w')
    print(len(dataDataVecs))
    #print(dataDataVecs[0])

    m = 0
    for i in range(len(dataDataVecs)):
        line = ''
        for j in range(len(dataDataVecs[0])):
            if j == len(dataDataVecs[0])-1:
                line += str(j+1)+':'+str(dataDataVecs[i][j])+'\n'
            else:
                line += str(j+1)+':'+str(dataDataVecs[i][j])+'\t'
        m += 1
        if m < (pos_num+1):
            g.write('1\t'+line)
        else:
            g.write('0\t'+line)
            
def TXTtoCSV(svmfile, csvfile):
	f = open(svmfile,'r')
	g = open(csvfile,'w')
	lines = f.readlines()
	legth = len(lines[0].split('	'))-1
	#print(legth)
	classline = 'class'
	for i in range(legth):
		classline += ',%d'%(i+1)
	g.write(classline+'\n')

	for line in lines:
		line = line.strip('\n').split('	')
		g.write(line[0]+',')

		legth2 = len(line[1:])
		m = 0
		for j in line[1:]:
			if m == legth2-1:
				j = j.split(':')[-1]
				g.write(j)
				m += 1
			else:
				j = j.split(':')[-1]
				g.write(j+',')
				m += 1
		g.write('\n')

	f.close()
	g.close()
    
npyToTXT("222_vecs.npy", '222_vecs.txt',222)#positive number
TXTtoCSV('222_vecs.txt', '222_vecs.csv')
