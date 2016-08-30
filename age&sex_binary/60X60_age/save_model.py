import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip
import pdb

import struct as sr


if __name__=="__main__":
    f=np.load('60X60_parameters_small_net_age_no_flip.npz')
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #import pdb
    #pdb.set_trace()
    fw=open("60X60_small_model.bin",'wb')
    ff=open("60X60_small_model.txt",'w')
    for i in xrange(len(param_values)):
        data=param_values[i]
        shape=data.shape
	#pdb.set_trace()
        for j in xrange(len(shape)):
	    ff.write(str(shape[j])+' ')
        #ff.write(sr.pack('i',(shape[-1])))
	ff.write('\n')
        data=data.flatten()
        #pdb.set_trace()
        for nNum in xrange(len(data)):
	    fw.write(sr.pack('f',data[nNum]))
        #fw.write(sr.pack('f',data[-1]))
	#if i!=(len(param_values)-1):
	    #ff.write('\n')
	    #fw.write(' ')
    ff.close()
    fw.close()
    #fi=open("b.txt",'w')
    #data=param_values[37]
    #for i in xrange(200):
#	fi.write(str(data[i])+' ')
   # fi.write('\n')
   # fi.close()
