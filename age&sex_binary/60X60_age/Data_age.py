import PIL.Image as Image
import numpy as np
import pdb
import cPickle

import copy
def shuffle(X,y):
        
        # print(len(X))
        
        chunk_size = len(X)
        shuffled_range = range(chunk_size)
        
        X_buffer = copy.deepcopy(X)
        y_buffer = copy.deepcopy(y)
        np.random.shuffle(shuffled_range)
        
        for i in range(chunk_size):
                
            X_buffer[i] = X[shuffled_range[i]]
            y_buffer[i] = y[shuffled_range[i]]
            
        X = X_buffer
        y = y_buffer
            

        
        return X,y



def get_data():
    

    path_test='../afg_caffe_cnn/afg_caffe_cnn/60X60_model/test/'
    len_test=8198
    fi=open("file_test_relative_name.txt",'rb')
    data_test=[]#np.zeros([len_test,3,60,60])
    label_test=[]#np.zeros(len_test)
    for i in xrange(len_test):
        Line=fi.readline()
        str_ls1=Line.split(' ')
  
        name=str_ls1[0]
        path_name=path_test+name
        label=str_ls1[1].split('\r')[0]

        data_test.append(path_name)
        label_test.append(int(label))
    
    fi.close()



    len_train=65452
    path_train='../afg_caffe_cnn/afg_caffe_cnn/60X60_model/train/'
    fi=open("file_train_relative_name.txt",'rb')
    data_train=[]#np.zeros([len_train,3,60,60])
    label_train=[]#np.zeros(len_train)
    for i in xrange(len_train):
        Line=fi.readline()
        str_ls1=Line.split(' ')
        #pdb.set_trace()
        name=str_ls1[0]
        path_name=path_train+name
        label=str_ls1[1].split('\r')[0]
	    data_train.append(path_name)
        label_train.append(int(label))
        
    
    fi.close()
    path_test,path_label_test=shuffle(data_test,label_test)
    path_train,path_label_train=shuffle(data_train,label_train)
    #pdb.set_trace()

    len_test=8198
    data_test=np.zeros([len_test,3,60,60])
    label_test=np.zeros(len_test)
    for i in xrange(len_test):
        #print i
        path_name=path_test[i]
        im=Image.open(path_name)
        im=np.asarray(im,dtype='float32')
        data_test[i,0]=im[:,:,0]
        data_test[i,1]=im[:,:,1]
        data_test[i,2]=im[:,:,2]
        label_test[i]=path_label_test[i]
    #pdb.set_trace()
    ret=[] 
    ret.append((data_test,label_test))


    len_train=45000
    data_train=np.zeros([len_train,3,60,60])
    label_train=np.zeros(len_train)
    for i in xrange(len_train):
        path_name=path_train[i]
        im=Image.open(path_name)
        im=np.asarray(im,dtype='float32')
        data_train[i,0]=im[:,:,0]
        data_train[i,1]=im[:,:,1]
        data_train[i,2]=im[:,:,2]
        label_train[i]=path_label_train[i]
    
    ret.append((data_train,label_train)) 
   

    test_set_X = ret[0][0]
    test_set_y = ret[0][1]
    train_set_X =ret[1][0][0:40000,:,:,:]
    train_set_y = ret[1][1][0:40000]
    valid_set_X = ret[1][0][40000:45000,:,:,:]
    valid_set_y = ret[1][1][40000:45000]

    return [test_set_X,test_set_y,train_set_X,train_set_y,valid_set_X,valid_set_y]



if __name__=="__main__":
    ret=get_data()
    fi=open("data_age_test.pkl",'wb')
    test=[ret[0],ret[1]]
    cPickle.dump(test,fi);
    fi.close() 


    #fi=open("data_age_train.pkl",'wb')
    #test=[ret[2],ret[3]]
    #cPickle.dump(test,fi);
    #fi.close() 

    #fi=open("data_age_valid.pkl",'wb')
    #test=[ret[4],ret[5]]
    #cPickle.dump(test,fi);
    #fi.close() 
