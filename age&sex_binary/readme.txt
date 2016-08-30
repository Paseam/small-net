This is a sdk for age recognition utilizing binary-net. To use thsi sdk,several things 
must be done 
1.run 60X60_samll_net.py in folder 60X60_age, then run the save_model.py and you will 
  get two files 60X60_model.bin and 60X60_model.txt respectively, and the first one 
  is the file for saving the net coefficients, second one is the file to describe the model.
2.copy 60X60_model.bin and 60X60_model.txt to the folder afg_caffe_cnn\afg_caffe_cnn\60X60_model which is 
  also the folder for saving test and train folders.
3.now you can run the sdk