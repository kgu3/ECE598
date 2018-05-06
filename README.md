# RepGAN
The python scripts to train the NN model on each dataset can be found in their folders.
Take MNIST as example, to train the model, call:
   python RepGAN_alterTrain_st1_sigmoid_uniform_normal.py --LR_recon=5e-4 --LR_adv=1e-3 --LR_adv_infoGAN=2e-4
The output will be store in a directory specified by "main_log_path", under the current directory.
Also, the MNIST data needs to be located in ./MNIST_data (namely, in a directory MNIST_data under the current directory)
