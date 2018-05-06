# RepGAN
The python scripts to train the NN model on each dataset can be found in their folders.  
**For MNIST, to train the model, call:**
   - python RepGAN_alterTrain_st1_sigmoid_uniform_normal.py --LR_recon=5e-4 --LR_adv=1e-3 --LR_adv_infoGAN=2e-4  

The output will be store in a directory specified by "main_log_path", under the current directory.  
Also, the MNIST data needs to be placed in ./MNIST/MNIST_data (namely, in a directory MNIST_data under the current directory)  

**For fashionMNIST, call:**
   - python RepGAN_4.py
   
The learning rates are hard coded in the script file, by variables "currLR_recon", "currLR_adv", "currLR_adv_infoGAN" at the end of the file, which can be changed if desired.
Also, fashionMNIST data needs to be placed in ./fashionMNIST/fashionMNIST_data

**For celebA, call:**
   - RepGAN_semiSup_RGB_run8e_pipeline_run4.py
   
The learning rates are also hard coded, as explained above.
The trainng data is located in ./celebA/celebA_data/, as tfRecord file.

**For svhn, call:**
   - RepGAN_semiSup_RGB_run8e.py
   
The learning rates are also hard coded, as explained above.
The trainng data is located in ./svhn/svhn_data/, as .mat file.
