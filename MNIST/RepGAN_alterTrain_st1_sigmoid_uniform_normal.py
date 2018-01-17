
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as distributions
import numpy as np
import os
import scipy as sp
import scipy.ndimage

### Define modules
def lrelu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)
def encoder(x,dim_categorical,dim_continuous, dim_noise, BatchNormDecay, flag_training=True):
    with slim.arg_scope([slim.fully_connected, slim.conv2d],
                        activation_fn = None,
                        normalizer_fn=slim.batch_norm, 
                        normalizer_params={'decay': BatchNormDecay, 'is_training': flag_training}):
        net = lrelu(slim.conv2d(x, num_outputs=64, kernel_size=4, stride=2))  # AAE paper use 20% dropout on this layer
        net = lrelu(slim.conv2d(net, num_outputs=128, kernel_size=4, stride=2))
        net = slim.flatten(net)
        net = lrelu(slim.fully_connected(net, 1024))
        
        latent_cat = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
        net_cont      = lrelu(slim.fully_connected(net, 2*dim_continuous))   # AAE paper use BN ONLY on all layers of encoder.
        mu    = net_cont[:, :dim_continuous]
        sigma = tf.exp(net_cont[:, dim_continuous:])
        noise = lrelu(slim.fully_connected(net, dim_noise))
    return latent_cat, mu, sigma, noise

def decoder(latent, BatchNormDecay, flag_training=True):
    with slim.arg_scope([slim.fully_connected, slim.conv2d],
                        activation_fn = tf.nn.relu,
                        normalizer_fn=slim.batch_norm,  # BN crucial for infoGAN here, otherwise no work
                        normalizer_params={'decay': BatchNormDecay, 'is_training': flag_training}
                       ):
        net = slim.fully_connected(latent, 1024)
        net = slim.fully_connected(net,7*7*128)
        net = tf.reshape(net, shape=[-1,7,7,128])
        net = slim.conv2d_transpose(net, 64, kernel_size = 4, stride = 2)
        xhat = slim.conv2d_transpose(net,  1, kernel_size = 4, stride = 2, activation_fn = tf.nn.sigmoid, normalizer_fn=None)
    return xhat

def Discriminator(latent_in, dim_DS_hiddn1, dim_DS_hiddn2, BatchNormDecay, flag_raw=False):
# For latent variables
    with slim.arg_scope([slim.fully_connected],
                                     activation_fn = None,
                                     #normalizer_fn=slim.batch_norm,
                                     normalizer_params={'decay': BatchNormDecay}):
        net = lrelu(slim.fully_connected(latent_in, dim_DS_hiddn1))
        net = lrelu(slim.fully_connected(net, dim_DS_hiddn2))
        if flag_raw:
            net = slim.fully_connected(net,1, activation_fn = None)
        else:
            net = slim.fully_connected(net,1, activation_fn = tf.nn.sigmoid)
    return net

def D2d(xhat, BatchNormDecay, flag_training=True):
# For decoder output
    with slim.arg_scope([slim.fully_connected, slim.conv2d],
                        activation_fn = None,
                        normalizer_fn=slim.batch_norm, 
                        normalizer_params={'decay': BatchNormDecay, 'is_training':flag_training}):
        net = lrelu(slim.conv2d(xhat, num_outputs=64, kernel_size=4, stride=2, normalizer_fn=None))
        net = lrelu(slim.conv2d(net, num_outputs=128, kernel_size=4, stride=2))
        net = slim.flatten(net)
        net = lrelu(slim.fully_connected(net, 1024))
        prob_realFake = slim.fully_connected(net,1, activation_fn = tf.nn.sigmoid, normalizer_fn=None)
    return prob_realFake

def Sample_categorical(size_batch, size_categorical):
    real_cat = np.zeros((size_batch, size_categorical))
    rand_idx = np.random.randint(0, size_categorical, size_batch)
    real_cat[np.arange(size_batch), rand_idx]=1.0
    return real_cat
def Sample_continuous(size_batch, size_continuous):
    real_cont = np.random.uniform(low=-1.0, high=1.0000001, size=(size_batch, size_continuous))
    return real_cont
def Sample_noise(size_batch, size_noise):
    noise = np.random.normal(loc=0.0, scale=1.0, size=(size_batch, size_noise))
    return noise

### Define RepGAN class ###
class RepGAN(object):
    def __init__(self, params):
    ### AAE ###
        with tf.variable_scope('encoder'):
            self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 28,28,1))
            self.latent_cat,\
            self.latent_cont_mu,\
            self.latent_cont_sigma,\
            self.latent_noise             = encoder(self.x,
                                             params.dim_cat,
                                             params.dim_cont,
                                             params.dim_noise,
                                             params.BatchNormDecay)
        with tf.variable_scope('decoder'):
            self.latent_cont_sample = self.latent_cont_mu + tf.multiply(self.latent_cont_sigma, tf.random_normal([1]))
            self.decoder_output = decoder(tf.concat([self.latent_cat, self.latent_cont_sample, self.latent_noise ],1),
                                          params.BatchNormDecay)
            self.xhat_flat_distribution = distributions.Bernoulli(probs = tf.clip_by_value(self.decoder_output, 1e-5, 1-1e-5))
            self.expected_log_likelihood = tf.reduce_sum(self.xhat_flat_distribution.log_prob(self.x),[1,2,3])
        with tf.variable_scope('decoder', reuse=True):
            self.sample_cat = tf.placeholder(tf.float32, shape=(params.Nsample, params.dim_cat))
            self.sample_cont= tf.placeholder(tf.float32, shape=(params.Nsample, params.dim_cont))
            self.sample_noise = tf.placeholder(tf.float32, shape=(params.Nsample, params.dim_noise))
            self.sample_output = decoder(tf.concat([self.sample_cat, self.sample_cont, self.sample_noise],1),  
                                         params.BatchNormDecay, 
                                         flag_training = False)
        self.real_cat = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_cat))
        self.real_cont = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_cont))
        self.real_noise = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_noise))
        with tf.variable_scope('Discri_cat'):
            self.real_cat_prob = Discriminator(self.real_cat,
                                               params.dim_DS_cat_hiddn1, params.dim_DS_cat_hiddn2,
                                               params.BatchNormDecay,
                                               flag_raw=True)
        with tf.variable_scope('Discri_cat', reuse=True):
            self.fake_cat_prob = Discriminator(self.latent_cat,
                                               params.dim_DS_cat_hiddn1, params.dim_DS_cat_hiddn2,
                                               params.BatchNormDecay,
                                               flag_raw=True)
        with tf.variable_scope('Discri_cont'):
            self.real_cont_prob = Discriminator(self.real_cont,
                                                params.dim_DS_cont_hiddn1, params.dim_DS_cont_hiddn2,
                                                params.BatchNormDecay,
                                                flag_raw=True)
        with tf.variable_scope('Discri_cont', reuse=True):
            self.fake_cont_prob = Discriminator(self.latent_cont_sample,
                                                params.dim_DS_cont_hiddn1, params.dim_DS_cont_hiddn2,
                                                params.BatchNormDecay,
                                                flag_raw=True)
        with tf.variable_scope('Discri_noise'):
            self.real_noise_prob = Discriminator(self.real_noise,
                                                params.dim_DS_cont_hiddn1, params.dim_DS_cont_hiddn2,
                                                params.BatchNormDecay,
                                                flag_raw=True)
        with tf.variable_scope('Discri_noise', reuse=True):
            self.fake_noise_prob = Discriminator(self.latent_noise,
                                                params.dim_DS_cont_hiddn1, params.dim_DS_cont_hiddn2,
                                                params.BatchNormDecay,
                                                flag_raw=True)
    ### infoGAN ###
        # The generator/discriminator training part of infoGAN
        self.infoGAN_cat = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_cat))
        self.infoGAN_cont = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_cont))
        self.infoGAN_noise = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_noise))
        self.real_data = tf.placeholder(tf.float32, shape=(params.batch_size, 28,28,1))
        with tf.variable_scope('decoder', reuse=True):
            self.decoderGene_output = decoder(tf.concat([self.infoGAN_cat, self.infoGAN_cont, self.infoGAN_noise],1),
                                                         params.BatchNormDecay)
        # Use same encoder (variable share) to model Q()
        with tf.variable_scope('encoder', reuse=True): 
            self.Q_cat,\
            self.Q_cont_mu,\
            self.Q_cont_sigma,_= encoder(self.decoderGene_output, 
                                        params.dim_cat, 
                                        params.dim_cont, 
                                        params.dim_noise,
                                        params.BatchNormDecay)
            self.Q_cont_distribution = distributions.MultivariateNormalDiag(loc=self.Q_cont_mu, scale_diag=self.Q_cont_sigma)
        with tf.variable_scope('D2d'):
            self.decoderGene_data_prob = D2d(self.decoderGene_output, params.BatchNormDecay)
        with tf.variable_scope('D2d', reuse=True):
            self.real_data_prob = D2d(self.real_data, params.BatchNormDecay)
        
# Loss functions
    # 1. reconstruction loss
        self.loss_recon = -tf.reduce_mean(self.expected_log_likelihood)
    # 2. mutual information loss
        self.loss_Q_cat = -tf.reduce_mean(tf.reduce_sum(tf.log(self.Q_cat+1e-8)*self.infoGAN_cat,1))
        self.loss_Q_cont = -tf.reduce_mean(self.Q_cont_distribution.log_prob(self.infoGAN_cont))
        self.loss_Q = self.loss_Q_cat + self.loss_Q_cont
    # 3. adversarial loss
        self.loss_Discri_cat  = -tf.reduce_mean(self.real_cat_prob - self.fake_cat_prob)
        self.loss_Gene_cat    = -tf.reduce_mean(self.fake_cat_prob)
        self.loss_Discri_cont = -tf.reduce_mean(self.real_cont_prob - self.fake_cont_prob)
        self.loss_Gene_cont   = -tf.reduce_mean(self.fake_cont_prob)
        self.loss_Discri_noise = -tf.reduce_mean(self.real_noise_prob - self.fake_noise_prob)
        self.loss_Gene_noise   = -tf.reduce_mean(self.fake_noise_prob)
        self.loss_Discri      = self.loss_Discri_cat + self.loss_Discri_cont + self.loss_Discri_noise
        self.loss_Gene        = self.loss_Gene_cat + self.loss_Gene_cont + self.loss_Gene_noise
        
        self.loss_Discri_decoder = -tf.reduce_mean(tf.log(self.real_data_prob+1e-8) + tf.log(1 - self.decoderGene_data_prob+1e-8))
        self.loss_Gene_decoder = -tf.reduce_mean(tf.log(self.decoderGene_data_prob+1e-8))
# Variable list
        self.varlist_E = [v for v in tf.trainable_variables() if v.name.startswith('encoder/')]
        self.varlist_D = [v for v in tf.trainable_variables() if v.name.startswith('decoder/')]
        self.varlist_Q = [v for v in tf.trainable_variables() if v.name.startswith('encoder/')]
        self.varlist_Discri_cat = [v for v in tf.trainable_variables() if v.name.startswith('Discri_cat/')]
        self.varlist_Discri_cont = [v for v in tf.trainable_variables() if v.name.startswith('Discri_cont/')]
        self.varlist_Discri_noise = [v for v in tf.trainable_variables() if v.name.startswith('Discri_noise/')]
        self.varlist_Discri_decoder = [v for v in tf.trainable_variables() if v.name.startswith('D2d/')]
        
        self.varlist_ED = self.varlist_E + self.varlist_D
        self.varlist_Discri = self.varlist_Discri_cat + self.varlist_Discri_cont + self.varlist_Discri_noise
# Optimizer - with dependency on 'update_ops' for tf.summary
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
        ### AAE ###
            self.optimizer_ED = slim.learning.create_train_op(self.loss_recon,
                                                              params.optimizer_recon,
                                                              variables_to_train=self.varlist_ED)
            self.optimizer_Discri = slim.learning.create_train_op(self.loss_Discri,
                                                                  params.optimizer_adv_discriminator,
                                                                  variables_to_train=self.varlist_Discri)
            self.optimizer_Gene = slim.learning.create_train_op(self.loss_Gene,
                                                                params.optimizer_adv_generator,
                                                                variables_to_train=self.varlist_E)
        ### infoGAN ###
            self.optimizer_Q_infoGAN = slim.learning.create_train_op(self.loss_Q,
                                                             params.optimizer_Q,
                                                             variables_to_train=self.varlist_Q+self.varlist_D)  
            self.optimizer_Discri_infoGAN = slim.learning.create_train_op(self.loss_Discri_decoder,
                                                                          params.optimizer_adv_discriminator_infoGAN,
                                                                          variables_to_train=self.varlist_Discri_decoder)
            self.optimizer_Gene_infoGAN = slim.learning.create_train_op(self.loss_Gene_decoder,
                                                                        params.optimizer_adv_generator_infoGAN,
                                                                        variables_to_train=self.varlist_D)
# Clip the Discriminator parameters - WGAN
        self.clip_discriminator=[]
        for currVars in self.varlist_Discri:
            self.clip_discriminator.append(tf.assign(currVars, tf.clip_by_value(currVars, -params.c, params.c)))

### Define trainning process ###
import matplotlib.pyplot as plt
def train(model, data, params, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(log_path+'reconstruction/'):
        os.makedirs(log_path+'reconstruction/')
    if not os.path.exists(log_path+'generation/'):
        os.makedirs(log_path+'generation/')
    with open(log_path+'log.txt', "a") as myfile:
        currHeader = 'Model = RepGAN'+'\n'+\
                    'dim_cat = '+'{:d}'.format(params.dim_cat)+'\n'+\
                    'dim_cont = '+'{:d}'.format(params.dim_cont)+'\n'+\
                    'dim_noise = '+'{:d}'.format(params.dim_noise)+'\n'+\
                    'noise_tunneling = True'+'\n'+\
                    'WGAN_on_decoder = False'+'\n'+\
                    'dim_Discriminator_cat = '+'{:d}'.format(params.dim_DS_cat_hiddn1)+'/'+\
                    '{:d}'.format(params.dim_DS_cat_hiddn2)+'\n'+\
                    'dim_Discriminator_cont = '+'{:d}'.format(params.dim_DS_cont_hiddn1)+'/'+\
                    '{:d}'.format(params.dim_DS_cont_hiddn2)+'\n'+\
                    'LR_recon = '+'{:.0e}'.format(params.LR_recon)+'\n'+\
                    'LR_adv   = '+'{:.0e}'.format(params.LR_adv)+'\n'+\
                    'LR_adv_infoGAN = '+'{:.0e}'.format(params.LR_adv_infoGAN)+'\n'+\
                    'BatchNormDecay = '+'{:.2f}'.format(params.BatchNormDecay)+'\n'+\
                    'num_critis = '+'{:d}'.format(params.num_critis)+'\n'+\
                    'clip_c = '+'{:.3f}'.format(params.c)+'\n'+\
                    '######  End of Header \####################################################\n'
        myfile.write(currHeader)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        eye_cat = np.zeros((params.Nsample, params.dim_cat))
        for idx in range(params.Nsample):
            eye_cat[idx,idx]=1.0
        zero_cont = np.zeros((params.Nsample, params.dim_cont))
    # alterTrain
        for step in range(params.num_steps):   
            currBatch,currLabel = data.train.next_batch(params.batch_size)
            currBatch = currBatch.reshape([-1, 28, 28, 1])
            currBatch_perm = np.random.permutation(currBatch)
            real_cat = Sample_categorical(params.batch_size, params.dim_cat)
            real_cont = Sample_continuous(params.batch_size, params.dim_cont)
            real_noise = Sample_noise(params.batch_size, params.dim_noise)
            infoGAN_cat = Sample_categorical(params.batch_size, params.dim_cat)
            infoGAN_cont = Sample_continuous(params.batch_size, params.dim_cont)
            infoGAN_noise = Sample_noise(params.batch_size, params.dim_noise)
            test_noise = Sample_noise(params.Nsample, params.dim_noise)
            feedDict = {model.x:           currBatch,
                        model.infoGAN_cat: infoGAN_cat,
                        model.infoGAN_cont: infoGAN_cont,
                        model.infoGAN_noise: infoGAN_noise,
                        model.real_cat:    real_cat,
                        model.real_cont:   real_cont,
                        model.real_noise:  real_noise,
                        model.real_data:   currBatch_perm,
                       }
        # 1. infoGAN - emphasized
            for _ in range(params.num_critis):
                sess.run(model.optimizer_Discri_infoGAN, feedDict)
                sess.run(model.optimizer_Gene_infoGAN,   feedDict)
                sess.run(model.optimizer_Q_infoGAN,      feedDict)
        # 2. AAE
            sess.run(model.optimizer_ED, feedDict)
            for _ in range(params.num_critis):
                sess.run(model.optimizer_Discri,     feedDict)
                sess.run(model.clip_discriminator,   feedDict)
            sess.run(model.optimizer_Gene,           feedDict)
        # Save model      
            if step%params.saveStep ==0:
                saver.save(sess, log_path+'ckpt/model.ckpt')
            # Generation image
            zero_noise = np.zeros((params.Nsample, params.dim_noise))
            if step%params.showStep ==0:     
                Ntrial=10
                decoderSample_small = np.ones((Ntrial, params.Nsample, 28,28,1), dtype=float) 
                for idx in range(Ntrial):
                    if idx == 0:
                        feedDict_rand = {
                                         model.sample_cat:   eye_cat,
                                         model.sample_cont:  zero_cont,
                                         model.sample_noise: zero_noise,
                                        }
                        decoderSample_small[idx,:,:,:] = sess.run(model.sample_output,  feedDict_rand) 
                    else:
                        sample_cont_rand = Sample_continuous(1, params.dim_cont)
                        sample_cont_rand = np.tile(sample_cont_rand,(params.Nsample,1))
                        test_noise = Sample_noise(params.Nsample, params.dim_noise)
                        feedDict_rand = {
                                         model.sample_cat:   eye_cat,
                                         model.sample_cont:  sample_cont_rand,
                                         model.sample_noise: test_noise,
                                        }
                        decoderSample_small[idx,:,:,:] = sess.run(model.sample_output,  feedDict_rand)

                d = np.empty((Ntrial,params.Nsample,28*3,28*3), dtype=float) # reconstructed from ancestrol sampling
                for idx_t in range(Ntrial):   # for each trial
                    for idx_cat in range(params.Nsample):   # for each category
                        temp_d             = decoderSample_small[idx_t,idx_cat].reshape([28,28])
                        d[idx_t,idx_cat] = sp.ndimage.interpolation.zoom(temp_d,3)
                outplot = np.swapaxes(d,1,2).reshape(Ntrial*28*3, params.Nsample*28*3)
                plt.imsave(log_path+'generation/'+str(step)+'.png',outplot)
            # Reconstruction image
                decoder_output = sess.run(model.decoder_output, feedDict)
                sampleIdx=3

                g = np.empty((8,28*3,28*3), dtype=float)  # reconstructed from encoder output
                for idx in range(8):
                    if idx <4:
                        temp_g = currBatch[idx].reshape([28,28])
                    else:
                        temp_g = decoder_output[idx-4].reshape([28,28])
                    g[idx] = sp.ndimage.interpolation.zoom(temp_g,3)
                outplot1=np.concatenate((g[0],g[1],g[2],g[3]),axis=1)
                outplot2=np.concatenate((g[4],g[5],g[6],g[7]),axis=1)
                outplot =np.concatenate((outplot1,outplot2),axis=0)
                plt.imsave(log_path+'reconstruction/'+str(step)+'.png',outplot)

class trainParameter():
    def __init__(self, LR_recon, LR_adv, LR_adv_infoGAN):
        self.batch_size = 128
        self.dim_x = 784
        self.dim_cat=10
        self.dim_cont = 2
        self.dim_noise = 20
    # Hidden dimension
        self.dim_DS_cat_hiddn1=3000   # Discriminator - categorical
        self.dim_DS_cat_hiddn2=3000
        self.dim_DS_cont_hiddn1=3000   # Discriminator - continuous
        self.dim_DS_cont_hiddn2=3000
    # Other
        self.num_steps =80000
        self.showStep=500
        self.saveStep=2000
        self.BatchNormDecay=0.95
        self.num_critis=5
        self.c=0.01
    # Optimizer
        self.LR_recon = LR_recon
        self.LR_adv   = LR_adv
        self.LR_adv_infoGAN = LR_adv_infoGAN
        
        self.optimizer_recon = tf.train.RMSPropOptimizer(LR_recon)
        self.optimizer_adv_discriminator = tf.train.RMSPropOptimizer(LR_adv)
        self.optimizer_adv_generator = tf.train.RMSPropOptimizer(LR_recon)
        
        self.optimizer_Q = tf.train.AdamOptimizer(LR_adv_infoGAN)
        self.optimizer_adv_generator_infoGAN     = tf.train.AdamOptimizer(LR_recon)
        self.optimizer_adv_discriminator_infoGAN = tf.train.AdamOptimizer(LR_adv_infoGAN)
    # Ancestrol sampling (test the reconstruction ability of decoder)
        self.Nsample=10 # number of batch in model.sample_output


# Load data & Define parameters
def main(log_path, LR_recon, LR_adv, LR_adv_infoGAN):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    params = trainParameter(LR_recon, LR_adv, LR_adv_infoGAN)
    # Traininng
    tf.reset_default_graph()
    RepGAN_1 = RepGAN(params)
    train(RepGAN_1, mnist, params, log_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LR_adv_infoGAN_idx', required=False) #given by SLURM_ARRAY_TASK_ID
    parser.add_argument('--LR_recon', required=False)
    parser.add_argument('--LR_adv',   required=False)
    parser.add_argument('--LR_adv_infoGAN', required=False)
    args = parser.parse_args()

    main_log_path = './LOG_RepGAN_alterTrain_st1_sigmoid_uniform_normal_reTest'
    if args.LR_recon != None and args.LR_adv != None and args.LR_adv_infoGAN != None and args.LR_adv_infoGAN_idx ==None:
        currLR_recon = float(args.LR_recon)
        currLR_adv   = float(args.LR_adv)
        currLR_adv_infoGAN = float(args.LR_adv_infoGAN)
        curr_log_path = main_log_path+'_singleTest/'+\
                            'LOG_RepGAN_'+\
                            '{:.0e}'.format(currLR_recon)+'_'+\
                            '{:.0e}'.format(currLR_adv)+'_'+\
                            '{:.0e}'.format(currLR_adv_infoGAN)+'/' 
        main(curr_log_path, currLR_recon, currLR_adv, currLR_adv_infoGAN)
    elif args.LR_recon == None and args.LR_adv == None and args.LR_adv_infoGAN == None and args.LR_adv_infoGAN_idx != None:
        LR_recon_pool = [1e-3, 5e-4, 2e-4]
        LR_adv_pool   = [5e-3, 1e-3, 1e-4]
        LR_adv_infoGAN_pool = [1e-3, 5e-4, 1e-4, 1e-5]
        
        currLR_adv_infoGAN = LR_adv_infoGAN_pool[int(args.LR_adv_infoGAN_idx)]
        for currLR_recon in LR_recon_pool:
            for currLR_adv in LR_adv_pool:
                try:
                    curr_log_path = main_log_path+'_LRTuning/'+\
                                    'LOG_RepGAN_'+'{:.0e}'.format(currLR_recon)+'_'+\
                                    '{:.0e}'.format(currLR_adv)+'_'+\
                                    '{:.0e}'.format(currLR_adv_infoGAN)+'/' 
                    main(curr_log_path, currLR_recon, currLR_adv, currLR_adv_infoGAN)
                except:
                    with open(main_log_path+'_LRTuning/crashed.txt','a') as myfile:
                        currLog = 'LR_recon = '+'{:.0e}'.format(currLR_recon)+'\n'+\
                                        'LR_adv   = '+'{:.0e}'.format(currLR_adv)+'\n'+\
                                        'LR_adv_infoGAN = '+'{:.0e}'.format(currLR_adv_infoGAN)+'\n'+\
                                        '#########################################################\n'
                        myfile.write(currLog)
                    continue
    else:
        print('Specify "--LR_adv_infoGAN_idx" or all of "--LR_recon  --LR_adv  --LR_adv_infoGAN"')

