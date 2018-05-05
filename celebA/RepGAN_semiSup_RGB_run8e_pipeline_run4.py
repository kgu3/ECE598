import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as distributions
import numpy as np
import os
import scipy as sp
import scipy.ndimage
import scipy.io as sio
import sklearn.preprocessing as skp
from utils import *

### Define modules
def variable_summaries(var):
    scopeName = 'summary/'+var.name.replace(':','/')
    with tf.name_scope(scopeName):
        tf.summary.histogram('histogram', var)
def lrelu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)
def encoder(x,N_cat, dim_categorical, dim_noise, BatchNormDecay, flag_training=True):
    with slim.arg_scope([slim.fully_connected, slim.conv2d],
                        activation_fn = None,
                        normalizer_fn=slim.batch_norm, 
                        normalizer_params={'decay': BatchNormDecay, 'is_training': flag_training}):
        net = lrelu(slim.conv2d(x, num_outputs=64, kernel_size=4, stride=2))
        net = slim.dropout(net, keep_prob = 0.8, is_training = flag_training)
        net = lrelu(slim.conv2d(net, num_outputs=128, kernel_size=4, stride=2))
        net = lrelu(slim.conv2d(net, num_outputs=256, kernel_size=4, stride=2))
        net = slim.flatten(net)
        net = lrelu(slim.fully_connected(net, 1024))
        
        with tf.variable_scope('FC_cat'):
            _cat0 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            _cat1 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            _cat2 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            _cat3 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            _cat4 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            _cat5 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            _cat6 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            _cat7 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            _cat8 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            _cat9 = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
            latent_cat = tf.concat([_cat0,_cat1,_cat2,_cat3,_cat4,_cat5,
                                    _cat6,_cat7,_cat8,_cat9],axis=1)      
        with tf.variable_scope('FC_noise'):
            noise = slim.fully_connected(net, dim_noise, activation_fn = None)
    return latent_cat, noise

def decoder(latent, BatchNormDecay, flag_training=True):
    with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose],
                        activation_fn = tf.nn.relu,
                        normalizer_fn=slim.batch_norm,  # BN crucial for infoGAN here, otherwise no work
                        normalizer_params={'decay': BatchNormDecay, 'is_training': flag_training}
                       ):
        net = slim.fully_connected(latent, 1024)
        net = slim.fully_connected(net,8*8*256)
        net = tf.reshape(net, shape=[-1,8,8,256])
        net = slim.conv2d_transpose(net, 128, kernel_size = 4, stride = 2)
        net = slim.conv2d_transpose(net, 64, kernel_size = 4, stride = 2)
        xhat = slim.conv2d_transpose(net,  3, kernel_size = 4, stride = 2, activation_fn = None, normalizer_fn=None)
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

def Sample_categorical(size_batch, N_cat, size_categorical):
    real_cat = np.zeros((size_batch*N_cat, size_categorical))
    rand_idx = np.random.randint(0, size_categorical, N_cat*size_batch)
    real_cat[np.arange(size_batch*N_cat), rand_idx]=1.0
    return real_cat     # real_cat.shape = [N_cat*size_batch, size_categorical]

def Sample_noise(size_batch, size_noise):
    noise = np.random.normal(loc=0.0, scale=0.5, size=(size_batch, size_noise))
    return noise

### Define RepGAN class ###
class RepGAN(object):
    def __init__(self, params):
    ### AAE ###
        with tf.variable_scope('encoder'):
            self.x = tf.placeholder(tf.float32, shape=(params.batch_size, params.img_W,params.img_H,3))
            self.latent_cat,\
            self.latent_noise      = encoder(self.x,
                                             params.N_cat, 
                                             params.dim_cat,
                                             params.dim_noise,
                                             params.BatchNormDecay)
            tf.summary.histogram('latent_cat', self.latent_cat)
            tf.summary.histogram('latent_noise', self.latent_noise)

        with tf.variable_scope('decoder'):
            self.decoder_output = decoder(tf.concat([self.latent_cat, self.latent_noise ],1),
                                          params.BatchNormDecay)
            #self.xhat_flat_distribution = distributions.Bernoulli(probs = tf.clip_by_value(self.decoder_output, 1e-5, 1-1e-5))
            #self.expected_log_likelihood = tf.reduce_sum(self.xhat_flat_distribution.log_prob(self.x),[1,2,3])
            
        with tf.variable_scope('decoder', reuse=True):
            self.sample_cat = tf.placeholder(tf.float32, shape=(params.Nsample, params.N_cat*params.dim_cat))
            self.sample_noise = tf.placeholder(tf.float32, shape=(params.Nsample, params.dim_noise))
            self.sample_output = decoder(tf.concat([self.sample_cat, self.sample_noise],1),  
                                         params.BatchNormDecay, 
                                         flag_training = False)
            
        self.real_cat = tf.placeholder(tf.float32, shape=(params.N_cat*params.batch_size, params.dim_cat))
        self.real_noise = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_noise))
        with tf.variable_scope('Discri_cat'):
            self.real_cat_prob = Discriminator(self.real_cat,
                                               params.dim_DS_cat_hiddn1, params.dim_DS_cat_hiddn2,
                                               params.BatchNormDecay,
                                               flag_raw=True)
        with tf.variable_scope('Discri_cat', reuse=True):
            self.latent_cat_reshaped = tf.reshape(self.latent_cat, shape=(-1,params.dim_cat))
            self.fake_cat_prob = Discriminator(self.latent_cat_reshaped,
                                               params.dim_DS_cat_hiddn1, params.dim_DS_cat_hiddn2,
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
        self.infoGAN_cat = tf.placeholder(tf.float32, shape=(params.batch_size, params.N_cat*params.dim_cat))
        self.infoGAN_noise = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_noise))
        self.real_data = tf.random_shuffle(self.x)
        with tf.variable_scope('decoder', reuse=True):
            self.decoderGene_output = decoder(tf.concat([self.infoGAN_cat, self.infoGAN_noise],1),
                                                         params.BatchNormDecay)
        # Use same encoder (variable share) to model Q()
        with tf.variable_scope('encoder', reuse=True): 
            self.Q_cat, _     = encoder(self.decoderGene_output, 
                                        params.N_cat,
                                        params.dim_cat, 
                                        params.dim_noise,
                                        params.BatchNormDecay)
            
            tf.summary.histogram('debug_sampleNoise', self.infoGAN_noise)
        with tf.variable_scope('D2d'):
            self.decoderGene_data_prob = D2d(self.decoderGene_output, params.BatchNormDecay)
        with tf.variable_scope('D2d', reuse=True):
            self.real_data_prob = D2d(self.real_data, params.BatchNormDecay)
# Loss functions
    # 1. reconstruction loss
        #self.loss_recon = -tf.reduce_mean(self.expected_log_likelihood)
        self.loss_recon = tf.reduce_mean(tf.reduce_sum(tf.square(self.x - self.decoder_output),axis=(1,2,3)))
        tf.summary.scalar('loss_recon', self.loss_recon)
    # 2. mutual information loss
        self.loss_Q_cat = -tf.reduce_mean(tf.reduce_sum(tf.log(self.Q_cat+1e-8)*self.infoGAN_cat,1))

        tf.summary.scalar('loss_Q_cat', self.loss_Q_cat)
    # 3. adversarial loss
        self.loss_Discri_cat  = -tf.reduce_mean(self.real_cat_prob - self.fake_cat_prob)
        self.loss_Gene_cat    = -tf.reduce_mean(self.fake_cat_prob)
        self.loss_Discri_noise = -tf.reduce_mean(self.real_noise_prob - self.fake_noise_prob)
        self.loss_Gene_noise   = -tf.reduce_mean(self.fake_noise_prob)
        self.loss_Discri      = self.loss_Discri_cat + self.loss_Discri_noise
        self.loss_Gene        = self.loss_Gene_cat + self.loss_Gene_noise
        
        self.loss_Discri_decoder = -tf.reduce_mean(tf.log(self.real_data_prob+1e-8) +\
                                                   tf.log(1 - self.decoderGene_data_prob+1e-8))
        self.loss_Gene_decoder = -tf.reduce_mean(tf.log(self.decoderGene_data_prob+1e-8))
        
        tf.summary.scalar('loss_Discri_cat', self.loss_Discri_cat)
        tf.summary.scalar('loss_Discri_noise', self.loss_Discri_noise)
        tf.summary.scalar('loss_Gene_cat', self.loss_Gene_cat)
        tf.summary.scalar('loss_Gene_noise', self.loss_Gene_noise)
        tf.summary.scalar('loss_Discri', self.loss_Discri)
        tf.summary.scalar('loss_Gene', self.loss_Gene)
        
        tf.summary.scalar('loss_D2d', self.loss_Discri_decoder)
        tf.summary.scalar('loss_generator', self.loss_Gene_decoder)
# Variable list
        self.varlist_E = [v for v in tf.trainable_variables() if v.name.startswith('encoder/')]
        self.varlist_D = [v for v in tf.trainable_variables() if v.name.startswith('decoder/')]
        self.varlist_Q = [v for v in tf.trainable_variables() if v.name.startswith('encoder/')]
        self.varlist_Discri_cat = [v for v in tf.trainable_variables() if v.name.startswith('Discri_cat/')]
        self.varlist_Discri_noise = [v for v in tf.trainable_variables() if v.name.startswith('Discri_noise/')]
        self.varlist_Discri_decoder = [v for v in tf.trainable_variables() if v.name.startswith('D2d/')]
        
        self.varlist_ED = self.varlist_E + self.varlist_D
        self.varlist_Discri = self.varlist_Discri_cat + self.varlist_Discri_noise
# Summary for model variables
        for var in self.varlist_ED:
            variable_summaries(var)
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
            self.optimizer_Q_infoGAN = slim.learning.create_train_op(self.loss_Q_cat,
                                                             params.optimizer_Q,
                                                             variables_to_train=self.varlist_D+self.varlist_Q)  
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
# Merge all summary nodes
        self.merged = tf.summary.merge_all()
    
### Define trainning process ###
import matplotlib.pyplot as plt
def train(model, trainBatch, params, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(log_path+'reconstruction/'):
        os.makedirs(log_path+'reconstruction/')
    if not os.path.exists(log_path+'generation/'):
        os.makedirs(log_path+'generation/')
    # log all trainable variables
    with open(log_path+'model_variables.txt','a') as myfile:
        tempText=''
        for var in tf.trainable_variables():
            tempText = tempText + str(var)+'\n'
        myfile.write(tempText)
        
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        eye_cat = np.zeros((params.Nsample, params.N_cat, params.dim_cat))
        for idx in range(params.Nsample):
            eye_cat[idx,idx, 1] = 1.0
        eye_cat = eye_cat.reshape(-1, params.N_cat*params.dim_cat)
        zero_noise = np.zeros((params.Nsample, params.dim_noise))
        train_writer = tf.summary.FileWriter(log_path + 'Summary/train')
    # alterTrain
        bestAccuracy = -1.0
        count_batch = 0
        preSave_step = -1.0
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for step in range(params.num_steps):   
            currBatch = sess.run(trainBatch)
            count_batch = count_batch + 1
            real_cat = Sample_categorical(params.batch_size, params.N_cat, params.dim_cat)
            real_noise = Sample_noise(params.batch_size, params.dim_noise)
            infoGAN_cat = Sample_categorical(params.batch_size, params.N_cat, params.dim_cat).\
                          reshape(params.batch_size, params.N_cat*params.dim_cat)
            infoGAN_noise = Sample_noise(params.batch_size, params.dim_noise)
            feedDict = {model.x:           currBatch,
                        model.infoGAN_cat: infoGAN_cat,
                        model.infoGAN_noise: infoGAN_noise,
                        model.real_cat:    real_cat,
                        model.real_noise:  real_noise,
                       }
        # 1. infoGAN - emphasized
            for _ in range(params.num_critis):
                sess.run([model.optimizer_Discri_infoGAN,
                          model.optimizer_Gene_infoGAN], feedDict)
            sess.run(model.optimizer_Q_infoGAN,      feedDict)
        # 2. AAE
            sess.run(model.optimizer_ED, feedDict)
            for _ in range(params.num_critis):
                sess.run(model.optimizer_Discri,     feedDict)
                sess.run(model.clip_discriminator,   feedDict)
            sess.run(model.optimizer_Gene,           feedDict)
        # Summary nodes
            if step%params.summaryStep ==0:
                summary = sess.run(model.merged,feedDict)
                train_writer.add_summary(summary, step)
        # Save model & compute accuracy      
            if step%params.saveStep ==0:
                saver.save(sess, log_path+'ckpt/model_'+str(step)+'.ckpt')
        # Generation image
            zero_noise = np.zeros((params.Nsample, params.dim_noise))
            if step%params.showStep ==0:     
                Ntrial=10
                decoderSample_small = np.ones((Ntrial, params.Nsample, 64,64,3), dtype=float) 
                for idx in range(Ntrial):
                    if idx == 0:
                        feedDict_rand = {
                                         model.sample_cat:   eye_cat,
                                         model.sample_noise: zero_noise,
                                        }
                        decoderSample_small[idx,:,:,:] = sess.run(model.sample_output,  feedDict_rand) 
                    else:
                        test_noise = Sample_noise(params.Nsample, params.dim_noise)
                        feedDict_rand = {
                                         model.sample_cat:   eye_cat,
                                         model.sample_noise: test_noise,
                                        }
                        decoderSample_small[idx,:,:,:] = sess.run(model.sample_output,  feedDict_rand)

                d = np.empty((Ntrial,params.Nsample,params.img_W*2,params.img_H*2,3), dtype=float) 
                for idx_t in range(Ntrial):   # for each trial
                    for idx_cat in range(params.Nsample):   # for each category
                        temp_d             = decoderSample_small[idx_t,idx_cat]
                        d[idx_t,idx_cat] = sp.ndimage.interpolation.zoom(temp_d,(2,2,1))
                outplot = np.swapaxes(d,1,2).reshape(Ntrial*params.img_W*2, params.Nsample*params.img_H*2,3)
                outplot = np.clip(outplot,0.0,255.0)/255.0
                plt.imsave(log_path+'generation/'+str(step)+'.png',outplot)
            # Reconstruction image
                decoder_output = sess.run(model.decoder_output, feedDict)
                g = np.empty((2,5,params.img_W*2,params.img_H*2,3), dtype=float)  # reconstructed from encoder output
                for idx in range(5):
                    temp_g_real = currBatch[idx]
                    temp_g_recon = decoder_output[idx]
                    g[0,idx] = sp.ndimage.interpolation.zoom(temp_g_real,(2,2,1))
                    g[1,idx] = sp.ndimage.interpolation.zoom(temp_g_recon,(2,2,1))
                outplot =np.swapaxes(g,1,2).reshape(2*params.img_W*2, 5*params.img_H*2, 3)
                outplot = np.clip(outplot,0.0,255.0)/255.0
                plt.imsave(log_path+'reconstruction/'+str(step)+'.png',outplot)
                ### DEBUG ##############################
                np.set_printoptions(precision=2)
                recon_cat,recon_noise = sess.run([model.latent_cat, 
                                                  model.latent_noise],
                                                 feedDict)
                recon_cat = recon_cat[0].reshape(params.N_cat, params.dim_cat)
                recon_noise = recon_noise[0]
                with open(log_path+'latent_var.txt', "a") as myfile:
                    tempText = 'Step = '+str(step)+'\n'+\
                               'recon_cat = \n'+str(recon_cat)+'\n'+\
                               'recon_noise = \n'+str(recon_noise)+'\n'+\
                               '#################################################\n'
                    myfile.write(tempText)
        # Stop the threads
        coord.request_stop()
        # Wait for threads to stop
        coord.join(threads)
        sess.close()
class trainParameter():
    def __init__(self, LR_recon, LR_adv, LR_adv_infoGAN):
        self.batch_size = 128
        self.N_cat = 10
        self.dim_cat=10
        self.dim_noise = 128
        self.img_W = 64
        self.img_H = 64
    # Hidden dimension
        self.dim_DS_cat_hiddn1=3000   # Discriminator - categorical
        self.dim_DS_cat_hiddn2=3000
        self.dim_DS_cont_hiddn1=3000   # Discriminator - continuous
        self.dim_DS_cont_hiddn2=3000
    # Other
        self.num_steps =100000
        self.showStep=500
        self.saveStep=500
        self.summaryStep = 200
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
        self.optimizer_supervised = tf.train.RMSPropOptimizer(10*LR_recon)
        
        self.optimizer_Q = tf.train.AdamOptimizer(LR_adv_infoGAN)
        self.optimizer_adv_generator_infoGAN     = tf.train.AdamOptimizer(LR_recon)
        self.optimizer_adv_discriminator_infoGAN = tf.train.AdamOptimizer(LR_adv_infoGAN)
    # Ancestrol sampling (test the reconstruction ability of decoder)
        self.Nsample=10 # number of batch in model.sample_output


# Load data & Define parameters
def main(log_path, LR_recon, LR_adv, LR_adv_infoGAN):
    #trainFiles = './celebA_data/trainData.tfrecords' # 64x64, [0,255]
    trainFiles = './celebA_data/trainData_crop_float32_0_255.tfrecords'
    trainFeature = {'img_raw': tf.FixedLenFeature([], tf.string)}
    # Initialize the graph, set parameters
    tf.reset_default_graph()
    params = trainParameter(LR_recon, LR_adv, LR_adv_infoGAN)
    
    # Build data-reading pipeline
    trainFiles_queue = tf.train.string_input_producer([trainFiles])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(trainFiles_queue)
    features = tf.parse_single_example(serialized_example, features=trainFeature)
    image = tf.decode_raw(features['img_raw'], tf.float32)
    image = tf.reshape(image,(64,64,3))
    trainBatch = tf.train.shuffle_batch([image], batch_size=params.batch_size,
                                        capacity=4096, num_threads=1, min_after_dequeue=2048)
    # Build the model, feed in data pipeline
    RepGAN_1 = RepGAN(params)
    # Training
    train(RepGAN_1, trainBatch, params, log_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LR_adv_infoGAN_idx', required=False) #given by SLURM_ARRAY_TASK_ID
    parser.add_argument('--LR_recon', required=False)
    parser.add_argument('--LR_adv',   required=False)
    parser.add_argument('--LR_adv_infoGAN', required=False)
    args = parser.parse_args()

    main_log_path = './LOG_RepGAN_semiSup_RGB_pipeline'
    
    currLR_recon = 1e-4
    currLR_adv   = 1e-3
    currLR_adv_infoGAN = 2e-5
    curr_log_path = main_log_path+'_singleTest/'+\
                        'LOG_RepGAN_'+\
                        '{:.0e}'.format(currLR_recon)+'_'+\
                        '{:.0e}'.format(currLR_adv)+'_'+\
                        '{:.0e}'.format(currLR_adv_infoGAN)+'_run4/' 
    main(curr_log_path, currLR_recon, currLR_adv, currLR_adv_infoGAN)
