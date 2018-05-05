import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as distributions
import numpy as np
import os
import scipy as sp
import scipy.ndimage
import scipy.io as sio
import sklearn.preprocessing as skp

### Define modules
def variable_summaries(var):
    scopeName = 'summary/'+var.name.replace(':','/')
    with tf.name_scope(scopeName):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
def lrelu(x, alpha=0.1):
    return tf.maximum(x, alpha * x)
def encoder(x,dim_categorical,dim_continuous, dim_noise, BatchNormDecay, flag_training=True):
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
            latent_cat = slim.fully_connected(net, dim_categorical,activation_fn=tf.nn.softmax)
        with tf.variable_scope('FC_mu'):
            mu = slim.fully_connected(net, dim_continuous, activation_fn = None)
        with tf.variable_scope('FC_sigma'):
            sigma =  slim.fully_connected(net, dim_continuous, activation_fn = tf.sigmoid)
        with tf.variable_scope('FC_noise'):
            noise = slim.fully_connected(net, dim_noise, activation_fn = None)
    return latent_cat, mu, sigma, noise

def decoder(latent, BatchNormDecay, flag_training=True):
    with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose],
                        activation_fn = tf.nn.relu,
                        normalizer_fn=slim.batch_norm,  # BN crucial for infoGAN here, otherwise no work
                        normalizer_params={'decay': BatchNormDecay, 'is_training': flag_training}
                       ):
        net = slim.fully_connected(latent, 1024)
        net = slim.fully_connected(net,4*4*256)
        net = tf.reshape(net, shape=[-1,4,4,256])
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

def Sample_categorical(size_batch, size_categorical):
    real_cat = np.zeros((size_batch, size_categorical))
    rand_idx = np.random.randint(0, size_categorical, size_batch)
    real_cat[np.arange(size_batch), rand_idx]=1.0
    return real_cat
def Sample_continuous(size_batch, size_continuous):
    #real_cont = np.random.uniform(low=-1.0, high=1.0000001, size=(size_batch, size_continuous))
    real_cont = np.random.normal(loc=0.0, scale=1.0, size=(size_batch, size_continuous))
    return real_cont
def Sample_noise(size_batch, size_noise):
    noise = np.random.normal(loc=0.0, scale=1.0, size=(size_batch, size_noise))
    return noise

### Define RepGAN class ###
class RepGAN(object):
    def __init__(self, params):
    ### AAE ###
        with tf.variable_scope('encoder'):
            self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 32,32,3))
            self.currTrainLabel = tf.placeholder(tf.float32, shape=(params.batch_size, 10))
            self.latent_cat,\
            self.latent_cont_mu,\
            self.latent_cont_sigma,\
            self.latent_noise             = encoder(self.x,
                                             params.dim_cat,
                                             params.dim_cont,
                                             params.dim_noise,
                                             params.BatchNormDecay)
            
            self.latent_cont_sigma_clip = tf.clip_by_value(self.latent_cont_sigma, -1.0, 1.0)
            self.latent_noise_clip = tf.clip_by_value(self.latent_noise, -1.0, 1.0)
            tf.summary.histogram('latent_noise', self.latent_noise)
            tf.summary.histogram('latent_noise_clip', self.latent_noise_clip)
            tf.summary.histogram('latent_cont_mu', self.latent_cont_mu)
            tf.summary.histogram('latent_cont_sigma', self.latent_cont_sigma)
            tf.summary.histogram('latent_cont_sigma_clip', self.latent_cont_sigma_clip)
        with tf.variable_scope('encoder', reuse=True):
            self.x_sup = tf.placeholder(tf.float32, shape=(params.batch_size, 32,32,3))
            self.currTrainLabel_sup = tf.placeholder(tf.float32, shape=(params.batch_size, 10))
            self.latent_cat_sup,_,_,_ = encoder(self.x_sup,
                                                 params.dim_cat,
                                                 params.dim_cont,
                                                 params.dim_noise,
                                                 params.BatchNormDecay)
        with tf.variable_scope('encoder', reuse=True):
            self.test_data = tf.placeholder(tf.float32, shape=(None, 32,32,3))
            self.test_cat,_,_,_= encoder(self.test_data,
                                                             params.dim_cat,
                                                             params.dim_cont,
                                                             params.dim_noise,
                                                             params.BatchNormDecay,
                                                             flag_training = False)
        with tf.variable_scope('decoder'):
            self.latent_cont_sample = self.latent_cont_mu + tf.multiply(self.latent_cont_sigma, tf.random_normal([1]))
            self.decoder_output = decoder(tf.concat([self.latent_cat, self.latent_cont_sample, self.latent_noise ],1),
                                          params.BatchNormDecay)
            self.xhat_flat_distribution = distributions.Bernoulli(probs = tf.clip_by_value(self.decoder_output, 1e-5, 1-1e-5))
            self.expected_log_likelihood = tf.reduce_sum(self.xhat_flat_distribution.log_prob(self.x),[1,2,3])
            
            self.latent_cont_sample_clip = tf.clip_by_value(self.latent_cont_sample, -2.0, 2.0)
            tf.summary.histogram('latent_cont_sample', self.latent_cont_sample)
            tf.summary.histogram('latent_cont_sample_clip', self.latent_cont_sample_clip)
            
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
        self.real_data = tf.placeholder(tf.float32, shape=(params.batch_size, 32,32,3))
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
            tf.summary.histogram('debug_sampleCont', self.infoGAN_cont)
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
    # 0. classification loss 
        self.loss_classify_sup = -tf.reduce_mean(tf.reduce_sum(tf.log(self.latent_cat_sup+1e-8)*self.currTrainLabel_sup,1))
        
        self.debug_loss_classify_unsup = -tf.reduce_mean(tf.reduce_sum(tf.log(self.latent_cat+1e-8)*self.currTrainLabel,1))
        tf.summary.scalar('loss_classify_sup', self.loss_classify_sup)
        tf.summary.scalar('loss_classify_unsup', self.debug_loss_classify_unsup)
    # 2. mutual information loss
        self.loss_Q_cat = -tf.reduce_mean(tf.reduce_sum(tf.log(self.Q_cat+1e-8)*self.infoGAN_cat,1))
        self.loss_Q_cont = -tf.reduce_mean(self.Q_cont_distribution.log_prob(self.infoGAN_cont))
        self.loss_Q = self.loss_Q_cat + self.loss_Q_cont
        
        self.debug_Q_L2 = tf.reduce_mean(tf.reduce_sum(tf.abs(self.infoGAN_cont - self.Q_cont_mu), axis=1))
        self.debug_Q_percnt = tf.reduce_mean(tf.reduce_sum(tf.abs(self.infoGAN_cont - self.Q_cont_mu),axis=1) / tf.reduce_sum(tf.abs(self.infoGAN_cont),axis=1))
        tf.summary.scalar('loss_Q_cat', self.loss_Q_cat)
        tf.summary.scalar('loss_Q_cont', self.loss_Q_cont)
        tf.summary.scalar('loss_Q', self.loss_Q)
        tf.summary.scalar('Q_L2_loss', self.debug_Q_L2)
        tf.summary.scalar('Q_percnt_loss', self.debug_Q_percnt)
    # 3. adversarial loss
        self.loss_Discri_cat  = -tf.reduce_mean(self.real_cat_prob - self.fake_cat_prob)
        self.loss_Gene_cat    = -tf.reduce_mean(self.fake_cat_prob)
        self.loss_Discri_cont = -tf.reduce_mean(self.real_cont_prob - self.fake_cont_prob)
        self.loss_Gene_cont   = -tf.reduce_mean(self.fake_cont_prob)
        self.loss_Discri_noise = -tf.reduce_mean(self.real_noise_prob - self.fake_noise_prob)
        self.loss_Gene_noise   = -tf.reduce_mean(self.fake_noise_prob)
        self.loss_Discri      = self.loss_Discri_cat + self.loss_Discri_cont + self.loss_Discri_noise
        self.loss_Gene        = self.loss_Gene_cat + self.loss_Gene_cont + self.loss_Gene_noise
        
        self.loss_Discri_decoder = -tf.reduce_mean(tf.log(self.real_data_prob+1e-8) +\
                                                   tf.log(1 - self.decoderGene_data_prob+1e-8))
        self.loss_Gene_decoder = -tf.reduce_mean(tf.log(self.decoderGene_data_prob+1e-8))
        
        tf.summary.scalar('loss_Discri_cat', self.loss_Discri_cat)
        tf.summary.scalar('loss_Discri_cont', self.loss_Discri_cont)
        tf.summary.scalar('loss_Discri_noise', self.loss_Discri_noise)
        tf.summary.scalar('loss_Gene_cat', self.loss_Gene_cat)
        tf.summary.scalar('loss_Gene_cont', self.loss_Gene_cont)
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
        self.varlist_Discri_cont = [v for v in tf.trainable_variables() if v.name.startswith('Discri_cont/')]
        self.varlist_Discri_noise = [v for v in tf.trainable_variables() if v.name.startswith('Discri_noise/')]
        self.varlist_Discri_decoder = [v for v in tf.trainable_variables() if v.name.startswith('D2d/')]
        
        self.varlist_ED = self.varlist_E + self.varlist_D
        self.varlist_Discri = self.varlist_Discri_cat + self.varlist_Discri_cont + self.varlist_Discri_noise
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
            self.optimizer_classify = slim.learning.create_train_op(self.loss_classify_sup,
                                                                  params.optimizer_supervised,
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
# Merge all summary nodes
        self.merged = tf.summary.merge_all()
    
### Define trainning process ###
import matplotlib.pyplot as plt
def train(model, trainImg, trainImg_sup, trainLabel, trainLabel_sup, testImg, testLabel, params, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(log_path+'reconstruction/'):
        os.makedirs(log_path+'reconstruction/')
    if not os.path.exists(log_path+'generation/'):
        os.makedirs(log_path+'generation/')
    # Data statistics
    with open(log_path+'data_stat.txt','a') as myfile:
        np.set_printoptions(precision=2, linewidth=10000)
        np.set_printoptions(threshold=10000)
        tempText = 'min_train = '+str(np.min(trainImg))+'  max_train = '+str(np.max(trainImg))+'\n'+\
                   'min_test = '+str(np.min(testImg))+'  max_test = '+str(np.max(testImg))+'\n'
        myfile.write(tempText)
    # log all trainable variables
    with open(log_path+'model_variables.txt','a') as myfile:
        tempText=''
        for var in tf.trainable_variables():
            tempText = tempText + str(var)+'\n'
        myfile.write(tempText)
        
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        eye_cat = np.zeros((params.Nsample, params.dim_cat))
        for idx in range(params.Nsample):
            eye_cat[idx,idx]=1.0
        zero_cont = np.zeros((params.Nsample, params.dim_cont))
        zero_noise = np.zeros((params.Nsample, params.dim_noise))
        train_writer = tf.summary.FileWriter(log_path + 'Summary/train')
        Nbatch_train = np.floor(trainImg.shape[0]/params.batch_size)
    # alterTrain
        bestAccuracy = -1.0
        count_batch = 0
        preSave_step = -1.0
        for step in range(params.num_steps):   
            if count_batch >= Nbatch_train:
                count_batch = 0
                idx = np.arange(trainImg.shape[0])
                np.random.shuffle(idx)
                trainImg = trainImg[idx]
                trainLabel = trainLabel[idx]
            # Get current training batch
            currBatch = trainImg[count_batch*params.batch_size:(count_batch+1)*params.batch_size]
            currtrainLabel = trainLabel[count_batch*params.batch_size:(count_batch+1)*params.batch_size]
            # Get current supervised training batch
            Idxsup = np.random.choice(trainImg_sup.shape[0], size = params.batch_size, replace=False)
            currBatch_sup = trainImg_sup[Idxsup]
            currtrainLabel_sup = trainLabel_sup[Idxsup]
            
            count_batch = count_batch + 1

            currBatch = currBatch.reshape([-1, 32, 32, 3])
            currBatch_sup = currBatch_sup.reshape([-1, 32, 32, 3])
            currtrainLabel_oneHot = np.zeros((currtrainLabel.shape[0],10))
            currtrainLabel_oneHot[np.arange(currtrainLabel.shape[0]),currtrainLabel-1] = 1
            currtrainLabel_sup_oneHot = np.zeros((currtrainLabel_sup.shape[0],10))
            currtrainLabel_sup_oneHot[np.arange(currtrainLabel_sup.shape[0]),currtrainLabel_sup-1] = 1
            currBatch_perm = np.random.permutation(currBatch)
            real_cat = Sample_categorical(params.batch_size, params.dim_cat)
            real_cont = Sample_continuous(params.batch_size, params.dim_cont)
            real_noise = Sample_noise(params.batch_size, params.dim_noise)
            infoGAN_cat = Sample_categorical(params.batch_size, params.dim_cat)
            infoGAN_cont = Sample_continuous(params.batch_size, params.dim_cont)
            infoGAN_noise = Sample_noise(params.batch_size, params.dim_noise)
            test_noise = Sample_noise(params.Nsample, params.dim_noise)
            feedDict = {model.x:           currBatch,
                        model.currTrainLabel: currtrainLabel_oneHot,
                        model.x_sup:              currBatch_sup,
                        model.currTrainLabel_sup: currtrainLabel_sup_oneHot,
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
            sess.run(model.optimizer_classify, feedDict)
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
                # Compute classification accuracy
                batch_test = 1000
                N_testEpoch = int(np.floor(testImg.shape[0]/batch_test))
                total_disagree=0.0
                for testEpoch in range(N_testEpoch+1):
                    if testEpoch == N_testEpoch:
                        currtestImg   = testImg[testEpoch*batch_test:]
                        currtestLabel = testLabel[testEpoch*batch_test:]
                    else:
                        currtestImg   = testImg[testEpoch*batch_test:(1+testEpoch)*batch_test]
                        currtestLabel = testLabel[testEpoch*batch_test:(1+testEpoch)*batch_test]
                    
                    currtestImg = currtestImg.reshape([-1, 32, 32, 3])
                    feedDict_test = {model.test_data:          currtestImg}
                    latent_cat = sess.run(model.test_cat, feedDict_test)   
                    predLabel = 1 + np.argmax(latent_cat, axis=1)
                    total_disagree = total_disagree + np.sum(currtestLabel != predLabel)
                currAccuracy = 1.0 - total_disagree/testImg.shape[0]
                with open(log_path+'accuracy.txt', "a") as myfile:
                    tempText = 'Step = '+str(step)+'  count_batch = '+str(count_batch)+\
                               '  Accuracy = '+'{:.2f}'.format(currAccuracy)+'\n'
                    myfile.write(tempText)
                if currAccuracy>0.5 and (currAccuracy>bestAccuracy or step - preSave_step >=5000):
                    preSave_step = step
                    if currAccuracy>bestAccuracy:
                        bestAccuracy = currAccuracy
                    saver.save(sess, log_path+'ckpt/model_'+str(step)+'.ckpt')
        # Generation image
            zero_noise = np.zeros((params.Nsample, params.dim_noise))
            if step%params.showStep ==0:     
                Ntrial=10
                decoderSample_small = np.ones((Ntrial, params.Nsample, 32,32,3), dtype=float) 
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

                d = np.empty((Ntrial,params.Nsample,32*3,32*3,3), dtype=float) # reconstructed from ancestrol sampling
                for idx_t in range(Ntrial):   # for each trial
                    for idx_cat in range(params.Nsample):   # for each category
                        temp_d             = decoderSample_small[idx_t,idx_cat]
                        d[idx_t,idx_cat] = sp.ndimage.interpolation.zoom(temp_d,(3,3,1))
                outplot = np.swapaxes(d,1,2).reshape(Ntrial*32*3, params.Nsample*32*3,3)
                outplot -= np.min(outplot)
                outplot /= np.max(outplot)
                plt.imsave(log_path+'generation/'+str(step)+'.png',outplot)
            # Reconstruction image
                decoder_output = sess.run(model.decoder_output, feedDict)
                sampleIdx=3

                g = np.empty((2,5,32*3,32*3,3), dtype=float)  # reconstructed from encoder output
                for idx in range(5):
                    temp_g_real = currBatch[idx]
                    temp_g_recon = decoder_output[idx]
                    g[0,idx] = sp.ndimage.interpolation.zoom(temp_g_real,(3,3,1))
                    g[1,idx] = sp.ndimage.interpolation.zoom(temp_g_recon,(3,3,1))
                outplot =np.swapaxes(g,1,2).reshape(2*32*3, 5*32*3, 3)
                outplot -= np.min(outplot)
                outplot /= np.max(outplot)
                plt.imsave(log_path+'reconstruction/'+str(step)+'.png',outplot)
                ### DEBUG ##############################
                np.set_printoptions(precision=2)
                recon_cat,recon_mu,recon_sgm,recon_noise = sess.run([model.latent_cat, 
                                                                     model.latent_cont_mu, 
                                                                     model.latent_cont_sigma,
                                                                     model.latent_noise],
                                                                    feedDict)
                with open(log_path+'latent_var.txt', "a") as myfile:
                    tempText = 'Step = '+str(step)+'\n'+\
                               'real_label = '+str(currtrainLabel[0:5])+'\n'+\
                               'recon_cat = \n'+str(recon_cat[0:5])+'\n'+\
                               'recon_mu = \n'+str(recon_mu[0:5])+'\n'+\
                               'recon_sgm = \n'+str(recon_sgm[0:5])+'\n'+\
                               'recon_noise = \n'+str(recon_noise[0:5])+'\n'+\
                               '#################################################\n'
                    myfile.write(tempText)
class trainParameter():
    def __init__(self, LR_recon, LR_adv, LR_adv_infoGAN):
        self.batch_size = 128
        self.dim_cat=10
        self.dim_cont = 2
        self.dim_noise = 20
    # Hidden dimension
        self.dim_DS_cat_hiddn1=3000   # Discriminator - categorical
        self.dim_DS_cat_hiddn2=3000
        self.dim_DS_cont_hiddn1=3000   # Discriminator - continuous
        self.dim_DS_cont_hiddn2=3000
    # Other
        self.num_steps =50000
        self.showStep=500
        self.saveStep=500
        self.summaryStep = 100
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
    # Read data
    train_data = sio.loadmat('./svhn_data/train_32x32.mat')
    test_data = sio.loadmat('./svhn_data/test_32x32.mat')
    trainImg = train_data['X']
    trainLabel = train_data['y']
    testImg = test_data['X']
    testLabel = test_data['y']
    
    trainLabel = np.squeeze(trainLabel)
    testLabel = np.squeeze(testLabel)
    trainImg = np.moveaxis(trainImg, -1, 0)
    testImg  = np.moveaxis(testImg, -1, 0)
    # Preprocessing
    trainImg = trainImg/255.0
    testImg = testImg/255.0
    # Pick the supervised samples
    N_supervised = 1000
    N_sup_perclass = int(np.floor(N_supervised/10))  # currently assume 10 classes
    trainImg_sup = np.zeros((N_sup_perclass*10,32,32,3))
    trainLabel_sup = np.zeros((N_sup_perclass*10,),dtype=np.int)
    counter = np.zeros((10,))
    while np.sum(counter)< N_sup_perclass*10:
        currIdx = np.random.randint(0, trainLabel.shape[0])
        currTrainLabel = trainLabel[currIdx] - 1    # from 0 to 9
        if counter[currTrainLabel] < N_sup_perclass:
            tempIdx = N_sup_perclass*currTrainLabel+counter[currTrainLabel]
            trainImg_sup[int(tempIdx)] = trainImg[currIdx]
            trainLabel_sup[int(tempIdx)] = trainLabel[currIdx]   # from 1 to 10
            counter[currTrainLabel] += 1
    

    '''
    trainImg = trainImg.reshape(-1,32*32)
    trainImg = skp.scale(trainImg)
    trainImg = trainImg.reshape(-1,32,32)
    testImg = testImg.reshape(-1,32*32)
    testImg = skp.scale(testImg)
    testImg = testImg.reshape(-1,32,32)
    '''
    # Traininng
    tf.reset_default_graph()
    params = trainParameter(LR_recon, LR_adv, LR_adv_infoGAN)
    RepGAN_1 = RepGAN(params)
    train(RepGAN_1, trainImg, trainImg_sup, trainLabel, trainLabel_sup, testImg, testLabel, params, log_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--LR_adv_infoGAN_idx', required=False) #given by SLURM_ARRAY_TASK_ID
    parser.add_argument('--LR_recon', required=False)
    parser.add_argument('--LR_adv',   required=False)
    parser.add_argument('--LR_adv_infoGAN', required=False)
    args = parser.parse_args()

    main_log_path = './LOG_RepGAN_semiSup_RGB'
    
    currLR_recon = 1e-4
    currLR_adv   = 1e-3
    currLR_adv_infoGAN = 2e-5
    curr_log_path = main_log_path+'_singleTest/'+\
                        'LOG_RepGAN_'+\
                        '{:.0e}'.format(currLR_recon)+'_'+\
                        '{:.0e}'.format(currLR_adv)+'_'+\
                        '{:.0e}'.format(currLR_adv_infoGAN)+'_run8b/' 
    main(curr_log_path, currLR_recon, currLR_adv, currLR_adv_infoGAN)
    
    '''
    if args.LR_recon != None and args.LR_adv != None and args.LR_adv_infoGAN != None and args.LR_adv_infoGAN_idx ==None:
        currLR_recon = float(args.LR_recon)
        currLR_adv   = float(args.LR_adv)
        currLR_adv_infoGAN = float(args.LR_adv_infoGAN)
        curr_log_path = main_log_path+'_singleTest/'+\
                            'LOG_RepGAN_'+\
                            '{:.0e}'.format(currLR_recon)+'_'+\
                            '{:.0e}'.format(currLR_adv)+'_'+\
                            '{:.0e}'.format(currLR_adv_infoGAN)+'_base/' 
        main(curr_log_path, currLR_recon, currLR_adv, currLR_adv_infoGAN)
    elif args.LR_recon == None and args.LR_adv == None and args.LR_adv_infoGAN == None and args.LR_adv_infoGAN_idx != None:
        LR_recon_pool = [1e-4, 1e-5]
        LR_adv_pool   = [1e-3, 1e-4]
        LR_adv_infoGAN_pool = [1e-3, 1e-4, 1e-5]
        
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
    '''