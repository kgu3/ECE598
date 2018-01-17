
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
        mu    = slim.fully_connected(net, dim_continuous, activation_fn = None)
        debug_sigma_in = slim.fully_connected(net, dim_continuous, activation_fn = None)
        sigma = tf.sigmoid(debug_sigma_in)
        noise = slim.fully_connected(net, dim_noise, activation_fn = None)
    return latent_cat, mu, sigma, noise, debug_sigma_in

def decoder(latent, BatchNormDecay, flag_training=True):
    with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose],
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
    #real_cont = np.random.uniform(low=-1.0, high=1.0000001, size=(size_batch, size_continuous))
    real_cont = np.random.normal(loc=0.0, scale=0.5, size=(size_batch, size_continuous))
    return real_cont
def Sample_noise(size_batch, size_noise):
    noise = np.random.normal(loc=0.0, scale=0.3, size=(size_batch, size_noise))
    return noise

### Define RepGAN class ###
class RepGAN(object):
    def __init__(self, params):
    ### AAE ###
        with tf.variable_scope('encoder'):
            self.debug_sampleNoise = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_noise))
            self.debug_sampleCont = tf.placeholder(tf.float32, shape=(params.batch_size, params.dim_cont))
            self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 28,28,1))
            self.latent_cat,\
            self.latent_cont_mu,\
            self.latent_cont_sigma,\
            self.latent_noise,\
            self.debug_sigma_in    = encoder(self.x,
                                             params.dim_cat,
                                             params.dim_cont,
                                             params.dim_noise,
                                             params.BatchNormDecay)
            
            tf.summary.histogram('latent_noise', self.latent_noise)
            tf.summary.histogram('latent_cont_mu', self.latent_cont_mu)
            tf.summary.histogram('latent_cont_sigma', self.latent_cont_sigma)
            tf.summary.histogram('sigma_in', self.debug_sigma_in)
            tf.summary.histogram('sample_noise', self.debug_sampleNoise)
            tf.summary.histogram('sample_cont', self.debug_sampleCont)
        with tf.variable_scope('encoder', reuse=True):
            self.test_data = tf.placeholder(tf.float32, shape=(None, 28,28,1))
            self.test_cat,_,_,_,_= encoder(self.test_data,
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
        zero_cont = tf.constant(0.0, shape=(params.batch_size, params.dim_cont))
        zero_noise = tf.constant(0.0, shape=(params.batch_size, params.dim_noise))
        self.real_data = tf.placeholder(tf.float32, shape=(params.batch_size, 28,28,1))
        with tf.variable_scope('decoder', reuse=True):
            self.decoderGene_output = decoder(tf.concat([self.infoGAN_cat, self.infoGAN_cont, self.infoGAN_noise],1),
                                                         params.BatchNormDecay)

        with tf.variable_scope('decoder', reuse=True):
            self.heads = decoder(tf.concat([self.infoGAN_cat, zero_cont, zero_noise],1),
                                                         params.BatchNormDecay)
        # Use same encoder (variable share) to model Q()
        with tf.variable_scope('encoder', reuse=True): 
            self.Q_cat,\
            self.Q_cont_mu,\
            self.Q_cont_sigma,_,\
            self.Q_cont_sigma_in = encoder(self.decoderGene_output, 
                                            params.dim_cat, 
                                            params.dim_cont, 
                                            params.dim_noise,
                                            params.BatchNormDecay)
            self.Q_cont_distribution = distributions.MultivariateNormalDiag(loc=self.Q_cont_mu, scale_diag=self.Q_cont_sigma)
        with tf.variable_scope('D2d'):
            #self.decoderGene_data_prob = D2d(self.decoderGene_output, params.BatchNormDecay)
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
        
        self.debug_Q_L2 = tf.reduce_mean(tf.reduce_sum(tf.abs(self.infoGAN_cont - self.Q_cont_mu), axis=1))
        tf.summary.scalar('loss_Q_cat', self.loss_Q_cat)
        tf.summary.scalar('loss_Q_cont', self.loss_Q_cont)
        tf.summary.scalar('loss_Q', self.loss_Q)
        tf.summary.scalar('Q_L2_loss', self.debug_Q_L2)
    # 3. adversarial loss
        self.loss_Discri_cat  = -tf.reduce_mean(self.real_cat_prob - self.fake_cat_prob)
        self.loss_Gene_cat    = -tf.reduce_mean(self.fake_cat_prob)
        self.loss_Discri_cont = -tf.reduce_mean(self.real_cont_prob - self.fake_cont_prob)
        self.loss_Gene_cont   = -tf.reduce_mean(self.fake_cont_prob)
        self.loss_Discri_noise = -tf.reduce_mean(self.real_noise_prob - self.fake_noise_prob)
        self.loss_Gene_noise   = -tf.reduce_mean(self.fake_noise_prob)
        self.loss_Discri      = self.loss_Discri_cat + self.loss_Discri_cont + self.loss_Discri_noise
        self.loss_Gene        = self.loss_Gene_cat + self.loss_Gene_cont + self.loss_Gene_noise
        
        self.loss_Discri_decoder = -tf.reduce_mean(tf.log(self.real_data_prob+1e-8) + 
                                                   tf.log(1 - self.decoderGene_data_prob+1e-8))
        self.loss_Gene_decoder = -tf.reduce_mean(tf.log(self.decoderGene_data_prob+1e-8))
        ### DEBUG #############################################
        self.debug_prob_real = tf.reduce_mean(self.real_data_prob)
        self.debug_prob_fake = tf.reduce_mean(self.decoderGene_data_prob)
    #4. xcorr loss -- different category
        '''
        temp = tf.expand_dims(self.infoGAN_cat, axis=-1)-tf.transpose(tf.expand_dims(self.infoGAN_cat,axis=0),[0,2,1])
        dist_cat = 0.5*tf.reduce_sum(tf.abs(temp),axis=1)

        head_flat = tf.reshape(self.heads, [-1,784]) # shape = [batch_size, 784]
        head_flat = head_flat - tf.reduce_mean(head_flat, axis=1, keep_dims=True)
        head_flat = head_flat / tf.sqrt(tf.reduce_sum(head_flat*head_flat, axis=1, keep_dims=True))
        temp = tf.multiply(tf.expand_dims(head_flat, axis=-1),
                           tf.transpose(tf.expand_dims(head_flat,axis=0),[0,2,1]))
        dist_head= tf.reduce_sum(temp,axis=1) #this is xcorr, larger means more similar
        self.dist_head = dist_head
        
        gene_flat = tf.reshape(self.decoderGene_output, [-1,784]) # shape = [batch_size, 784]
        gene_flat = gene_flat - tf.reduce_mean(gene_flat, axis=1, keep_dims=True)
        gene_flat = gene_flat / tf.sqrt(tf.reduce_sum(gene_flat*gene_flat, axis=1, keep_dims=True))
        temp = tf.multiply(tf.expand_dims(gene_flat, axis=-1),
                           tf.transpose(tf.expand_dims(gene_flat,axis=0),[0,2,1]))
        dist_gene= tf.reduce_sum(temp,axis=1)
        self.dist_gene = dist_gene
        
        self.loss_xcorr_diffCat = 0.001*tf.reduce_mean(tf.multiply(dist_cat,dist_head) + 
                                                         tf.multiply(dist_cat,dist_gene))
        self.loss_xcorr_sameCat = -0.001*tf.reduce_mean(tf.multiply(-(dist_cat-1.0),dist_gene))
        self.loss_xcorr = self.loss_xcorr_sameCat + self.loss_xcorr_diffCat
        '''
# Variable list
        self.varlist_E = [v for v in tf.trainable_variables() if v.name.startswith('encoder/')]
        self.varlist_D = [v for v in tf.trainable_variables() if v.name.startswith('decoder/')]
        self.varlist_Discri_cat = [v for v in tf.trainable_variables() if v.name.startswith('Discri_cat/')]
        self.varlist_Discri_cont = [v for v in tf.trainable_variables() if v.name.startswith('Discri_cont/')]
        self.varlist_Discri_noise = [v for v in tf.trainable_variables() if v.name.startswith('Discri_noise/')]
        self.varlist_Discri_decoder = [v for v in tf.trainable_variables() if v.name.startswith('D2d/')]
        
        self.varlist_ED = self.varlist_E + self.varlist_D
        self.varlist_Discri = self.varlist_Discri_cat + self.varlist_Discri_cont + self.varlist_Discri_noise
# Updates ops for BatchNorm
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.BN_E = [v for v in self.update_ops if v.name.startswith('encoder')]
        self.BN_D = [v for v in self.update_ops if v.name.startswith('decoder')]
        self.BN_D2d = [v for v in self.update_ops if v.name.startswith('D2d')]
# Optimizer - with dependency on 'update_ops' for tf.summary
        ### AAE ###
        self.optimizer_ED = slim.learning.create_train_op(self.loss_recon,
                                                          params.optimizer_recon,
                                                          variables_to_train=self.varlist_ED,
                                                          update_ops = self.BN_E + self.BN_D)
        
        self.optimizer_Discri = slim.learning.create_train_op(self.loss_Discri,
                                                              params.optimizer_adv_discriminator,
                                                              variables_to_train=self.varlist_Discri,
                                                              update_ops = [])
        
        self.optimizer_Gene = slim.learning.create_train_op(self.loss_Gene,
                                                            params.optimizer_adv_generator,
                                                            variables_to_train=self.varlist_E,
                                                            update_ops = self.BN_E)
        ### infoGAN ###
        self.optimizer_Q_infoGAN = slim.learning.create_train_op(self.loss_Q,
                                                         params.optimizer_Q,
                                                         variables_to_train=self.varlist_ED,
                                                         update_ops = self.BN_E + self.BN_D)  
        
        self.optimizer_Discri_infoGAN = slim.learning.create_train_op(self.loss_Discri_decoder,
                                                                      params.optimizer_adv_discriminator_infoGAN,
                                                                      variables_to_train=self.varlist_Discri_decoder,
                                                                      update_ops = self.BN_D2d)
        
       # self.optimizer_Gene_infoGAN = slim.learning.create_train_op(self.loss_Gene_decoder + self.loss_xcorr,
        self.optimizer_Gene_infoGAN = slim.learning.create_train_op(self.loss_Gene_decoder,
                                                                    params.optimizer_adv_generator_infoGAN,
                                                                    variables_to_train=self.varlist_D,
                                                                    update_ops = self.BN_D)
# Clip the Discriminator parameters - WGAN
        self.clip_discriminator=[]
        for currVars in self.varlist_Discri:
            self.clip_discriminator.append(tf.assign(currVars, tf.clip_by_value(currVars, -params.c, params.c)))
# Merge all summary nodes
        self.merged = tf.summary.merge_all()

### Define trainning process ###
import matplotlib.pyplot as plt
def train(model, data, params, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(log_path+'reconstruction/'):
        os.makedirs(log_path+'reconstruction/')
    if not os.path.exists(log_path+'generation/'):
        os.makedirs(log_path+'generation/')
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        eye_cat = np.zeros((params.Nsample, params.dim_cat))
        for idx in range(params.Nsample):
            eye_cat[idx,idx]=1.0
        zero_cont = np.zeros((params.Nsample, params.dim_cont))
        zero_noise = np.zeros((10, params.dim_noise))
        testImg, trueLabel = data.test.images, data.test.labels
        testImg = testImg.reshape([-1, 28, 28, 1])
        train_writer = tf.summary.FileWriter(log_path + 'Summary/train')
    # alterTrain
        bestAccuracy = -1.0
        flag_D2d_train = True
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
            debug_sampleNoise = Sample_noise(params.batch_size, params.dim_noise)
            debug_sampleCont  = Sample_continuous(params.batch_size, params.dim_cont)
            feedDict = {model.x:           currBatch,
                        model.infoGAN_cat: infoGAN_cat,
                        model.infoGAN_cont: infoGAN_cont,
                        model.infoGAN_noise: infoGAN_noise,
                        model.real_cat:    real_cat,
                        model.real_cont:   real_cont,
                        model.real_noise:  real_noise,
                        model.real_data:   currBatch_perm,
                        model.debug_sampleNoise: debug_sampleNoise,
                        model.debug_sampleCont:  debug_sampleCont,
                       }
        # 1. infoGAN - emphasized
            for _ in range(params.num_critis):
                #if flag_D2d_train:
                sess.run(model.optimizer_Discri_infoGAN, feedDict)
                sess.run(model.optimizer_Gene_infoGAN,   feedDict)
                sess.run(model.optimizer_Q_infoGAN,      feedDict)
        # 2. AAE
            sess.run(model.optimizer_ED, feedDict)
            for _ in range(params.num_critis):
                sess.run(model.optimizer_Discri,     feedDict)
                sess.run(model.clip_discriminator,   feedDict)
            sess.run(model.optimizer_Gene,           feedDict)
        # 3. Balance
            debug_prob_real = sess.run(model.debug_prob_real, feedDict)
            debug_prob_fake = sess.run(model.debug_prob_fake, feedDict)
            currRatio = debug_prob_fake/debug_prob_real
            '''
            flag_D2d_train=True
            if currRatio<0.2:
                flag_D2d_train=False
            '''
        # Summary nodes
            if step%500 ==0:
                summary = sess.run(model.merged,feedDict)
                train_writer.add_summary(summary, step)
                latent_noise = sess.run(model.latent_noise, feedDict)
                with open(log_path+'latent_noise.txt', "a") as myfile:
                    np.set_printoptions(precision=3)
                    np.set_printoptions(linewidth=10000)
                    tempText = 'Step = '+str(step)+'\n'+\
                               'latent_noise = \n'+str(latent_noise[0:10,:])+'\n'
                    myfile.write(tempText)
        # Save model & compute accuracy      
            if step%params.saveStep ==0:
                feedDict_save = {
                                model.sample_cat:      eye_cat,
                                model.sample_cont:    zero_cont,
                                model.sample_noise:  zero_noise,
                               }
                clusterHead = sess.run(model.sample_output,  feedDict_save)
                # Find the true label of clusterHead
                headLabel = np.zeros((clusterHead.shape[0],params.dim_cat))
                for idxHead in range(clusterHead.shape[0]):
                    currMSE=((testImg - clusterHead[idxHead]) ** 2).mean(axis=(1,2))
                    bestMatch = np.argmin(currMSE)
                    headLabel[idxHead] = trueLabel[bestMatch]
                # Compute classification accuracy
                # test set in batches (solve the OOM problem)
                testBatch=1000
                testIttrN = int(testImg.shape[0]/testBatch)
                total_disagree=0
                for testIdx in range(testIttrN):
                    currtestImg,currtrueLabel = data.test.next_batch(testBatch)
                    currtestImg = currtestImg.reshape([-1, 28, 28, 1])
                    feedDict_test = {model.test_data:          currtestImg,
                                    }
                    latent_cat = sess.run(model.test_cat, feedDict_test)   
                    # Compute classification accuracy
                    latent_cat_idx = np.argmax(latent_cat, axis=1) # classification label of test images
                    total_disagree = total_disagree + np.absolute((currtrueLabel-headLabel[latent_cat_idx])).sum()/2.0
                currAccuracy = 1.0 - total_disagree/testImg.shape[0]
                with open(log_path+'accuracy.txt', "a") as myfile:
                    tempText = 'Step = '+str(step)+'    Accuracy = '+'{:.2f}'.format(currAccuracy)+'\n'
                    myfile.write(tempText)
                if currAccuracy>bestAccuracy:
                    bestAccuracy = currAccuracy
                    #saver.save(sess, log_path+'ckpt/model_'+str(step)+'.ckpt')
                ### DEBUG ##################################
                loss_D2d  = sess.run(model.loss_Discri_decoder, feedDict)
                loss_Gene = sess.run(model.loss_Gene_decoder,   feedDict)
                cont_mu   = sess.run(model.Q_cont_mu,  feedDict)
                cont_sigma = sess.run(model.Q_cont_sigma,  feedDict)
                sigma_in   = sess.run(model.Q_cont_sigma_in,  feedDict)
                np.set_printoptions(linewidth=10000)
                np.set_printoptions(precision=2)
                with open(log_path+'DEBUG.txt','a') as myfile:
                    tempText = 'Step = '+str(step)+\
                               ' prob_real = '+'{:.3f}'.format(debug_prob_real)+\
                               ' prob_fake = '+'{:.3f}'.format(debug_prob_fake)+\
                               ' currRatio = '+str(currRatio)+\
                               ' flag_D2d_train = '+str(flag_D2d_train)+\
                               ' L_D2d = '+'{:.3f}'.format(loss_D2d)+\
                               ' L_Gene = '+'{:.3f}'.format(loss_Gene)+'\n'
                    myfile.write(tempText)
                with open(log_path+'continuous.txt','a') as myfile:
                    tempText = 'Step = '+str(step)+'\n'+\
                               ' real_cont =\n'+str(infoGAN_cont[0:10])+'\n'+\
                               ' mu = \n'+str(cont_mu[0:10])+'\n'+\
                               ' sigma = \n'+str(cont_sigma[0:10])+'\n'+\
                               ' sigma_in = \n'+str(sigma_in[0:10])+'\n'
                    myfile.write(tempText)
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

                g = np.empty((2,5,28*3,28*3), dtype=float)  # reconstructed from encoder output
                for idx in range(5):
                    temp_g_real = currBatch[idx].reshape([28,28])
                    temp_g_recon = decoder_output[idx].reshape([28,28])
                    g[0,idx] = sp.ndimage.interpolation.zoom(temp_g_real,3) 
                    g[1,idx] = sp.ndimage.interpolation.zoom(temp_g_recon,3) 
                outplot = np.swapaxes(g,1,2).reshape(2*28*3, 5*28*3)
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
        self.num_steps =60000
        self.showStep=500
        self.saveStep=500
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
    mnist = input_data.read_data_sets('fashionMNIST_data', one_hot=True)
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

    main_log_path = './LOG_RepGAN_4'
    currLR_recon = 5e-5
    currLR_adv   = 1e-3
    currLR_adv_infoGAN = 2e-5
    curr_log_path = main_log_path+'_singleTest/'+\
                        'LOG_RepGAN_'+\
                        '{:.0e}'.format(currLR_recon)+'_'+\
                        '{:.0e}'.format(currLR_adv)+'_'+\
                        '{:.0e}'.format(currLR_adv_infoGAN)+'_base/' 
    main(curr_log_path, currLR_recon, currLR_adv, currLR_adv_infoGAN)