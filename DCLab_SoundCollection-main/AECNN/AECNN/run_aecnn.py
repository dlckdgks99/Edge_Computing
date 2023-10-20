"""
Reimplementing AECNN model minimizing L1 loss.
No discriminator.

Written by Deepak Baby, UGent, Oct 2018.
Adjusted by Fotis Drakopoulos, UGent, Jan 2019.
"""
from __future__ import print_function

import tensorflow as tf
# from tensorflow.contrib.layers import xavier_initializer, flatten, fully_connected
import numpy as np
import keras
from keras.layers import Input, Dense, Conv1D, Conv2D, Conv2DTranspose, BatchNormalization
from keras.layers import LeakyReLU, PReLU, Reshape, Concatenate, Flatten, Add, Lambda
from keras.models import Sequential, Model
# from keras.optimizers import Adam
from keras.callbacks import TensorBoard
keras_backend = tf.keras.backend
keras_initializers = tf.keras.initializers
from data_ops import *
from file_ops import *
from models import *
import keras.backend as K

import time
from tqdm import *
import h5py
import os,sys
import scipy.io.wavfile as wavfile
#from aecnn_ashutosh_is2018 import *

#n_dft = 512
#n_shift = 256
#dft_real_kernels, dft_imag_kernels = get_stft_kernels(n_dft)

#print ("DFT matrix sizes")
#print ("Real mat : " + str(dft_real_kernels.shape))
#print ("Imag mat : " + str(dft_imag_kernels.shape))

#exit()

if __name__ == '__main__':

    # Various GAN options
    opts = {}
    opts ['dirhead'] = "AECNN"
    opts ['z_off'] = True # set to True to omit the latent noise input
    opts ['Gtanh'] = False # set to True if G uses tanh output activation
    # normalization
    #################################
    # Only one of the follwoing should be set to True
    opts ['applyinstancenorm'] = False
    opts ['applybatchrenorm'] = False
    opts ['applybatchnorm'] = False
    opts ['applygroupnorm'] = False
    opts ['applyspectralnorm'] = False
    opts ['applyinstancenorm_G'] = False
    ##################################
    # label smoothing (set to 1 for no label smoothing)
    opts ['D_real_target'] = 1.0
    # GT initialization
    opts ['GT_init_G'] = False
    opts ['GT_init_D'] = False
    opts ['gt_fixed'] = False # set to True for fixed GT layer
    opts['gt_stride'] = 2 # stride to be applied on GT layer
    # PreEmph layer
    opts ['preemph_G'] = False
    opts ['preemph_D'] = False
    opts ['preemph_init'] = np.array([[-0.95, 1]]) # initializer for preemph layer
    opts ['preemph_stride'] = 1 # stride for preemph layer
    # Show model summary
    opts ['show_summary'] = True
   

    ####################################################
    # Other fixed options
    opts ['window_length'] =  2**8
    opts ['featdim'] = 1 # 1 since it is just 1d time samples
    opts ['filterlength'] =  11
    opts ['strides'] = 2
    opts ['padding'] = 'SAME'
    opts ['g_enc_numkernels'] = [16, 16, 16, 32, 32]
    opts ['d_fmaps'] = opts ['g_enc_numkernels'] # We use the same structure for discriminator
    opts['leakyrelualpha'] = 0.3
    opts ['batch_size'] = 100
    opts ['applyprelu'] = True

    opts ['use_bias'] = False
    opts ['d_activation'] = 'leakyrelu'
    g_enc_numkernels = opts ['g_enc_numkernels']
    opts ['g_dec_numkernels'] = g_enc_numkernels[:-1][::-1] + [1]
    opts ['gt_stride'] = 2
    opts ['g_l1loss'] = 100.
    opts ['d_lr'] = 0.0002
    opts ['g_lr'] = 0.0002
    opts ['random_seed'] = 111
    
    ## Set the matfiles
    clean_train_matfile = "./data/clean_train_segan1d_%s.mat" % opts['window_length']
    noisy_train_matfile = "./data/noisy_train_segan1d_%s.mat" % opts['window_length']
    noisy_test_matfile = "./data/noisy_test_segan1d_%s.mat" % opts['window_length']
    
    # load GT filter coef
    if opts['GT_init_G'] or opts['GT_init_D'] :
        gtfile = h5py.File('GT_16channel_31tap.mat')
        gt = np.array(gtfile['gt'])
        print ("Shape of GT matrix " + str(gt.shape))
        opts ['gt'] = gt

    opts['preemph'] = 0
    # set preemph if there is no preemph layer
    #if not (opts ['preemph_D'] or opts['preemph_G']):
    #    opts ['preemph'] = 0.95
    #else :
    #    opts ['preemph'] = 0  
 
    n_epochs = 41
    fs = 16000
    
    # set flags for training or testing
    TRAIN_SEGAN =  True
    SAVE_MODEL =  True
    LOAD_SAVED_MODEL = False
    TEST_SEGAN =  True

    #modeldir = get_modeldirname(opts)
    modeldir = "./AECNN_%s_k%s_emph%s_%s_b%d_pr%d/" % (opts['window_length'],opts['filterlength'],opts['preemph'],''.join(str(e) for e in opts['g_enc_numkernels']),int(opts['use_bias']),int(opts['applyprelu']))
    print ("The model directory is " + modeldir)
    print ("_____________________________________")

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    # The G model has the wav and the noise inputs
    wav_shape = (opts['window_length'], opts['featdim'])
    opts ['G_input'] =  Input(shape=wav_shape, name="main_input_noisy")
    #opts['G_input'] = Input(tensor=tf.placeholder(tf.float32, shape=(None,opts['window_length'],opts['featdim']), name='main_input_noisy'))

    # Obtain the generator and the discriminator
    G = generator(opts)

    # Define optimizers
    # g_opt = keras.optimizers.Adam(lr=opts['g_lr'])
    g_opt = tf.optimizers.Adam(lr=opts['g_lr'])

    #G_wav = G(wav_in_noisy)
    #G = Model(wav_in_noisy, G_wav)
    G.summary()
  
    # compile individual models
    G.compile(loss='mean_absolute_error', optimizer=g_opt)

    if TEST_SEGAN:
        ftestnoisy = h5py.File(noisy_test_matfile)
        noisy_test_data = ftestnoisy['feat_data']
        noisy_test_dfi = ftestnoisy['dfi']
        print ("Number of test files: " +  str(noisy_test_dfi.shape[1]) )

    # Begin the training part
    if TRAIN_SEGAN:   
        fclean = h5py.File(clean_train_matfile)
        clean_train_data = np.array(fclean['feat_data']).astype('float32')
        fnoisy = h5py.File(noisy_train_matfile)
        noisy_train_data = np.array(fnoisy['feat_data']).astype('float32')
        numtrainsamples = clean_train_data.shape[1]
        idx_all = np.arange(numtrainsamples)
        # set random seed
        np.random.seed(opts['random_seed'])
        batch_size = opts['batch_size']

        print ("********************************************")
        print ("               SEGAN TRAINING               ")
        print ("********************************************")
        print ("Shape of clean feats mat " + str(clean_train_data.shape))
        print ("Shape of noisy feats mat " + str(noisy_train_data.shape))
        numtrainsamples = clean_train_data.shape[1]

        # Tensorboard stuff
        log_path = './logs/' + modeldir
        callback = TensorBoard(log_path)
        callback.set_model(G)
        train_names = ['G_loss']

        idx_all = np.arange(numtrainsamples)
        # set random seed
        np.random.seed(opts['random_seed'])

        batch_size = opts['batch_size']
        num_batches_per_epoch = int(np.floor(clean_train_data.shape[1]/batch_size))
        for epoch in range(n_epochs):
            # train D with  minibatch
            np.random.shuffle(idx_all) # shuffle the indices for the next epoch
            for batch_idx in range(num_batches_per_epoch):
                start_time = time.time()
                idx_beg = batch_idx * batch_size
                idx_end = idx_beg + batch_size
                idx = np.sort(np.array(idx_all[idx_beg:idx_end]))
                #print ("Batch idx " + str(idx[:5]) +" ... " + str(idx[-5:]))
                cleanwavs = np.array(clean_train_data[:,idx]).T
                cleanwavs = data_preprocess(cleanwavs, preemph=opts['preemph'])
                cleanwavs = np.expand_dims(cleanwavs, axis = 2)
                noisywavs = np.array(noisy_train_data[:,idx]).T
                noisywavs = data_preprocess(noisywavs, preemph=opts['preemph'])
                noisywavs = np.expand_dims(noisywavs, axis = 2)
                #np.set_printoptions(threshold=np.inf,precision=3)
                #print(noisywavs[80,:,0])
                #sys.exit()

                g_loss = G.train_on_batch(noisywavs, cleanwavs)
             
                time_taken = time.time() - start_time

                printlog = "E%d/%d:B%d/%d [G loss: %f] [Exec. time: %f]" %  (epoch, n_epochs, batch_idx, num_batches_per_epoch, g_loss, time_taken)
        
                print (printlog)
                # Tensorboard stuff 
                logs = [g_loss]
                # write_log(callback, train_names, logs, epoch)

            if epoch % 10 == 0:
                model_json = G.to_json()
                with open(modeldir + "/Gmodel.json", "w") as json_file:
                    json_file.write(model_json)
                G.save_weights(modeldir + "/Gmodel.h5")
                G.save(modeldir + "/Gmodel_full.h5")
                print ("Model saved to " + modeldir)

            if (TEST_SEGAN and epoch == n_epochs - 1): #if (TEST_SEGAN and epoch % 10 == 0): #or epoch == n_epochs - 1:
                print ("********************************************")
                print ("               SEGAN TESTING                ")
                print ("********************************************")

                resultsdir = modeldir + "/test_results_epoch" + str(epoch)
                if not os.path.exists(resultsdir):
                    os.makedirs(resultsdir)

                if LOAD_SAVED_MODEL:
                    print ("Loading model from " + modeldir + "/Gmodel")
                    json_file = open(modeldir + "/Gmodel.json", "r")
                    loaded_model_json = json_file.read()
                    json_file.close()
                    G_loaded = model_from_json(loaded_model_json)
                    G_loaded.compile(loss='mean_squared_error', optimizer=g_opt)
                    G_loaded.load_weights(modeldir + "/Gmodel.h5")
                else:
                    G_loaded = G

                print ("Saving Results to " + resultsdir)

                for test_num in tqdm(range(noisy_test_dfi.shape[1])) :
                    test_beg = noisy_test_dfi[0, test_num]
                    test_end = noisy_test_dfi[1, test_num]
                    #print ("Reading indices " + str(test_beg) + " to " + str(test_end))
                    noisywavs = np.array(noisy_test_data[:,test_beg:test_end]).T
                    noisywavs = data_preprocess(noisywavs, preemph=opts['preemph'])
                    noisywavs = np.expand_dims(noisywavs, axis = 2)
                    if not opts['z_off']:
                        noiseinput = np.random.normal(0, 1, (noisywavs.shape[0], z_dim1, z_dim2))
                        cleaned_wavs = G_loaded.predict([noisywavs, noiseinput])
                    else :
                        cleaned_wavs = G_loaded.predict(noisywavs)
          
                    cleaned_wavs = np.reshape(cleaned_wavs, (noisywavs.shape[0], noisywavs.shape[1]))
                    cleanwav = reconstruct_wav(cleaned_wavs)
                    cleanwav = np.reshape(cleanwav, (-1,)) # make it to 1d by dropping the extra dimension
                    
                    if opts['preemph'] > 0:
                        cleanwav = de_emph(cleanwav, coeff=opts['preemph'])

                    destfilename = resultsdir +  "/testwav_%d.wav" % (test_num)
                    wavfile.write(destfilename, fs, cleanwav)



        # Finally, save the model
        if SAVE_MODEL:
            model_json = G.to_json()
            with open(modeldir + "/Gmodel.json", "w") as json_file:
                json_file.write(model_json)
            G.save_weights(modeldir + "/Gmodel.h5")
            G.save(modeldir + "/Gmodel_full.h5")
            print ("Model saved to " + modeldir)
