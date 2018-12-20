"""
Use New data as both vol and ref input, seg part removed

"""

# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('../ext/medipy-lib')
import medipy
import networks
# import util
from medipy.metrics import dice
import datagenerators

# Test file and anatomical labels we want to evaluate
test_brain_file = open('../data/test_examples_new1.txt') ##
test_brain_strings = test_brain_file.readlines()
test_brain_strings = [x.strip() for x in test_brain_strings]
n_batches = len(test_brain_strings)
# good_labels = sio.loadmat('../data/labels.mat')['labels'][0]

# atlas files
atlas = sio.loadmat('../data/atlas_ref.mat')['at_ref']  # register to another
atlas_vol = atlas[np.newaxis, ..., np.newaxis]
    
# atlas_seg = atlas['seg']

def test(gpu_id, model_dir, iter_num, 
         compute_type = 'GPU',  # GPU or CPU
         vol_size=(160,192,224),
         nf_enc=[16,32,32,32],
         nf_dec=[32,32,32,32,16,3],
         save_file='../data/results.mat'):
    """
    test via segmetnation propagation
    works by iterating over some iamge files, registering them to atlas,
    propagating the warps, then computing Dice with atlas segmentations
    """  

    # GPU handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        # if testing miccai run, should be xy indexing.
        net = networks.miccai2018_net(vol_size, nf_enc, nf_dec, use_miccai_int=False, indexing='ij')  
        net.load_weights(os.path.join(model_dir, str(iter_num) + '.h5'))

        # compose diffeomorphic flow output model
        diff_net = keras.models.Model(net.inputs, net.get_layer('diffflow').output)

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

    # if CPU, prepare grid
    if compute_type == 'CPU':
        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)
    
    # prepare a matrix of dice values
    # dice_vals = np.zeros((len(good_labels), n_batches))
    # import pdb; pdb.set_trace()
    for k in range(n_batches):
        # get data, to revise!!!
        vol_name = test_brain_strings[k] #.split(",")!!
        X_vol = datagenerators.load_example_mat1(vol_name)[0]  ## mat

        # predict transform
        with tf.device(gpu):
            pred = diff_net.predict([X_vol, atlas_vol])

        # Warp segments with flow
        if compute_type == 'CPU':
            flow = pred[0, :, :, :, :]
            warp_seg = util.warp_seg(X_seg, flow, grid=grid, xx=xx, yy=yy, zz=zz)

        else:  # GPU
            # warp_seg = nn_trf_model.predict([X_seg, pred])[0,...,0]
            warp_vol = nn_trf_model.predict([X_vol, pred])[0,...,0] ## ???!!!
        
        # compute Volume Overlap (Dice)
        ## dice_vals[:, k] = dice(warp_seg, atlas_seg, labels=good_labels)
        ## print('%3d %5.3f %5.3f' % (k, np.mean(dice_vals[:, k]), np.mean(np.mean(dice_vals[:, :k+1]))))

        if save_file is not None:
            # sio.savemat(save_file, {'dice_vals': dice_vals, 'labels': good_labels, 'pred': pred, 'warpvol': warp_vol, 'warpseg': warp_seg, 'vol': X_vol, 'seg': X_seg, 'atvol': atlas_vol, 'atseg': atlas_seg})
            sio.savemat(save_file, {'pred': pred, 'warpvol': warp_vol, 'vol': X_vol, 'atvol': atlas_vol})

if __name__ == "__main__":
    """
    assuming the model is model_dir/iter_num.h5
    python test_miccai2018.py gpu_id model_dir iter_num
    """
    test(sys.argv[1], sys.argv[2], sys.argv[3])