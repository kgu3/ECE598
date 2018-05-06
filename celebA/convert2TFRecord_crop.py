import os
from glob import glob
import tensorflow as tf
import scipy.misc as misc
from skimage import io
import numpy as np
import time
import sys

def get_image(image_path,
              resize_height=-1, resize_width=-1):
    image = io.imread(image_path)
    image = image[50:175, 40:130]
    if resize_height>0 and resize_width>0:
        image = misc.imresize(image,(resize_height,resize_width),'lanczos')
    image = np.array(image).astype(np.float32)
    return image
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

trainFiles = glob(os.path.join("./celebA_data/img_align_celeba", "*.jpg"))
trainFiles = np.array(trainFiles)
print(trainFiles.shape[0])

# Write images to TFRecord file
tfrecords_filename = './celebA_data/trainData_crop_float32_0_255.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
count=0
t1=time.time()
for idx in range(trainFiles.shape[0]):
    currFile = trainFiles[idx]
    img = get_image(currFile,resize_height=64,resize_width=64)
    img_raw = img.tostring()
    feature = {'img_raw': _bytes_feature(img_raw)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    count += 1
    if count%5000 == 0:
        print(count)
        sys.stdout.flush()
writer.close()
t2=time.time()
print('total time = ', t2-t1)