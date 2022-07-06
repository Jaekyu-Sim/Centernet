import tensorflow as tf
import numpy as np

#resnet 18 -> [2, 2, 2, 2] block 갯수

"""def basic_block(_input, _num_of_filter, _stride = 1):
    residual = tf.keras.layers.Lambda(lambda x: _input)
    x = tf.layers.conv2d(inputs = _input, filters = _num_of_filter, kernel_size = (3, 3), padding = "SAME", strides = (_stride, _stride))
    #x = tf.kereas.layers.BatchNormalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs = x, filters = _num_of_filter, kernel_size = (3, 3), padding = "SAME", strides = (_stride, _stride))
    result = tf.nn.relu(tf.keras.layers.Add()([residual, x]))
    
    return result"""

def basic_block(_input, _num_of_filter, _stride):
    if(_stride == 1):
        residual = _input
    else:
        
        residual = tf.layers.conv2d(_input, filters = _num_of_filter, kernel_size = 1, strides = _stride)
        
    x = tf.layers.conv2d(inputs = _input, filters = _num_of_filter, kernel_size = (3, 3), padding = "SAME", strides = (_stride, _stride))
    #x = tf.kereas.layers.BatchNormalization(x)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs = x, filters = _num_of_filter, kernel_size = (3, 3), padding = "SAME", strides = (1, 1))
    result = tf.nn.relu(tf.math.add(residual, x))
    
    return result
    
def make_basic_block_layer1(_input, _num_of_filter, _stride = 1):#block 갯수 2개
    x = basic_block(_input = _input, _num_of_filter = _num_of_filter, _stride = _stride)
    
    x = basic_block(_input = x, _num_of_filter = _num_of_filter, _stride = 1)
    result = basic_block(_input = x, _num_of_filter = _num_of_filter, _stride = 1)
    
    return result
    
def make_basic_block_layer2(_input, _num_of_filter, _stride = 1):#block 갯수 2개
    x = basic_block(_input = _input, _num_of_filter = _num_of_filter, _stride = _stride)
    
    x = basic_block(_input = x, _num_of_filter = _num_of_filter, _stride = 1)
    result = basic_block(_input = x, _num_of_filter = _num_of_filter, _stride = 1)
    
    return result    

def make_basic_block_layer3(_input, _num_of_filter, _stride = 1):#block 갯수 2개
    x = basic_block(_input = _input, _num_of_filter = _num_of_filter, _stride = _stride)
    
    x = basic_block(_input = x, _num_of_filter = _num_of_filter, _stride = 1)
    result = basic_block(_input = x, _num_of_filter = _num_of_filter, _stride = 1)
    
    return result    

def make_basic_block_layer4(_input, _num_of_filter, _stride = 1):#block 갯수 2개
    x = basic_block(_input = _input, _num_of_filter = _num_of_filter, _stride = _stride)
    
    x = basic_block(_input = x, _num_of_filter = _num_of_filter, _stride = 1)
    result = basic_block(_input = x, _num_of_filter = _num_of_filter, _stride = 1)
    
    return result
def make_transposed_conv_layer(_input, _num_of_filter, _num_of_kernel):
    #num_of_layer = 3
    x = tf.keras.layers.Conv2DTranspose(filters = _num_of_filter, kernel_size = _num_of_kernel, strides = 2, padding = "SAME")(_input)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2DTranspose(filters = _num_of_filter, kernel_size = _num_of_kernel, strides = 2, padding = "SAME")(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.Conv2DTranspose(filters = _num_of_filter, kernel_size = _num_of_kernel, strides = 2, padding = "SAME")(x)
    result = tf.nn.relu(x)
    
    return result

def make_heatmap_layer(_input, _num_of_class):
    heatmap = tf.layers.conv2d(inputs = _input, filters=_num_of_class, kernel_size=(1, 1), strides=1, padding="same")
    
    return heatmap