"""
This scripts uses the Inception model trained on ImageNet 2012 Challenge data
set and transfers it to a network used for Global Policy Search (GPS).
"""
import tensorflow as tf
import os
import sys
import tarfile
import urllib.request

# FLAGS = None
model_dir = "/tmp/imagenet"
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def fetch_inception_model():
  """Download and extract model tar file trained on ImageNet."""
  dest_directory = model_dir
  print("destiny directory: "+str(dest_directory))
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def conv2d(img, w, b, name, strides=[1, 1, 1, 1]):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=strides, padding='SAME', name=name), b))

def build_gps_graph():
    """ Builds the NN graph to be used for GPS"""
    nn_input = tf.placeholder(tf.float32, shape=[None, 299*299*3], name='input')
    #nn_input = tf.placeholder(tf.float32, shape=[None, 240*240*3], name='input')
    #y_true = tf.placeholder(tf.float32, [None, dim_output], name='y_true')
    #y_true_cls = tf.argmax(y_true, dimension=1)

    # image goes through 3 convnet layers
    num_filters = [64, 32, 32]

    # Store layers weight & bias
    with tf.variable_scope('conv_params'):
        weights = {
            'wc1': tf.get_variable("wc1", initializer=tf.random_normal([5, 5, 3, num_filters[0]], stddev=0.01)),
            'wc2': tf.get_variable("wc2", initializer=tf.random_normal([5, 5, num_filters[0], num_filters[1]], stddev=0.01)),
            'wc3': tf.get_variable("wc3", initializer=tf.random_normal([5, 5, num_filters[1], num_filters[2]], stddev=0.01)),
        }
        biases = {
            'bc1': tf.get_variable("bc1", initializer=tf.zeros([num_filters[0]], dtype='float')),
            'bc2': tf.get_variable("bc2", initializer=tf.zeros([num_filters[1]], dtype='float')),
            'bc3': tf.get_variable("bc3", initializer=tf.zeros([num_filters[2]], dtype='float')),
        }

    image_input = tf.reshape(nn_input, [-1, 3, 299, 299])
    image_input = tf.transpose(image_input, perm=[0,3,2,1])
    conv_layer_1 = conv2d(img=image_input, w=weights['wc1'], b=biases['bc1'], name="conv1", strides=[1,2,2,1])
    conv_layer_2 = conv2d(img=conv_layer_1, w=weights['wc2'], b=biases['bc2'], name="conv2")
    conv_layer_3 = conv2d(img=conv_layer_2, w=weights['wc3'], b=biases['bc3'], name="conv3")

    print(image_input)
    print(conv_layer_1)


# Fetch the model to use for transferring the knowledge
fetch_inception_model()

# Creates graph from saved GraphDef.
create_graph()

with tf.Session() as sess:
    # Just log everything to debug stuff in tensorboard
    writer = tf.summary.FileWriter('logs/', sess.graph)

    # fetch tensors from the graph
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    conv = sess.graph.get_tensor_by_name('conv:0')

    print(softmax_tensor)
    print(conv)

    # # Runs the softmax tensor by feeding the image_data as input to the graph.
    # predictions = sess.run(softmax_tensor,
    #                        {'DecodeJpeg/contents:0': image_data})
    # predictions = np.squeeze(predictions)

    # Instead of running the prediction, we will add layers here starting from the
    # the "conv" layer which outputs a Tensor of the type:
    #    Tensor("conv:0", shape=(1, 149, 149, 32), dtype=float32)
    # Code that defines the network is available at
    # https://github.com/tensorflow/models/blob/master/inception/inception/slim/inception_model.py#L85-L107
    # from here one can tell that the in put is of the kind 299 x 299 x 3 (while the output 147 x 147 x 32)
    build_gps_graph()
