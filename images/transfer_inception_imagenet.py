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
