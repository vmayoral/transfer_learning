"""
Restore previous runs
"""
import tensorflow as tf

# --------------
# XOR definitions
# Desired input output mapping of XOR function:
x_ = [[0, 0], [0, 1], [1, 0], [1, 1]] # input
#labels=[0,      1,      1,      0]   # output =>
expect=[[1,0],  [0,1],  [0,1], [1,0]] # ONE HOT REPRESENTATION! 'class' [1,0]==0 [0,1]==1
# --------------

# run_number = 9 # does bad
run_number = 6 # does good
sess = tf.Session()

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('checkpoints/'+str(run_number)+'/xor.ckpt.meta' )
saver.restore(sess,tf.train.latest_checkpoint('checkpoints/'+str(run_number)+'/'))

# Access saved Variables directly
# print(sess.run('layer3/b3:0'))

# restore the graph based on latest checkpoints
graph = tf.get_default_graph()
inputs = graph.get_tensor_by_name("inputs:0")
outputs = graph.get_tensor_by_name("outputs:0")
y = graph.get_tensor_by_name("y:0")
hidden = graph.get_tensor_by_name("layer1/hidden:0")

feed_dict={inputs: x_, outputs:expect } # feed the net with our inputs and desired outputs.
print("inputs: "+ str(x_))
print("expected outputs: " +str(expect))
print(sess.run(y,feed_dict))
print(sess.run(tf.argmax(y,1),feed_dict))
