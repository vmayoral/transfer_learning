"""
Repurpose (add layers) and train for another problem
"""
import tensorflow as tf

# --------------
# # XOR definitions
# # Desired input output mapping of XOR function:
# x_ = [[0, 0], [0, 1], [1, 0], [1, 1]] # input
# #labels=[0,      1,      1,      0]   # output =>
# expect=[[1,0],  [0,1],  [0,1], [1,0]] # ONE HOT REPRESENTATION! 'class' [1,0]==0 [0,1]==1

# AND definitions
# Desired input output mapping of AND function:
x_ = [[0, 0], [0, 1], [1, 0], [1, 1]] # input
#labels=[0,      1,      1,      0]   # output =>
expect=[[1,0],  [1,0],  [1,0], [0,1]] # ONE HOT REPRESENTATION! 'class' [1,0]==0 [0,1]==1
# --------------

# run_number = 9 # does bad
run_number = 8 # does good
sess = tf.Session()

# First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('checkpoints/'+str(run_number)+'/xor.ckpt.meta' )
saver.restore(sess,tf.train.latest_checkpoint('checkpoints/'+str(run_number)+'/'))
number_hidden_nodes = 20 # 20 outputs to create some room for negatives and positives

# restore the graph based on latest checkpoints
graph = tf.get_default_graph()
inputs = graph.get_tensor_by_name("inputs:0")
outputs = graph.get_tensor_by_name("outputs:0")
hidden2 = graph.get_tensor_by_name("layer2/hidden2:0")

# Extend the model, repurpose the third layer and add a fourth layer
# Third layer (repurpose it)
with tf.name_scope('layer3_3'):
    W3_3 = tf.Variable(tf.random_uniform([number_hidden_nodes,number_hidden_nodes], -.1, .1), name='w3_3')
    b3_3 = tf.Variable(tf.random_uniform([number_hidden_nodes], -.01, .01), name='b3_3')
    hidden3_3 = tf.nn.relu(tf.matmul(hidden2, W3_3) + b3_3)
    hidden3_3 = tf.identity(hidden2, name="hidden3_3")

# Fourth layer
with tf.name_scope('layer4'):
    W4 = tf.Variable(tf.random_uniform([number_hidden_nodes,2], -.1, .1), name='w3')
    b4 = tf.Variable(tf.zeros([2]), name='b4')
    hidden4 = tf.matmul(hidden3_3, W4) + b4
    hidden4 = tf.identity(hidden4, name="hidden4")

y_3 = tf.nn.softmax(hidden4, name="y_3")
cross_entropy_3 = -tf.reduce_sum(outputs*tf.log(y_3), name="cross_entropy_3")
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy_3)
tf.summary.scalar("cross_entropy_3", cross_entropy_3)

# Train
merged = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/'+str(run_number)+'_3/', sess.graph)
    sess.run(tf.initialize_all_variables())
    # writer = tf.train.SummaryWriter("logs/", graph=tf.get_default_graph())

    for step in range(5000):
        feed_dict={inputs: x_, outputs:expect } # feed the net with our inputs and desired outputs.
        e,a,summary=sess.run([cross_entropy_3, train_step, merged],feed_dict)
        # if e<1:break # early stopping yay
        print("step %d : entropy %s" % (step,e)) # error/loss should decrease over time
        # write log
        writer.add_summary(summary, step)

    saver.save(sess, "checkpoints/"+str(run_number)+"_3/xor.ckpt") # save data

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y_3,1), tf.argmax(outputs,1)) # argmax along dim-1
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # [True, False, True, True] -> [1,0,1,1] -> 0.75.

    print("accuracy %s"%(accuracy.eval({inputs: x_, outputs: expect})))

    learned_output=tf.argmax(y_3,1)
    print(learned_output.eval({inputs: x_}))
