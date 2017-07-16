#!/usr/bin/env PYTHONIOENCODING="utf-8" python
"""
A simple neural network learning the XOR function
"""
import tensorflow as tf

# Desired input output mapping of XOR function:
x_ = [[0, 0], [0, 1], [1, 0], [1, 1]] # input
#labels=[0,      1,      1,      0]   # output =>
expect=[[1,0],  [0,1],  [0,1], [1,0]] # ONE HOT REPRESENTATION! 'class' [1,0]==0 [0,1]==1

# # Desired input output mapping of AND function:
# x_ = [[0, 0], [0, 1], [1, 0], [1, 1]] # input
# #labels=[0,      1,      1,      0]   # output =>
# expect=[[1,0],  [1,0],  [1,0], [0,1]] # ONE HOT REPRESENTATION! 'class' [1,0]==0 [0,1]==1

# x = tf.Variable(x_)
x = tf.placeholder("float", [None,2], name='inputs') #  can we feed directly?
y_ = tf.placeholder("float", [None, 2], name='outputs') # two output classes

number_hidden_nodes = 20 # 20 outputs to create some room for negatives and positives

# First layer
with tf.name_scope('layer1'):
    W = tf.Variable(tf.random_uniform([2, number_hidden_nodes], -.01, .01), name='w')
    b = tf.Variable(tf.random_uniform([number_hidden_nodes], -.01, .01), name='b')
    hidden  = tf.nn.relu(tf.matmul(x,W) + b) # first layer.
    hidden = tf.identity(hidden, name="hidden")

# Second layer
with tf.name_scope('layer2'):
    # W2 = tf.Variable(tf.random_uniform([number_hidden_nodes,2], -.1, .1))
    W2 = tf.Variable(tf.random_uniform([number_hidden_nodes,number_hidden_nodes], -.1, .1), name='w2')
    # b2 = tf.Variable(tf.zeros([2]))
    b2 = tf.Variable(tf.random_uniform([number_hidden_nodes], -.01, .01), name='b2')
    hidden2 = tf.nn.relu(tf.matmul(hidden, W2) + b2)
    hidden2 = tf.identity(hidden2, name="hidden2")

# Third layer
with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_uniform([number_hidden_nodes,2], -.1, .1), name='w3')
    b3 = tf.Variable(tf.zeros([2]), name='b3')
    hidden3 = tf.matmul(hidden2, W3) + b3
    hidden3 = tf.identity(hidden3, name="hidden3")

y = tf.nn.softmax(hidden3, name="y")
# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_*tf.log(y), name="cross_entropy")
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

# create a summary for our cost and accuracy
tf.summary.scalar("cross_entropy", cross_entropy)
# tf.scalar_summary("accuracy", accuracy)

# Train
merged = tf.summary.merge_all()
saver = tf.train.Saver()
for i in range(10):
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs/'+str(i)+'/', sess.graph)
        sess.run(tf.initialize_all_variables())
        # writer = tf.train.SummaryWriter("logs/", graph=tf.get_default_graph())

        for step in range(1000):
            feed_dict={x: x_, y_:expect } # feed the net with our inputs and desired outputs.
            e,a, summary=sess.run([cross_entropy, train_step, merged],feed_dict)
            # if e<1:break # early stopping yay
            # print("step %d : entropy %s" % (step,e)) # error/loss should decrease over time
            # write log
            writer.add_summary(summary, step)

        saver.save(sess, "checkpoints/"+str(i)+"/xor.ckpt") # save data

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # argmax along dim-1
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # [True, False, True, True] -> [1,0,1,1] -> 0.75.

        print("accuracy %s"%(accuracy.eval({x: x_, y_: expect})))

        learned_output=tf.argmax(y,1)
        print(learned_output.eval({x: x_}))
