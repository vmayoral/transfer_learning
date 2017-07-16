"""
Retrain (continue training) an existing model
    the idea is to load a model that's doing bad and
    analyze how further training affects the model itself
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
run_number = 8 # does good
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
# cross_entropy = graph.get_tensor_by_name("cross_entropy:0")

cross_entropy = -tf.reduce_sum(outputs*tf.log(y), name="cross_entropy")
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
tf.summary.scalar("cross_entropy", cross_entropy)

# Train
merged = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/'+str(run_number)+'/', sess.graph)
    sess.run(tf.initialize_all_variables())
    # writer = tf.train.SummaryWriter("logs/", graph=tf.get_default_graph())

    for step in range(5000,15000):
        feed_dict={inputs: x_, outputs:expect } # feed the net with our inputs and desired outputs.
        e,a,summary=sess.run([cross_entropy, train_step, merged],feed_dict)
        # if e<1:break # early stopping yay
        print("step %d : entropy %s" % (step,e)) # error/loss should decrease over time
        # write log
        writer.add_summary(summary, step)

    saver.save(sess, "checkpoints/"+str(run_number)+"/xor.ckpt") # save data

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(outputs,1)) # argmax along dim-1
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # [True, False, True, True] -> [1,0,1,1] -> 0.75.

    print("accuracy %s"%(accuracy.eval({inputs: x_, outputs: expect})))

    learned_output=tf.argmax(y,1)
    print(learned_output.eval({inputs: x_}))
