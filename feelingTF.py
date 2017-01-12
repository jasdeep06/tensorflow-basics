#simple neuron in TF
import tensorflow as tf
#coz input is constant
x=tf.constant(1.0,name="input")
#coz training step will vary weights to reach the final value
w=tf.Variable(0.8,name="weight")
#y=x*w
y=tf.mul(x,w,name="output")
#coz final value is constant
y_=tf.constant(0.0,name="final_value")
#squared error loss function
loss=tf.pow(y-y_,2,name="loss_function")
#Gradient descent to minimize loss
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#writing summaries
for value in [x,w,y,y_,loss]:
#summary buffer
    tf.scalar_summary(value.op.name,value)

#merges all summaries
summaries=tf.merge_all_summaries()
sess=tf.Session()
summary_writer=tf.train.SummaryWriter('log_simple_stats',sess.graph)

sess.run(tf.initialize_all_variables())

for i in range(100):
    print(sess.run(loss))
    sess.run(train_step)