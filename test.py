import tensorflow as tf


def variable_exp():
    with tf.variable_scope("foo") as scope:
        v=tf.get_variable("v",[1])
    with tf.variable_scope("bar"):
        v1=tf.get_variable("v1",[1])
    return v,v1
v,v1=variable_exp()
print(v.name)
print(v1.name)
with tf.variable_scope("foo",reuse=True):
    k=tf.get_variable("v",[1])
    print(k.name)
