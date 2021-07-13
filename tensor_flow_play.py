import tensorflow as tf

x = tf.constant('hello world')
sess = tf.compat.v1.Session()
print(sess.run(x))