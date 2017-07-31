import tensorflow as tf
import layer

emb_in = tf.placeholder(tf.float32, [None, 32] , name="emb_in")
category_in = tf.placeholder(tf.float32, [None, 2] , name="category")
dropout = tf.placeholder(tf.float32)

with tf.variable_scope("layer0") :
  W0 = layer.weight_variable( [32, 32], name=("layer0_weight") )
  b0 = layer.bias_variable( [32], name=("layer0_bias") )
  layer0 = tf.add( tf.matmul(emb_in, W0) , b0 )
  layer0 = tf.tanh( layer0 )
  layer0 = tf.nn.dropout( layer0 , dropout )

with tf.variable_scope("layer1") :
  W1 = layer.weight_variable( [32, 2], name=("layer1_weight") )
  b1 = layer.bias_variable( [2], name=("layer1_bias") )
  layer1 = tf.add( tf.matmul(layer0, W1) , b1 )

category_out = tf.nn.softmax( layer1 )

loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels=category_in, logits=layer1) )
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

correct = tf.reduce_mean( tf.cast( tf.equal( tf.argmax( category_in , 1 ) , tf.argmax( layer1 , 1 ) ) , dtype=tf.float32 ) )

init_new_vars_op = tf.variables_initializer([W0,b0,W1,b1])
