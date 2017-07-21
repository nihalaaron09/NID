
import tensorflow as tf
import numpy as np


RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    #yhat = tf.nn.sigmoid(tf.matmul(h, w_2) )
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def read_csv(fname):
    with open(fname) as fin:
        lines = fin.readlines();
    keys = lines[0].strip().split(',')
    data = []
    for l in lines[1:]:
        data.append(dict(zip(keys, l.strip().split(','))))
    return data


def One_Hot(y_val):


    new=[]
    for val in y_val:
	if val==[0.0]:
		new.append([1,0])
	else:
		new.append([0,1])
    return new

def load_data(label_dict):
    x = []
    y = []
    x_keys = ['FEAT_1','FEAT_2','FEAT_3','FEAT_4','FEAT_5','FEAT_6']
    y_keys = ['LABEL'];
    for item in label_dict:
        x.append([float(item[k]) for k in x_keys])
        y.append([float(item[k]) for k in y_keys])
   # y=One_Hot(y)
    
    return (x,y)


#train_labels_filename = '/cfarhomes/nihal09/Documents/KDDlabels/Py_Extracted_train_full.txt'
train_labels_filename = '/cfarhomes/nihal09/Documents/KDDlabels/Extracted_KDD_train_full.txt'
train_labels_dict = read_csv(train_labels_filename);


#test_labels_filename = '/cfarhomes/nihal09/Documents/KDDlabels/Py_Extracted_test_full.txt'
test_labels_filename = '/cfarhomes/nihal09/Documents/KDDlabels/Extracted_KDD_test_full.txt'
test_labels_dict = read_csv(test_labels_filename);

train_X, train_y = load_data(train_labels_dict);
test_X, test_y = load_data(test_labels_dict);



# Layer's sizes
x_size = len(train_X[0])   # Number of input nodes: 4 features and 1 bias
h_size = 8               # Number of hidden nodes
y_size = len(train_y[0])   # Number of outcomes (3 iris flowers)


# Symbols
X = tf.placeholder("float", shape=[None, x_size])
y = tf.placeholder("float", shape=[ None,y_size])   # shape=[None,y_size]


# Weight initializations
w_1 = init_weights((x_size, h_size))
w_2 = init_weights((h_size, y_size))

    
# Forward propagation
yhat    = forwardprop(X, w_1, w_2)
#predict = tf.argmax(yhat, axis=1)


#tester-------
constant_val=tf.constant(0.5)
predict=tf.cast(tf.less(constant_val,yhat), tf.float32  )
#predict=tf.less(yhat,constant_val)
#-------


# Backward propagation 
cost=tf.squared_difference(yhat,y)
#cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
updates = tf.train.GradientDescentOptimizer(0.000001).minimize(cost)


# accuracy test
#accuracy=  tf.reduce_mean(tf.cast(tf.equal(predict, y), tf.float32)) 

# Run SGD
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for epoch in range(20):
	for i in range(len(train_X)):
		sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
		#sess.run(yhat, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
		#sess.run(cost, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]}) 
		#sess.run(predict, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
	train_accuracy = np.mean(train_y == sess.run(predict, feed_dict={X:train_X, y: train_y}))
	test_accuracy  = np.mean(test_y == sess.run(predict, feed_dict={X:test_X, y: test_y}))
	print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"% (epoch + 1, 100. * train_accuracy, 100. *test_accuracy))





#for epoch in range(20):
    # Train with each example
 #   for i in range(len(train_X)):
  #      sess.run( updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]} )         
   # train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X:train_X, y: train_y}))
 #   test_accuracy  = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X:test_X, y: test_y}))
  #  print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"% (epoch + 1, 100. * train_accuracy, 100. *test_accuracy))



	#print sess.run(yhat, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
#print sess.run(predict, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
#print sess.run(y, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})
#print sess.run(cost, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]}) 


#sess.close()

