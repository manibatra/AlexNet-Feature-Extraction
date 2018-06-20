import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time

from alexnet import AlexNet

nb_classes = 43
epochs = 10
batch_size = 128

# TODO: Load traffic signs data.
training_file = "train.p"
with open(training_file, mode='rb') as f:
	data = pickle.load(f)

X_features, y_labels = data['features'], data['labels']

# TODO: Split data into training and validation sets.

X_train, X_valid, y_train, y_valid = train_test_split(X_features, y_labels, test_size=0.33, random_state=42)


# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, (None))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
w1 = tf.Variable(tf.truncated_normal(shape, 0, 0.1))
b1 = tf.Variable(tf.zeros([nb_classes]))
fc = tf.nn.xw_plus_b(fc7, w1, b1)
probs = tf.nn.softmax(fc)

# TODO: Define loss, training, accuracy operations.
rate = 0.001
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=probs, labels=y))
optmizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss_op, var_list=[w1, b1])

correct_prediction = tf.equal(tf.argmax(probs, 1), y)
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

def evaluate(X_data, y_data):
	num_examples = len(X_data)
	total_accuracy = 0
	total_loss = 0
	sess = tf.get_default_session()
	for offset in range(0, num_examples, batch_size):
		end = offset + batch_size
		batch_x, batch_y = X_data[offset:end], y_data[offset:end]
		loss, accuracy = sess.run(accuracy_op, feed_dict = {x: batch_x,
													  y: batch_y})
		total_accuracy += (accuracy * len(batch_x))
		total_loss += (loss * len(batch_x))
	return total_accuracy / num_examples, total_loss / num_examples

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	num_examples = len(X_train)

	for epoch in range(epochs):
		X_train, y_train = shuffle(X_train, y_train)
		t0 = time.time()

		for offset in range(0, num_examples, batch_size):
			end = offset + batch_size
			batch_x, batch_y = X_train[offset:end], y_train[offset:end]
			sess.run(optmizer, feed_dict={x:batch_x, y:batch_y})
		validation_accuracy, validation_loss = evaluate(X_valid, y_valid)
		print("EPOCH {}.......".format(epoch + 1))
		print("Time: %.3f seconds" % (time.time() - t0))
		print("Validation Loss = {:.3f}".format(validation_loss))
		print("Validation Accuracy = {:.3f}".format(validation_accuracy))
		print()
