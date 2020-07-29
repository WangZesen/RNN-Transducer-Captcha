import tensorflow as tf
import warprnnt_tensorflow
import argparse
import numpy as np
from tool import dataloader, visual
from model import transducer

parser = argparse.ArgumentParser(description = 'train RNN transducer', formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser_group = parser.add_argument_group('Data Parameters')
parser_group.add_argument('--data', type = str, default = '../Data/TFRecords', help = 'data directory')
parser_group.add_argument('--train_prefix', type = str, default = 'data', help = 'Prefix of Train Dataset')
parser_group.add_argument('--test_prefix', type = str, default = 'test', help = 'Prefix of Test Dataset')
parser_group.add_argument('--alphabet', action = 'store_true', help = 'Enable If Including Alphabet')
parser_group.add_argument('--slice_width', type = int, default = 20, help = 'Width of Image Slice')
parser_group.add_argument('--slice_height', type = int, default = 32, help = 'Height of Image Slice')
parser_group.add_argument('--num_channels', type = int, default = 3, help = 'Number of Channels of Image')

parser_group = parser.add_argument_group('Model Parameters')
parser_group.add_argument('--enc_num_layers', type = int, default = 2, help = 'Number of RNN Layers in Encoder')
parser_group.add_argument('--enc_num_units', type = int, default = 128, help = 'Dimension of LSTM Cell in RNN Layers of Encoder')
parser_group.add_argument('--pred_num_layers', type = int, default = 2, help = 'Number of RNN Layers in Predictor')
parser_group.add_argument('--pred_num_units', type = int, default = 128, help = 'Dimension of LSTM Cell in RNN Layers of Predictor')
parser_group.add_argument('--joint_num_units', type = int, default = 128, help = 'Dimension of Hidden Layer of Joint Network')
parser_group.add_argument('--embed_size', type = int, default = 64, help = 'Dimension of Embedding')

parser_group = parser.add_argument_group('Training Parameters')
parser_group.add_argument('--batch_size', type = int, default = 128, help = 'Batch Size')
parser_group.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')

@tf.function
def train_step(slice_input, label_input, slice_len, label_len):
	with tf.GradientTape() as tape:
		outputs = model([slice_input, label_input])
		loss = tf.reduce_mean(warprnnt_tensorflow.rnnt_loss(outputs, label_input[:, 1:], slice_len, label_len))
		tf.print(loss)
	gradient = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradient, model.trainable_variables))

@tf.function
def train_epoch(train_ds):
	for batch in train_ds:
		train_step(batch[0], batch[1], batch[2], batch[3])

@tf.function
def test_step(slice_input, label_input, slice_len, label_len):
	outputs = model([slice_input, label_input])
	loss = tf.reduce_mean(warprnnt_tensorflow.rnnt_loss(outputs, label_input[:, 1:], slice_len, label_len))
	tf.print(loss)

@tf.function
def test_epoch(test_ds):
	for batch in test_ds:
		test_step(batch[0], batch[1], batch[2], batch[3])


# Main Starts Here
args = parser.parse_args()

args.vocab_size = 11 # Blank + Digits
if args.alphabet:
	args.vocab_size += 26 * 2 # Lower + Upper

train_ds = dataloader.get_dataset(args.data, args.train_prefix, args.batch_size)
test_ds = dataloader.get_dataset(args.data, args.test_prefix, args.batch_size)
model = transducer.build_transducer(args)
optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)

for batch in test_ds.take(1):
	res = model([batch[0], batch[1]])
	probs = np.array(tf.nn.softmax(res, axis=-1))[0:1]
	label = np.array(batch[1])[0:1]
	label_len = np.array(batch[3][0:1])[0]
	visual.heatmap(probs, label, label_len)

for i in range(2):
	print ('Test!')
	test_epoch(test_ds)
	print ('Train!')
	train_epoch(train_ds)
	for batch in test_ds.take(1):
		res = model([batch[0], batch[1]])
		probs = np.array(tf.nn.softmax(res, axis=-1))[0:1]
		label = np.array(batch[1])[0:1]
		label_len = np.array(batch[3][0:1])[0]
		visual.heatmap(probs, label, label_len)

