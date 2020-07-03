import tensorflow as tf

def build_encoder(input_shape, num_layers, num_units):
	slice_input = tf.keras.Input(shape = input_shape, dtype = tf.float32)

	# Feature Extraction
	conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation = tf.nn.relu)
	feature = tf.keras.layers.TimeDistributed(conv1)(slice_input)
	conv2 = tf.keras.layers.Conv2D(4, (3, 3), strides = 2)
	feature = tf.keras.layers.TimeDistributed(conv2)(feature)
	flatten = tf.keras.layers.Flatten()
	feature = tf.keras.layers.TimeDistributed(flatten)(feature) # [Batch, T, F]
	output = tf.keras.layers.BatchNormalization()(feature)

	# RNN
	for i in range(num_layers):
		cell = tf.keras.layers.LSTMCell(num_units)
		rnn = tf.keras.layers.RNN(cell, return_sequences = True)
		output = rnn(output)
		output = tf.keras.layers.LayerNormalization()(output)

	return tf.keras.Model(inputs = [slice_input], outputs = [output], name = 'encoder')

def build_predictor(vocab_size, embed_size, num_layers, num_units):
	label_input = tf.keras.Input(shape = [None], dtype = tf.int64)

	# Label to Embedding
	embed = tf.keras.layers.Embedding(vocab_size, embed_size)(label_input)

	output = embed

	for i in range(num_layers):
		cell = tf.keras.layers.LSTMCell(num_units)
		rnn = tf.keras.layers.RNN(cell, return_sequences = True)
		output = rnn(output)
		output = tf.keras.layers.LayerNormalization()(output)

	return tf.keras.Model(inputs = [label_input], outputs = [output], name = 'predictor')

def build_joint(pred_input_shape, enc_input_shape, vocab_size, hidden_units):
	pred_input = tf.keras.Input(shape = pred_input_shape, dtype = tf.float32)
	enc_input = tf.keras.Input(shape = enc_input_shape, dtype = tf.float32)

	pred_dense = tf.keras.layers.Dense(hidden_units)
	enc_dense = tf.keras.layers.Dense(hidden_units)

	hidden_output = tf.expand_dims(pred_dense(pred_input), axis = 1) + tf.expand_dims(enc_dense(enc_input), axis = 2)
	hidden_output = tf.keras.activations.relu(hidden_output)

	dense_logit = tf.keras.layers.Dense(vocab_size)
	output = dense_logit(hidden_output)

	return tf.keras.Model(inputs = [pred_input, enc_input], outputs = [output], name = 'joint')

def build_transducer(args):
	slice_input_shape = [None, args.slice_width, args.slice_height, args.num_channels]
	joint_pred_input_shape = [None, args.enc_num_units] # [Batch, Time, h_Enc]
	joint_enc_input_shape = [None, args.pred_num_units] # [Batch, Label, h_Pred]

	slice_input = tf.keras.Input(shape = slice_input_shape, dtype = tf.float32, name = 'image_slice')
	label_input = tf.keras.Input(shape = [None], dtype = tf.float32, name = 'label')

	encoder = build_encoder(slice_input_shape, args.enc_num_layers, args.enc_num_units)
	predictor = build_predictor(args.vocab_size, args.embed_size, args.pred_num_layers, args.pred_num_units)
	joint = build_joint(joint_pred_input_shape, joint_enc_input_shape, args.vocab_size, args.joint_num_units)

	h_enc = encoder(slice_input)
	h_pred = predictor(label_input)
	logit = joint([h_pred, h_enc])

	return tf.keras.Model(inputs = [slice_input, label_input], outputs = [logit], name = 'transducer')
