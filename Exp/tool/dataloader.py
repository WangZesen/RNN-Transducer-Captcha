import tensorflow as tf
import os

def get_dataset(data_dir, prefix, batch_size):
	# Collect Filenames of Data Files
    filenames = os.listdir(data_dir)
    data_files = [os.path.join(data_dir, x) for x in filenames if x.startswith(prefix) and x.endswith('.tfrec')]
    
    def parse_fn(example):
        example_struct = {
            "data": tf.io.VarLenFeature(dtype = tf.float32),
            "label": tf.io.VarLenFeature(dtype = tf.int64),
            "data_len": tf.io.FixedLenFeature(shape = (), dtype = tf.int64),
            "label_len": tf.io.FixedLenFeature(shape = (), dtype = tf.int64)
        }
        raw_data = tf.io.parse_single_example(example, example_struct) 
        data = [tf.reshape(tf.sparse.to_dense(raw_data["data"]), [-1, 20, 32, 3]), # Reshape For Feeding to CNN
            tf.cast(tf.sparse.to_dense(raw_data["label"]), tf.int32),
            tf.cast(raw_data["data_len"], tf.int32),
            tf.cast(raw_data["label_len"], tf.int32)]
        return data

    # Read Dataset
    dataset = tf.data.TFRecordDataset(filenames = data_files)
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(map_func = parse_fn, num_parallel_calls = 4) # process data with 4 processes
    dataset = dataset.padded_batch(batch_size = batch_size)
    return dataset