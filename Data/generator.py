from captcha.image import ImageCaptcha
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse
import random
import os

parser = argparse.ArgumentParser(description = 'captcha generator', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--samples', type = int, default = 50000, help = 'Number of Generated Samples')
parser.add_argument('-g', '--group', type = int, default = 1000, help = 'Number of Generated Samples in One TF-Record')
parser.add_argument('--maxwidth', type = int, default = 12, help = 'Maximal Number of Strides')
parser.add_argument('--minwidth', type = int, default = 6, help = 'Minimal Number of Strides')
parser.add_argument('--minlen', type = int, default = 1, help = 'Minimal Number of Chars in One Captcha')
parser.add_argument('--maxlen', type = int, default = 5, help = 'Maximal Number of Chars in One Captcha')
parser.add_argument('-p', '--prefix', type = str, default = 'data', help = 'Prefix of Dataset')
parser.add_argument('-a', '--alphabet', action = 'store_true', help = 'Add to Include Alphabet')

STRIDE = 10
WIDTH = STRIDE * 2

def gen(random_char, min_len, max_len):
	rt = ''
	length = random.randint(min_len, max_len)
	for i in range(length):
		rt += random_char[random.randint(0, len(random_char) - 1)]
	rt = [rt[i] for i in range(len(rt))]
	rt.sort()
	rt = ''.join(rt)
	if len(rt.replace(' ', '')) == 0:
		rt = gen(random_char, min_len, max_len)
	return rt

def time_slice(data, width = None, stride = None):
	width = width or WIDTH
	stride = stride or STRIDE
	t = (data.shape[0] - width) // stride + 1
	res = np.zeros((t, width, data.shape[1], data.shape[2]))
	offset = 0
	for i in range(t):
		res[i, :, :, :] = data[offset:offset+width,:,:]
		offset += stride
	return res, np.array(t, dtype = np.int32)

def get_label(captcha, random_char, max_len):
	res = [0] # prepend blank label
	# res = [] 
	for i in range(len(captcha)):
		res.append(random_char.index(captcha[i]))
	return np.array(res, dtype = np.int32), np.array(len(captcha), dtype = np.int32)

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(data, label, data_len, label_len):
	feature = {
		"data": _float_feature(data.reshape(-1)),
		"label": _int64_feature(label.reshape(-1)),
		"data_len": _int64_feature(data_len.reshape(-1)),
		"label_len": _int64_feature(label_len.reshape(-1))
	}
	example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return example_proto.SerializeToString()

args = parser.parse_args()

random_char = [' '] + [str(i) for i in range(10)]
if args.alphabet:
	lower = 'abcdefghijklmnopqrstuvwxyz'
	random_char.extend([lower[i] for i in range(len(lower))])
	upper = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
	random_char.extend([upper[i] for i in range(len(upper))])

fonts = [os.path.join('Fonts', x) for x in os.listdir('Fonts')]

for i in range(args.samples):
	captcha = gen(random_char, args.minlen, args.maxlen)

	max_width = args.maxwidth
	min_width = max(len(captcha) * 2, args.minwidth)
	width = random.randint(min_width, max_width) * STRIDE

	image = ImageCaptcha(fonts = fonts, width = width, height = 32, font_sizes = (24, 26, 28))
	data = image.generate(captcha)
	filename = args.prefix + '_' + str(i).zfill(6) + '_' + captcha.replace(' ', '') + '.png'
	image.write(captcha, os.path.join('Samples', filename))

record_cnt = 0
writer = None
group_cnt = 0 

filenames = os.listdir('Samples')

for i in range(len(filenames)):
	if not filenames[i].startswith(args.prefix):
		continue
	captcha = filenames[i].rstrip('.png').split('_')[-1]
	raw = Image.open(os.path.join('Samples', filenames[i]))
	raw = np.swapaxes(raw, 0, 1)
	raw = (raw - 127.5) / 127.5

	data, data_len = time_slice(raw)
	label, label_len = get_label(captcha, random_char, args.maxlen)
	example = serialize_example(data, label, data_len, label_len)

	if writer is None:
		writer = tf.io.TFRecordWriter('TFRecords/%s_%s.tfrec' % (args.prefix, str(record_cnt + 1).zfill(4)))
	writer.write(example)
	group_cnt += 1
	if group_cnt >= args.group:
		writer.close()
		record_cnt += 1
		writer = None
		group_cnt = 0

if group_cnt > 0:
	writer.close()
