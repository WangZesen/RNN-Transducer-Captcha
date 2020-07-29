import matplotlib.pyplot as plt
import numpy as np

def heatmap(probs, label, label_len):
	assert probs.shape[0] == 1
	t = probs.shape[1]
	u = label_len + 1
	image = np.zeros((t + 2, u))
	
	for i in range(t):
		for j in range(u):
			print (np.sum(probs[0, i, j, :]))
			image[i][j] = probs[0, i, j, label[0, j]]

	image[t + 1, :] = 1.
	image[t, :] = 0.

	plt.imshow(image)
	plt.ylabel('time')
	plt.xlabel('label')
	
	plt.colorbar()
	plt.show()
	
