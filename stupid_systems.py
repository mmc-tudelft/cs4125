# Some examples of stupid systems as discussed in the lecture

import numpy as np
import skimage
from skimage import io
import os.path
import sklearn
from sklearn import metrics


# Returns the values of the first ten red pixels on the first row of the image
def get_ten_red_pixels(dataset, data_folder):
    sample_ids = dataset['sample_ids']
    
    # prepopulate a data matrix. It will be N x 10, with N the number of samples in the data.
    data = np.zeros((len(sample_ids), 10))

    for i in range(0, len(sample_ids)):
        # what is this sample called?
        sample = sample_ids[i]
        # find corresponding image file
        image_path = os.path.join(data_folder, 'jpg', sample + '.jpg')
        imdata = skimage.io.imread(image_path)

        # image is M x N x 3 (3 because of R, G, B)
        # we want row 0, the first 10 columns, and the R-value out of the (R, G, B) triples
        # when sub-indexing the array, it may still 'think' it is multidimensional (check imdata[0, 0:10, 0].shape)
        # with flatten we ensure it actually is 1 x 10
        data[i,:] = imdata[0, 0:10, 0].flatten()
        
    return data


# Counts all tokens (aka whitespace-separated strings) in the tweet text file
def count_tokens(dataset, data_folder):
    sample_ids = dataset['sample_ids']
    # prepopulate a data matrix. It will be N x 1, with N the number of samples in the data.
    data = np.zeros((len(sample_ids), 1))
    
    for i in range(0, len(sample_ids)):
        # what is this sample called?
        sample = sample_ids[i]
        # find corresponding text file
        text_path = os.path.join(data_folder, 'txt', sample + '.txt')
        
        f = open(text_path, 'r')
        text_lines = f.readlines()
        
        word_count = 0
        
        for line in text_lines:
            word_count += len(str.split(line))
        
        data[i] = word_count

    return data

# Gives random labels in the range[0, 1] (inclusive)
def give_random_answers(dataset):
	# upper bound of randint is exclusive so we give 2 as second parameter
	return np.random.randint(0, 2, len(dataset['sample_ids']))

# Always return class label 0
def always_say_zero(dataset):
	return np.zeros(len(dataset['sample_ids']))

# Always return class label 1
def always_say_one(dataset):
	return np.ones(len(dataset['sample_ids']))





