# utility functions for Assignment B of CS4125

import numpy as np
import re
from sklearn import metrics
from sklearn.model_selection import KFold

# returns a phototweet_dataset dictionary for the Phototweet dataset
# phototweet_dataset['target'] will contain class IDs for each item in the dataset
# phototweet_dataset['sample_ids'] will contain sample IDs for each item in the dataset
def initialize_phototweet_dataset(data_folder):
    phototweet_dataset = {}
    ground_truth_lines = open(data_folder + 'groundTruth.txt', 'r').readlines()
    
    phototweet_dataset['target_names'] = ['negative', 'positive']
    
    targets = []
    sample_ids = []
    
    
    for line in ground_truth_lines:
    	# ground truth data is formatted as [sample_id][tab][negative/positive].
    	# use regular expression matching to parse the sample id (one or more numbers)
    	# and the negative/positive class (all word tokens following the tab character)
        match = re.match(r'(\d+)\t(\w+)', line)

        # use try/except if exceptions may occur
        try:
        	# class targets need to be numbers. Map negative -> 0, positive -> 1 by a list index lookup
            targets.append(phototweet_dataset['target_names'].index(match.group(2)))
            sample_ids.append(match.group(1))
        except:
        	# if an exception occurs, print the line that caused an issue
            print 'problem with %s' % line
    
    phototweet_dataset['target'] = np.array(targets)
    phototweet_dataset['sample_ids'] = sample_ids
    
    return phototweet_dataset


# perform K-fold cross-validation with training
#
# required input:
# data - N x M numpy array with N the amount of dataset samples, and M the feature dimensionality
# target - N true class labels
# classifier - scikit-learn classifier to be used
# 
# optional input:
# k - number of folds (default 5)
def validate_kfold(data, target, classifier, k=5):
    kf = KFold(n_splits=k, random_state=0)
    fold_index = 1

    average_accuracy = 0.0

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]

        classifier.fit(X_train, y_train)

        print '**** fold %d ****' % fold_index
        print metrics.confusion_matrix(y_test, classifier.predict(X_test))
        print metrics.accuracy_score(y_test, classifier.predict(X_test))
        fold_index += 1
        average_accuracy += 1.0/k * metrics.accuracy_score(y_test, classifier.predict(X_test))

    print '\n**** AVERAGE ACCURACY OVER ALL %d FOLDS ****' % k
    print average_accuracy
    
    return average_accuracy

