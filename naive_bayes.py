'''Naive bayes classifier'''
from sys import argv
import math
import pandas as pd

class NaiveBayes:
    '''Naive bayes class'''

    def __init__(self):
        self.class_prob = []

    def mean(self, dset):
        '''Return mean of an array (a class)'''
        try:
           return sum(dset) / float(len(dset))
        except ZeroDivisionError as e:
            print e
            return 0

    def std(self, dset):
        """Compute standard deviation of an array (a class)"""
        avg = self.mean(dset)
        variance = sum([math.pow(x - avg, 2) for x in dset])
        std = math.sqrt(variance)
        return std



    def summarize(self, dset):
        '''Summarizes a dset into mean and standard
        deviation for each feature (each column)'''
        summarized_ds = [(self.mean(x), self.std(x)) for x in zip(*dset)]
        # delete last row with class values
        del summarized_ds[-1]
        return summarized_ds


    def separate_classes(self, dset, n_classes):
        """Separate the classes into 2D array"""
        separated = [[] for i in range(n_classes)]
        for i in range(len(dset)):
            val = dset[i]
            separated[int(val[-1])].append(val)
        print "separate_classes: " + str(separated)
        return separated


    def summarize_by_class(self, dset, n_classes):
        """Summarizes mean and std deviation by class"""
        final_summarized = []
        sep_dset = self.separate_classes(dset, n_classes)
        for class_dset in sep_dset:
            summarized = self.summarize(class_dset)
            final_summarized.append(summarized)
        return final_summarized


def gaussian_prob_density(datapt):
    '''Returns gaussian probability density given an
    array of [datapoint, average, std deviation]'''
    data, avg, sd = datapt[0], datapt[1], datapt[2]
    lhs = (1 / math.sqrt(2 * math.pi * math.pow(sd, 2)))
    rhs = math.exp(-((math.pow(data - avg, 2) / (2 * math.pow(sd, 2)))))
    pdf = lhs * rhs
    return pdf


def calc_class_prob(summaries, data_pt):
    '''Transform a dset (list of lists, with each
    list being values for a feature'''
    probabilities = {}
    for i in range(len(summaries)):
        class_summary = summaries[i]
        probabilities[i] = 1
        for j in range(len(class_summary)):
            mean, stdev = class_summary[i]
            x_val = data_pt[i]
            stats = [x_val, mean, stdev]
            probabilities[i] *= gaussian_prob_density(stats)
            probabilities.append(transformed)
    # print gaussian_input
    # pool = multiprocessing.Pool(processes=len(gaussian_input))
    # probabilities = pool.map(gaussian_prob_density, gaussian_input)

    return probabilities


def get_backblaze_df(path, features):
    dframe = pd.read_csv(path, usecols=features)
    dframe = dframe.fillna(value=-1)
    dframe = dframe.reindex(columns=features)

    return dframe


def testing():
    '''Testing the NB model written from scratch'''
    nb = NaiveBayes()
    if argv[-1] == "single":
        # test a single datapoint
        dset = [[1, 20, 1], [2, 21, 0], [3, 22, 1], [4, 22, 0]]
        sep = nb.summarize_by_class(dset, 2)
        print "sum by class: " + str(sep)
        pt = [1, 19]
        prob = nb.calc_class_prob(sep, pt)
        print "Probability: " + str(prob)

    elif argv[-1] == "backblaze":
        features = ['smart_5_raw', 'smart_12_raw', 'smart_184_raw',
                    'smart_187_raw', 'smart_197_raw', 'failure']
        dframe = get_backblaze_df("jan-feb-data.csv", features)
        dset = dframe.values
        target = dset[:, -1]
        dset_list = dset.tolist()
        sum_ds = nb.summarize_by_class(dset_list, 2)
        dset_nolabels = dset[:, 0:-1]
        print "summarized: " + str(sum_ds)
        # dset_nolabels = dset_nolabels.tolist()
        y_pred = []
        # print dset_nolabels
        # for row in dset_nolabels:
        # print "testing dset point: " + str(dset_nolabels[0])
        for i in range(len(dset_nolabels)):
            probs = nb.calc_class_prob(sum_ds, dset_nolabels[i])
            if probs[0] > probs[1]:
                y_pred.append(0)
            else:
                y_pred.append(1)
        print "Probabilities: " + str(y_pred)
        mislabeled = []
        for i in range(len(y_pred)):
            if target[i] != y_pred[i]:
                mislabeled.append([target[i], y_pred[i]])
        print "Num mislabeled pts out of total %d points: %d" % (dset.shape[0], len(mislabeled))