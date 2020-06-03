import numpy as np


class Stats:
    def __init__(self, actual, predicted):
        self.TP = 'TP'
        self.TN = 'TN'
        self.FP = 'FP'
        self.FN = 'FN'
        actual = actual.tolist()
        predicted = predicted.tolist()
        zipped = zip(actual, predicted)

        def map_to_label(actual_label, pred_label):
            if actual_label == 1:
                if pred_label == 1:
                    return self.TP
                if pred_label == 0:
                    return self.FN
            if actual_label == 0:
                if pred_label == 0:
                    return self.TN
                if pred_label == 1:
                    return self.FP

        labeled = list(map(lambda x: map_to_label(x[0], x[1]), zipped))
        self.tp_count = labeled.count(self.TP)
        self.tn_count = labeled.count(self.TN)
        self.fp_count = labeled.count(self.FP)
        self.fn_count = labeled.count(self.FN)

    def recall(self):
        if self.tp_count == 0 and self.fn_count == 0:
            return 0
        return self.tp_count / (self.tp_count + self.fn_count)

    def precision(self):
        if self.tp_count == 0 and self.fp_count == 0:
            return 0
        return self.tp_count / (self.tp_count + self.fp_count)

    def f_measure(self):
        if (self.precision() + self.recall()) == 0:
            return 0
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall())

    def accuracy(self):
        return (self.tp_count + self.tn_count) / (self.tp_count + self.tn_count + self.fp_count + self.fn_count)

    def confusion_matrix(self):
        return np.array([[self.tn_count, self.fn_count], [self.fp_count, self.tp_count]])
