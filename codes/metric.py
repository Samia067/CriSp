import numpy as np

# code from https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code/notebook
class MAP_At_K:
    def __init__(self, k=100):
        self.k = k
        self.pred_count = 0
        self.map = np.zeros(k)

    def apk(self, labels, predictions):
        """
        Computes the average precision at k.
        This function computes the average prescision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
        # assert ( len(np.unique(predictions)) == self.k)
        if not labels:
            return 0.0
        if len(predictions) > self.k:
            predictions = predictions[:self.k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predictions):
            # first condition checks whether it is valid prediction
            # second condition checks if prediction is not repeated
            if p in labels: #  and p not in predictions[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
            ap = score / min(len(labels), i+1)
            self.map[i] += ap

        # ap = score / min(len(labels), self.k)
        # self.aps.append(ap)


    def add(self, predicted, actual):
        """
        Computes the mean average precision at k.
        This function computes the mean average prescision at k between two lists
        of items.
        Parameters
        ----------
        actual : list
                 A list of lists of elements that are to be predicted
                 (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The mean average precision at k over the input lists
        """

        # return np.mean([self.apk(a, p, self.k) for a, p in zip(actual, predicted)])
        self.pred_count += 1
        self.apk(actual, predicted)


    def get_scores(self):
        # return self.aps / self.pred_count
        self.map = self.map / self.pred_count if self.pred_count else [0]*self.k
        auc = np.average(self.map)
        return auc, self.map


class Hit_At_K:
    def __init__(self, k=100):
        self.k = k
        self.hits = np.zeros(k)
        self.pred_count = 0

    def add(self, predictions, labels):
        # true_positive_count = np.cumsum(np.array([1 if prediction in labels else 0 for prediction in predictions]))
        true_positive_count = np.cumsum(np.sum(np.array(predictions)[:, np.newaxis].repeat(len(labels), axis=1) == np.array(list(labels)), axis=1))
        for i in range(len(self.hits)):
            self.hits[i] += true_positive_count[i] > 0 # / (i + 1)
        # if true_positive_count[-1] == 0:
        #     print(labels, 'not found in predictions')
        self.pred_count += 1

    def get_scores(self):
        self.hits = self.hits / self.pred_count  if self.pred_count else [0]*self.k
        auc = np.average(self.hits)
        return auc, self.hits
