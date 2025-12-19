import numpy as np

class Metrics(object):
    def __init__(self):
        super().__init__()
        self.PAD = 0

    def apk(self, actual, predicted, k=10):
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        return score / min(len(actual), k)

    def ndcg_k(self, actual, predicted, k=10):
        ideal_ranking = [1.0] + [0.0] * (k - 1)
        actual_ranking = []
        for item in predicted[:k]:
            if item == actual:
                actual_ranking.append(1.0)
            else:
                actual_ranking.append(0.0)
        dcg = 0.0
        for i, rel in enumerate(actual_ranking):
            dcg += rel / np.log2(i + 2)  # i+2因为索引从0开始，且log2(i+1)中的i+1
        idcg = 0.0
        for i, rel in enumerate(ideal_ranking):
            idcg += rel / np.log2(i + 2)
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def compute_metric(self, y_prob, y_true, k_list=[10, 20, 50, 100]):
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        scores.update({'ndcg@' + str(k): [] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                scores_len += 1.0
                p_sort = p_.argsort()
                for k in k_list:
                    topk = p_sort[-k:][::-1]
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                    scores['map@' + str(k)].extend([self.apk([y_], topk, k)])
                    scores['ndcg@' + str(k)].extend([self.ndcg_k(y_, topk, k)])

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len