from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.add_state('tp', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(n_classes), dist_reduce_fx='sum')
    
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        
        for i in range(self.n_classes):
            self.tp[i] += torch.sum((preds == i) & (target == i))
            self.fp[i] += torch.sum((preds == i) & (target != i))
            self.fn[i] += torch.sum((preds != i) & (target == i))

    def compute(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = torch.nan_to_num(f1_score, nan=0.0)
        return f1_score[0]

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError("preds and target do not have the same shape")

        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
