import requests
import torch
import pandas as pd
from sklearn.metrics import f1_score


class MetricMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = pd.DataFrame()
        self.y_pred = pd.DataFrame()
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = pd.DataFrame([x.cpu().numpy() for x in y_true]).T
        y_pred = pd.DataFrame([x.cpu().detach().numpy().argmax(axis=1) for x in y_pred]).T
        self.y_true = pd.concat((self.y_true, y_true))
        self.y_pred = pd.concat((self.y_pred, y_pred))
        self.score = f1_score(self.y_true, self.y_pred, average='micro')

    @property
    def avg(self):
        return self.score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SlackWebhook():
    def __init__(self, webhook_url, verbose=False):
        self.webhook_url = webhook_url
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        message, sep = "", " - "
        for a in args:
            message += "{}".format(a) + sep
        for key, val in kwargs.items():
            message += "{}: {}".format(key, val) + sep
        message = message.strip(sep)
        self.send(message)

    def log(self, message):
        if self.verbose:
            print("{}".format(message))

    def send(self, message):
        try:
            res = requests.post(url=self.webhook_url, headers={
                                "Content-type": "application/json"}, json={"text": message})
            self.log(res.status_code)
        except Exception as e:
            self.log(e)


class EarlyStopping(object):
    def __init__(self, patience, higher_is_better, tolerance, save_path, model):
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.best_score = -100000 if self.higher_is_better else 100000
        self.tolerance = tolerance  # score should inc/dec by this much to by pass early stopping
        self.ctr = 0
        self.save_path = save_path
        self.model = model

    def check(self, metric):
        if self.ctr < self.patience:

            if self.higher_is_better:
                if metric >= (self.best_score+self.tolerance):
                    self.best_score = metric
                    self.ctr = 0
                    torch.save(self.model.state_dict(), self.save_path)
                else:
                    self.ctr += 1
            else:
                if metric <= (self.best_score-self.tolerance):
                    self.best_score = metric
                    self.ctr = 0
                    torch.save(self.model.state_dict(), self.save_path)
                else:
                    self.ctr += 1
            return True

        else:
            return False
