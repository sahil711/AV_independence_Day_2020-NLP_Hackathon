from utils import AverageMeter, MetricMeter
import time
import torch
from tqdm import tqdm


class GpuEngine():

    def __init__(self, model, device, optimizer, criterion, schedular, es, log_path, num_epochs, slack_loggger, slack_header, verbose=True):
        '''
        please note that the model here is assumed to be device compatible i.e. no need to run the to(device) command
        '''

        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.schedular = schedular
        self.log_path = log_path
        self.verbose = verbose
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.es = es
        self.slack_loggger = slack_loggger
        self.slack_header = slack_header

    def fit(self, train_dl, valid_dl):
        '''
        valid and train data loaders can be instances the pytorch's dataloader class
        they can be with distributed or without one
        '''

        for i in range(self.num_epochs):

            st = time.time()
            train_loss, train_metric = self.train_one_epoch(train_dl)
            val_loss, val_metric = self.validation(valid_dl)

            if self.schedular is not None:
                self.schedular.step(metrics=val_metric)

            # early stopping condition
            if not self.es.check(val_metric):
                break

            et = time.time()

            if self.verbose:
                mssg = 'Epoch: {} train loss: {} train_metric: {} val_loss: {} val_metric: {} time {}'.format(
                    i, train_loss, train_metric, val_loss, val_metric, round((et-st)/60, 2))
                self.log(mssg)

    def validation(self, data_loader):

        self.model.eval()
        losses = AverageMeter()
        metric = MetricMeter()

        with torch.no_grad():

            for i, data_ in enumerate(tqdm(data_loader, total=len(data_loader))):

                input_ids = data_['input_ids'].to(self.device)
                token_type_ids = data_['token_type_ids'].to(self.device)
                attention_mask = data_['attention_mask'].to(self.device)
                cs = data_['Computer Science'].to(self.device)
                phy = data_['Physics'].to(self.device)
                math = data_['Mathematics'].to(self.device)
                stat = data_['Statistics'].to(self.device)
                bio = data_['Quantitative Biology'].to(self.device)
                fin = data_['Quantitative Finance'].to(self.device)
                n = input_ids.size(0)

                target = [cs, phy, math, stat, bio, fin]

                preds = self.model(input_ids, token_type_ids, attention_mask)
                preds = list(preds.values())

                # make loss a class variable and use the class variable to calculate the loss
                loss = self.criterion(preds, target)

                # upating the loss with the new results
                losses.update(loss.detach().item(), n)
                # updating the metric value with the new sample
                metric.update(target, preds)

        return losses.avg, metric.avg

    def train_one_epoch(self, data_loader):

        self.model.train()

        losses = AverageMeter()
        metric = MetricMeter()

        for i, data_ in enumerate(tqdm(data_loader, total=len(data_loader))):

            input_ids = data_['input_ids'].to(self.device)
            token_type_ids = data_['token_type_ids'].to(self.device)
            attention_mask = data_['attention_mask'].to(self.device)
            cs = data_['Computer Science'].to(self.device)
            phy = data_['Physics'].to(self.device)
            math = data_['Mathematics'].to(self.device)
            stat = data_['Statistics'].to(self.device)
            bio = data_['Quantitative Biology'].to(self.device)
            fin = data_['Quantitative Finance'].to(self.device)
            n = input_ids.size(0)

            target = [cs, phy, math, stat, bio, fin]

            self.optimizer.zero_grad()
            preds = self.model(input_ids, token_type_ids, attention_mask)
            preds = list(preds.values())

            # make loss a class variable and use the class variable to calculate the loss
            loss = self.criterion(preds, target)
            loss.backward()

            self.optimizer.step()

            # upating the loss with the new results
            losses.update(loss.detach().item(), n)
            # updating the metric value with the new sample
            metric.update(target, preds)

        return losses.avg, metric.avg

    def log(self, message):
        if self.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write('{}\n'.format(message))

        self.slack_loggger('{} : '.format(self.slack_header), message)
