import os
import numpy as np
import torch
import torch.nn as nn




def train_model(model, loader, val_loader,test_loader, criterion, metric, optimizer,
                num_epoch, device='cpu', scheduler=None):
    best_score = None
    loss_history = []
    val_loss_history = []
    val_score_history = []
    test_loss_history = []
    test_score_history = []


    for epoch in range(num_epoch):
        model.train()  # enter train mode
        loss_accum = 0
        count = 0
        for x, y in loader:
            logits = model(x.to(device))
            loss = criterion(logits, y.to(device))
            # optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss accumulation
            count += 1
            loss_accum += loss
        loss_history.append(float(loss_accum / count))  # average loss over epoch

        model.eval()  # enter evaluation mode
        with torch.no_grad():
            loss_accum = 0
            score_accum = 0
            count = 0
            for x, y in val_loader:
                logits = model(x.to(device))
                y = y.to(device)
                count += 1
                loss_accum += criterion(logits, y)
                score_accum += metric(logits, y)
            val_loss_history.append(float(loss_accum / count))
            val_score_history.append(float(score_accum / count))

            if best_score is None or best_score < np.mean(val_score_history[-1]):
                best_score = np.mean(val_score_history[-1])
                torch.save(model.state_dict(), os.path.join('data', model.__class__.__name__))  # save best model

            if scheduler:
                scheduler.step()  # make scheduler step
            for x, y in test_loader:
                logits = model(x.to(device))
                y = y.to(device)
                count += 1
                loss_accum += criterion(logits, y)
                score_accum += metric(logits, y)
            test_loss_history.append(float(loss_accum / count))
            test_score_history.append(float(score_accum / count))


            print('Epoch #{}, train loss: {:.4f}, val loss: {:.4f}, {}: {:.4f}'.format(
                epoch,
                loss_history[-1],
                val_loss_history[-1],
                metric.__name__,
                val_score_history[-1]
            ))

    return loss_history, val_loss_history, val_score_history,test_loss_history
def accuracy(logits, y_true):
    '''
    logits: torch.tensor on the device, output of the model
    y_true: torch.tensor on the device
    '''
    _, indices = torch.max(logits, 1)
    correct_samples = torch.sum(indices == y_true)
    total_samples = y_true.shape[0]
    return float(correct_samples) / total_samples


def compute_accuracy(model, loader, device):
    model.eval()
    score_accum = 0
    count = 0
    for x, y in loader:
        logits = model(x.to(device))
        count += 1
        score_accum += accuracy(logits, y.to(device))
    return float(score_accum / count)



