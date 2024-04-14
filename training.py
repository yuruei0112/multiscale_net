def train_one_epoch(train_data_loader):
    for images, labels in train_data_loader:
        labels = labels.reshape((labels.shape[0], 6))
        optimizer.zero_grad()
        preds =Multiscale_Net(images)
        _loss = criterion(preds, labels)
        _loss.backward()
        optimizer.step()
    for i in range(6):
        epoch_score.append(roc_auc_score(np.array(labels)[:, i], np.array(preds)[:, i]))

    return epoch_score
