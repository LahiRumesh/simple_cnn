
class EarlyStopping():
    """
    Early stopping to stop the training when the loss accuracy not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when accuracy is
               not improving
        :param min_delta: minimum difference between new accuracy and old accuracy for
               new accuracy to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False
        
        
    def __call__(self, val_acc,epoch):

        if self.best_acc == None:
            self.best_acc = val_acc
        elif val_acc - self.best_acc > self.min_delta:
            self.best_acc = val_acc
        elif val_acc - self.best_acc < self.min_delta:
            self.counter += 1
            print(f" Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('Early stopping...')
                self.early_stop = True