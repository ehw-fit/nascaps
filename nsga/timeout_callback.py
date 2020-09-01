

from keras import callbacks
import datetime
from time import time



class TimeoutCallback(callbacks.Callback):
    """Sets timeout for keras training phase. If the timeout is reached, the epoch ends 

    Arguments:
        train_time {int} -- Maximal training time in seconds, set None if not used
    """    
    def __init__(self, train_time):
        super(TimeoutCallback, self).__init__()

        self.train_time = train_time

    def on_train_begin(self, logs = None):
        self.train_start = time()
        print('Training starts at {}'.format(self.train_start))

    def on_epoch_end(self, epoch, logs = None):
        if self.train_time:
            remains = self.train_time - (time() - self.train_start) 
            if remains <= 0: # limit reached
                print('Training interrupted by TimeoutCallback({}) in epoch {} (ren: {:.1f})'.format(self.train_time, epoch, remains))
                self.model.stop_training = True
            
            else:
                print('Remaining timeout {:.0f} sec in epoch {}'.format(remains, epoch))
