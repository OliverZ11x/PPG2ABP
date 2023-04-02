import torch.nn as nn
import torch

class UNetDS64(nn.Module):
    """
    Deeply supervised U-Net with kernels multiples of 64
    
    Arguments:
        length {int} -- length of the input signal
    
    Keyword Arguments:
        n_channel {int} -- number of channels in the output (default: {1})
    
    Returns:
        PyTorch model -- created model
    """
    
    def __init__(self, length, n_channel=1):
        super(UNetDS64, self).__init__()

        x = 64

        self.inputs = nn.Sequential(nn.Conv1d(n_channel, x, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x),
                                    nn.Conv1d(x, x, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x))

        self.pool1 = nn.Sequential(nn.MaxPool1d(2),
                                    nn.Conv1d(x, x*2, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x*2),
                                    nn.Conv1d(x*2, x*2, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x*2))

        self.pool2 = nn.Sequential(nn.MaxPool1d(2),
                                    nn.Conv1d(x*2, x*4, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x*4),
                                    nn.Conv1d(x*4, x*4, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x*4))

        self.pool3 = nn.Sequential(nn.MaxPool1d(2),
                                    nn.Conv1d(x*4, x*8, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x*8),
                                    nn.Conv1d(x*8, x*8, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x*8))

        self.pool4 = nn.Sequential(nn.MaxPool1d(2),
                                    nn.Conv1d(x*8, x*16, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x*16),
                                    nn.Conv1d(x*16, x*16, 3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(x*16))

        self.up6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv1d(x*16+x*8, x*8, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(x*8),
                                  nn.Conv1d(x*8, x*8, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(x*8))

        self.up7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv1d(x*8+x*4, x*4, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(x*4),
                                  nn.Conv1d(x*4, x*4, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(x*4))

        self.up8 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                  nn.Conv1d(x*4+x*2, x*2, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(x*2),
                                  nn.Conv1d(x*2, x*2, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(x*2))
        
        self.up9 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv1d(x*2+x, x, 3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm1d(x),
                                nn.Conv1d(x, x, 3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm1d(x))

        self.outputs = nn.Sequential(nn.Conv1d(x, n_channel, 1),
                                    nn.Sigmoid())

        self.ds4 = nn.Sequential(nn.Conv1d(x*16, n_channel, 1),
                                    nn.Sigmoid())

        self.ds3 = nn.Sequential(nn.Conv1d(x*8, n_channel, 1),
                                    nn.Sigmoid())

        self.ds2 = nn.Sequential(nn.Conv1d(x*4, n_channel, 1),
                                    nn.Sigmoid())

        self.ds1 = nn.Sequential(nn.Conv1d(x*2, n_channel, 1),
                                    nn.Sigmoid())
        
    def forward(self, x):
        x1 = self.inputs(x)
        x2 = self.pool1(x1)
        x3 = self.pool2(x2)
        x4 = self.pool3(x3)
        x5 = self.pool4(x4)

        x6 = self.up6(torch.cat([x5, x4], dim=1))
        ds4 = self.ds4(x6)

        x7 = self.up7(torch.cat([x6, x3], dim=1))
        ds3 = self.ds3(x7)

        x8 = self.up8(torch.cat([x7, x2], dim=1))
        ds2 = self.ds2(x8)

        x9 = self.up9(torch.cat([x8, x1], dim=1))
        ds1 = self.ds1(x9)

        outputs = self.outputs(x9)

        return outputs, ds1, ds2, ds3, ds4
    
    
    
    



from helper_functions import *
from models import *
import time
from tqdm import tqdm
import pickle
import os
from tensorflow.keras.optimizers import Adam

"""
    Trains the refinement network in 10 fold cross validation manner
"""

model_dict = {}                                             # all the different models
model_dict['UNet'] = UNet
model_dict['UNetLite'] = UNetLite
model_dict['UNetWide40'] = UNetWide40
model_dict['UNetWide48'] = UNetWide48
model_dict['UNetDS64'] = UNetDS64
model_dict['UNetWide64'] = UNetWide64
model_dict['MultiResUNet1D'] = MultiResUNet1D
model_dict['MultiResUNetDS'] = MultiResUNetDS


mdlName1 = 'UNetDS64'                                       # approximation network
mdlName2 = 'MultiResUNet1D'                                 # refinement network

length = 1024                                               # length of the signal

                                                            # 10 fold cross validation
for foldname in range(1):

    print('----------------')
    print('Training Fold {}'.format(foldname+1))
    print('----------------')
                                                                                        # loading training data
    dt = pickle.load(open(os.path.join('data','train{}.p'.format(foldname)),'rb'))
    X_train = dt['X_train']
    Y_train = dt['Y_train']
                                                                                        # loading validation data
    dt = pickle.load(open(os.path.join('data','val{}.p'.format(foldname)),'rb'))
    X_val = dt['X_val']
    Y_val = dt['Y_val']

                                                                                        # loading metadata
    dt = pickle.load(open(os.path.join('data','meta{}.p'.format(foldname)),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']


    Y_train = prepareLabel(Y_train)                                         # prepare labels for training deep supervision
    
    Y_val = prepareLabel(Y_val)                                             # prepare labels for training deep supervision

    mdl1 = model_dict[mdlName1](length)             # create approximation network
    
    # loss = mae, with deep supervision weights
    mdl1.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mean_squared_error'], loss_weights=[1., 0.9, 0.8, 0.7, 0.6])                                                         


    checkpoint1_ = ModelCheckpoint(os.path.join('models','{}_model1_fold{}.h5'.format(mdlName1,foldname)), verbose=1, monitor='val_out_loss',save_best_only=True, mode='auto')  
                                                                    # train approximation network for 100 epochs
    history1 = mdl1.fit(X_train,{'out': Y_train['out'], 'level1': Y_train['level1'], 'level2':Y_train['level2'], 'level3':Y_train['level3'] , 'level4':Y_train['level4']},epochs=100,batch_size=256,validation_data=(X_val,{'out': Y_val['out'], 'level1': Y_val['level1'], 'level2':Y_val['level2'], 'level3':Y_val['level3'] , 'level4':Y_val['level4']}),callbacks=[checkpoint1_],verbose=1)

    pickle.dump(history1.history, open('History/{}_model1_fold{}.p'.format(mdlName1,foldname),'wb'))    # save training history

    mdl1 = None                                             # garbage collection

    time.sleep(10)                                          # pause execution for a while to free the gpu
    
    
    
    
    
# ----------------------------------------------------------------------------


from helper_functions import *
from models import *
import torch
import torch.optim as optim
import time
from tqdm import tqdm
import pickle
import os

"""
    Trains the refinement network in 10 fold cross validation manner
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dict = {}                                             # all the different models
model_dict['UNet'] = UNet
model_dict['UNetLite'] = UNetLite
model_dict['UNetWide40'] = UNetWide40
model_dict['UNetWide48'] = UNetWide48
model_dict['UNetDS64'] = UNetDS64
model_dict['UNetWide64'] = UNetWide64
model_dict['MultiResUNet1D'] = MultiResUNet1D
model_dict['MultiResUNetDS'] = MultiResUNetDS

mdlName1 = 'UNetDS64'                                       # approximation network
mdlName2 = 'MultiResUNet1D'                                 # refinement network

length = 1024                                               # length of the signal

                                                            # 10 fold cross validation
for foldname in range(1):

    print('----------------')
    print('Training Fold {}'.format(foldname+1))
    print('----------------')
                                                                                        # loading training data
    dt = pickle.load(open(os.path.join('data','train{}.p'.format(foldname)),'rb'))
    X_train = dt['X_train']
    Y_train = dt['Y_train']
                                                                                        # loading validation data
    dt = pickle.load(open(os.path.join('data','val{}.p'.format(foldname)),'rb'))
    X_val = dt['X_val']
    Y_val = dt['Y_val']

                                                                                        # loading metadata
    dt = pickle.load(open(os.path.join('data','meta{}.p'.format(foldname)),'rb'))
    max_ppg = dt['max_ppg']
    min_ppg = dt['min_ppg']
    max_abp = dt['max_abp']
    min_abp = dt['min_abp']

    Y_train = prepareLabel(Y_train)                                         # prepare labels for training deep supervision
    
    Y_val = prepareLabel(Y_val)                                             # prepare labels for training deep supervision

    mdl1 = model_dict[mdlName1](length).to(device)    # create approximation network
    
    # loss = mae, with deep supervision weights
    optimizer = optim.Adam(mdl1.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    
    checkpoint1_ = {'val_out_loss': float('inf')}
    best_val_loss = float('inf')
    
    # train approximation network for 100 epochs
    for epoch in range(100):
        mdl1.train()
        total_train_loss = 0
        for i in range(0, len(X_train), 256):
            inputs = torch.tensor(X_train[i:i+256]).to(device)
            targets = {
                'out': torch.tensor(Y_train['out'][i:i+256]).to(device),
                'level1': torch.tensor(Y_train['level1'][i:i+256]).to(device),
                'level2': torch.tensor(Y_train['level2'][i:i+256]).to(device),
                'level3': torch.tensor(Y_train['level3'][i:i+256]).to(device),
                'level4': torch.tensor(Y_train['level4'][i:i+256]).to(device),
            }
            optimizer.zero_grad()
            outputs = mdl1(inputs)      # forward pass
            loss = criterion(outputs['out'], targets['out'])
            loss += 0.2 * criterion(outputs['level1'], targets['level1'])
            loss += 0.1 * criterion(outputs['level2'], targets['level2'])
            loss += 0.05 * criterion(outputs['level3'], targets['level3'])
            loss += 0.025 * criterion(outputs['level4'], targets['level4'])
            total_train_loss += loss.item() * inputs.size(0)
            loss.backward()             # backward pass
            optimizer.step()
            
        avg_train_loss = total_train_loss / len(X_train)
        
        with torch.no_grad():
            mdl1.eval()
            total_val_loss = 0
            for i in range(0, len(X_val), 256):
                inputs = torch.tensor(X_val[i:i+256]).to(device)
                targets = {
                    'out': torch.tensor(Y_val['out'][i:i+256]).to(device),
                    'level1': torch.tensor(Y_val['level1'][i:i+256]).to(device),
                    'level2': torch.tensor(Y_val['level2'][i:i+256]).to(device),
                    'level3': torch.tensor(Y_val['level3'][i:i+256]).to(device),
                    'level4': torch.tensor(Y_val['level4'][i:i+256]).to(device),
                }
                outputs = mdl1(inputs)
                loss = criterion(outputs['out'], targets['out'])
                loss += 0.2 * criterion(outputs['level1'], targets['level1'])
                loss += 0.1 * criterion(outputs['level2'], targets['level2'])
                loss += 0.05 * criterion(outputs['level3'], targets['level3'])
                loss += 0.025 * criterion(outputs['level4'], targets['level4'])
                total_val_loss += loss.item() * inputs.size(0)
                
            avg_val_loss = total_val_loss / len(X_val)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint1_ = {
                    'state_dict': mdl1.state_dict(),
                    'val_out_loss': avg_val_loss,
                    'epoch': epoch,
                }
                
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, 100, avg_train_loss, avg_val_loss))

    # save the best model checkpoint
    checkpoint1_path = os.path.join('checkpoints', 'mdl1_fold{}.pth'.format(foldname))
    torch.save(checkpoint1_, checkpoint1_path)
    print('Approximation Network Training Done.')