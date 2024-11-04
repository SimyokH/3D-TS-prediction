# -*- coding: utf-8 -*-
"""
Created on Thu May 16 10:03:42 2024

@author: Administrator
"""
import numpy as np
from tqdm import tqdm
# from sklearn.model_selection import train_test_split 
import torch
import torch.nn as nn
# from torch.optim import SGD,Adam
import torch.utils.data as Data

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.0)
    
    def forward(self, in_data):
        # in_data: [B, D, N]
        out = self.act(self.fc1(in_data))
        out = self.fc2(self.drop(out))
        return out + in_data


class TS_3D(nn.Module):
    def __init__(self):
        super(TS_3D, self).__init__()
        # Spatial feature
        self.num_nodes = 5086
        self.if_spatial = False
        self.node_dim = 8
        
        # Time feature
        self.if_day_in_year = False
        self.day_in_year_size = 366
        self.day_in_year_dim = 2
        
        self.if_month_in_year = False
        self.month_in_year_size = 12
        self.month_in_year_dim = 2
        
        #-----
        self.input_dim = 1     #[ts,dim,miy]
        self.sigma_layer = 3
        self.in_window = 12
        self.pre_window = 12
        self.num_layer = 3
        
        # Additional items
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.drop_out_rate = 0
        
        self.fc1 = nn.Linear(self.in_window*self.input_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64,32)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=self.drop_out_rate)
        
        #spatial embedding
        self.node_emb_sigma1 = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        self.node_emb_sigma2 = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        self.node_emb_sigma3 = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        
        # self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial)
        # MLP
        self.encoder = nn.Sequential(
            *[MLP(32, 32) for _ in range(self.num_layer)]
            )
        
        # # final-regression
        self.regression_layer = nn.Linear(32,self.pre_window, bias=True)


    def forward(self, in_data):
        #check
        # print(in_data.shape)
        # in_data = in_data.transpose(1,3)
        
        # in_data = in_data.unsqueeze(1)
        # in_data = in_data.repeat(1,self.num_nodes,1)
            
        #--------------embedding layer-------------------------------------------------------------------------------#
        nodes_indx = torch.Tensor([list(range(self.num_nodes)) for _ in range(len(in_data))]).long().to(self.device) 
        # print(nodes_indx.shape)
        # nodes_indx = nodes_indx.repeat(3,1,1)
        # print(nodes_indx.shape)
        # nodes_indx = nodes_indx.permute(1,0,2)
        # print(nodes_indx.shape)
        
        node_emb_sigma1 = []
        if self.if_spatial:
            node_emb_sigma1.append(self.node_emb_sigma1(nodes_indx)) # torch.Size([64, 5086, 8])
            # node_emb = self.node_emb(nodes_indx)
            pass
        
        node_emb_sigma2 = []
        if self.if_spatial:
            node_emb_sigma2.append(self.node_emb_sigma2(nodes_indx)) # torch.Size([64, 5086, 8])
            # node_emb = self.node_emb(nodes_indx)
            pass
        
        node_emb_sigma3 = []
        if self.if_spatial:
            node_emb_sigma3.append(self.node_emb_sigma3(nodes_indx)) # torch.Size([64, 5086, 8])
            # node_emb = self.node_emb(nodes_indx)
            pass
        
        
        # vanilla net
        # print(in_data.shape)
        # batch_size,sigma_layer,num_nodes,_ = in_data.shape
        # tsdata = in_data.view(batch_size,sigma_layer,num_nodes,-1)
        # print(tsdata.shape)
        x = self.act(self.fc1(in_data))
        x = self.act(self.fc2(self.drop(x)))
        x = self.act(self.fc3(self.drop(x)))
        # x = x.squeeze(2)
        
        # sigma_1 = torch.cat([x[:,0,...]]+node_emb_sigma1,dim=2)
        # sigma_1 = torch.unsqueeze(sigma_1, dim=1)
        # sigma_2 = torch.cat([x[:,1,...]]+node_emb_sigma2,dim=2)
        # sigma_2 = torch.unsqueeze(sigma_2, dim=1)
        # sigma_3 = torch.cat([x[:,2,...]]+node_emb_sigma3,dim=2)
        # sigma_3 = torch.unsqueeze(sigma_3, dim=1)
        
        
        # hidden_with_emb = torch.cat([sigma_1]+[sigma_2]+[sigma_3],dim=1)
        encoder_ = self.encoder(x)
        # print('hidden_with_emb=',hidden_with_emb.shape)
        outputfinal = self.regression_layer(encoder_)
        # print('outputshape=',outputfinal.shape)
        # output = outputfinal.squeeze(2)
        # print('xshape=',x.shape)
        return outputfinal
    
    
#--------------------------------------------DATA AQURI & prepossing----------------------------------------------#
dataset_name = '12-12'
Xtrain = np.load('dataset/'+dataset_name+'/Xtrain.npy')
Ytrain = np.load('dataset/'+dataset_name+'/Ytrain.npy')
Xval = np.load('dataset/'+dataset_name+'/Xval.npy')
Yval = np.load('dataset/'+dataset_name+'/Yval.npy')
Xtest = np.load('dataset/'+dataset_name+'/Xtest.npy')
Ytest = np.load('dataset/'+dataset_name+'/Ytest.npy')

X_train=torch.from_numpy(Xtrain.astype(np.float32))
y_train=torch.from_numpy(Ytrain.astype(np.float32))

X_val=torch.from_numpy(Xval.astype(np.float32))
y_val=torch.from_numpy(Yval.astype(np.float32))

X_test=torch.from_numpy(Xtest.astype(np.float32))
y_test=torch.from_numpy(Ytest.astype(np.float32))

train_data=Data.TensorDataset(X_train,y_train)
train_loader=Data.DataLoader(dataset=train_data,batch_size=64,shuffle=False)
val_data=Data.TensorDataset(X_val,y_val)
val_loader=Data.DataLoader(dataset=val_data,batch_size=32,shuffle=False)

test_data=Data.TensorDataset(X_test,y_test)
test_loader=Data.DataLoader(dataset=test_data,batch_size=32,shuffle=False)

#--------------------------------------------TRAINING----------------------------------------------#  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TS_3D()
model.to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 50
for epoch in range(num_epochs):
    loop = tqdm(enumerate(train_loader), total =len(train_loader))
    loop.set_description(f'[ Epoch {epoch+1}/{num_epochs} ]')
    # forward
    train_loss=0
    for step,(batch_x,batch_y) in loop:
        batch_x.to(device)
        batch_y.to(device)
        output=model(batch_x)
        loss=criterion(output,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix({'MAEloss': f'{loss:.4f}'})
        loop.update()
        pass
    # test
    totaltestloss = 0 
    with torch.no_grad():
        for step,(batchx,batchy) in enumerate(val_loader):
            outputtest = model(batchx)
            # testloss 
            testloss_batch = criterion(outputtest,batchy)
            totaltestloss += testloss_batch.data.item()
            pass
        pass
    # if epoch % 10 == 0:
    #   torch.save(model,'checkpoint/diy_3year/'+'%d.pt'%epoch)

    print('IN Validating--Epoch:{},Val Loss:{:.4}'.format(epoch, totaltestloss/len(val_loader)))
    
    pass
#------------------final test----------------------------#
prediction = []
real = []
total_ = 0
with torch.no_grad():
    for step,(batchx,batchy) in enumerate(test_loader):
        output_ = model(batchx)
        testloss_batch_ = criterion(output_,batchy)
        prediction.append(output_)
        real.append(batchy)
        total_ += testloss_batch_.data.item()
        pass
    pass
print('Final Testing--Test Loss:{:.4}'.format(total_/len(test_loader)))
#------------------save the last epoch-------------------#
# prediction = torch.cat(prediction, dim=0)
# real = torch.cat(real, dim=0)
# predictionnp = prediction.cpu().numpy()
# np.save("prediction.npy", predictionnp)
# realnp = real.cpu().numpy()
# np.save("real.npy", realnp)

#-----------------------------------------------MODEL SAVING $ FEATURE VISULIZATION---------------------#

# torch.save(model, 'checkpoint/model3Dspatial.pt')