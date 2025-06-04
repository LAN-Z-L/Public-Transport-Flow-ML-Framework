import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
from torch.nn import Conv2d
from einops.layers.torch import Rearrange, Reduce
from tensorboardX import SummaryWriter

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            #由此可以看出 FeedForward 的输入和输出维度是一致的
            nn.Linear(dim,hidden_dim),
            #激活函数
            nn.GELU(),
            #防止过拟合
            nn.Dropout(dropout),
            #重复上述过程
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x=self.net(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self,dim,token_dim,channel_dim,dropout=0.):
        super().__init__()
        self.row_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b h w -> b w h'),
            FeedForward(dim,token_dim,dropout),
            Rearrange('b w h -> b h w')
 
         )
        self.column_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim,channel_dim,dropout)
        )
    def forward(self,x):
        
        x = x+self.row_mixer(x)
        
        x = x+self.column_mixer(x)
        
        return x
    

class MLPMixer(nn.Module):
    def __init__(self,in_channels,dim,num_classes,image_size,depth,token_dim,channel_dim,dropout=0.):
        super().__init__()
        
        # original paper 'b c h w -> b (h w) c'
        self.to_input_arrange = nn.Sequential(Reduce('b c h w -> b h w', 'mean'), Rearrange('b h w -> b h w'))
        # w as the channels -> input size (N,48,48)
 
        # 输入为48*48的table
        # 以下为row-mixing MLPs（MLP1）和column-mixing MLPs（MLP2）各一层
        self.mixer_blocks=nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim,token_dim,channel_dim,dropout))
 
        
        self.linear = nn.Linear(dim,dim)
        
        self.layer_normal=nn.LayerNorm(dim)
        
        self.relu = nn.ReLU(inplace=True)
 
        
        self.mlp_head=nn.Sequential(
            nn.Linear(dim,num_classes),
            
        )
        
    def forward(self,x):
        #print('x.shape:', x.shape)
        x = self.to_input_arrange(x)
        #print('input_arrange.shape:', x.shape)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        #[N,48,48]    
        
        x = self.layer_normal(x)
        #print(x)
        x = self.linear(x)

        x = self.layer_normal(x)
        
        for i in range(427):
            x[:,i,i] = 0
        
        x = torch.abs(x)
        
        return x



if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPMixer(in_channels=1, dim=427, num_classes=427*427, image_size=427, depth=1, token_dim=427,
                     channel_dim=427).to(device)
    summary(model,(1,427,427))
 
    inputs = torch.Tensor(1, 1, 427, 427)
    inputs = inputs.to(device)
    print(inputs.shape)
 
    # 将model保存为graph
    with SummaryWriter(log_dir='logs', comment='model') as w:
        w.add_graph(model, (inputs,))
        print("success")