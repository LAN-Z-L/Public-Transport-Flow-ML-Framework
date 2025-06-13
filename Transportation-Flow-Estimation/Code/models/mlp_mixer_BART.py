


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
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x=self.net(x)
        return x




class MixerBlock(nn.Module):
    def __init__(self,dim,token_dim,channel_dim,dropout=0.):
        super().__init__()
        self.token_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b h w -> b w h'),
            FeedForward(dim,token_dim,dropout),
            Rearrange('b w h -> b h w')

         )
        self.channel_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim,channel_dim,dropout)
        )
    def forward(self,x):

        x = x+self.token_mixer(x)

        x = x+self.channel_mixer(x)

        return x




class MLPMixer(nn.Module):
    def __init__(self,in_channels,dim,num_classes,image_size,depth,token_dim,channel_dim,dropout=0.):
        super().__init__()

        self.to_input_arrange = nn.Sequential(Rearrange('b c h w -> b h (c w)'))

        self.mixer_blocks=nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim,token_dim,channel_dim,dropout))

        self.layer_normal=nn.LayerNorm(dim)

        self.mlp_head=nn.Sequential(
            nn.ReLU()
        )

    def forward(self,x):
        print('x.shape:', x.shape)
        x = self.to_input_arrange(x)
        print('input_arrange.shape:', x.shape)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_normal(x)    
        print('x_combine.shape:', x.shape)


        x = self.mlp_head(x)

        '''
        for a in x:
            size = a.shape[0]
            size = int(np.sqrt(size))
            for i in range(size):
                x[:, i + i * size] = 0
        '''

        return x




a = torch.randn(2,3)
print(a)
w1 = torch.nn.Linear(3,3)
b= w1(a)
print(b)
w2 = nn.LayerNorm(3)
c = w2(b)
print(c)




if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPMixer(in_channels=1, dim=48, num_classes=48*48, image_size=48, depth=1, token_dim=48,
                     channel_dim=48).to(device)
    summary(model,(1,48,48))

    inputs = torch.Tensor(1, 1, 48, 48)
    inputs = inputs.to(device)
    print(inputs.shape)

    with SummaryWriter(log_dir='logs', comment='model') as w:
        w.add_graph(model, (inputs,))
        print("success")
