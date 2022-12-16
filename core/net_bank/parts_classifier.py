import torch
from torch import nn
from torch.nn import functional as F
from .oflow_point import ResnetPointnet
from .dgcnn import DGCNN

class Parts_classifier(nn.Module):
    def __init__(self, num_t=4):
        super().__init__()

        self.feature_net=nn.Sequential(*[
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 256, 1),
        ])

        self.cls_net=nn.Sequential(*[
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        ])

        self.res_point_net = ResnetPointnet(
            c_dim=1,
            dim=3+256,
            hidden_dim=1,
        )

        self.dgcnn = DGCNN(emb_dims=256, in_dims=3+num_t)
        
    def forward2(self, x, c_global):
        '''
        In:
            x: BP,TN,3+T
            c: BP, 256
        Out:
            pr: BP,TN
        '''
        BP, TN, _ = x.shape
        
        feature = self.feature_net(x.permute(0,2,1))
        feature = torch.max(feature, dim=-1).values #BP, 256
        input = torch.cat([x, feature.unsqueeze(1).expand(-1, TN, -1)], dim=-1).permute(0,2,1)
        y = self.cls_net(input).unsqueeze(1)
        '''
        input = torch.cat([x, c_global.unsqueeze(1).expand(-1,x.shape[1],-1)], dim=-1)
        _, y = self.res_point_net(input, return_unpooled=True)
        y_1 = self.mlp(y).unsqueeze(-1)
        '''
        return y
    
    def forward(self, query):
        '''
        query: BP,TN,3+T
        '''
        x = self.dgcnn(query)#BP,TN,256
        y = self.cls_net(x.permute(0,2,1))#BP,1,TN
        return y



if __name__ == '__main__':
    model = Parts_classifier(num_t=4)
    x = torch.randn(32, 1200, 7)
    y = model(x)
    print(y.shape)
