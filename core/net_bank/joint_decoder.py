import torch
from torch import nn
from torch.nn import functional as F

class Joint_decoder(nn.Module):
    def __init__(self, num_p, num_c=256):
        super().__init__()
        self.num_p = num_p
        self.joint_net=nn.Sequential(*[
            nn.Linear(num_c, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 6*num_p)
        ])
        
        self.rotation_net=nn.Sequential(*[
            nn.Linear(7+num_c, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 1)
        ])
        
    def forward(self, c_joint, theta):
        '''
        In:
            c_joint: B,C
            theta: B,T,P-1
        Out:
            trans: B,P,3
            axis: B,P,3
            rotation: B,P,T
        '''
        B, T= theta.shape[0],theta.shape[1]
        P = self.num_p

        y_joint = self.joint_net(c_joint).reshape(B, P, 6) #B,P,6
        y_trans, y_axis = y_joint[:,:,:3], y_joint[:,:,3:]

        input_rotation = torch.cat(
                [
                    c_joint.unsqueeze(1).unsqueeze(1).expand(-1, P, T, -1), #B,P,T,C
                    y_joint.unsqueeze(2).expand(-1,-1,T,-1), #B,P,T,6
                    torch.cat([theta.permute(0,2,1).unsqueeze(-1),torch.zeros_like(theta[:,:,:1]).reshape(B,1,T,1)], dim=1) #B,P,T,1
                ],
                dim=-1)#B,P,T,7+C
        y_rotation = self.rotation_net(input_rotation.reshape(B*P*T,-1)).reshape(B,P,T,1).squeeze(-1)#B,P,T

        return y_trans, y_axis, y_rotation



if __name__ == '__main__':
    model = Joint_decoder(num_p=2, num_c=256)
    c_joint = torch.randn(16,256)
    theta = torch.randn(16,8,2)
    t,a,r = model(c_joint, theta)
    for i in [t,a,r]:
        print(i.shape)
    
    x = torch.randn(8,2)
    print(F.softmax(x, dim=1))