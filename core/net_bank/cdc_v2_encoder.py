from .oflow_point import ResnetPointnet
import torch
from torch import nn
from .lpdc_encoder import SpatioTemporalResnetPointnet
from .dgcnn import DGCNN


class GeometrySetEncoder(nn.Module):
    def __init__(self, frame_c_dim=128, frame_hidden_dim=256, hidden_dim=256, c_dim=128):
        super().__init__()
        self.frame_pointnet = ResnetPointnet(
            dim=3,
            c_dim=frame_c_dim,
            hidden_dim=frame_hidden_dim,
        )
        self.set_pointnet = ResnetPointnet(
            dim=frame_c_dim,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
        )
        return

    def forward(self, pc_seq):
        B, T, N, D = pc_seq.shape
        frame_pc = pc_seq.reshape(-1, N, D)
        frame_abs = self.frame_pointnet(frame_pc).reshape(B, T, -1)
        z_g = self.set_pointnet(frame_abs)
        return z_g


class ATCSetEncoder(nn.Module):
    def __init__(
        self,
        atc_num,
        c_dim,
        ci_dim,
        hidden_dim,
    ) -> None:
        super().__init__()
        self.atc_num = atc_num
        self.backbone_pointnet = ResnetPointnet(
            dim=3,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
        )
        self.set_mlp_layers = nn.ModuleList(
            [nn.Linear(c_dim * 2, c_dim), nn.Linear(c_dim * 2, c_dim), nn.Linear(c_dim * 2, c_dim)]
        )
        self.theta_fc = nn.Sequential(nn.Linear(512, c_dim), nn.ReLU(), nn.Linear(c_dim, c_dim), nn.ReLU(), nn.Linear(c_dim, atc_num))
        self.axis_fc = nn.Sequential(nn.Linear(c_dim, c_dim), nn.ReLU(), nn.Linear(c_dim, c_dim), nn.ReLU(), nn.Linear(c_dim, 6 * (atc_num+1)))
        self.c_fc = nn.Sequential(nn.Linear(c_dim, c_dim), nn.ReLU(), nn.Linear(c_dim, ci_dim))
        self.elu = nn.ELU()

        self.feature_net = nn.Sequential(*[
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 256, 1),
        ])

        self.joint_net = nn.Sequential(*[
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4*atc_num + 6*(atc_num+1))
        ])

        self.dgcnn_encoder = DGCNN(k=20, emb_dims=256, dropout=0., in_dims=7)

        self.LSTM = []
        for i in range(atc_num):
            self.LSTM.append(nn.LSTM(
                input_size=512+1,
                hidden_size=512,
                num_layers=512,
                batch_first=True,
            ).cuda())

    '''
    def forward2(self, pc_set):
        B, T, N, D = pc_set.shape
        set_feature = self.backbone_pointnet(pc_set.reshape(-1, N, 3)).reshape(B, T, -1)  # B,T, C
        pooled = torch.max(set_feature, dim=1, keepdim=True).values
        pooled = pooled.expand(-1, T, -1)
        f = torch.cat([set_feature, pooled], dim=-1)
        for layer in self.set_mlp_layers:
            new_f = self.elu(layer(f))
            pooled_f = torch.max(new_f, dim=1, keepdim=True).values
            f = torch.cat([new_f, pooled_f.expand(-1, T, -1)], dim=-1)
        theta = self.theta_fc(new_f).reshape(B, T, -1)
        axis = self.axis_fc(pooled_f.squeeze(1))
        c_global = self.c_fc(pooled_f.squeeze(1))
        return c_global, axis, theta
    '''

    def forward(self, pc_set):
        '''
        pc_set: B,T,N,3
        theta:B,T,P-1
        label:B,P,T,N
        '''
        B, T, N, D = pc_set.shape

        one_hot = torch.zeros_like(pc_set[:,:,:,:1]).repeat(1,1,1,T)#B,T,N,T
        for t in range(T):
            one_hot[:,t,:,t] += 1
        x = torch.cat([pc_set, one_hot], dim=-1)#B,T,N,3+T
        x = x.reshape(B,T*N,-1)
        y = self.dgcnn_encoder(x)
        f_global = torch.max(y, dim=1).values#B,256
        f_t = torch.max(y.reshape(B,T,N,-1), dim=2).values#B,T,256
        f = torch.cat([f_global.unsqueeze(1).expand(-1, T, -1), f_t], dim=-1)#B,T,512

        '''
        for i in range(self.atc_num):
            self.LSTM[i].flatten_parameters()
            _, indices = torch.sort(theta_gt[:,:,i], dim=-1)
            f_i = torch.cat([f, theta_gt[:,:,i].unsqueeze(-1)], dim=-1)
            for b in range(B):
                f_i[b] = f_i[b,indices[b],:]
            out_i, (h_n, h_c) = self.LSTM[i](f_i, None)#B,T,256
        '''        
        for layer in self.set_mlp_layers:
            new_f = self.elu(layer(f))
            pooled_f = torch.max(new_f, dim=1, keepdim=True).values
            f = torch.cat([new_f, pooled_f.expand(-1, T, -1)], dim=-1)
        
        axis = self.axis_fc(pooled_f)
        theta_pred = self.theta_fc(f)#B,T,P-1
        c_global = self.c_fc(pooled_f)

        return c_global, axis, theta_pred



class ATCSetEncoder2(nn.Module):
    def __init__(
        self,
        atc_num,
        c_dim,
        ci_dim,
        hidden_dim,
    ) -> None:
        super().__init__()
        self.backbone_pointnet = ResnetPointnet(
            dim=3,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
        )
        self.set_mlp_layers = ResnetPointnet(
            dim=c_dim,
            c_dim=ci_dim,
            hidden_dim=hidden_dim,
        )
        self.theta_fc = nn.Linear(c_dim, atc_num)

    def forward(self, pc_set):
        B, T, N, D = pc_set.shape
        set_feature = self.backbone_pointnet(pc_set.reshape(-1, N, 3)).reshape(B, T, -1)  # B,T, C
        c_global, state_feat = self.set_mlp_layers(set_feature, return_unpooled=True)
        theta = self.theta_fc(state_feat)
        return c_global, theta


class Query1D(nn.Module):
    def __init__(self, ci_dim, c_dim, t_dim, amp_dim, hidden_dim) -> None:
        super().__init__()
        self.amp = nn.Sequential(
            nn.Linear(t_dim, amp_dim // 2), nn.LeakyReLU(), nn.Linear(amp_dim // 2, amp_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(c_dim + amp_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(c_dim, ci_dim), # ! warning, check this and fix this bug, c_dim != hidden dim
        )

    def forward(self, code, theta):
        # B,C; B,T,ATC
        B, T, _ = theta.shape
        amp = self.amp(theta)
        f = torch.cat([amp, code.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        ci = self.mlp(f)
        return ci


class Query1DLarger(nn.Module):
    def __init__(self, ci_dim, c_dim, t_dim, amp_dim, hidden_dim) -> None:
        super().__init__()
        self.amp = nn.Sequential(
            nn.Linear(t_dim, amp_dim // 2), nn.LeakyReLU(), nn.Linear(amp_dim // 2, amp_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(c_dim + amp_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, ci_dim),
        )

    def forward(self, code, theta):
        # B,C; B,T,ATC
        B, T, _ = theta.shape
        amp = self.amp(theta)
        f = torch.cat([amp, code.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        ci = self.mlp(f)
        return ci


# class LPDC_Encoder(nn.Module):
#     def __init__(self, c_dim, hidden_dim) -> None:
#         super().__init__()
#         self.lpdc_encoder = SpatioTemporalResnetPointnet(
#             c_dim=c_dim,
#             dim=3,
#             hidden_dim=hidden_dim,
#             use_only_first_pcl=False,
#             pool_once=False,
#         )
#         self.geometry_set_encoder = nn.Sequential(
#             nn.Linear(c_dim, c_dim), nn.LeakyReLU(), nn.Linear(c_dim, c_dim)
#         )

#     def forward(self, x):
#         return
