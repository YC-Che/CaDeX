# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds

#DCD temrature parameter
alpha = 5

def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction = "mean",
    point_reduction = "mean",
    norm: int = 2,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_x = 1 - torch.exp(-alpha * cham_x)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if weights is not None:
        cham_x *= weights

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)

    if point_reduction == "mean":
        x_lengths_clamped = x_lengths.clamp(min=1)
        cham_x /= x_lengths_clamped
        cham_x /= N

    return cham_x.sum()

def align_chamfer(
    x,#BPT,N,3
    y,#BPT,(T-1)N,3
    B,
    P,
    T,
    c_trans,
    c_axis,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    batch_reduction = "mean",
    point_reduction = "mean",
    norm: int = 2,
    classifier_label=None,
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)
    N = x.shape[1]

    if classifier_label == None:
        #Only for the first time
        static_x = x.reshape(B,P,T,-1,3)[:,-1,:,:,:]#B,T,N,3
        static_y = y.reshape(B,P,T,-1,3)[:,-1,:,:,:]#B,T,(T-1)N,3
        static_knn = knn_points(static_x.reshape(B*T,-1,3), static_y.reshape(B*T,-1,3), lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=20, return_nn=True)
        static_neighbor = static_knn.idx.reshape(B,T,N,-1)#B,T,N,K
        for i in range(T):
            exclude_upper = i * N
            static_neighbor[:,i,:,:] = torch.where(static_neighbor[:,i,:,:]>=exclude_upper, static_neighbor[:,i,:,:]+N, static_neighbor[:,i,:,:])
        static_neighbor = static_neighbor.reshape(B, T*N, -1)#B,TN,K
        static_cham = static_knn.dists[..., 0].reshape(B,-1)#B, TN

        static_neighbor_cham = []
        for b in range(B):
            c = static_cham[b][static_neighbor[b]]#TN,K
            static_neighbor_cham.append(c)
        static_neighbor_cham = torch.stack(static_neighbor_cham, dim = 0)#B,TN,K

        #threshold = torch.quantile(static_cham, q=0.8, dim=-1)
        threshold = torch.mean(static_cham, dim=-1)#B
        active_mask = static_cham >= threshold.unsqueeze(-1)
        active_mask = active_mask.reshape(B,T,-1) #B,T,N
        static_mask = ~active_mask
    else:
        active_mask = torch.sum(classifier_label[:,:-1,:,:], dim=1) > classifier_label[:,-1,:,:]
        static_mask = ~active_mask#B,T,N

    #torch.save(static_x, './chamfer_x.pt')
    #torch.save(static_cham, './chamfer_c.pt')

    active_x = []
    for b in range(B):
        for p in range(P-1):
            for t in range(T):
                active = x.reshape(B,P,T,-1,3)[b,p,t][active_mask[b,t],:]
                if active.shape[0] == 0:
                    active_x.append(10 * torch.ones((N,4), device=x.device, dtype=x.dtype))#N,4
                else:
                    joint_trans = c_trans[b,p]
                    joint_axis = c_axis[b,p]
                    active_cham = active - joint_trans.unsqueeze(0)#M,3
                    M = active_cham.shape[0]
                    active_cham = torch.norm(active - joint_trans.unsqueeze(0), dim=-1, keepdim=True)
                    active = torch.cat([active, active_cham], dim=-1)
                    active_x.append(torch.cat([active, 10 * torch.ones((N-active.shape[0], 4), device=x.device, dtype=x.dtype)]))#N,4
    active_x = torch.stack(active_x, dim=0)
    active_x = active_x.reshape(B,P-1,T,-1,4)#B,P-1,T,N,4

    exclude = [[list(range(i)), list(range(i+1, T))]for i in range(T)]
    exclude = [exclude[i][0] + exclude[i][1] for i in range(T)] #T,T-1
    active_y = torch.stack([active_x[:,:,exclude[i],:,:] for i in range(T)], dim=2)#B,P-1,T,T-1,N,4

    active_x = active_x.reshape(B*(P-1)*T,-1,4)
    active_y = active_y.reshape(B*(P-1)*T,-1,4)
    active_cham = knn_points(active_x, active_y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    active_cham = active_cham.dists[..., 0].reshape(B,P-1,-1)#B,P-1,TN
    active_cham = 1 - torch.exp(-alpha * active_cham)
    if P>2:
        active_cham = torch.min(active_cham, dim=1).values #B,TN
    return active_cham.flatten().mean(), static_mask, active_mask

    '''
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # BPT,N
    cham_x = cham_x.reshape(B,P,-1)#B,P, TN
    static_mean = torch.mean(cham_x[:,-1,:], dim=-1)#B
    active_mask = cham_x[:,-1,:] > static_mean.unsqueeze(-1) #B,TN
    cham_x = torch.min(cham_x[:,:-1,:], dim=1).values#B,TN
    cha_sum = cham_x[active_mask].mean()
    return cha_sum
    '''
if __name__ == '__main__':
    x = torch.randn((6*2*4,500,3)).cuda()
    y = torch.randn((6*2*4,1500,3)).cuda()
    print(align_chamfer(x,y,6,2,4))

def consistent_loss(self, query, label):
    B,P,T,N = label.shape

    label = label.reshape(B,P,T*N).permute(0,2,1)#B,TN,P
    query = query.reshape(B,P,T*N,3)[:,-1]#B,TN,3
    lengths = torch.full((B,), T*N, dtype=torch.int64, device=query.device)
    nn_points = knn_points(query, query, lengths1=lengths, lengths2=lengths, norm=2, K=5).idx #B,TN,K
    loss_sum = 0
    for b in range(B):
        nn_label = label[b][nn_points[b], :]#TN,K,P
        nn_diff = torch.sum((label[b].unsqueeze(1) - nn_label)**2, dim=-1)#TN,K
        nn_diff = torch.mean(torch.mean(nn_diff, dim=-1))
        loss_sum += nn_diff
    return loss_sum