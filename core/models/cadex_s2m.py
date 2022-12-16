import math
from .model_base import ModelBase
import torch
import copy
import trimesh
from torch import nn
from torch.nn import functional as F
import time
from core.net_bank.oflow_point import ResnetPointnet
from core.net_bank.oflow_decoder import DecoderCBatchNorm, Decoder
from core.net_bank.nvp_v2 import NVP_v2_5
from core.net_bank.cdc_v2_encoder import ATCSetEncoder, Query1D
from core.net_bank.joint_decoder import Joint_decoder
from core.net_bank.parts_classifier import Parts_classifier
from .utils.chamfer_loss import chamfer_distance, align_chamfer
import logging
from .utils.occnet_utils import get_generator
from torch import distributions as dist
import numpy as np
from copy import deepcopy
from pytorch3d.ops.knn import knn_gather, knn_points

from core.models.utils.viz_cdc_render import viz_cdc
from core.models.utils.oflow_eval.evaluator import MeshEvaluator
from core.models.utils.oflow_common import eval_atc_all, eval_iou


class Model(ModelBase):
    def __init__(self, cfg):
        network = CaDeX_S2M(cfg)
        super().__init__(cfg, network)

        self.input_num = cfg["dataset"]["input_num"]
        self.num_atc = cfg["dataset"]["num_atc"]

        viz_mesh = []
        T = cfg["dataset"]["set_size"]
        for t in range(T):
            viz_mesh += ["mesh_t%d" % t]
        self.output_specs = {
            "metric": [
                "batch_loss",
                "loss_recon",
                "loss_corr",
                "loss_supervised_theta",
                "loss_joint",
                "loss_classifier",
                "loss_align",
                "loss_supervised_axis",
                "loss_axis",
                "loss_norm",
                "iou",
                "iou_gen",
                "iou_obs",
            ]
            + ["loss_reg_shift_len"],
            "image": ["mesh_viz_image", "query_viz_image"],
            "mesh": viz_mesh + ["cdc_mesh"],
            "video": ["flow_video"],
            "hist": ["loss_recon_i", "loss_corr_i", "cdc_shift", "theta_error"],
            "xls": ["running_metric_report", "results_observed", "results_generated"],
        }

        self.viz_one = cfg["logging"]["viz_one_per_batch"]
        self.iou_threshold = cfg["evaluation"]["iou_threshold"]
        self.corr_eval_project_to_final_mesh = cfg["evaluation"]["project_to_final_mesh"]
        if self.corr_eval_project_to_final_mesh:
            logging.warning("In config set Corr-Proj-To-Mesh true, ignore it, set to false")
            self.corr_eval_project_to_final_mesh = False
        self.mesh_extractor = get_generator(cfg)
        self.evaluator = MeshEvaluator(cfg["dataset"]["n_query_sample_eval"])

        self.viz_use_T = cfg["dataset"]["input_type"] != "pcl"

    def generate_mesh(self, c_trans, c_rotation, c_g, use_uncomp_cdc=True):
        mesh_t_list = []
        net = self.network.module if self.__dataparallel_flag__ else self.network
        P, T = c_rotation.shape[0], c_rotation.shape[1]
        # extract t0 mesh by query t0 space
        observation_c = {
            "c_trans": c_trans.unsqueeze(0).detach(),
            "c_rotation": c_rotation.unsqueeze(0).detach(),
            "c_g": c_g.unsqueeze(0).detach(),
            "query_t": torch.zeros((1, 1)).to(c_trans.device),
        }
        mesh_t0 = self.mesh_extractor.generate_from_latent(c=observation_c, F=net.decode_by_current)
        # Safe operation, if no mesh is extracted, replace by a fake one
        if mesh_t0.vertices.shape[0] == 0:
            mesh_t0 = trimesh.primitives.Box(extents=(1.0, 1.0, 1.0))
            logging.warning("Mesh extraction fail, replace by a place holder")
        # get deformation code
        #c_homeo = c_t.unsqueeze(0).transpose(2, 1)  # B,C,T
        # convert t0 mesh to cdc
        t0_mesh_vtx = np.array(mesh_t0.vertices).copy()
        t0_mesh_vtx = torch.Tensor(t0_mesh_vtx).cuda().unsqueeze(0).unsqueeze(0)  # B,T,N,3 = 1,1,Pts,3
        #t0_mesh_vtx_cdc, t0_mesh_vtx_cdc_uncompressed = net.map2canonical(
        #    c_homeo[:, :, :1], t0_mesh_vtx.unsqueeze(1), return_uncompressed=True
        #)  # code: B,C,T, query: B,T,N,3
        t0_c_rotation = observation_c["c_rotation"][:,:,:1,:,:]
        t0_c_trans = observation_c["c_trans"]
        t0_mesh_vtx_cdc_uncompressed = net.multi_flame_align(
            t0_mesh_vtx, t0_c_rotation, t0_c_trans)#B,P,1,N,3
        
        if net.compress_cdc:
            t0_mesh_vtx_cdc = torch.sigmoid(t0_mesh_vtx_cdc_uncompressed) - 0.5
        _, t0_mesh_vtx_label = net.decode_by_multi_flame(t0_mesh_vtx_cdc, c_g)
        # get all frames vtx by mapping cdc to each frame
        # soruce_vtx_cdc = t0_mesh_vtx_cdc.expand(-1, T, -1, -1)
        soruce_vtx_cdc = t0_mesh_vtx_cdc_uncompressed.expand(-1, -1, T, -1, -1)

        # surface_vtx = net.map2current(c_homeo, soruce_vtx_cdc).squeeze(0)
        surface_vtx = net.multi_flame_convergence(soruce_vtx_cdc, observation_c["c_rotation"], observation_c["c_trans"], t0_mesh_vtx_label)
        #surface_vtx = net.map2current(c_homeo, soruce_vtx_cdc, compressed=False).squeeze(0)
        # ! clamp all vtx to unit cube
        surface_vtx = torch.clamp(surface_vtx, -1.0, 1.0).squeeze(0)
        surface_vtx = surface_vtx.detach().cpu().numpy()  # T,Pts,3
        # make meshes for each frame
        for t in range(0, T):
            mesh_t = deepcopy(mesh_t0)
            mesh_t.vertices = surface_vtx[t]
            mesh_t.update_vertices(mask=np.array([True] * surface_vtx.shape[0]))
            mesh_t_list.append(mesh_t)
        mesh_cdc = deepcopy(mesh_t0)
        mesh_cdc_vtx = surface_vtx[:1]
        mesh_cdc.vertices = mesh_cdc_vtx.squeeze(0)
        mesh_cdc.update_vertices(mask=np.array([True] * surface_vtx.shape[0]))
        return mesh_t_list, surface_vtx, mesh_cdc

    def map_pc2cdc(self, batch, bid, key="seq_pc"):
        net = self.network.module if self.__dataparallel_flag__ else self.network
        #_c_t = batch["c_t"][bid].unsqueeze(0)
        c_trans = batch["c_trans"][bid].unsqueeze(0) #B,P,3
        c_rotation = batch["c_rotation"][bid].unsqueeze(0) #B,P,T,3,3
        c_g = batch["c_g"][bid].unsqueeze(0)
        _obs_pc = batch[key][bid].unsqueeze(0)
        #_, input_cdc_un = net.map2canonical(_c_t.transpose(2, 1), _obs_pc, return_uncompressed=True)
        _obs_pc_muilti = net.multi_flame_align(_obs_pc, c_rotation, c_trans) #B,P,T,N,3
        if net.compress_cdc:
            _obs_pc_muilti_query = torch.sigmoid(_obs_pc_muilti) - 0.5
        parts_logits, parts_label = net.decode_by_multi_flame(_obs_pc_muilti_query, c_g) #B,P,T,N
        _obs_pc_muilti -= c_trans.unsqueeze(2).unsqueeze(2)
        _obs_pc_muilti = (torch.inverse(c_rotation.unsqueeze(3)) @ _obs_pc_muilti .unsqueeze(-1)).squeeze(-1)
        _obs_pc_muilti += c_trans.unsqueeze(2).unsqueeze(2)

        input_cdc_un = torch.sum(_obs_pc_muilti * parts_label.unsqueeze(-1), dim=1)
        input_cdc_un = input_cdc_un.detach().cpu().numpy().squeeze(0)
        return input_cdc_un

    def _postprocess_after_optim(self, batch):
        # eval iou
        if "occ_hat_iou" in batch.keys():
            report = {}

            occ_pred = batch["occ_hat_iou"].detach().cpu().numpy()
            occ_gt = batch["model_input"]["points.occ"].detach().cpu().numpy()
            iou = eval_iou(occ_gt, occ_pred, threshold=self.iou_threshold)  # B,T_all
            # make metric tensorboard
            iou_observed_theta_gt = iou[:, : self.input_num]
            iou_generated = iou[:, self.input_num :]
            batch["iou_obs"] = iou_observed_theta_gt.mean()
            batch["iou_gen"] = iou_generated.mean()
            batch["iou"] = iou.mean()
            # make report
            report["iou"] = iou.mean(axis=1).tolist()
            report["iou_obs"] = iou_observed_theta_gt.mean(axis=1).tolist()
            report["iou_gen"] = iou_generated.mean(axis=1).tolist()
            batch["running_metric_report"] = report

        if "c_trans" in batch.keys():
            self.network.eval()
            phase = batch["model_input"]["phase"]
            viz_flag = batch["model_input"]["viz_flag"]
            TEST_RESULT_OBS = {}
            TEST_RESULT_GEN = {}
            B, P, T, _, _ = batch["c_rotation"].shape
            with torch.no_grad():
                # prepare viz mesh lists
                for t in range(T):
                    batch["mesh_t%d" % t] = []
                batch["cdc_mesh"] = []
                rendered_fig_list, rendered_fig_query_list, video_list = [], [], []
                for bid in range(B):
                    # generate mesh
                    # * With GT Theta
                    logging.info("Generating Mesh Observed/Unobserved with GT theta")
                    start_t = time.time()
                    mesh_t_list, _, mesh_cdc = self.generate_mesh(
                        batch["c_trans"][bid], batch['c_rotation'][bid], batch["c_g"][bid]
                    )
                    recon_time = time.time() - start_t
                    for t in range(0, T):  # if generate mesh, then save it
                        batch["mesh_t%d" % t].append(mesh_t_list[t])
                    batch["cdc_mesh"].append(mesh_cdc)
                    if phase.startswith("test"):
                        # evaluate the generated mesh list
                        logging.info("Generating Mesh Observed/Unobserved with GT theta")
                        # * With predict Theta
                        mesh_t_list_pred_theta, _, _ = self.generate_mesh(
                            batch["c_trans"][bid], batch['c_rotation'][bid], batch["c_g"][bid]
                        )
                        logging.warning("Start eval")
                        eval_dict_mean_gt_observed, _ = eval_atc_all(
                            pcl_corr=batch["model_input"]["points_mesh"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            pcl_chamfer=batch["model_input"]["points_chamfer"][bid][
                                : self.input_num
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            points_tgt=batch["model_input"]["points"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            occ_tgt=batch["model_input"]["points.occ"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            mesh_t_list=mesh_t_list[: self.input_num],
                            evaluator=self.evaluator,
                            corr_project_to_final_mesh=self.corr_eval_project_to_final_mesh,
                            eval_corr=self.input_num > 1,
                        )
                        eval_dict_mean_gt_generated, _ = eval_atc_all(
                            pcl_corr=batch["model_input"]["points_mesh"][bid][self.input_num :]
                            .detach()
                            .cpu()
                            .numpy(),
                            pcl_chamfer=batch["model_input"]["points_chamfer"][bid][
                                self.input_num :
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            points_tgt=batch["model_input"]["points"][bid][self.input_num :]
                            .detach()
                            .cpu()
                            .numpy(),
                            occ_tgt=batch["model_input"]["points.occ"][bid][self.input_num :]
                            .detach()
                            .cpu()
                            .numpy(),
                            mesh_t_list=mesh_t_list[self.input_num :],
                            evaluator=self.evaluator,
                            corr_project_to_final_mesh=self.corr_eval_project_to_final_mesh,
                        )
                        eval_dict_mean_pred_observed, _ = eval_atc_all(
                            pcl_corr=batch["model_input"]["points_mesh"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            pcl_chamfer=batch["model_input"]["points_chamfer"][bid][
                                : self.input_num
                            ]
                            .detach()
                            .cpu()
                            .numpy(),
                            points_tgt=batch["model_input"]["points"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            occ_tgt=batch["model_input"]["points.occ"][bid][: self.input_num]
                            .detach()
                            .cpu()
                            .numpy(),
                            mesh_t_list=mesh_t_list_pred_theta,
                            evaluator=self.evaluator,
                            corr_project_to_final_mesh=self.corr_eval_project_to_final_mesh,
                            eval_corr=self.input_num > 1,
                        )
                        logging.warning("End eval")
                        # record the batch results
                        for k, v in eval_dict_mean_gt_observed.items():
                            _k = f"{k}(G)"
                            if _k not in TEST_RESULT_OBS.keys():
                                TEST_RESULT_OBS[_k] = [v]
                            else:
                                TEST_RESULT_OBS[_k].append(v)
                        for k, v in eval_dict_mean_pred_observed.items():
                            _k = f"{k}(P)"
                            if _k not in TEST_RESULT_OBS.keys():
                                TEST_RESULT_OBS[_k] = [v]
                            else:
                                TEST_RESULT_OBS[_k].append(v)
                        for k, v in eval_dict_mean_gt_generated.items():
                            _k = f"{k}(G)"
                            if _k not in TEST_RESULT_GEN.keys():
                                TEST_RESULT_GEN[_k] = [v]
                            else:
                                TEST_RESULT_GEN[_k].append(v)
                        theta_hat = batch["theta_hat"][bid].detach().cpu().numpy()
                        theta_gt = batch["theta_gt"][bid].detach().cpu().numpy()
                        for atc_i in range(self.num_atc):
                            error = abs(theta_hat[:, atc_i] - theta_gt[:, atc_i]).mean()
                            error = error / np.pi * 180.0
                            k = f"theta-{atc_i}-error(degree)"
                            if k not in TEST_RESULT_OBS.keys():
                                TEST_RESULT_OBS[k] = [error]
                            else:
                                TEST_RESULT_OBS[k].append(error)
                        if "time-all" not in TEST_RESULT_OBS.keys():
                            TEST_RESULT_OBS["time-all"] = [recon_time]
                        else:
                            TEST_RESULT_OBS["time-all"].append(recon_time)
                        logging.info("Test OBS: {}".format(TEST_RESULT_OBS))
                        logging.info("Test GEN: {}".format(TEST_RESULT_GEN))

                    # render an image of the mesh
                    if viz_flag:
                        scale_cdc = True
                        if "viz_cdc_scale" in self.cfg["logging"].keys():
                            scale_cdc = self.cfg["logging"]["viz_cdc_scale"]
                        viz_align_cdc = False
                        if "viz_align_cdc" in self.cfg["logging"].keys():
                            viz_align_cdc = self.cfg["logging"]["viz_align_cdc"]
                        fig_t_list, fig_query_list = viz_cdc(
                            mesh_t_list,
                            mesh_cdc,
                            input_pc=batch["seq_pc"][bid].detach().cpu().numpy(),
                            input_cdc=self.map_pc2cdc(batch, bid, key="seq_pc"),
                            corr_pc=batch["corr_pc"][bid].detach().cpu().numpy(),
                            corr_cdc=self.map_pc2cdc(batch, bid, key="corr_pc"),
                            object_T=batch["object_T"][bid].detach().cpu().numpy()
                            if self.viz_use_T
                            else None,
                            scale_cdc=scale_cdc,
                            interval=self.cfg["logging"]["mesh_viz_interval"],
                            query=batch["query"][bid].detach().cpu().numpy(),
                            query_occ=batch["query_occ"][bid].detach().cpu().numpy(),
                            align_cdc=viz_align_cdc,
                            cam_dst_default=1.7,
                        )
                        cat_fig = np.concatenate(fig_t_list, axis=0).transpose(2, 0, 1)
                        cat_fig = np.expand_dims(cat_fig, axis=0).astype(np.float) / 255.0
                        rendered_fig_list.append(cat_fig)
                        cat_fig2 = np.concatenate(fig_query_list, axis=1).transpose(2, 0, 1)
                        cat_fig2 = np.expand_dims(cat_fig2, axis=0).astype(np.float) / 255.0
                        rendered_fig_query_list.append(cat_fig2)
                        # pack a video
                        video = np.concatenate(
                            [i.transpose(2, 0, 1)[np.newaxis, ...] for i in fig_t_list], axis=0
                        )  # T,3,H,W
                        video = np.expand_dims(video, axis=0).astype(np.float) / 255.0
                        video_list.append(video)

                    # if not in test
                    if self.viz_one and not phase.startswith("test"):
                        break
                if viz_flag:
                    batch["mesh_viz_image"] = torch.Tensor(
                        np.concatenate(rendered_fig_list, axis=0)
                    )  # B,3,H,W
                    batch["query_viz_image"] = torch.Tensor(
                        np.concatenate(rendered_fig_query_list, axis=0)
                    )  # B,3,H,W
                    batch["flow_video"] = torch.Tensor(
                        np.concatenate(video_list, axis=0)
                    )  # B,T,3,H,W
            if phase.startswith("test"):
                batch["results_observed"] = TEST_RESULT_OBS
                batch["results_generated"] = TEST_RESULT_GEN
        del batch["model_input"]
        return batch


class CaDeX_S2M(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)

        self.num_atc = cfg["dataset"]["num_atc"]
        self.input_num = cfg["dataset"]["input_num"]

        cg_decoder = Decoder #DecoderCBatchNorm
        if "cg_cbatchnorm" in cfg["model"].keys():
            if not cfg["model"]["cg_cbatchnorm"]:
                logging.info("Canonical Geometry Decoder not using CBatchNorm")
                cg_decoder = Decoder
        H = NVP_v2_5
        H_act = nn.LeakyReLU
        if "homeo_act" in cfg["model"].keys():
            act_type = cfg["model"]["homeo_act"]
            act_dict = {"relu": nn.ReLU, "elu": nn.ELU, "leakyrelu": nn.LeakyReLU}
            assert act_type in act_dict.keys(), "Homeo Activation not support"
            H_act = act_dict[act_type]

        self.network_dict = torch.nn.ModuleDict(
            {
                "homeomorphism_encoder": ATCSetEncoder(
                    **cfg["model"]["homeomorphism_encoder"], atc_num=self.num_atc
                ),
                "ci_decoder": Query1D(**cfg["model"]["ci_decoder"], t_dim=self.num_atc),
                "canonical_geometry_encoder": ResnetPointnet(
                    **cfg["model"]["canonical_geometry_encoder"]
                ),
                "canonical_geometry_decoder": cg_decoder(
                    dim=3, z_dim=0, **cfg["model"]["canonical_geometry_decoder"]
                ),
                "homeomorphism_decoder": H(
                    **cfg["model"]["homeomorphism_decoder"], activation=H_act
                ),
                "joint_decoder": Joint_decoder(
                    num_p=self.num_atc+1, num_c=256
                ),
                "parts_classifier": Parts_classifier(
                    num_t=4
                )
            }
        )

        for k in self.network_dict:
            logging.info(
                "{} params in {}".format(
                    sum(param.numel() for param in self.network_dict[k].parameters()), k
                )
            )

        self.compress_cdc = cfg["model"]["compress_cdc"]

        self.corr_loss_weight = 1.0
        if "corr_weight" in cfg["model"].keys():
            self.corr_loss_weight = cfg["model"]["corr_weight"]
        self.corr_square = False
        if "corr_square" in cfg["model"].keys():
            self.corr_square = cfg["model"]["corr_square"]


        self.use_corr_loss = cfg["model"]["loss_corr"]

    @staticmethod
    def logit(x, safe=False):
        eps = 1e-16 if safe else 0.0
        return -torch.log((1 / (x + eps)) - 1)

    def map2canonical(self, code, query, return_uncompressed=False):
        # code: B,C,T, query: B,T,N,3
        # B1, M1, _ = F.shape # batch, templates, C
        # B2, _, M2, D = x.shape # batch, Npts, templates, 3
        coordinates = self.network_dict["homeomorphism_decoder"].forward(
            code.transpose(2, 1), query.transpose(2, 1)
        )
        if self.compress_cdc:
            out = torch.sigmoid(coordinates) - 0.5
        else:
            out = coordinates
        if return_uncompressed:
            return out.transpose(2, 1), coordinates.transpose(2, 1)  # B,T,N,3
        else:
            return out.transpose(2, 1)  # B,T,N,3

    def map2current(self, code, query, compressed=True):
        # code: B,C,T, query: B,T,N,3
        # B1, M1, _ = F.shape # batch, templates, C
        # B2, _, M2, D = x.shape # batch, Npts, templates, 3
        coordinates = self.logit(query + 0.5) if (self.compress_cdc and compressed) else query
        coordinates, _ = self.network_dict["homeomorphism_decoder"].inverse(
            code.transpose(2, 1), coordinates.transpose(2, 1)
        )
        return coordinates.transpose(2, 1)  # B,T,N,3

    def forward(self, input_pack, viz_flag, refine_optimizer=None):
        output = {}
        phase = input_pack["phase"]
        axis_gt = []
        for b in range(input_pack["theta"].shape[0]):
            if input_pack["category"][b] == "laptop":
                axis_object = [-1,0,0]
            elif input_pack["category"][b] == "door":
                axis_object = [0,1,0]
            elif input_pack["category"][b] == "stapler":
                axis_object = [-1,0,0]
            else:
                axis_object = [0,0,0]
            axis_gt.append(torch.tensor(axis_object, device=input_pack["inputs"].device, dtype=input_pack["inputs"].dtype))
        axis_gt = torch.stack(axis_gt, dim=0)#B,3

        if phase.startswith("val"):
            pass
        else:
            pass
            '''
            #random rotation
            rand_vec = torch.rand((input_pack["theta"].shape[0], 3), device=input_pack["theta"].device, dtype=input_pack["theta"].dtype) - 0.5
            rand_vec = ExpSO3(2 * math.pi * rand_vec).unsqueeze(1).unsqueeze(1)#B,1,1,3,3
            axis_gt = (rand_vec[:,0,0,:,:] @ axis_gt.unsqueeze(-1))
            axis_gt = axis_gt.squeeze(-1).unsqueeze(1)#B,1,3
            for i in ["inputs", "points", "pointcloud"]:
                input_pack[i] = (rand_vec @ input_pack[i].unsqueeze(-1)).squeeze(-1)
            '''
            
            #random exchange axis
            choices = []
            for b in range(input_pack["theta"].shape[0]):
                choice = np.random.choice([0, 1, 2], size=1)
                choices.append(choice)
                if choice == 0:
                    pass
                elif choice == 1:
                    #exchange into YXZ
                    axis_gt[b] = axis_gt[b,[1,2,0]]
                    for i in ["inputs", "points", "pointcloud"]:
                        input_pack[i][b, :, :, :] = input_pack[i][b, :, :, [1, 2 ,0]]
                elif choice == 2:
                    #exchange into XZY
                    axis_gt[b] = axis_gt[b,[2,0,1]]
                    for i in ["inputs", "points", "pointcloud"]:
                        input_pack[i][b, :, :, :] = input_pack[i][b, :, :, [2, 0, 1]]
                elif choice ==3:
                    axis_gt[b,1] *= -1
                    input_pack[i][b, :, :, 1] *= -1
                elif choice == 4:
                    #exchange into YXZ
                    axis_gt[b] = axis_gt[b,[1,0,2]]
                    axis_gt[b,1] *= -1
                    for i in ["inputs", "points", "pointcloud"]:
                        input_pack[i][b, :, :, :] = input_pack[i][b, :, :, [1, 0 ,2]]
                        input_pack[i][b, :, :, 1] *= -1
                elif choice == 5:
                    #exchange into XZY
                    axis_gt[b] = axis_gt[b,[0,2,1]]
                    axis_gt[b,1] *= -1
                    for i in ["inputs", "points", "pointcloud"]:
                        input_pack[i][b, :, :, :] = input_pack[i][b, :, :, [0, 2 ,1]]
                        input_pack[i][b, :, :, 1] *= -1
            

        set_pc, theta_gt = input_pack["inputs"][:, : self.input_num], input_pack["theta"]
        B, T_all, _ = input_pack["theta"].shape
        P = self.num_atc+1
        T_in = self.input_num
        N = input_pack["inputs"].shape[2]
        c_length = torch.cat([- input_pack["theta"], torch.zeros_like(input_pack["theta"][:,:,:1])], dim=-1).permute(0,2,1)#B,P,T
        refine_T = 1
        '''
        if phase.startswith("val"): 
            self.train()
            refine_params = list(self.network_dict["homeomorphism_encoder"].parameters()) + list(self.network_dict["parts_classifier"].parameters())
            refine_T = 1
        '''
        for refine_t in range(refine_T):
            c_global, c_joint, theta_hat = self.network_dict["homeomorphism_encoder"](set_pc)
            axis = c_joint.reshape(B, P, 6)[:,:,:3]#B,P,3
            c_axis = axis / torch.norm(axis, dim=-1).unsqueeze(-1)
            #print(c_axis[0,0])
            #print(axis_gt[0])
            c_trans = c_joint.reshape(B, P, 6)[:,:,3:]
            c_trans = torch.cat([c_trans[:,:-1,:], torch.zeros_like(c_trans[:,:1,:])], dim=1)#B,P,3
            c_rotation = self.rotation_vec_2_matrix(c_axis, c_length)
            input_multi = self.multi_flame_align(set_pc, c_rotation[:,:,:T_in,:,:], c_trans, cat_flame_label=True)#B,P,T,N,(3+T)
            input_multi_query, input_multi_others = self.binary_split(input_multi)#BPT,N,3, BPT,(t-1)N,3

            input_multi_w = self.network_dict["parts_classifier"](input_multi.reshape(B*P, T_in*N, -1)).reshape(B,P,T_in,N)
            input_multi_label = F.softmax(input_multi_w, dim=1)#B,P,T,N
        '''
            if phase.startswith("val") and refine_t < refine_T - 1:
                print("!!!shoud not into this loop!!!")
                refine_align_loss, static_mask,_ = self.align_loss(input_multi_query, input_multi_others,B,P,T_in, c_trans, c_axis, label=input_multi_label.detach())
                refine_classifier_loss = self.chamfer_loss(input_multi_query.detach(), input_multi_others.detach(), input_multi_label)
                refine_joint_loss = self.joint_decoder_loss(axis, c_length, theta_gt, c_trans, set_pc, static_mask)
                refine_loss = refine_joint_loss + 10*refine_align_loss + 10*refine_classifier_loss
                refine_optimizer.zero_grad()
                refine_loss.backward()
                refine_optimizer.step()
        if phase.startswith("val"): 
            self.eval()
        '''
        input_multi_label_hard = F.one_hot(torch.argmax(input_multi_label, dim=1), num_classes=P).permute(0,3,1,2)#B,P,T,N
        input_multi_classified = input_multi[:,:,:,:,:3] * input_multi_label_hard.unsqueeze(-1) #B,P,T,N,3
        #detach
        #input_multi_classified = input_multi_classified.detach()
        if self.compress_cdc:
            input_multi_classified[:,:,:,:,:3] = torch.sigmoid(input_multi_classified[:,:,:,:,:3]) - 0.5
        c_g = self.network_dict["canonical_geometry_encoder"](input_multi_classified.reshape(B*P, T_in*N, 3)).reshape(B,P,-1)

        # visualize
        if viz_flag:
            output["c_trans"] = c_trans.detach()
            output["c_rotation"] = c_rotation.detach()
            output["c_g"] = c_g.detach()
            output["seq_pc"] = input_pack["inputs"]
            output["object_T"] = input_pack["object_T"].detach()
            output["corr_pc"] = input_pack["pointcloud"].detach()
            output["query"] = input_pack["points"].detach()
            output["query_occ"] = input_pack["points.occ"].detach()

        # if test, direct return
        if phase.startswith("test"):
            output["c_trans"] = c_trans.detach()
            output["c_rotation"] = c_rotation.detach()
            output["c_g"] = c_g.detach()
            # also test with pred theta
            c_t_pred_theta = self.network_dict["ci_decoder"](c_global, theta_hat)
            output["c_t_pred_theta"] = c_t_pred_theta.detach()
            output["theta_hat"] = theta_hat.detach()
            output["theta_gt"] = input_pack["theta"][:, : self.input_num]
            output["theta_all"] = input_pack["theta"]

            return output

        points_multi = self.multi_flame_align(input_pack["points"], c_rotation, c_trans)
        if self.compress_cdc:
            points_multi = torch.sigmoid(points_multi) - 0.5
        #detach
        #points_multi = points_multi.detach()
        pr, _ = self.decode_by_multi_flame(points_multi, c_g)#B,P,T,N
        pr = torch.max(pr, dim=1).values# B,T,N
        pr = dist.Bernoulli(logits=pr)
        # occ_hat = pr.probs
        # compute corr loss
        '''
        if self.use_corr_loss:
            _, cdc_first_frame_un = self.map2canonical(
                c_t[:, 0].unsqueeze(2),
                input_pack["pointcloud"][:, 0].unsqueeze(1),
                return_uncompressed=True,
            )  # B,1,M,3
            cdc_forward_frames = self.map2current(
                c_t[:, 1:].transpose(2, 1),
                cdc_first_frame_un.expand(-1, T_all - 1, -1, -1),
                compressed=False,
            )
            if self.corr_square:
                corr_loss_i = (
                    torch.norm(
                        cdc_forward_frames - input_pack["pointcloud"][:, 1:].detach(), dim=-1
                    )
                    ** 2
                )
            else:
                corr_loss_i = torch.abs(
                    cdc_forward_frames - input_pack["pointcloud"][:, 1:].detach()
                ).sum(-1)
            corr_loss = corr_loss_i.mean()
        '''
        reconstruction_loss_i = torch.nn.functional.binary_cross_entropy_with_logits(
            pr.logits, input_pack["points.occ"], reduction="none"
            )
        reconstruction_loss = reconstruction_loss_i.mean()
        align_loss, static_mask, _ = self.align_loss(input_multi_query, input_multi_others,B,P,T_in, c_trans, c_axis)
        classifier_loss = self.chamfer_loss(input_multi_query, input_multi_others, input_multi_label)
        joint_loss = self.joint_decoder_loss(axis, c_length, theta_gt, c_trans, set_pc, static_mask)
        supervised_axis_loss = torch.mean((axis_gt.flatten() - c_axis[:,:-1,:].flatten()) ** 2)
        supervised_theta_loss = torch.mean((theta_hat - theta_gt[:, :T_in]) ** 2)
        norm_loss = self.norm_projection_loss(set_pc, c_axis, c_trans)

        output["batch_loss"] = joint_loss + supervised_theta_loss + 10*classifier_loss + 10*align_loss + reconstruction_loss #+ supervised_axis_loss
        output["loss_recon"] = reconstruction_loss.detach()
        output["loss_supervised_theta"] = supervised_theta_loss.detach()
        output["loss_joint"] = joint_loss.detach()
        output["loss_supervised_axis"] = supervised_axis_loss.detach()
        output["loss_classifier"] = 10*classifier_loss.detach()
        output["loss_align"] = 10*align_loss.detach()
        output["loss_norm"] = norm_loss.detach()
        output["loss_recon_i"] = reconstruction_loss_i.detach().reshape(-1)
        output["theta_error"] = torch.abs(theta_hat - theta_gt[:, :T_in]).view(-1).detach()
        output["viz_query"] = input_multi_query
        output["viz_weights"] = input_multi_label
        output["viz_trans"] = c_trans
        output["viz_axis"] = c_axis
        output["viz_length"] = c_length
        '''
        if self.use_corr_loss:
            output["batch_loss"] = output["batch_loss"] + corr_loss * self.corr_loss_weight
            output["loss_corr"] = corr_loss.detach()
            output["loss_corr_i"] = corr_loss_i.detach().reshape(-1)
        '''
        if phase.startswith("val"):  # add eval
            output["occ_hat_iou"] = pr.probs

        return output

    def decode_by_cdc(self, observation_c, query):
        B, T, N, _ = query.shape
        query = query.reshape(B, -1, 3)
        logits = self.network_dict["canonical_geometry_decoder"](
            query, None, observation_c
        ).reshape(B, T, N)
        return dist.Bernoulli(logits=logits)

    def decode_by_current(self, query, z_none, c):
        '''
        # ! decoder by coordinate in current coordinate space, only used in viz
        c_t = c["c_t"]
        c_g = c["c_g"]
        query_t = c["query_t"]
        assert query.ndim == 3
        query = query.unsqueeze(1)
        idx = (query_t * (c_t.shape[1] - 1)).long()  # B,t
        c_homeomorphism = torch.gather(
            c_t, dim=1, index=idx.unsqueeze(2).expand(-1, -1, c_t.shape[-1])
        ).transpose(
            2, 1
        )  # B,C,T
        # transform to canonical frame
        cdc = self.map2canonical(c_homeomorphism, query)  # B,T,N,3
        logits = self.decode_by_cdc(observation_c=c_g, query=cdc).logits
        pr = dist.Bernoulli(logits=logits.squeeze(1))
        #return pr
        '''
        c_trans = c['c_trans']
        c_rotation = c['c_rotation']
        c_g = c["c_g"]
        query_multi = self.multi_flame_align(query.unsqueeze(0), c_rotation[:,:,:1,:,:], c_trans)
        if self.compress_cdc:
            query_multi = torch.sigmoid(query_multi) - 0.5
        parts_logits, parts_label = self.decode_by_multi_flame(query_multi, c_g)
        parts_logits = torch.max(parts_logits, dim=1).values.squeeze(1)
        pr = dist.Bernoulli(logits=parts_logits)
        return pr

    def multi_flame_align(self, x, rotation, translation, cat_flame_label=False):
        '''
        x: B,T,N,3
        rotation: B,P,T,3,3
        translation: B,P,3
        Outout: B,P,T,N,3 or B,P,T,N,(3+T)
        '''
        B,P,T = rotation.shape[:3]
        N = x.shape[2]
        x_multi = x.squeeze(-1).unsqueeze(1).repeat(1,P,1,1,1) #B,P,T,N,3
        x_multi -= translation.unsqueeze(-2).unsqueeze(-2)
        x_multi = (rotation.unsqueeze(3).expand(-1,-1,-1,N,-1,-1) @ x_multi.unsqueeze(-1)).squeeze(-1)
        x_multi += translation.unsqueeze(-2).unsqueeze(-2)

        if cat_flame_label:
            flame_vec = torch.zeros_like(x_multi[:,:,:,:,0], dtype=torch.long) #B,P,T,N
            for i in range(T):
                flame_vec[:,:,i,:] = i
            flame_vec = F.one_hot(flame_vec, num_classes=T)
            x_multi = torch.cat([x_multi, flame_vec], dim=-1)#B,P,T,N,(3+T)

        return x_multi
    
    def multi_flame_convergence(self, x_muiti, rotation, translation, parts_label):
        '''
        x_multi: B,P,T,N,3
        rotation: B,P,T,3,3
        translation: B,P,3
        parts_label: B,P,T,N
        '''
        B,P,T,N = parts_label.shape
        rotation_inv = torch.inverse(rotation)
        x_convergence = x_muiti - translation.unsqueeze(-2).unsqueeze(-2) ##B,P,T,N,3
        x_convergence = (rotation_inv.unsqueeze(3).expand(-1,-1,-1,N,-1,-1) @ x_convergence.unsqueeze(-1)).squeeze(-1)#B,P,T,N,3
        x_convergence += translation.unsqueeze(-2).unsqueeze(-2) ##B,P,T,N,3
        x_convergence = torch.sum(x_convergence * parts_label.unsqueeze(-1), dim=1)#B,T,N,3
        return x_convergence

    
    def decode_by_multi_flame(self, x_multi, c_g):
        '''
        x_multi: B,P,T,N,3
        c_g: B,P,256
        '''
        B,P,T,N,_ = x_multi.shape

        x = x_multi.reshape(B*P, T*N, 3)
        g = c_g.reshape(B*P, -1)

        parts_logits = self.network_dict["canonical_geometry_decoder"](x, None, g).reshape(B,P,T,N) #B,P,T,N
        parts_label = F.one_hot(torch.argmax(parts_logits, dim=1), num_classes=P).permute(0,3,1,2) #B,P,T,N

        return parts_logits, parts_label
    
    def joint_decoder_loss(self, axis, length, theta_gt, trans, set_pc, static_mask):
        '''
        axis: B,P,3
        trans: B,P,3
        length: B,P,T
        theta_gt: B,T,P-1
        trans:B,P,3
        set_pc: B,T,N,3
        static_mask:B,T,N
        axis_gt:B,P,3
        '''
        B,P,T = length.shape

        #rotation axis norm remain 1
        axis_length = torch.norm(axis.reshape(B*P, 3), dim=1)
        axis_loss = torch.mean((axis_length - 1) ** 2)

        #all rotation axis element >=0
        abs_loss = torch.mean(torch.abs(axis.reshape(-1)) - axis.reshape(-1))

        #-2pi <= length < 2pi
        range_diff = length.flatten()**2 - 4*math.pi**2
        range_loss = torch.mean(torch.maximum(range_diff, torch.zeros_like(range_diff)))

        #trans should be closed to static parts
        pc = set_pc.reshape(B,-1,3)#B,TN,3
        mask = static_mask.reshape(B,-1)#B,TN
        trans_loss = []
        for b in range(B):
            static_pc = pc[b, mask[b], :]
            distance = trans[b,:-1,:].unsqueeze(1) - static_pc.unsqueeze(0).expand(P-1,-1,-1)#P-1,M,3
            distance = torch.sum(distance ** 2, dim=-1) #P-1,M
            distance =  torch.mean(1 - torch.exp(-5 * distance), dim=-1)#P-1
            distance = torch.mean(distance)
            trans_loss.append(distance)
        trans_loss = torch.mean(torch.stack(trans_loss, dim=0), dim=0)
        #trans_loss = torch.mean(torch.maximum(torch.abs(trans.flatten()) - 1, torch.zeros_like(trans.flatten())))

        
        #static parts should have same rotation length
        static_loss = torch.mean(torch.var(length[:,-1,:], dim=-1))

        #theta difference should be the same as the theta_gt's difference
        theta_gt_diff = theta_gt[:,1:,:] - theta_gt[:,0,:].unsqueeze(1)
        length_diff = (length[:,:P-1,1:] - length[:,:P-1,0].unsqueeze(2)).permute(0,2,1)
        #theta_gt_diff = math.pi/2 - theta_gt
        #length_diff = length[:,:P-1,:].permute(0,2,1)
        theta_loss = torch.mean((theta_gt_diff.flatten() + length_diff.flatten()) ** 2)
        
        return axis_loss + trans_loss #+ abs_loss + static_loss + theta_loss + range_loss 
    
    def norm_projection_loss(self, point, axis, trans):
        '''
        point: B,T,N,3
        axis:B,P,3
        trans:B,P,3
        '''
        B,T,N,_ = point.shape
        P = axis.shape[1]
        part_point = point.unsqueeze(1).repeat(1,P,1,1,1) - trans.reshape(B,P,1,1,3)#B,P,T,N,3
        distribution = torch.sum(part_point * axis.reshape(B,P,1,1,3), dim=-1)#B,P,T,N
        distance = torch.mean(distribution, dim=-1)[:,:-1,:]#B,P-1,T
        variance = torch.var(distance, dim=-1)#B,P-1
        projection_loss = 100 * torch.mean(variance.flatten())

        return projection_loss
    
    def chamfer_loss(self, query, others, label=None):
        if label!=None:
            label = label.reshape(-1, label.shape[-1]) #BPT,N
        loss_sum = chamfer_distance(query, others, weights=label)
        return loss_sum
    
    def align_loss(self, query, others, B, P, T, c_trans, c_axis, label=None):
        loss_sum, static_mask, active_mask = align_chamfer(query, others, B, P, T, c_trans, c_axis, classifier_label=label)
        return loss_sum, static_mask, active_mask

    def binary_split(self, x):
        B,P,T,N,_ = x.shape
        exclude = [[list(range(i)), list(range(i+1, T))]for i in range(T)]
        exclude = [exclude[i][0] + exclude[i][1] for i in range(T)] #T,T-1
        others = torch.stack([x.unsqueeze(2)[:,:,:,exclude[i],:,:3] for i in range(T)], dim=2)#B,P,T,T-1,N,3
        others = others.reshape(B*P*T,-1,3)#BPT,(T-1)N,3
        query = x.reshape(B*P*T, N, -1)[:,:,:3] #BPT,N,3
        return query, others

    def rotation_vec_2_matrix(self, axis, length):
        '''
        axis: B,P,3
        lenght: B,P,T
        mtx: B,P,T,3,3
        '''
        B,P,T = length.shape
        rotation_vec = axis.unsqueeze(2).expand(-1,-1,T,-1) * length.unsqueeze(-1).expand(-1,-1,-1,3)
        rotation_mtx = ExpSO3(rotation_vec)

        return rotation_mtx


    #https://albert.growi.cloud/627ceb255286c3b02691384d
def hat(phi):
    phi_x = phi[..., 0]
    phi_y = phi[..., 1]
    phi_z = phi[..., 2]
    zeros = torch.zeros_like(phi_x)

    phi_hat = torch.stack([
        torch.stack([ zeros, -phi_z,  phi_y], dim=-1),
        torch.stack([ phi_z,  zeros, -phi_x], dim=-1),
        torch.stack([-phi_y,  phi_x,  zeros], dim=-1)
    ], dim=-2)
    return phi_hat


def ExpSO3(phi, eps=1e-4):
    theta = torch.norm(phi, dim=-1)
    phi_hat = hat(phi)
    I = torch.eye(3, device=phi.device)
    coef1 = torch.zeros_like(theta)
    coef2 = torch.zeros_like(theta)

    ind = theta < eps

    # strict
    _theta = theta[~ind]
    coef1[~ind] = torch.sin(_theta) / _theta
    coef2[~ind] = (1 - torch.cos(_theta)) / _theta**2

    # approximate
    _theta = theta[ind]
    _theta2 = _theta**2
    _theta4 = _theta**4
    coef1[ind] = 1 - _theta2/6 + _theta4/120
    coef2[ind] = .5 - _theta2/24 + _theta4/720

    coef1 = coef1[..., None, None]
    coef2 = coef2[..., None, None]
    return I + coef1 * phi_hat + coef2 * phi_hat @ phi_hat