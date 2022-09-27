import torch
import copy
import os

from pcdet.datasets.augmentor.augmentor_utils import *
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from .detector3d_template import Detector3DTemplate
from .second_net import SECONDNet

class SECOND_SSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)
        self.second = SECONDNet(model_cfg=model_cfg,num_class=num_class,dataset=dataset)
        
        
        self.second_ema = SECONDNet(model_cfg=model_cfg_copy,num_class=num_class,dataset=dataset_copy)
        for param in self.second_ema.parameters():
            param.detach_()
        self.add_module('second', self.second)
        self.add_module('second_ema', self.second_ema)
        
        self.thresh = model_cfg.THRESH
        self.unlabeled_supervise_cls = model_cfg.UNLABELED_SUPERVISE_CLS
        self.unlabeled_weight = model_cfg.UNLABELED_WEIGHT
        self.register_buffer("precision", torch.zeros(1))
        self.register_buffer("recall", torch.zeros(1))
        
        
    def forward(self, batch_dict):
        if self.training:
            mask = batch_dict['mask'].view(-1)
            
            labeled_mask = torch.nonzero(mask).squeeze(1).long()
            unlabeled_mask = torch.nonzero(1-mask).squeeze(1).long()
            batch_dict_ema = {}
            keys = list(batch_dict.keys())
            for k in keys:
                if k + '_ema' in keys:
                    continue
                if k.endswith('_ema'):
                    batch_dict_ema[k[:-4]] = batch_dict[k]
                else:
                    batch_dict_ema[k] = batch_dict[k]
            
            with torch.no_grad():
                for cur_module in self.second_ema.module_list:
                    try:
                        batch_dict_ema = cur_module(batch_dict_ema, disable_gt_roi_when_pseudo_labeling=True)
                    except:
                        batch_dict_ema = cur_module(batch_dict_ema)
                pred_dicts, recall_dicts = self.second_ema.post_processing(batch_dict_ema, no_recall_dict=True)
                
                pseudo_boxes = []
                pseudo_scores = []
                pseudo_labels = []
                max_box_num = batch_dict['gt_boxes'].shape[1]
                max_pseudo_box_num = 0
                for ind in unlabeled_mask:
                    pseudo_score = pred_dicts[ind]['pred_scores']
                    pseudo_box = pred_dicts[ind]['pred_boxes']
                    pseudo_label = pred_dicts[ind]['pred_labels']
                    #无预测就送一个空label进去
                    if len(pseudo_label)==0:
                        pseudo_boxes.append(pseudo_label.new_zeros((0,8)).float())
                        continue
                    #按照类别选取了阈值
                    print(self.thresh)
                    conf_thresh = torch.tensor(self.thresh, device=pseudo_label.device)
                    conf_thresh = conf_thresh.unsqueeze(0).repeat(len(pseudo_label),1)
                    conf_thresh = conf_thresh.gather(dim=1, index = (pseudo_label-1).unsqueeze(-1))
                    
                    valid_inds = pseudo_score> conf_thresh.squeeze()  #bool
                    
                    pseudo_box = pseudo_box[valid_inds]
                    pseudo_label = pseudo_label[valid_inds]
                    
                    pseudo_boxes.append(torch.cat([pseudo_box, pseudo_label.view(-1,1).float()],dim=1))
                    if pseudo_box.shape[0]> max_pseudo_box_num:
                        max_pseudo_box_num = pseudo_box.shape[0]
                
                max_box_num = batch_dict['gt_boxes'].shape[1]
                
                ori_unlabeled_boxes = batch_dict['gt_boxes'][unlabeled_mask, ...]
                
                #填充整齐各个label
                if max_box_num >= max_pseudo_box_num:
                    for i, pseudo_box in enumerate(pseudo_boxes):
                        diff = max_box_num- pseudo_box.shape[0]
                        if diff >0:
                            pseudo_box = torch.cat([pseudo_box, torch.zeros((diff,8), device=pseudo_box.device)], dim=0)  #0补全
                        batch_dict['gt_boxes'][unlabeled_mask[i]] = pseudo_box
                else:
                    ori_boxes = batch_dict['gt_boxes']
                    new_boxes = torch.zeros((ori_boxes.shape[0], max_pseudo_box_num,ori_boxes.shape[2]),
                                            device = ori_boxes.device)
                    #分别用0去填补有标签和为标签数据
                    for i, inds in enumerate(labeled_mask):
                        diff = max_pseudo_box_num- ori_boxes[inds].shape[0]
                        new_box = torch.cat([ori_boxes[inds],torch.zeros((diff,8),device=ori_boxes.device)], dim=0)
                        new_boxes[inds] = new_box
                    
                    for i ,pseudo_box in enumerate(pseudo_boxes):
                        diff = max_pseudo_box_num- pseudo_box.shape[0]
                        if diff>0:
                            pseudo_box = torch.cat([pseudo_box,torch.zeros((diff,8), device=pseudo_box.device)],dim=0)
                        new_boxes[inds] = pseudo_box
                    batch_dict['gt_boxes'] = new_boxes
                
                batch_dict['gt_boxes'][unlabeled_mask, ...] = random_flip_along_x_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['flip_x'][unlabeled_mask, ...]
                )
                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_rotation_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['noise_rot'][unlabeled_mask, ...]
                )
                batch_dict['gt_boxes'][unlabeled_mask, ...] = global_scaling_bbox(
                    batch_dict['gt_boxes'][unlabeled_mask, ...], batch_dict['noise_scale'][unlabeled_mask, ...]
                )

                precision, recall = self.get_pseudo_label_quality(
                        unlabeled_mask, batch_dict['gt_boxes'], ori_unlabeled_boxes)
                self.precision = precision
                self.recall = recall
                #self.precision = 0.9*self.precision + precision*0.1
                #self.recall = 0.9*self.recall + recall*0.1
                        
            for cur_module in self.second.module_list:
                batch_dict = cur_module(batch_dict)
            
            disp_dict= {}
            loss_rpn_cls, loss_rpn_box, tb_dict = self.second.dense_head.get_loss(scalar = False)
            
            if not self.unlabeled_supervise_cls:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum()
            else:
                loss_rpn_cls = loss_rpn_cls[labeled_mask, ...].sum() +  loss_rpn_cls[unlabeled_mask, ...].sum() * self.unlabeled_weight
            
            loss_rpn_box = loss_rpn_box[labeled_mask, ...].sum() + loss_rpn_box[unlabeled_mask, ...].sum() * self.unlabeled_weight
            
            loss = loss_rpn_box+loss_rpn_cls
            
            tb_dict_ = {}
            for key in tb_dict.keys():
                if 'loss' in key:
                    tb_dict_[key+ "_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key+ "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                elif 'acc' in key:
                    tb_dict_[key + "_labeled"] = tb_dict[key][labeled_mask, ...].sum()
                    tb_dict_[key + "_unlabeled"] = tb_dict[key][unlabeled_mask, ...].sum()
                else:
                    tb_dict_[key] = tb_dict[key]
            
            tb_dict_['max_box_num'] = max_box_num
            tb_dict_['max_pseudo_box_num'] = max_pseudo_box_num
            tb_dict_['precision'] = self.precision
            tb_dict_['recall'] = self.recall
            
            ret_dict = {
                'loss': loss
            }
            
            return ret_dict, tb_dict_, disp_dict
        else:
            for cur_module in self.second.module_list:
                batch_dict = cur_module(batch_dict)
            
            pred_dicts, recall_dicts = self.second.post_processing(batch_dict)
            
            return pred_dicts, recall_dicts
    
    def update_global_step(self):
        self.global_step +=1
        alpha = 0.999
        for ema_param , param in zip(self.second_ema.parameters(), self.second.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)
    
    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        
        logger.info('==> Loading parameters from checkpoint %s to %s' %(filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        
        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' %checkpoint['version'])
            logger.info('==>1111111111111111111111111111111111111111111111')
        
        updata_model_state = {}
        for key, val in model_state_disk.items():
            new_key = 'second.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                updata_model_state[new_key] = val
            new_key = 'second_ema.'+ key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                updata_model_state[new_key] = val
            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                updata_model_state[new_key] = val
                
        state_dict = self.state_dict()
        state_dict.update(updata_model_state)
        self.load_state_dict(state_dict)
        
        for key in state_dict :
            if key not in updata_model_state:
                logger.info('Not updated weight %s: %s' %(key, str(state_dict[key].shape)))
                
        logger.info('-->Done (loaded %d/%d)' %(len(updata_model_state), len(self.state_dict())))
    
    def get_pseudo_label_quality(self, unlabeled_mask, det_bboxes, gt_boxes):
        r = []
        p = []
        for i, ind in enumerate(unlabeled_mask) :
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(
                det_bboxes[ind, ...][:, 0:7],
                gt_boxes[i, :, 0:7])
            cls_pseudo = det_bboxes[ind, ...][:, 7]
            unzero_inds = torch.nonzero(cls_pseudo).squeeze(1).long()
            cls_pseudo = cls_pseudo[unzero_inds]
            if len(unzero_inds) > 0:
                iou_max, asgn = anchor_by_gt_overlap[unzero_inds, :].max(dim=1)
                fg = (iou_max > 0.5) & (gt_boxes[i][:, 7].gather(dim=0, index=asgn) == cls_pseudo)
                num_tp, num_fp = fg.sum(), (~fg).sum()
                num_gt = gt_boxes[i].shape[0]
                recall = num_tp/num_gt
                precision = num_tp/(num_tp+num_fp)
            else:
                recall = torch.tensor(0, device = unlabeled_mask.device)
                precision = torch.tensor(0, device = unlabeled_mask.device)
            r.append(recall)
            p.append(precision)
        print(p)
        print(r)
        #p = torch.cat(p).float().mean()
        #r = torch.cat(r).float().mean()
        return precision, recall