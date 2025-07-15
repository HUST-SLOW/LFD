import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import time
from thop import profile
import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve
from skimage import measure
import PIL
import matplotlib.pyplot as plt
import csv
from torch.optim import SGD, Adam
from models.scheduler.lr_scheduler import LR_Scheduler
from models.loss.LFD_loss import FSOhemCELoss
from models.net.LFD_core import LFD_Core
from utils.utils import AverageMeter, intersectionAndUnion, iou_curve, normalization01, denormalization
from datasets import mvtec_ad, mvtec_3d_ad,visa

class LFD():
    def __init__(self, cfg):
        self.cfg = cfg
        self.scale_factor = cfg['training']['scale_factor']
        self.weight_model = ""

        if cfg['training']['state'] and cfg['training']['start_epoch'] > 0:
            self.weight_model = os.path.join(
                cfg['outputs']['ckpt_path'], 'model_epoch_{}.pth'.format(cfg['training']['start_epoch']))
            assert os.path.exists(self.weight_model), "checkpoint does not exist!"

    def _load_dataloader(self, dataset_name, class_name=None, train=True):
        if dataset_name == 'mvtec_ad':
            if train:
                return mvtec_ad.mvtec_train_dataloader(self.cfg, class_name=class_name)
            else:
                return mvtec_ad.mvtec_test_dataloader(self.cfg, class_name=class_name)
        elif dataset_name == 'mvtec_3d_ad':
            if train:
                return mvtec_3d_ad.mvtec3d_train_dataloader(self.cfg, class_name=class_name)
            else:
                return mvtec_3d_ad.mvtec3d_test_dataloader(self.cfg, class_name=class_name)
        elif dataset_name == 'visa':
            if train:
                return visa.visa_train_dataloader(self.cfg, class_name=class_name)
            else:
                return visa.visa_test_dataloader(self.cfg, class_name=class_name)
        else:
            raise NotImplementedError("Dataset not implemented")

    def _train(self, dataset_name):
        # uni_training: Training all classes in one model
        self.model = LFD_Core(self.scale_factor)
        if len(self.weight_model) > 0:
            param_dict = torch.load(self.weight_model, map_location='cpu')
            param_dict = {k.replace("module.", ""): param_dict.pop(k) for k in list(param_dict.keys())}
            self.model.load_state_dict(param_dict)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()

        self.train_dataloader = self._load_dataloader(dataset_name, train=True)
        self.optimizer = SGD(self.model.parameters(), lr=self.cfg['training']['lr'], momentum=0.9, weight_decay=1e-4)
        self.scheduler = LR_Scheduler(self.optimizer, 0, 0, self.cfg['training']['start_epoch'], self.cfg['training']['max_epochs'], self.cfg['training']['lr'], 1e-5, len(self.train_dataloader))
        self.global_progress = tqdm(range(self.cfg['training']['start_epoch'], self.cfg['training']['max_epochs']), desc="Training")
        self.loss_func = FSOhemCELoss(ignore_index=-1)

        for epoch in self.global_progress:
            local_progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.cfg['training']['max_epochs']}", 
                                    disable=self.cfg['training']['hide_bar'], dynamic_ncols=True)
            ave_loss = AverageMeter()
            self.model.train()
            for i, batch in enumerate(local_progress):
                self.model.zero_grad()
                batch = {k: batch[k].cuda() for k in ["img", "mask"]}
                y_pred = self.model(batch)
                loss = self.loss_func(y_pred, batch["mask"])
                loss.backward()
                lr = self.scheduler.step()
                self.optimizer.step()
                ave_loss.update(loss.data.item())
                local_progress.set_postfix({"lr": lr, "loss": ave_loss.average()})
            if (epoch + 1) % self.cfg['training']['save_epoch'] == 0:
                torch.save(self.model.state_dict(), f'{self.cfg["outputs"]["ckpt_path"]}/model_epoch_{epoch+1}.pth')

    def train(self):
        dataset_name = self.cfg['datasets']['dataset_name']
        # Training all classes in one model
        if self.cfg['datasets']['uni_training']:
            self._train(dataset_name)
        # Training each class in one model
        else:
            for class_name in self._get_classnames(dataset_name):
                os.makedirs(os.path.join(self.cfg['outputs']['ckpt_path'], class_name), exist_ok=True)
                print(f'training------:{class_name}')
                self._train(dataset_name, class_name)

    def _get_classnames(self, dataset_name):
        if dataset_name == 'mvtec_ad':
            return mvtec_ad._CLASSNAMES
        elif dataset_name == 'mvtec_3d_ad':
            return mvtec_3d_ad._CLASSNAMES
        elif dataset_name == 'visa':
            return visa._CLASSNAMES
        else:
            raise NotImplementedError("Dataset not implemented")

    def test(self, class_name=None):
        classname_list = self._get_classnames(self.cfg['datasets']['dataset_name'])
        epoch = self.cfg['testing']['test_epoch']
        print(f'testing epoch: {epoch}')
        self.cfg['testing']['test_epoch'] = epoch
        
        metrics_dict = {}
        auroc_sp_ls, f1_sp_ls, ap_sp_ls = [], [], []
        auroc_px_ls, f1_px_ls, ap_px_ls = [], [], []
        iou_px_ls = []

        for class_name in classname_list:
            if self.cfg['testing']['state']:
                if self.cfg['datasets']['uni_training']:
                    self.weight_model = os.path.join(
                        self.cfg['outputs']['ckpt_path'], 'model_epoch_{}.pth'.format(self.cfg['testing']['test_epoch']))
                else:
                    self.weight_model = os.path.join(
                        self.cfg['outputs']['ckpt_path'], class_name, 'model_epoch_{}.pth'.format(self.cfg['testing']['test_epoch']))
                assert os.path.exists(self.weight_model), "checkpoint does not exist!"

            self.model = LFD_Core(self.scale_factor)
            if len(self.weight_model) > 0:
                param_dict = torch.load(self.weight_model, map_location='cpu')
                param_dict = {k.replace("module.", ""): param_dict.pop(k) for k in list(param_dict.keys())}
                self.model.load_state_dict(param_dict)
            self.model = self.model.cuda()

            self.model.eval()

            print(f'testing------:{class_name}')
            self.test_dataloader = self._load_dataloader(self.cfg['datasets']['dataset_name'], class_name=class_name, train=False)

            intersection_meter = AverageMeter()
            union_meter = AverageMeter()

            image_pred, image_gt = [], []
            pixel_pred, pixel_gt = [], []
            image_path_list, org_image_list = [], []
            IoU_list = []   # for iou

            for i, batch in tqdm(enumerate(self.test_dataloader)):
                assert batch["img"].shape[0] == 1
                batch["img"] = batch["img"].cuda()
                gt_mask = batch['mask'].cpu().numpy()

                image_path_list.extend(batch['image_path'])

                with torch.no_grad():
                    y_pred = self.model(batch)
                y_pred = F.softmax(y_pred, dim=1)
                pred = y_pred[:, 1, :, :]

                pred = pred.squeeze(0).cpu().numpy()

                intersection, union = intersectionAndUnion(pred, gt_mask, 2)
                intersection_meter.update(intersection), union_meter.update(union)

                image_pred.append(pred.max()), image_gt.append(gt_mask.max())
                pixel_pred.append(pred), pixel_gt.append(gt_mask)
                org_image_list.append(batch['img'].squeeze().permute(1, 2, 0).cpu().numpy())

                if gt_mask.astype(np.uint8).max() != 0:
                    fpr, iou, thresholds = iou_curve(gt_mask.astype(np.uint8).ravel(), pred.ravel())
                    iou_score = iou.max()
                    IoU_list.append(iou_score)

            pr_sp, gt_sp = np.array(image_pred).squeeze(), np.array(image_gt).squeeze()
            pr_px, gt_px = np.array(pixel_pred).squeeze(), np.array(pixel_gt).squeeze()
            gt_sp[gt_sp != 0] = 1
            gt_px[gt_px != 0] = 1
            gt_sp = gt_sp.astype(np.uint8)
            gt_px = gt_px.astype(np.uint8)

            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            ap_sp = average_precision_score(gt_sp, pr_sp)
            precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
            auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
            ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
            precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
            f1_scores = (2 * precisions * recalls) / (precisions + recalls)
            f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
            iou_px = np.mean(IoU_list)

            if self.cfg['testing']['vis']:
                class_vis_save_path = os.path.join(self.cfg['outputs']['vis_path'], f"Epoch_{self.cfg['testing']['test_epoch']}", class_name)
                visualization(pr_px, gt_px, org_image_list, image_path_list, self.cfg, class_vis_save_path)

            image_metric = auroc_sp, f1_sp, ap_sp
            pixel_metric = auroc_px, f1_px, ap_px
            iou_metric = iou_px
            metrics_dict[class_name] = (image_metric, pixel_metric, iou_metric)
            auroc_sp_ls.append(auroc_sp), f1_sp_ls.append(f1_sp), ap_sp_ls.append(ap_sp)
            auroc_px_ls.append(auroc_px), f1_px_ls.append(f1_px), ap_px_ls.append(ap_px)
            iou_px_ls.append(iou_px)

        auroc_sp_mean, f1_sp_mean, ap_sp_mean = np.mean(auroc_sp_ls), np.mean(f1_sp_ls), np.mean(ap_sp_ls)
        auroc_px_mean, f1_px_mean, ap_px_mean = np.mean(auroc_px_ls), np.mean(f1_px_ls), np.mean(ap_px_ls)
        iou_px_mean = np.mean(iou_px_ls)

        for i, class_name in enumerate(classname_list):
            print(class_name)
            print('image-level, auroc:{:.2f}, f1:{:.2f}, ap:{:.2f}'.format(auroc_sp_ls[i]*100, f1_sp_ls[i]*100, ap_sp_ls[i]*100))
            print('pixel-level, auroc:{:.2f}, f1:{:.2f}, ap:{:.2f}'.format(auroc_px_ls[i]*100, f1_px_ls[i]*100, ap_px_ls[i]*100))
            print('iou:{:.2f}'.format(iou_px_ls[i]*100))
            print("="*30)
        print('mean')
        print('image-level, auroc:{:.2f}, f1:{:.2f}, ap:{:.2f}'.format(auroc_sp_mean*100, f1_sp_mean*100, ap_sp_mean*100))
        print('pixel-level, auroc:{:.2f}, f1:{:.2f}, ap:{:.2f}'.format(auroc_px_mean*100, f1_px_mean*100, ap_px_mean*100))
        print('iou:{:.2f}'.format(iou_px_mean*100))

        csv_save_path = os.path.join(self.cfg['outputs']['vis_path'], f'metrics_epoch_{self.cfg["testing"]["test_epoch"]}.csv')
        with open(csv_save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class_name'] + list(metrics_dict.keys()) + ['mean'])
            writer.writerow(['auroc_sp'] + [f'{metrics_dict[class_name][0][0]*100:.2f}' for class_name in metrics_dict.keys()] + [f'{auroc_sp_mean*100:.2f}'])
            writer.writerow(['f1_sp'] + [f'{metrics_dict[class_name][0][1]*100:.2f}' for class_name in metrics_dict.keys()] + [f'{f1_sp_mean*100:.2f}'])
            writer.writerow(['ap_sp'] + [f'{metrics_dict[class_name][0][2]*100:.2f}' for class_name in metrics_dict.keys()] + [f'{ap_sp_mean*100:.2f}'])
            writer.writerow(['auroc_px'] + [f'{metrics_dict[class_name][1][0]*100:.2f}' for class_name in metrics_dict.keys()] + [f'{auroc_px_mean*100:.2f}'])
            writer.writerow(['f1_px'] + [f'{metrics_dict[class_name][1][1]*100:.2f}' for class_name in metrics_dict.keys()] + [f'{f1_px_mean*100:.2f}'])
            writer.writerow(['ap_px'] + [f'{metrics_dict[class_name][1][2]*100:.2f}' for class_name in metrics_dict.keys()] + [f'{ap_px_mean*100:.2f}'])
            writer.writerow(['iou_px'] + [f'{metrics_dict[class_name][2]*100:.2f}' for class_name in metrics_dict.keys()] + [f'{iou_px_mean*100:.2f}'])

        return

    def eval_fps(self):
        cudnn.benchmark = True

        self.model.eval()
        self.model = self.model.cuda()

        img_size = self.cfg['eval_speed']['size']
        input_size = (self.cfg['eval_speed']['batch_size'], self.cfg['eval_speed']['num_channels'], img_size[0], img_size[1])
        img = torch.randn(input_size, device='cuda:0')
        input = {'img': img}
        iteration = self.cfg['eval_speed']['iter']
        macs, params = profile(self.model, inputs=(input,))
        print("macs:", macs, "params:", params)
        for _ in range(50):
            self.model(input)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iteration):
            self.model(input)
            print("iteration {}/{}".format(_, iteration), end='\r')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start

        speed_time = elapsed_time / iteration * 1000
        fps = iteration / elapsed_time

        print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
        print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
        return speed_time, fps

def visualization(pr_px, gt_px, org_image_list, image_path_list, cfg, save_dir):
    """Visualization of the segmentation results
    Args:
        pr_px (np.array): predicted pixel-level segmentation mask
        gt_px (np.array): ground truth pixel-level segmentation mask
        org_image_list (list): original image list
        image_path_list (list): original image path list
        cfg (dict): configuration file
        save_dir (str): save directory
    """
    pr_px = normalization01(pr_px)
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(len(image_path_list)):
        image_path = image_path_list[idx].split('/')
        anomaly_type = image_path[-2]
        image_name = image_path[-1]

        save_segmentation_vis_path = os.path.join(save_dir, 'segmentation_vis', anomaly_type)
        os.makedirs(save_segmentation_vis_path, exist_ok=True)
        save_segmentation_vis_path = os.path.join(save_segmentation_vis_path, image_name)

        save_anomaly_map_path = os.path.join(save_dir, 'anomaly_map_vis', anomaly_type)
        os.makedirs(save_anomaly_map_path, exist_ok=True)
        save_anomaly_map_path = os.path.join(save_anomaly_map_path, image_name)

        anomaly_map = pr_px[idx].squeeze()
        anomaly_map = normalization01(anomaly_map) * 255
        anomaly_map = cv2.applyColorMap(anomaly_map.astype(np.uint8), cv2.COLORMAP_JET)

        gt_mask = gt_px[idx].squeeze()
        gt_mask = normalization01(gt_mask) * 255
        gt_mask = cv2.applyColorMap(gt_mask.astype(np.uint8), cv2.COLORMAP_JET)

        image = denormalization(org_image_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_anomaly_map = cv2.addWeighted(image, 0.5, anomaly_map, 0.5, 0)
        image_gt_mask = cv2.addWeighted(image, 0.5, gt_mask, 0.5, 0)

        gap = 10
        output_image = np.zeros((cfg['datasets']['img_resize'], cfg['datasets']['img_resize']*3+2*gap, 3), dtype=np.uint8) + 255
        output_image[:cfg['datasets']['img_resize'], :cfg['datasets']['img_resize']] = image
        output_image[:cfg['datasets']['img_resize'], cfg['datasets']['img_resize']+gap:cfg['datasets']['img_resize']*2+gap] = image_gt_mask
        output_image[:cfg['datasets']['img_resize'], cfg['datasets']['img_resize']*2+2*gap:cfg['datasets']['img_resize']*3+2*gap] = image_anomaly_map

        cv2.imwrite(save_anomaly_map_path, anomaly_map)
        cv2.imwrite(save_segmentation_vis_path, output_image)

if __name__ == "__main__":
    model = LFD_Core()
    model.eval()
    print(model)
    x = torch.rand(2, 3, 256, 256)
    pred = model({"img": x})
    print(pred.shape)