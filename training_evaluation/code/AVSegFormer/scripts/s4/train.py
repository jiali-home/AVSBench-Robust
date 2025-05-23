import torch
import time
import torch.nn
import torch.nn.functional as F
import os
import random
import numpy as np
# from mmcv import Config
from mmengine.config import Config

import sys
sys.path.append('xxx/AVSegformer/')
import argparse
from utils import pyutils
from utils.loss_util import LossUtil
from utility import mask_iou
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
from loss import IouSemanticAwareLoss


def main():
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # logger
    log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, dir_name)):
        os.mkdir(os.path.join(args.log_dir, dir_name))
    log_file = os.path.join(args.log_dir, dir_name, f'{log_name}.log')
    logger = getLogger(log_file, __name__)
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)
    checkpoint_dir = os.path.join(args.checkpoint_dir, dir_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # model
    if args.model is not None:
        cfg.model['type'] = args.model

    model = build_model(**cfg.model)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    logger.info("Total params: %.2fM" % (sum(p.numel()
                for p in model.parameters()) / 1e6))

    # dataset
    if args.misaligned_percentage is not None:
        cfg.dataset['train']['misaligned_percentage'] = args.misaligned_percentage
    train_dataset = build_dataset(**cfg.dataset.train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.dataset.train.batch_size,
                                                   shuffle=True,
                                                   num_workers=cfg.process.num_works,
                                                   pin_memory=True)
    max_step = (len(train_dataset) // cfg.dataset.train.batch_size) * \
        cfg.process.train_epochs
    val_dataset = build_dataset(**cfg.dataset.val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg.dataset.val.batch_size,
                                                 shuffle=False,
                                                 num_workers=cfg.process.num_works,
                                                 pin_memory=True)

    # optimizer
    optimizer = pyutils.get_optimizer(model, cfg.optimizer)
    loss_util = LossUtil(**cfg.loss)
    avg_meter_miou = pyutils.AverageMeter('miou')

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    for epoch in range(cfg.process.train_epochs):
        if epoch == cfg.process.freeze_epochs:
            model.module.freeze_backbone(False)

        for n_iter, batch_data in enumerate(train_dataloader):
            # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            imgs, audio, mask, label = batch_data
            imgs, audio, mask = imgs.cuda(non_blocking=True), audio.cuda(non_blocking=True), mask.cuda(non_blocking=True)
            label = label.cuda()

            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask = mask.view(B, H, W)
            audio = audio.view(-1, audio.shape[2],
                               audio.shape[3], audio.shape[4])


            if args.model == 'AVSegFormer':
                output, mask_feature = model(audio, imgs)
            else:
                output, mask_feature, similarity = model(audio, imgs)
                label = label.expand(similarity.size(0))  # [1] -> [5]
                bce_loss = F.binary_cross_entropy(torch.sigmoid(similarity), label)

            loss, loss_dict = IouSemanticAwareLoss(
                output, mask_feature, mask.unsqueeze(1).unsqueeze(1), **cfg.loss)
                        
            if args.bce:
                loss += bce_loss

            loss_util.add_loss(loss, loss_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if (global_step - 1) % 100 == 0:
                base_log = 'Iter:%5d/%5d, %slr: %.6f, loss: %.4f' % (
                    global_step - 1, 
                    max_step, 
                    loss_util.pretty_out(), 
                    optimizer.param_groups[0]['lr'], 
                    loss.item()
                )
                
                if args.bce and 'bce_loss' in locals():
                    train_log = base_log + ', bce_loss: %.4f' % bce_loss.item()
                else:
                    train_log = base_log
                logger.info(train_log)

        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
                imgs, audio, mask, _, _ = batch_data

                imgs = imgs.cuda()
                audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * frame, H, W)
                audio = audio.view(-1, audio.shape[2],
                                audio.shape[3], audio.shape[4])

                if args.model == 'AVSegFormer':
                    output, _ = model(audio, imgs)
                else:
                    output, _, _ = model(audio, imgs)

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join(
                    checkpoint_dir, '%s_best.pth' % (args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s' % model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = f'Epoch: {epoch}, Miou: {miou}, maxMiou: {max_miou}'
            # print(val_log)
            logger.info(val_log)
    
        torch.cuda.empty_cache()
        model.train()
        
    logger.info(f'Best val Miou {max_miou} at epoch: {best_epoch}')
    # Save the final model after training
    final_model_path = os.path.join(
        checkpoint_dir, '%s_final.pth' % (args.session_name))
    torch.save(model.module.state_dict(), final_model_path)
    logger.info('Saved final model to %s' % final_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str,
                        default='work_dir', help='log dir')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='work_dir', help='dir to save checkpoints')
    parser.add_argument("--session_name", default="S4",
                        type=str, help="the S4 setting")

    parser.add_argument("--empty_loss", default="False", action='store_true')
    parser.add_argument("--gumbel", default="False", action='store_true')
    parser.add_argument('--misaligned_percentage', type=int, default=None, help="Percentage of misaligned data for MS3Dataset_ood")
    parser.add_argument('--empty_lambda', type=float, default=0.5)
    parser.add_argument("--model", default="AVSegFormer", type=str)
    parser.add_argument("--bce", default="False", action='store_true')

    args = parser.parse_args()
    main()
