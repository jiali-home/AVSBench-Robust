import torch
import torch.nn
import os
# from mmcv import Config
from mmengine.config import Config
import argparse
import sys
sys.path.append('path/AVSegformer/')
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure, save_mask
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset

def calculate_FPR(pred, eps=1e-7, size_average=True):
    r"""
    Calculate False Positive Rate (FPR).
    
    Args:
        pred: Predicted localization map, size [N x H x W].
        eps: Small epsilon to avoid division by zero.
        size_average: If True, returns the average FPR over the batch; if False, returns FPR for each sample.
    
    Returns:
        FPR: Tensor of size [1] if size_average=True, else size [N].
    """
    assert len(pred.shape) == 3  # Ensure input is [N x H x W]

    # Apply sigmoid to convert predictions to probability map, then threshold to binary mask
    pred = torch.sigmoid(pred)
    binary_pred = (pred > 0.5).int()
    
    # Calculate the percentage of activated area for each image in the batch
    N, H, W = binary_pred.shape
    num_pixels = H * W
    FPR_per_sample = binary_pred.sum(dim=(1, 2)) / (num_pixels + eps)  # [N]

    # Return the average FPR if size_average is True, otherwise return FPR per sample
    if size_average:
        return FPR_per_sample.mean()
    else:
        return FPR_per_sample


def main():
    # logger
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)

    # model
    if args.model is not None:
        cfg.model['type'] = args.model

    model = build_model(**cfg.model)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    logger.info('Load trained model %s' % args.weights)

    # Test data
    test_dataset = build_dataset(**cfg.dataset.test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.dataset.test.batch_size,
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')
    avg_meter_FPR = pyutils.AverageMeter('FPR')

    # Test
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
            imgs, audio, mask, category_list, video_name_list = batch_data

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
                output, _, similarity = model(audio, imgs)
                print('similarity', torch.sigmoid(similarity), (similarity))

            if args.save_pred_mask:
                mask_save_path = os.path.join(
                    args.save_dir, dir_name, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path,
                          category_list, video_name_list)

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(output.squeeze(1), mask)
            avg_meter_F.add({'F_score': F_score})

            FPR = calculate_FPR(output.squeeze(1))  # New metric for each batch/frame
            avg_meter_FPR.add({'FPR': FPR})
            logger.info(f'video: {video_name_list}, test miou: {miou.item()}, F_score: {F_score}, FPR: {FPR}')


        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        FPR = avg_meter_FPR.pop('FPR')

        print('test miou:', miou.item())
        print('test F_score:', F_score)
        print('test FPR:', FPR)
        logger.info(f'test miou: {miou.item()}, F_score: {F_score}, FPR: {FPR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('weights', type=str, help='model weights path')
    parser.add_argument("--save_pred_mask", action='store_true',
                        default=False, help="save predited masks or not")
    parser.add_argument('--save_dir', type=str,
                        default='work_dir', help='save path')
    parser.add_argument("--model", default="AVSegFormer", type=str)

    args = parser.parse_args()
    main()
