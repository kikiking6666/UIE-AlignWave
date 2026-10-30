import json
import warnings

import torch.optim as optim
from accelerate import Accelerator

from torch.utils.data import DataLoader

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from config import Config
from data import get_data

from DWT import DWT

from metrics.uciqe import batch_uciqe

from models import *
from utils import *

warnings.filterwarnings('ignore')





def train():
    # Accelerate
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator(log_with='wandb') if opt.OPTIM.WANDB else Accelerator()
    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)
    device = accelerator.device

    config = {
        "dataset": opt.TRAINING.TRAIN_DIR
    }
    accelerator.init_trackers("UW", config=config)

    # Data Loader
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = get_data(train_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
                             {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    trainloader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, num_workers=16,
                             drop_last=False, pin_memory=True, shuffle=True)# 将ImbalancedDatasetSampler改成shuffle=True

    val_dataset = get_data(val_dir, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                            pin_memory=True)

    # Model & Loss
    # model = UIR_PolyKernel()
    model = UIR_PolyKernel_Wavelet()

    criterion_psnr = torch.nn.SmoothL1Loss()
    criterion_l1 = torch.nn.L1Loss()  # 用于子带监督

    # Optimizer & Scheduler
    optimizer_b = optim.AdamW(model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=opt.OPTIM.LR_MIN)

    start_epoch = 1

    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    best_psnr_epoch = 1
    best_psnr = 0

    size = len(testloader)

    # training
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        model.train()

        for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
            inp = data[0].contiguous()
            tar = data[1]

            # forward
            optimizer_b.zero_grad()
            res = model(inp)

            #子带监督
            dwt = DWT().to(res.device)

            # 对 GT 做 DWT 分解
            gt_subbands = dwt(tar)
            LL_gt, HL_gt, LH_gt, HH_gt = torch.chunk(gt_subbands, 4, dim=0)

            # 对输出也做 DWT 分解
            out_subbands = dwt(res)
            LL_out, HL_out, LH_out, HH_out = torch.chunk(out_subbands, 4, dim=0)

            # 定义子带 loss
            loss_LL = criterion_l1(LL_out, LL_gt)  # 低频 supervision，保证颜色
            loss_HF = (criterion_l1(HL_out, HL_gt) +
                       criterion_l1(LH_out, LH_gt) +
                       criterion_l1(HH_out, HH_gt))  # 高频 supervision，保证细节

            # loss-function setting
            loss_psnr = criterion_psnr(res, tar)
            loss_ssim = 1 - structural_similarity_index_measure(res, tar, data_range=1)
            loss_uciqe = 1 - batch_uciqe(res)

            lambda_LL, lambda_HF = 0.1, 0.1  # 可调超参数
            train_loss = (loss_psnr +
                          0.2 * loss_ssim +
                          0.01 * loss_uciqe +
                          lambda_LL * loss_LL +
                          lambda_HF * loss_HF)

            # backward
            accelerator.backward(train_loss)
            optimizer_b.step()

        scheduler_b.step()

        # testing
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model.eval()
            psnr = 0
            ssim = 0

            uciqe = 0
            val_loss = 0

            for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                inp = data[0].contiguous()
                tar = data[1]

                with torch.no_grad():
                    res = model(inp)

                res, tar = accelerator.gather((res, tar))

                psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
                ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
                uciqe += batch_uciqe(res)

            psnr /= size
            ssim /= size
            uciqe /= size
            val_loss /= size

            # 每一轮都保存
            save_checkpoint({
                'state_dict': model.state_dict(),
            }, epoch, opt.MODEL.SESSION + f"_epoch{epoch}", opt.TRAINING.SAVE_DIR)

            # 仍然单独保存 best
            if psnr > best_psnr:
                best_psnr = psnr
                best_psnr_epoch = epoch
                save_checkpoint({
                    'state_dict': model.state_dict(),
                }, epoch, opt.MODEL.SESSION + "_best", opt.TRAINING.SAVE_DIR)


            accelerator.log({
                "PSNR": psnr,
                "SSIM": ssim,
            }, step=epoch)

            if accelerator.is_local_main_process:
                log_stats = ("epoch: {}, PSNR: {}, SSIM: {}, UCIQE: {}, "
                             "best PSNR: {}, best epoch: {}"
                             .format(epoch, psnr, ssim, uciqe, best_psnr, best_psnr_epoch))

                print(log_stats)
                with open(os.path.join(opt.LOG.LOG_DIR, opt.TRAINING.LOG_FILE), mode='a', encoding='utf-8') as f:
                    f.write(json.dumps(log_stats) + '\n')

    accelerator.end_training()


if __name__ == '__main__':
    train()
