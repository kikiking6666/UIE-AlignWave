import json
import warnings
import os

import torchvision
from metrics.uciqe import batch_uciqe
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from tqdm import tqdm
from lpips import LPIPS

from config import Config
from models import *
from utils import *

warnings.filterwarnings("ignore")


def test():
    opt = Config("config.yml")
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator()
    device = accelerator.device

    # -------------------------------
    # Data Loader
    # -------------------------------
    val_dir = opt.TESTING.VAL_DIR
    img_options = {"w": opt.TRAINING.PS_W, "h": opt.TRAINING.PS_H}

    if opt.TESTING.MODE == "noref":
        from data.dataset_RGB import NonRefDataReader

        val_dataset = NonRefDataReader(
            val_dir,
            inp=opt.TESTING.INPUT,
            mode="test",
            ori=opt.TRAINING.ORI,
            img_options=img_options,
        )
    else:  # 默认 ref
        from data.dataset_RGB import DataReader

        val_dataset = DataReader(
            val_dir,
            inp=opt.TESTING.INPUT,
            tar=opt.TESTING.TARGET,
            mode="test",
            ori=opt.TRAINING.ORI,
            img_options=img_options,
        )

    testloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True,
    )

    # -------------------------------
    # Model
    # -------------------------------
    model = UIR_PolyKernel_Wavelet()
    load_checkpoint(model, opt.TESTING.WEIGHT)
    model, testloader = accelerator.prepare(model, testloader)

    # LPIPS (only for ref mode)
    lpips_model = LPIPS(net="vgg").to(device) if opt.TESTING.MODE == "ref" else None

    model.eval()

    # -------------------------------
    # Metrics accumulator
    # -------------------------------
    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_uciqe = 0
    stat_lpips = 0

    # -------------------------------
    # Evaluation loop
    # -------------------------------
    for _, test_data in enumerate(tqdm(testloader)):
        if opt.TESTING.MODE == "noref":
            inp, filename = test_data
            tar = None
        else:
            inp, tar, filename = test_data

        with torch.no_grad():
            res = model(inp)

        # Save result images
        if opt.TESTING.SAVE_IMAGES:
            if not os.path.isdir(opt.TESTING.RESULT_DIR):
                os.makedirs(opt.TESTING.RESULT_DIR)
            torchvision.utils.save_image(
                res, os.path.join(opt.TESTING.RESULT_DIR, filename[0])
            )

        # Metrics
        if opt.TESTING.MODE == "ref":
            stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
            stat_ssim += structural_similarity_index_measure(
                res, tar, data_range=1
            ).item()
            stat_uciqe += batch_uciqe(res)
            stat_lpips += lpips_model(res, tar).item()
        else:  # noref
            stat_uciqe += batch_uciqe(res)

    # -------------------------------
    # Finalize stats
    # -------------------------------
    if opt.TESTING.MODE == "ref":
        stat_psnr /= size
        stat_ssim /= size
        stat_uciqe /= size
        stat_lpips /= size
    else:
        stat_uciqe /= size

    test_info = "Test Result on {}, checkpoint {}, testing data {}".format(
        opt.MODEL.SESSION, opt.TESTING.WEIGHT, opt.TESTING.VAL_DIR
    )
    if opt.TESTING.MODE == "ref":
        log_stats = "PSNR: {:.4f}, SSIM: {:.4f}, UCIQE: {:.4f}, LPIPS: {:.4f}".format(
            stat_psnr, stat_ssim, stat_uciqe, stat_lpips
        )
    else:
        log_stats = "UCIQE: {:.4f}".format(stat_uciqe)

    print(test_info)
    print(log_stats)

    # Save log
    with open(
        os.path.join(opt.LOG.LOG_DIR, opt.TESTING.LOG_FILE), mode="a", encoding="utf-8"
    ) as f:
        f.write(json.dumps(test_info) + "\n")
        f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    test()


