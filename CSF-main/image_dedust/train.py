import os
import torch
from data import train_dataloader
from utils import Adder, Timer
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler
from losses import LossNetwork
from torchvision.models import vgg16

def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()

    vgg_model = vgg16(pretrained=True)
    vgg_model = vgg_model.features[:16].cuda()
    for param in vgg_model.parameters():
        param.requires_grad = False

    ploss = LossNetwork(vgg_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)

    warmup_epochs=3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter()
    epoch_l1_adder = Adder()
    epoch_per_adder = Adder()
    epoch_loss_adder = Adder()

    iter_l1_adder = Adder()
    iter_per_adder = Adder()
    iter_loss_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)
            loss_l1s = criterion(pred_img, label_img)
            label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(pred_img, dim=(-2,-1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)
            loss_l1f = criterion(pred_fft3, label_fft3)
            loss_per = ploss(pred_img, label_img)
            loss_l1 = loss_l1s + loss_l1f

            loss = 0.1 * loss_l1 + 0.05 * loss_per
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

            iter_l1_adder(loss_l1.item())
            iter_per_adder(loss_per.item())
            iter_loss_adder(loss.item())

            epoch_l1_adder(loss_l1.item())
            epoch_per_adder(loss_per.item())
            epoch_loss_adder(loss.item())

            if (iter_idx + 1) % args.print_freq == 0:
                print("Epoch: %03d LR: %.10f Total loss: %7.4f L1 loss: %7.4f Per loss: %7.4f " % (
                    epoch_idx, scheduler.get_lr()[0], iter_loss_adder.average(), iter_l1_adder.average(),
                    iter_per_adder.average()))

                writer.add_scalar('Total loss', iter_loss_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('L1 Loss', iter_l1_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('Per loss', iter_per_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

                iter_timer.tic()
                iter_l1_adder.reset()
                iter_per_adder.reset()

        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        if epoch_idx % args.save_freq == 0:
         print("EPOCH: %02d\nElapsed time: %4.2f Epoch l1 Loss: %7.4f Epoch per Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_l1_adder.average(), epoch_per_adder.average()))
        epoch_l1_adder.reset()
        epoch_per_adder.reset()
        scheduler.step()

        if epoch_idx % args.valid_freq == 0:
            val_its = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val_its))
            writer.add_scalar('PSNR', val_its, epoch_idx)
            if val_its >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
