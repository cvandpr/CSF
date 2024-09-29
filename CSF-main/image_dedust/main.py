import os
import torch
import argparse

from fvcore.nn import FlopCountAnalysis
from torch.backends import cudnn
from models.CSF import build_net
from eval import _eval
from train import _train
from train_ots import _train_ots

def main(args):
    cudnn.benchmark = True
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    mode = [args.mode, args.data]
    model = build_net(mode)

    print(model)
    print(f"network parameters: {sum(param.numel() for param in model.parameters()) / 1024 ** 2:.2f} M")

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train' and args.data == 'Indoor':
        _train(model, args)
    elif args.mode == 'train' and args.data == 'Outdoor':
        _train_ots(model, args)
    elif args.mode == 'test':
        _eval(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='CSF', type=str)
    parser.add_argument('--data', type=str, default='Indoor', choices=['Indoor', 'Outdoor'])


    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)
    parser.add_argument('--data_dir', type=str, default='/media/guo/test400')

    # Train
    parser.add_argument('--batch_size', type=int, default=2)      
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--exp', default='r1', type=str, help='experiment setting')

    # Test
    parser.add_argument('--test_model', type=str, default='./Best.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    args = parser.parse_args()
    args.model_save_dir = os.path.join('ckpts/', args.exp)
    args.result_dir = os.path.join('results/', args.exp, 'images')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    main(args)
