import os
import time
import math
import datetime
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from collections import OrderedDict

# Import các module của project
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace



# HÀM PARSE THAM SỐ DÒNG LỆNH
def get_args():
    parser = argparse.ArgumentParser(description='RetinaFace Training')
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='File label train')
    parser.add_argument('--network', default='mobile0.25', help='Backbone: mobile0.25 hoặc resnet50')
    parser.add_argument('--num_workers', default=4, type=int, help='Số luồng đọc dữ liệu')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum cho SGD')
    parser.add_argument('--resume_net', default=None, help='Path checkpoint nếu muốn tiếp tục train')
    parser.add_argument('--resume_epoch', default=0, type=int, help='Epoch bắt đầu lại')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay cho SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma giảm lr')
    parser.add_argument('--save_folder', default='./weights/', help='Folder lưu model')
    return parser.parse_args()



# HÀM ĐIỀU CHỈNH LEARNING RATE
def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size, initial_lr):
    warmup_epoch = -1
    # Nếu đang trong warmup thì tăng lr tuyến tính
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        # Sau warmup giảm lr theo step decay
        lr = initial_lr * (gamma ** step_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


#  HÀM TRAIN CHÍNH
def train():
    args = get_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Load config backbone
    cfg = cfg_mnet if args.network == "mobile0.25" else cfg_re50

    rgb_mean = (104, 117, 123)
    num_classes = 2
    img_dim = cfg['image_size']
    num_gpu = cfg['ngpu']
    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']
    gpu_train = cfg['gpu_train']

    cudnn.benchmark = True

    # Load model RetinaFace
    net = RetinaFace(cfg=cfg)
    print(f" Tổng số tham số model: {sum(p.numel() for p in net.parameters())}")

    # Nếu có checkpoint thì load
    if args.resume_net:
        print('🔄 Load checkpoint model...')
        state_dict = torch.load(args.resume_net)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Bỏ 'module.' nếu có
            new_state_dict[k[7:] if k.startswith('module.') else k] = v
        net.load_state_dict(new_state_dict)

    # Đưa model sang GPU
    net = torch.nn.DataParallel(net).cuda() if num_gpu > 1 and gpu_train else net.cuda()

    # Khai báo optimizer, loss function, prior box
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward().cuda()

    print(' Load Dataset...')
    dataset = WiderFaceDetection(args.training_dataset, preproc(img_dim, rgb_mean))

    # Tính số iteration/epoch
    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size
    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0
    start_iter = args.resume_epoch * epoch_size
    epoch = args.resume_epoch

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))

            # Lưu checkpoint theo mốc epoch
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), f"{args.save_folder}{cfg['name']}_epoch_{epoch}.pth")
            epoch += 1

        # Điều chỉnh learning rate
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size, args.lr)

        # Load batch dữ liệu
        load_t0 = time.time()
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # Forward model
        out = net(images)

        # Tính loss
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Tính thời gian và ETA
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))

        # Log training
        print(f'Epoch:{epoch}/{max_epoch} || Iter: {iteration+1}/{max_iter} || Loc: {loss_l.item():.4f} Cla: {loss_c.item():.4f} Landm: {loss_landm.item():.4f} || LR: {lr:.8f} || Time: {batch_time:.4f}s || ETA: {str(datetime.timedelta(seconds=eta))}')

    # Lưu model cuối cùng
    torch.save(net.state_dict(), f"{args.save_folder}{cfg['name']}_Final.pth")
    print('✅ Training hoàn tất.')

#  CHẠY TRAIN
if __name__ == '__main__':
    train()
