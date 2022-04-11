import torch.optim

from __init__ import *


def go_train(args):
    # 训练设备
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("backbone = " + args.backbone)

    print("metric = " + args.metric)

    # 清洗数据，生成训练所需csv及dict
    # train_csv_train, train_csv_val, dict_id_all = make_csv(data_csv_path,
    #                                                        low,
    #                                                        high,
    #                                                        val_number,
    #                                                        save_path)
    f2 = open(args.dict_id_path, 'r')
    dict_id_all = json.load(f2)
    train_csv_train = pd.read_csv(args.train_csv_train_path)
    train_csv_val = None

    num_classes = len(dict_id_all)

    # 加载模型的loss函数类型
    criterion = FocalLoss(gamma=2)

    # 加载backbone,默认convnext_tiny
    if args.backbone == 'convnext_small':
        model = timm.create_model('convnext_small', pretrained=args.pretrained, num_classes=512)
    elif args.backbone == 'convnext_base':
        model = timm.create_model('convnext_base', pretrained=args.pretrained, num_classes=512)
    elif args.backbone == 'convnext_large':
        model = timm.create_model('convnext_large', pretrained=args.pretrained, num_classes=512)
    else:
        model = timm.create_model('convnext_tiny', pretrained=args.pretrained, num_classes=512)
    # 加载模型的margin类型
    if args.metric == 'Arc':
        metric_fc = ArcMarginProduct(512, num_classes, m=args.m)
    elif args.metric == 'Add':
        metric_fc = AddMarginProduct(512, num_classes, m=args.m)
    else:
        metric_fc = SphereProduct(512, num_classes)

    # dataset
    train_dataset = ArcDataset(train_csv_train, dict_id_all, args.data_train_path, args.w,
                               args.h)
    if train_csv_val is not None:
        val_dataset = ArcDataset(train_csv_val, dict_id_all, args.data_train_path, args.w,
                                 args.h)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.Freeze_batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    else:
        val_dataloader = None

    # 训练前准备
    model.to(device)

    metric_fc.to(device)

    criterion.to(device)

    if args.Freeze_Epoch != 0:
        # -------------------------------#
        #   开始冻结训练
        # -------------------------------#
        print("--------冻结训练--------")
        # -------------------------------#
        #   生成冻结dataloader
        # -------------------------------#
        Freeze_train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.Freeze_batch_size, shuffle=True,
                                             num_workers=args.num_workers)
        # -------------------------------#
        #   选择优化器
        # -------------------------------#

        # optimizer = torch.optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
        # lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
        #                                    warmup=True, warmup_epochs=1)
        Freeze_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                             lr=args.Freeze_lr, weight_decay=args.Freeze_weight_decay)
        Freeze_scheduler = create_lr_scheduler(Freeze_optimizer, len(Freeze_train_dataloader), args.Freeze_Epoch,
                                               warmup=True, warmup_epochs=1)
        # -------------------------------#
        #   冻结措施
        # -------------------------------#
        for param in model.parameters():
            param.requires_grad = False
        model = make_train(model=model,
                           metric_fc=metric_fc,
                           criterion=criterion,
                           optimizer=Freeze_optimizer,
                           scheduler=Freeze_scheduler,
                           train_loader=Freeze_train_dataloader,
                           val_loader=val_dataloader,
                           device=device,
                           num_classes=num_classes,
                           max_epoch=args.Freeze_Epoch + args.Unfreeze_Epoch,
                           save_interval=args.save_interval,
                           save_path=args.save_path,
                           backbone=args.backbone,
                           epoch_start=1,
                           epoch_end=args.Freeze_Epoch,
                           Str=args.metric,
                           Freeze_Epoch=args.Freeze_Epoch)

    # -------------------------------#
    #   开始解冻训练
    # -------------------------------#
    print("--------解冻训练--------")

    # -------------------------------#
    #   生成解冻dataloader
    # -------------------------------#
    Unfreeze_train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.Unfreeze_batch_size, shuffle=True,
                                           num_workers=args.num_workers)
    # -------------------------------#
    #   选择优化器
    # -------------------------------#
    Unfreeze_optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                           lr=args.Unfreeze_lr, weight_decay=args.Unfreeze_weight_decay)
    Unfreeze_scheduler = create_lr_scheduler(Unfreeze_optimizer, len(Unfreeze_train_dataloader), args.Unfreeze_Epoch,
                                             warmup=True, warmup_epochs=1)
    # -------------------------------#
    #   解冻措施
    # -------------------------------#
    for param in model.parameters():
        param.requires_grad = True
    make_train(model=model,
               metric_fc=metric_fc,
               criterion=criterion,
               optimizer=Unfreeze_optimizer,
               scheduler=Unfreeze_scheduler,
               train_loader=Unfreeze_train_dataloader,
               val_loader=val_dataloader,
               device=device,
               num_classes=num_classes,
               max_epoch=args.Freeze_Epoch + args.Unfreeze_Epoch,
               save_interval=args.save_interval,
               save_path=args.save_path,
               backbone=args.backbone,
               epoch_start=args.Freeze_Epoch + 1,
               epoch_end=args.Freeze_Epoch + args.Unfreeze_Epoch,
               Str=args.metric,
               Freeze_Epoch=args.Freeze_Epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--backbone', type=str, default='resnet50', help='特征网络选择，默认resnet50')
    parser.add_argument('--data_train_path', type=str, help='训练集路径', required=True)
    parser.add_argument('--data_csv_path', type=str, help='全体训练集csv路径',
                        default=r'../input/happy-whale-and-dolphin/train.csv')
    parser.add_argument('--save_path', type=str, help='存储路径', default=r'./')
    parser.add_argument('--dict_id_path', type=str, help='训练类型对应字典路径', required=True)
    parser.add_argument('--train_csv_train_path', type=str, help='需要训练数据csv路径', required=True)
    parser.add_argument('--metric', type=str, help='Arc/Add/Sph', default='Arc')
    parser.add_argument('--pretrained', type=bool, help='是否需要预训练', default=True)
    parser.add_argument('--model_path', type=str, help='上次训练模型权重', default=r'')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_interval', type=int, help='保存间隔', default=3)
    parser.add_argument('--Freeze_Epoch', type=int, help='冻结训练轮次', default=12)
    parser.add_argument('--Freeze_lr', type=float, help='冻结训练lr', default=0.1)
    parser.add_argument('--Freeze_gamma', type=float, help='冻结训练gamma', default=0.1)
    parser.add_argument('--Freeze_lr_step', type=int, help='冻结训练lr衰减周期', default=10)
    parser.add_argument('--Freeze_weight_decay', type=float, help='冻结训练权重衰减率', default=5e-4)
    parser.add_argument('--Freeze_batch_size', type=int, help='冻结训练batch size', default=256)
    parser.add_argument('--Unfreeze_Epoch', type=int, help='解冻训练轮次', default=36)
    parser.add_argument('--Unfreeze_lr', type=float, help='解冻训练lr', default=0.05)
    parser.add_argument('--Unfreeze_gamma', type=float, help='解冻训练gamma', default=0.2)
    parser.add_argument('--Unfreeze_lr_step', type=int, help='解冻训练lr衰减周期', default=10)
    parser.add_argument('--Unfreeze_weight_decay', type=float, help='解冻训练权重衰减率', default=5e-4)
    parser.add_argument('--Unfreeze_batch_size', type=int, help='解冻训练batch size', default=64)
    parser.add_argument('--w', type=int, help='训练图片宽度', default=224)
    parser.add_argument('--h', type=int, help='训练图片高度', default=224)
    parser.add_argument('--m', type=float, help='Arc参数', default=0.4)
    args = parser.parse_args()

    go_train(args)
