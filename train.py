import argparse
import os
import time
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam, SGD
from torchvision import utils

from utils import create_dataloader, YOLOLoss, parse_cfg, build_model

parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument('--cfg', '-c', default='cfg/yolov1.yaml', help='YOLOv1 config file path')
parser.add_argument('--dataset_cfg', '-d', default='cfg/dataset.yaml', help='dataset config file path')
parser.add_argument('--weight', '-w', default='', help='pretrained model weight file path')
parser.add_argument('--output', '-o', default='output', help='output path')
parser.add_argument('--epochs', '-e', default=100, help='training epochs', type=int)
parser.add_argument('--lr', '-l', default=0.002, help='training learning rate', type=float)
parser.add_argument('--batch_size', '-bs', default=32, help='training batch size', type=int)
parser.add_argument('--save_freq', '-sq', default=10, help='frequency of saving model checkpoint while training', type=int)
args = parser.parse_args()

def train(model, train_loader, optimizer, epoch, device, S, B, train_loss_lst):
    model.train() # 设置为训练模式
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        t_start = time.time()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # 反向传播
        criterion = YOLOLoss(S, B)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        t_batch = time.time() - t_start

        # 展示第一个batch的数据
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1,2,0)))
            plt.savefig(os.path.join(output_path, 'batch0.png'))
            # plt.show()
            plt.close(fig)

        # 打印loss and accuracy
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Time: {:.4f}s  Loss: {:.6f}'
                    .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), t_batch, loss.item()))

    train_loss /= len(train_loader)
    train_loss_lst.append(train_loss)
    return train_loss_lst

def validate(model, val_loader, device, S, B, val_loss_lst):
    model.eval()
    val_loss = 0
    # 不需要计算梯度
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = YOLOLoss(S, B)
            val_loss += criterion(output, target).item()

    val_loss /= len(val_loader)
    print('Val set: Average loss: {:.4f}\n'.format(val_loss))

    # record validating loss
    val_loss_lst.append(val_loss)
    return val_loss_lst

def test(model, test_loader, device, S, B):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            criterion = YOLOLoss(S, B)
            test_loss += criterion(output, target).item()

    test_loss /= len(test_loader)
    print('Test set average loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    cfg = parse_cfg(args.cfg)
    dataset_cfg = parse_cfg(args.dataset_cfg)
    img_path, label_path = dataset_cfg['images'], dataset_cfg['labels']
    S, B, num_classes, input_size = cfg['S'], cfg['B'], cfg['num_classes'], cfg['input_size']
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('device: GPU')
    else:
        device = torch.device('cpu')
        print('device: CPU')

    # 输出
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    output_path = os.path.join(args.output, start)
    os.makedirs(output_path)

    model = build_model(args.weight, S, B, num_classes).to(device)

    train_loader, val_loader, test_loader = create_dataloader(img_path, label_path, 0.8, 0.1, 0.1, args.batch_size,input_size, S, B, num_classes)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)

    train_loss_lst, val_loss_lst = [], []

    for epoch in range(args.epochs):
        train_loss_lst = train(model, train_loader, optimizer, epoch, device, S, B, train_loss_lst)
        val_loss_lst = validate(model, val_loader, device, S, B, val_loss_lst)

        # 保存weight
        if epoch % args.save_freq == 0 and epoch >= args.epochs / 2:
            torch.save(model.state_dict(), os.path.join(output_path, 'epoch' + str(epoch) + '.pth'))
    
    test(model, test_loader, device, S, B)

    torch.save(model.state_dict(), os.path.join('weights/', 'last.pth'))

    fig = plt.figure()
    plt.plot(range(args.epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(args.epochs), val_loss_lst, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, 'loss_curve.jpg'))
    plt.show()
    plt.close(fig)