import sys
# -*-coding:utf-8-*-

from my_dataset import MyDataSet
from utils import read_split_data
import os
import json
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from LSR import LSR
import csv

# from model_ef import efficientnet_b0 as create_model
#from model_v3 import mobilenet_v3_large as create_model
from model_v3 import mobilenet_v3_small as create_model
root = "../data/train/lip_data"  # 数据集所在根目录


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(30),
                                     transforms.ToTensor()]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor()])}

    train_dataset = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'Yellow':0, 'white':1,'red_yellow':2,'Red':3,'black':4}
    
    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    # plot_data_loader_image(train_loader)
    validate_dataset = MyDataSet(images_path=val_images_path,
                                            images_class=val_images_label,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = create_model(num_classes=4).to(device)
    
    # loss_function = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    epochs = 300
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_line = []
    acc_line = []
    lr_line = []
    best_acc = 0.0
    savepath='./mobilenet_v3_small_checkpoints'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        lsr = LSR()
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = lsr(logits, labels.to(device))
            
            
            
            #loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        val_loss = running_loss / train_steps
        
        loss_line.append(val_loss)
        acc_line.append(val_accurate)
        lr_line.append(optimizer.param_groups[0]["lr"])
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, val_loss, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            savename = '{}_{:.5f}_mobilenet_v3_small.pth'.format(epoch,best_acc)
            torch.save(net.state_dict(), os.path.join(savepath,savename))
        scheduler.step()
    with open('mobilenet_v3_small.csv','w', newline='') as f:
             f_csv = csv.writer(f)
             f_csv.writerow(acc_line)
             f_csv.writerow(loss_line)
             f_csv.writerow(lr_line)
    print('Finished Training')




if __name__ == '__main__':
    main()
