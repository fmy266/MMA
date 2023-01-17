import torch
import torch.nn as nn
import multi_track_network
import sys
import os
sys.path.append("..")
import trainer
import black_box_attack
import utils
import time

def save_model(info):
    def val_func(model, loader, epoch_, device):
        checkpoint = {
            "model":model.state_dict(),
        }
        torch.save(checkpoint, os.path.join(os.getcwd(), "saved_models", info+"_"+str(epoch_)+".pth"))
    return val_func


class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, predict, label):
        loss = 0
        for predict_ in predict:
            loss += self.loss_func(predict_, label)
        return loss



dataset_name = "cifar10" # cifar100, mnist, fashionmnist
train_loader, test_loader = utils.DataManger.get_dataloader(dataset_name, root = "../data")
height = 5
for width in [2, 3, 4, 5]:
    model = multi_track_network.MultiTrackModel(channel = 3, num_classes = 10, block_width = width)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4, nesterov = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
    train = trainer.BaseTrainer(model, train_loader, LossFunc(), optimizer, torch.device("cuda:0"),
                use_auto_mixed_prec = True, scheduler = scheduler,
                val_loader = test_loader)
    consumed_time = train.train(epoch = 50, valid_func = save_model("{}_{}x{}".format(dataset_name, 5, width, torch.device("cuda:0"))))