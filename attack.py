#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# author：fmy

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
def load_model(model_name, dataset_name, epoch, device):
    if dataset_name in ["cifar10","cifar100"]:
        import models.mobilenet as mobilenet
        import models.googlenet as googlenet
        import models.resnet as resnet
        nets = {"mobilenet":mobilenet.MobileNet, "googlenet":googlenet.GoogLeNet,
                "resnet":resnet.ResNet18}

    elif dataset_name in ["mnist","fashionmnist"]:
        import models.mobilenet_mnist as mobilenet_mnist
        import models.googlenet_mnist as googlenet_mnist
        import models.resnet_mnist as resnet_mnist
        nets = {"mobilenet":mobilenet_mnist.MobileNet, "googlenet":googlenet_mnist.GoogLeNet,
                "resnet":resnet_mnist.ResNet18}

    if dataset_name == "cifar100":
        net = nets[model_name](num_classes=100).to(device)
    else:
        net = nets[model_name]().to(device)

    path = dataset_name + "_" + model_name + "_" + str(epoch) + ".pth"
    path = os.path.join(os.getcwd(), "saved_models", path)
    net.load_state_dict(torch.load(path)["model"])
    net.eval()
    return net

def load_multi_track_model(dataset_name, epoch, device, size = None):
    net = multi_track_network.MultiTrackModel(block_width = int(size[2])).to(device)
    path = dataset_name + "_" + size + "_" + str(epoch) + ".pth"
    path = os.path.join(os.getcwd(), "saved_models", path)
    net.load_state_dict(torch.load(path)["model"])
    net.eval()
    return net

def record(dataset,epoch,proxy_model,black_model,black_acc,acc,asr,method,epsilon, index = -1):
    with open("result.out", "a+") as f:
        f.write("{dataset}\t{epoch}\t{proxy_model}\t{black_model}\t{method}\t{epsilon}\t{black_acc:.2f}\t{acc:.2f}\t{asr:.2f}\t"
                "{index}\n".format(dataset = dataset, epoch = epoch, proxy_model = proxy_model, black_model = black_model,
            method = method, epsilon = epsilon, black_acc = black_acc, acc = acc, asr = asr, index=index
        ))

@torch.no_grad()
def multi_track_acc(model, dataloader, index, device):
    correct, num = 0., 0.
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        if index == "all":
            output = model(data)
            output = torch.stack(output, dim=0).sum(dim = 0)
            correct += (output.max(dim = 1)[1] == label).sum().item()
        else:
            correct += (model(data)[index].max(dim = 1)[1] == label).sum().item()
        num += label.size()[0]
    return correct/num*100.

class LossFunc(nn.Module):
    def __init__(self, layer_index):
        super(LossFunc, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()
        self.layer_index = layer_index

    def forward(self, predict, label):
        if self.layer_index == "all":
            loss = 0
            for predict_ in predict:
                loss += self.loss_func(predict_, label)
            return loss
        else:
            return self.loss_func(predict[self.layer_index], label)


if __name__ == "__main__":
    dataset_name = "cifar10"
    _, test_loader = utils.DataManger.get_dataloader(dataset_name, root="../data")
    device = torch.device("cuda:0")

    epsilon = 0.1
    FGSM = black_box_attack.attack_method.FGSM(epsilon=epsilon, device=device)
    BIM = black_box_attack.attack_method.BIM(epsilon=epsilon, iter_num=10, step=0.01, device=device)

    for black_model_name in ["mobilenet","googlenet","resnet"]: #

        black_model = load_model(black_model_name, "cifar10", 50, device)
        black_acc = utils.vanilla_acc(black_model, test_loader, device)

        # mobilenet
        for epoch in [5 * i for i in range(1,11)]:
            proxy_model = load_model("mobilenet", "cifar10", epoch, device)
            acc = utils.vanilla_acc(proxy_model, test_loader, device)
            one_step_asr = FGSM.attack(black_model, proxy_model, nn.CrossEntropyLoss(), test_loader)
            multi_step_asr = BIM.attack(black_model, proxy_model, nn.CrossEntropyLoss(), test_loader)
            record(dataset_name, epoch, "mobilenet", black_model_name, black_acc, acc, one_step_asr, "FGSM", epsilon)
            record(dataset_name, epoch, "mobilenet", black_model_name, black_acc, acc, multi_step_asr, "BIM", epsilon)

        # googlenet
        for epoch in [5 * i for i in range(1,11)]:
            proxy_model = load_model("googlenet", "cifar10", epoch, device)
            acc = utils.vanilla_acc(proxy_model, test_loader, device)
            one_step_asr = FGSM.attack(black_model, proxy_model, nn.CrossEntropyLoss(), test_loader)
            multi_step_asr = BIM.attack(black_model, proxy_model, nn.CrossEntropyLoss(), test_loader)
            record(dataset_name, epoch, "googlenet", black_model_name, black_acc, acc, one_step_asr, "FGSM", epsilon)
            record(dataset_name, epoch, "googlenet", black_model_name, black_acc, acc, multi_step_asr, "BIM", epsilon)

        # resnet
        for epoch in [5 * i for i in range(1,11)]:
            proxy_model = load_model("resnet", "cifar10", epoch, device)
            acc = utils.vanilla_acc(proxy_model, test_loader, device)
            one_step_asr = FGSM.attack(black_model, proxy_model, nn.CrossEntropyLoss(), test_loader)
            multi_step_asr = BIM.attack(black_model, proxy_model, nn.CrossEntropyLoss(), test_loader)
            record(dataset_name, epoch, "resnet", black_model_name, black_acc, acc, one_step_asr, "FGSM", epsilon)
            record(dataset_name, epoch, "resnet", black_model_name, black_acc, acc, multi_step_asr, "BIM", epsilon)


        # multi_track_model    2,3,4
        for epoch in [5 * i for i in range(1,11)]:
            for height in [2]:
                for width in [2,3,4,5]:
                    proxy_model = load_multi_track_model("cifar10", epoch, device, size = "{}x{}".format(height, width))

                    for attack_index in range(width):
                        acc = multi_track_acc(proxy_model, test_loader, attack_index, device)
                        one_step_asr = FGSM.attack(black_model, proxy_model, LossFunc(attack_index), test_loader)
                        multi_step_asr = BIM.attack(black_model, proxy_model, LossFunc(attack_index), test_loader)
                        record(dataset_name, epoch, "{}x{}".format(height,width), black_model_name, black_acc, acc, one_step_asr, "FGSM", epsilon, attack_index)
                        record(dataset_name, epoch, "{}x{}".format(height,width), black_model_name, black_acc, acc, multi_step_asr, "BIM", epsilon, attack_index)

                    acc = multi_track_acc(proxy_model, test_loader, "all", device)
                    one_step_asr = FGSM.attack(black_model, proxy_model, LossFunc("all"), test_loader)
                    multi_step_asr = BIM.attack(black_model, proxy_model, LossFunc("all"), test_loader)
                    record(dataset_name, epoch, "{}x{}".format(height,width), black_model_name, black_acc, acc, one_step_asr, "FGSM", epsilon, "all")
                    record(dataset_name, epoch, "{}x{}".format(height,width), black_model_name, black_acc, acc, multi_step_asr, "BIM", epsilon, "all")