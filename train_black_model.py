import torch
import sys
import os
sys.path.append("..")
import trainer
import utils
import time
# torch.backends.cudnn.benchmark = True

def save_model(info):
    def val_func(model, loader, epoch_, device):
        checkpoint = {
            "model":model.state_dict(),
        }
        torch.save(checkpoint, os.path.join(os.getcwd(), "saved_models", info+"_"+str(epoch_)+".pth"))
    return val_func


def train_black_model(dataset_name, model_name, device = torch.device("cuda:1"), epoch = 50, interval = 5):

    if dataset_name in ["cifar10","cifar100"]:
        import models.mobilenet as mobilenet
        import models.googlenet as googlenet
        import models.resnet as resnet
        nets = {"mobilenet":mobilenet.MobileNet, "googlenet":googlenet.GoogLeNet, "resnet":resnet.ResNet18}

    elif dataset_name in ["mnist","fashionmnist"]:
        import models.mobilenet_mnist as mobilenet_mnist
        import models.googlenet_mnist as googlenet_mnist
        import models.resnet_mnist as resnet_mnist
        nets = {"mobilenet":mobilenet_mnist.MobileNet, "googlenet":googlenet_mnist.GoogLeNet, "resnet":resnet_mnist.ResNet18}

    if dataset_name == "cifar100":
        net = nets[model_name](num_classes=100).to(device)
    else:
        net = nets[model_name]().to(device)

    train_loader, test_loader = utils.DataManger.get_dataloader(dataset_name, root="../data")
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 1e-4, nesterov = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
    train = trainer.BaseTrainer(net, train_loader, torch.nn.CrossEntropyLoss(), optimizer, device,
                use_auto_mixed_prec = True, scheduler = scheduler,
                val_loader = test_loader)

    consumed_time = train.train(epoch = epoch, valid_func = save_model(dataset_name+"_"+model_name), interval = interval)
    with open("log.txt", "a+") as f:
        f.write("{}, epoch = {}, consumed_time = {}\n".format(model_name, 50, consumed_time))



if __name__ == '__main__':

    # train_black_model("cifar10", "mobilenet", torch.device('cuda:1'), 50, 5)
    # train_black_model("cifar10", "googlenet", torch.device('cuda:1'), 50, 5)
    train_black_model("cifar10", "resnet", torch.device('cuda:0'), 1, 5)



    # train_black_model("cifar100", "mobilenet", torch.device('cuda:1'), 100, 10)
    # train_black_model("cifar100", "googlenet", torch.device('cuda:1'), 100, 10)
    # train_black_model("cifar100", "resnet", torch.device('cuda:0'), 1, 10)

    # train_black_model("mnist", "mobilenet", torch.device('cuda:1'), 10, 1)
    # train_black_model("mnist", "googlenet", torch.device('cuda:1'), 10, 1)
    # train_black_model("mnist", "resnet", torch.device('cuda:0'), 1, 1)

    # train_black_model("fashionmnist", "mobilenet", torch.device('cuda:1'), 20, 2)
    # train_black_model("fashionmnist", "googlenet", torch.device('cuda:1'), 20, 2)
    # train_black_model("fashionmnist", "resnet", torch.device('cuda:0'), 1, 2)

