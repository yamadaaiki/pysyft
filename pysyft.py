import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# PySyftをimport
import syft as sy

# hook PyTorch 
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# パラメータの設定
class Argments():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False

# パラメータの格納
args = Argments()
use_cuda = not args.no_cuda and torch.cuda.is_available
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers' : 1, 'pin_memory': True} if use_cuda else {}

federated_train_loader = sy.FederatedDataLoader(
    datasets.MNIST('./data', train=True, download=True,
                    transform =transforms.Compose([
                        transforms.toTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    .federate((bob, alice)),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.Data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)






