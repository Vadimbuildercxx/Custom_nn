import argparse
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import PixelCNN
from utils import imshow
from tqdm import tqdm
from consts import CLASSES

def discretize(sample):
    return (sample * 255).to(torch.long)

def train_loop(epochs, train_loader, val_loader, loss_fn, optimizer, scheduler):
    train_losses = []
    val_losses = []

    total_train_iter_len = epochs * len(train_loader)
    val_iter_len = len(val_loader)

    pbar_train = tqdm(total=total_train_iter_len, desc="Train")
    pbar_val = tqdm(total=val_iter_len, desc="Validation")

    for i in range(epochs):

        pbar_train.set_description(f"Train / epoch: {i}")

        loss_train = 0
        model.train()
        for batch, labels in train_loader:
            batch = batch.cuda()
            labels = labels.cuda()

            out = model(batch, labels)

            loss = loss_fn(out, batch)
            loss_train += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar_train.update(1)

        scheduler.step()

        loss_train = loss_train / len(train_loader)

        train_losses.append(loss_train)

        pbar_val.set_description(f"Val / epoch: {i}")
        loss_valid = 0
        model.eval()
        for batch, labels in val_loader:
            batch = batch.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                out = model(batch, labels)

            loss = loss_fn(out, batch)
            loss_valid += loss.item()

            pbar_val.update(1)

        pbar_val.reset()

        loss_valid = loss_valid / len(val_loader)

        val_losses.append(loss_valid)

        print(f"Epoch {i+1} / {epochs}, Train Loss: {loss_train}, Val Loss: {loss_valid}")

    return train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float,  default=1e-3)
    parser.add_argument('--gamma', type=float,  default=0.99)
    parser.add_argument('--in_channels', type=int,  default=1)
    parser.add_argument('--inner_dim', type=int,  default=64)
    parser.add_argument('--batch_size', type=int,  default=128)

    args = parser.parse_args()
    print(args)

    model = PixelCNN(args.in_channels, args.inner_dim).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    loss_fn = lambda pred, x: (F.cross_entropy(pred, x, reduction='mean'))

    transform = transforms.Compose([transforms.ToTensor(),
                                    discretize])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(trainset, [50000, 10000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)


    train_loop(args.epochs, trainloader, validloader, loss_fn, optimizer, scheduler)