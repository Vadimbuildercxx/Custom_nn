import argparse
import os

import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from model import Word2Vec
from utils import EarlyStopper
from data import Reader, Word2VecDataset, collate_fn

def train_loop(
        epochs,
        train_loader,
        val_loader,
        train_set,
        model,
        opt,
        early_stopper):

    p_bar = tqdm(total=epochs * len(train_loader))

    train_losses = []
    val_losses = []
    logs = []

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for i, (x, y, y_neg) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            y_neg = y_neg.cuda()
            loss = model(x, y, y_neg)

            train_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            p_bar.update(1)

            if i % 100 == 0:
                p_bar.set_description(f"Epoch {epoch}, Train loss: {train_loss / (i + 1)}")

        val_loss = 0.0
        model.eval()
        for i, (x, y, y_neg) in enumerate(val_loader):
            x = x.cuda()
            y = y.cuda()
            y_neg = y_neg.cuda()

            loss = model(x, y, y_neg)

            val_loss += loss.item()

            if i % 100 == 0:
                p_bar.set_description(f"Epoch {epoch}, Val loss: {val_loss / (i + 1)}")

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logs.append(f"Epoch {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")
        p_bar.set_description(f"Epoch {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")
        p_bar.refresh()

        if early_stopper.early_stop(val_loss):
            break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'id2label': train_set.dataset.data.id2label,
            'label2id': train_set.dataset.data.label2id,
        }, f"w2v_model_epoch_{epoch}.pt")

    return train_losses, val_losses, logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float,  default=1e-4)
    parser.add_argument('--emb_dim', type=int,  default=100)
    parser.add_argument('--batch_size', type=int,  default=256)
    parser.add_argument('--dataset_path', type=str,  default="data/wikitext-103/wiki.train.tokens")
    parser.add_argument('--min_words_count', type=int,  default=1)
    parser.add_argument('--window_size', type=int,  default=5)
    parser.add_argument('--early_stopper_patience', type=int,  default=2)
    parser.add_argument('--early_stopper_min_delta', type=float,  default=0.15)
    parser.add_argument('--train_val_split_coeff', type=float,  default=0.9)

    args = parser.parse_args()

    data = Reader(args.dataset_path, args.min_words_count)
    dataset = Word2VecDataset(data, args.window_size)

    coeff = args.train_val_split_coeff
    train_len = int(len(dataset)*coeff)
    test_len = len(dataset)-int(len(dataset)*coeff)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])

    BATCH_SIZE = args.batch_size
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=1)

    early_stopper = EarlyStopper(patience=args.early_stopper_patience, min_delta=args.early_stopper_min_delta)
    model = Word2Vec(num_tokens=train_set.dataset.data.id2label.__len__(), embedding_dim=args.emb_dim).cuda()

    opt = torch.optim.SparseAdam(model.parameters(), lr=args.lr)

    train_losses, val_losses, logs = train_loop(args.epochs, train_loader, val_loader,
                                                train_set, model, opt, early_stopper)

    dataframe = pd.DataFrame(list(map(list, zip(train_losses, val_losses, logs))), columns=['Train Loss', 'Val Loss', 'Logs'])
    print(f"Save train logs to {os.getcwd()}\\train_logs.csv")
    dataframe.to_csv("train_logs.csv", index=False)