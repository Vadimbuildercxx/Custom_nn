import torch
import matplotlib.pyplot as plt
import numpy as np

def imshow(imgs: torch.Tensor, classes, labels: torch.Tensor = None, rows = 4, cols = 10):
    count_imgs = imgs.shape[0]

    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)

    for i in range(rows):
        for j in range(cols):
            if i * cols + j < count_imgs:
                img = imgs[i * cols + j]
                npimg = img.numpy()

                if labels is not None:
                    axs[i, j].set_xlabel(classes[labels[i * cols + j].item()])
                    axs[i, j].imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray_r')
                else:
                    axs[i, j].imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray_r')
    fig.text(0.5, 0.04, 'Width', ha='center')
    fig.text(0.04, 0.5, 'Height', va='center', rotation='vertical')
    plt.show()


