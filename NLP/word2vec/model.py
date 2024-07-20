import torch
from torch import nn
import torch.nn.functional as F

class Word2Vec(nn.Module):
    def __init__(self, num_tokens, embedding_dim) -> None:
        super().__init__()
        self.dense_c = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_dim, sparse=True)
        self.dense_o = nn.Embedding(num_embeddings=num_tokens, embedding_dim=embedding_dim, sparse=True)

        initrange = 0.5 / embedding_dim
        self.dense_c.weight.data.uniform_(-initrange, initrange)
        self.dense_o.weight.data.uniform_(-0, 0)

    def forward(self, x_center, x_outer, x_negative) -> torch.Tensor:
        """
        Calculate loss with respect to pairs of tokens.

        :param x_center: tensor of center tokens ids.
        :param x_outer: tensor of outer tokens ids.
        :param x_negative: tensor of negative tokens.

        :return: loss
        """
        center = self.dense_c(x_center)
        outer = self.dense_o(x_outer)
        score = torch.mul(center, outer).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)

        negative = self.dense_o(x_negative)
        neg_score = torch.bmm(negative, center.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1/2 * (torch.mean(score) + torch.mean(neg_score))

    def get_vector(self, key):
        return self.dense_c(key)