from typing import List, Union
import torch


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Word2VecModel:
    def __init__(self, model, label2id, id2label):
        self.model = model
        self.id2label = id2label
        self.label2id = label2id
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)

    def get_vector_by_word(self, key: str) -> torch.Tensor:
        return self.get_vector_by_id(self.label2id[key])

    def get_vector_by_id(self, id: int) -> torch.Tensor:
        return self.model.dense_c(torch.tensor(id).cuda())

    def get_k_nearest_id(self, vector: torch.Tensor, k: int) -> List[int]:
        top_k = torch.topk(self.cos_sim(vector, self.model.dense_c.weight), k)
        idx_s = []
        for i in range(k):
            idx_s.append(top_k[i].item())
        return idx_s

    def get_k_nearest_word(self, vector: torch.Tensor, k: int) -> List[str]:
        top_k = torch.topk(self.cos_sim(vector, self.model.dense_c.weight), k)
        str_s = []
        for i in range(k):
            str_s.append(self.id2label[top_k.indices[i].item()])
        return str_s

    def compare_vectors(self, a: Union[torch.Tensor, str], b: Union[torch.Tensor, str]) -> float:
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return self.cos_sim(a, b).item()
        elif isinstance(a, str) and isinstance(b, str):
            return self.cos_sim(self[a], self[b]).item()
        else:
            raise TypeError

    def __getitem__(self, idx) -> torch.Tensor:
        if isinstance(idx, int):
            return self.get_vector_by_id(idx)
        elif isinstance(idx, str):
            return self.get_vector_by_word(idx)
        else:
            raise TypeError
