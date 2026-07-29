import torch
from torch import nn
import torch.nn.functional as F


class DeployFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        cf2_num_input_dim,
        hom_num_input_dim,
        base_num_input_dim,
        deposit_cat_num,
        deposit_embedding_dim,
        num_cat: list,
        embedding_dim: list,  # desired dimension of each cat
        app_rnn_length,
        hidden_size: list,  # hidden state of deposit rnn and application rnn accordingly
        num_layers: list,  # num_layers of deposit rnn and application rnn accordingly
        cf2_final_dim=128,
    ):
        super(DeployFeatureExtractionModel, self).__init__()

        self.cf2_num_input_dim = cf2_num_input_dim
        self.hom_num_input_dim = hom_num_input_dim
        self.base_num_input_dim = base_num_input_dim
        self.deposit_cat_num = deposit_cat_num
        self.deposit_embedding_dim = deposit_embedding_dim
        self.num_cat = num_cat
        self.embedding_dim = embedding_dim
        self.app_rnn_length = app_rnn_length
        self.cf2_final_dim = cf2_final_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cf2_embeddings = nn.ModuleList(
            [nn.Embedding(n, e) for n, e in zip(num_cat, embedding_dim)]
        )
        self.hom_embeddings = nn.ModuleList(
            [nn.Embedding(n, e) for n, e in zip(num_cat, embedding_dim[:3])]
        )

        # hom flow
        self.hom_fc1 = nn.Linear(hom_num_input_dim + sum(embedding_dim[0:3]), 128)
        self.hom_fc2 = nn.Linear(128, 256)
        self.hom_fc3 = nn.Linear(256, 256)
        self.hom_fc4 = nn.Linear(256, 128)
        self.hom_fc5 = nn.Linear(128, 16)
        self.application_gru = nn.GRU(
            16,
            hidden_size[1],
            num_layers=num_layers[1],
            batch_first=True,
        )

        # cf2 flow
        self.deposit_embedding = nn.Embedding(deposit_cat_num, deposit_embedding_dim)
        self.deposit_gru = nn.GRU(
            deposit_embedding_dim,
            hidden_size[0],
            num_layers=num_layers[0],
            batch_first=True,
        )
        self.cf2_fc1 = nn.Linear(
            cf2_num_input_dim + sum(embedding_dim) + hidden_size[0], 256
        )
        self.cf2_fc2 = nn.Linear(256, 512)
        self.cf2_fc3 = nn.Linear(512, 512)
        self.cf2_fc4 = nn.Linear(512, 256)
        self.cf2_fc5 = nn.Linear(256, cf2_final_dim)
        self.attention_weights = nn.Linear(cf2_final_dim, 1)

    def hom_forward(
        self,
        hom_idx,  # identifier show which row is from which data
        hom_num,  # 3D [batch, len, col_number]
        hom_cat,  # 3D [batch, len, col_number]
    ):
        if sum(hom_idx) != 0:
            hom_cat_embedding = [
                self.hom_embeddings[idx](hom_cat[:, idx].int())
                for idx in range(hom_cat.size(1))
            ]
            hom_cat_embedding = torch.flatten(
                torch.cat(hom_cat_embedding, dim=1), start_dim=1
            )
            x = torch.cat((hom_num, hom_cat_embedding), dim=1)
        else:
            x = torch.zeros(
                len(hom_idx),
                self.hom_num_input_dim + sum(self.embedding_dim[:3]),
                dtype=torch.float64,
            )
        not_0 = sum(hom_idx) > 0
        x = x * not_0

        temp_out = self.hom_fc1(x)
        temp_out = self.hom_fc2(temp_out)
        temp_out = nn.ReLU()(temp_out)
        temp_out = self.hom_fc3(temp_out)
        temp_out = nn.ReLU()(temp_out)
        temp_out = self.hom_fc4(temp_out)
        temp_out = nn.ReLU()(temp_out)
        temp_out = self.hom_fc5(temp_out)
        temp_out = nn.ReLU()(temp_out)

        out = torch.tensor([])
        cur_idx = 0
        for i in range(len(hom_idx)):
            pad_size = self.app_rnn_length - hom_idx[i]
            temp = (
                F.pad(temp_out[cur_idx : cur_idx + hom_idx[i]], (0, 0, pad_size, 0))
            ).unsqueeze(0)
            if i == 0:
                out = temp
            else:
                out = torch.cat((out, temp))
            cur_idx += hom_idx[i]

        h0 = torch.zeros(
            self.num_layers[1], len(hom_idx), self.hidden_size[1], dtype=torch.float64
        )
        out, _ = self.application_gru(out, h0)
        out = out[:, -1, :]
        return out

    def attention_combiner(self, embeddings):
        weights = self.attention_weights(embeddings)
        weights = nn.Softmax(dim=0)(weights)
        combined_embedding = torch.sum(weights * embeddings, dim=0)
        return combined_embedding

    def cf2_forward(
        self,
        cf2_idx,  # identifier show which row is from which data
        cf2_num,
        cf2_cat,
        cf2_deposit_hist,
        cf2_deposit_hist_mask,
    ):
        cf2_cat_embedding = [
            self.cf2_embeddings[idx](cf2_cat[:, idx].to(dtype=torch.int))
            for idx in range(len(self.num_cat))
        ]
        cf2_cat_embedding = torch.cat(cf2_cat_embedding, dim=1)
        deposit_hist_embedding = self.deposit_embedding(
            cf2_deposit_hist.to(dtype=torch.int)
        )
        deposit_hist_embedding = (
            deposit_hist_embedding * cf2_deposit_hist_mask.unsqueeze(-1)
        )
        h0 = torch.zeros(self.num_layers[0], cf2_num.size(0), self.hidden_size[0]).to(
            torch.float64
        )
        temp_out, _ = self.deposit_gru(deposit_hist_embedding, h0)
        temp_out = temp_out[:, -1, :]
        temp_out = torch.cat([cf2_num, cf2_cat_embedding, temp_out], dim=1)
        temp_out = self.cf2_fc1(temp_out)
        temp_out = nn.ReLU()(temp_out)
        temp_out = self.cf2_fc2(temp_out)
        temp_out = nn.ReLU()(temp_out)
        temp_out = self.cf2_fc3(temp_out)
        temp_out = nn.ReLU()(temp_out)
        temp_out = self.cf2_fc4(temp_out)
        temp_out = nn.ReLU()(temp_out)
        temp_out = self.cf2_fc5(temp_out)
        cur_idx = 0
        for i in range(len(cf2_idx)):
            combined_embedding = self.attention_combiner(
                temp_out[cur_idx : cur_idx + cf2_idx[i]]
            )
            if i == 0:
                out = combined_embedding.unsqueeze(0)
            else:
                out = torch.cat((out, combined_embedding.unsqueeze(0)))
            cur_idx += cf2_idx[i]
        return out

    def forward(
        self,
        cf2_idx,
        hom_idx,
        cf2_num,
        cf2_cat,
        cf2_deposit_hist,
        cf2_deposit_hist_mask,
        hom_num,
        hom_cat,
    ):
        hom = self.hom_forward(hom_idx, hom_num, hom_cat)
        cf2 = self.cf2_forward(
            cf2_idx, cf2_num, cf2_cat, cf2_deposit_hist, cf2_deposit_hist_mask
        )
        x = torch.cat([hom, cf2], dim=1)
        return x
