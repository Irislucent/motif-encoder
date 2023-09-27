import argparse
import math
import numpy as np
import torch
import torch.nn as nn


class TokenAndPositionalEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, emb_dim, max_len):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.token_emb = nn.Conv1d(self.input_dim, self.emb_dim, 1)
        self.pos_emb = self.positional_encoding(self.max_len, self.emb_dim)

    def get_angles(self, pos, i, emb_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(emb_dim))
        return pos * angle_rates

    def positional_encoding(self, position, emb_dim):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(emb_dim)[np.newaxis, :],
            emb_dim,
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return torch.tensor(pos_encoding, dtype=torch.float32)

    def forward(self, x):
        seq_len = x.shape[1]
        x = torch.permute(x, (0, 2, 1))
        x = self.token_emb(x)
        x *= torch.sqrt(torch.tensor(self.emb_dim, dtype=torch.float32))
        x = torch.permute(x, (0, 2, 1))
        return x + self.pos_emb.to(x.device)[:, : x.shape[1]]


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        d_model = config["d_model"]
        d_latent = config["d_latent"]
        n_layers = config["n_layers"]
        d_ff = d_model * 4 # seem to be a default
        d_expander = d_latent * 4 
        dropout_rate = config["dropout_rate"]
        n_heads = 8

        self.emb = TokenAndPositionalEmbeddingLayer(
            input_dim=config["d_input"], emb_dim=d_model, max_len=config["max_seq_len"]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.fc = nn.Linear(d_model, d_latent)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.expander = nn.Sequential(
            nn.Linear(d_latent, d_expander),
            nn.ReLU(),
            nn.Linear(d_expander, d_expander),
            nn.ReLU(),
            nn.Linear(d_expander, d_expander)
        )

    def forward(self, src):
        # src: (N, S, E)
        # src_key_padding_mask are all False, so the output is the same as src. This is because all inputs have the same length.
        src_key_padding_mask = torch.zeros((src.shape[0], src.shape[1])).bool()
        src_key_padding_mask = src_key_padding_mask.to(src.device)
        src_emb = self.emb(src)  # (N, S, D_MODEL)
        src_emb = self.transformer_encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )  # (N, S, D_MODEL)
        src_emb = self.fc(src_emb)  # (N, S, D_LATENT)
        src_emb = torch.permute(src_emb, (0, 2, 1))  # (N, D_LATENT, S)
        src_emb = self.pool(src_emb)  # (N, D_LATENT, 1)
        src_emb = torch.squeeze(src_emb, dim=2)  # (N, D_LATENT)
        src_emb_expanded = self.expander(src_emb)  # (N, S, D_EXPANDER)

        return src_emb, src_emb_expanded


def test_model(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expander_params = sum(p.numel() for p in model.expander.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    print(f"Total trainable parameters without expander: {total_params - expander_params}")
    src = np.random.randint(0, 2, (16, 64, 84))
    src = torch.from_numpy(src).float()
    emb, expanded = model(src)
    print(emb.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model Architecture
    parser.add_argument(
        "--pad_idx", type=int, default=0, help="Pad value of sequences."
    )
    parser.add_argument(
        "--d_input", type=int, default=84, help="Dimension of the transformer encoder input."
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Dimension of the transformer encoder/decoder input.",
    )
    parser.add_argument(
        "--d_latent", type=int, default=128, help="Dimension of the latent representation."
    )
    parser.add_argument(
        "--n_layers", type=int, default=5, help="Depth of the transformer."
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=64, help="Maximum length of the sequence."
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.2, help="Dropout rate of all layers."
    )
    args = parser.parse_args()
    model = BertEncoder(vars(args))

    test_model(model)
