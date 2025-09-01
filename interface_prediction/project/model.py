from lightning.pytorch import LightningModule
import torch
from torch_geometric import nn as graph_nn
from torch import nn
from torch.nn import functional as F
import torchmetrics
from lightning.pytorch.cli import LightningCLI


class MyModel(LightningModule):
    """
    LightningModule wrapping a GAT model.
    """
    def __init__(self, in_channels=20, hidden_channels=32, num_layers=2, heads=2, out_channels=1, dropout=0.01, jk="last"):
        super().__init__()
        self.model = graph_nn.GAT(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         heads=heads,
                         out_channels=out_channels,
                         dropout=dropout,
                         jk=jk, v2=True)
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, node_attributes, edge_index):
        return self.model(node_attributes, edge_index)

    def training_step(self, batch, batch_idx):
        out = self(batch.amino_acid_one_hot.float(), batch.edge_index)
        loss = self.loss_function(out, batch.interface_label.float().view(-1, 1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True,
                 batch_size=batch.batch_size)
        return loss

    def configure_optimizers(self):
      return torch.optim.Adam(params=self.model.parameters(), lr=0.001, weight_decay=0.0001)


def main():
    """
    Run with python main.py fit -c config.yaml
    Or in an sbatch script with srun python main.py fit -c config.yaml
    """
    torch.set_float32_matmul_precision("medium")
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
