import torch
from pathlib import Path
import pandas as pd
from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Dataset
import pickle
from tqdm import tqdm
from graphein.ml import GraphFormatConvertor
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.features.nodes import amino_acid as graphein_nodes
from graphein.protein import edges as graphein_edges
from graphein.protein.subgraphs import extract_subgraph
from graphein.protein.visualisation import plotly_protein_structure_graph
from functools import partial



def load_graph(path, chain):
    graph_config = ProteinGraphConfig(
        node_metadata_functions = [graphein_nodes.amino_acid_one_hot, graphein_nodes.meiler_embedding],
        edge_construction_functions = [graphein_edges.add_peptide_bonds, 
                                       partial(graphein_edges.add_distance_threshold, 
                                               threshold=8.,
                                               long_interaction_threshold=2)],
    )
    graph = construct_graph(path=path, config=graph_config, verbose=False)
    interface_residues = set()
    for source, target, kind in graph.edges(data=True):
        c1, c2 = source.split(":")[0], target.split(":")[0]
        if 'distance_threshold' in kind['kind']:
            if c1 != c2:
                if c1 == chain:
                    interface_residues.add(source)
                elif c2 == chain:
                    interface_residues.add(target)
    graph = extract_subgraph(graph, chains=chain)
    for node, data in graph.nodes(data=True):
        if node in interface_residues:
            data['interface_label'] = 1
        else:
            data['interface_label'] = 0
    return graph


class MyDataset(Dataset):
    """
    torch-geometric Dataset class for loading protein files as graphs.
    """
    def __init__(self, paths: list):
        columns = [
            "chain_id",
            "coords",
            "edge_index",
            "kind",
            "node_id",
            "residue_number",
            "amino_acid_one_hot",
            "meiler",
            "interface_label",
        ]
        self.convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg", columns=columns, verbose=None)
        self.paths = []
        for path in paths:
            self.paths.append((path, path.stem.split("--")[0], "R"))
            self.paths.append((path, path.stem.split("--")[1], "L"))
        super(MyDataset, self).__init__()

    def download(self):
        for path, name, chain in tqdm(self.paths):
            output = Path(self.raw_dir) / f'{name}-{chain}.pkl'
            if not output.exists():
                graphein_graph = load_graph(path, chain)
                with open(output, "wb") as f:
                    pickle.dump(graphein_graph, f)

    @property
    def raw_file_names(self):
        return [Path(self.raw_dir) / f"{name}-{chain}.pkl" for _, name, chain in self.paths]

    @property
    def processed_file_names(self):
        return [Path(self.processed_dir) / f"{name}-{chain}.pt" for _, name, chain in self.paths]

    def process(self):
        for _, name, chain in tqdm(self.paths):
            output = Path(self.processed_dir) / f'{name}-{chain}.pt'
            if not output.exists():
                with open(Path(self.raw_dir) / f"{name}-{chain}.pkl", "rb") as f:
                    graphein_graph = pickle.load(f)
                torch_data = self.convertor(graphein_graph)
                torch.save(torch_data, output)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_names[idx], weights_only=False)
        return data



class MyDataModule(LightningDataModule):
    def __init__(self, dataset_file: Path, batch_size=8):
        super().__init__()
        dataset_file = Path(dataset_file)
        self.dataset = pd.read_csv(dataset_file)
        self.dataset["holo_filenames"] = self.dataset["pinder_id"].apply(lambda x: f"{x.split('--')[0]}-R--{x.split('--')[1]}-L.pdb")
        self.dataset["holo_paths"] = self.dataset[["split", "pinder_id", "holo_filenames"]].apply(lambda x: dataset_file.parent / x["split"] / x["pinder_id"] / x["holo_filenames"], axis=1)
        self.batch_size = batch_size

    def prepare_data(self):
        for split in self.dataset["split"].unique():
            MyDataset(paths=self.dataset[self.dataset["split"] == split]["holo_paths"].values)
    
    def setup(self, stage):
        self.train = MyDataset(paths=self.dataset[self.dataset["split"] == "train"]["holo_paths"].values)
        self.val = MyDataset(paths=self.dataset[self.dataset["split"] == "val"]["holo_paths"].values)
        self.test = MyDataset(paths=self.dataset[self.dataset["split"] == "test"]["holo_paths"].values)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
