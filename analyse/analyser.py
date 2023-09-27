"""
(base class) MotiveAnalyser, for the inference and visualization of motives
The dataloader gets chunks from the target song
Try every chunk as the base, project the whole song into the base
Pick significant bases
"""
import os
import abc

import numpy as np
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
from scipy.interpolate import make_interp_spline

from analyse.dataloader import get_dataloader
from utils.midi_utils import *

plt.rc("font", family="Times New Roman")


class MotiveAnalyser:
    def __init__(self, config, input_path):
        self.config = config
        if os.path.splitext(input_path)[1] != ".mid":
            raise ValueError("input_path must be a .mid file")
        self.input_path = input_path
        self.chunk_size = 1
        self.grid = 1 / 32
        self.representation = (
            "piano_roll"  # now we only study the piano roll representation
        )
        self.cp_path = config["active_checkpoint"]

        self.dl = get_dataloader(
            self.input_path, self.chunk_size, self.grid, self.representation
        )

    def load_model(self):
        """
        Load model from checkpoint
        """
        config = self.config
        if config["method"] == "contrastive":
            from contrastive.train import PLWrapper
            if config["encoder"] == "bert":
                from contrastive.bert import BertEncoder
                model = BertEncoder(config["bert_config"])
        elif config["method"] == "regularized":
            from regularized.train import PLWrapper
            if config["encoder"] == "bert":
                from regularized.bert import BertEncoder
                model = BertEncoder(config["bert_config"])

        self.model = PLWrapper(model, config)
        checkpoint = torch.load(self.cp_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.encode_fn = self.model.forward

    def encode(self):
        # encode all chunks in the song, loaded by the dataloader
        self.model.eval()
        self.encodings = []
        for chunk in self.dl:
            if self.config["method"] == "contrastive":
                encoding = self.model(chunk)
            elif self.config["method"] == "regularized":
                encoding = self.model(chunk)[0]
            self.encodings.append(encoding.detach().cpu().numpy())
        self.encodings = np.concatenate(self.encodings, axis=0)  # (num_chunks, emb_dim)

    def perform_clustering(self, alg="dbscan"):
        """
        Use clustering to pick bases
        """
        if alg == "dbscan":
            self.clustering = DBSCAN(eps=5, min_samples=4).fit(
                self.encodings
            )  # clustering results in self.clustering.labels_

        print(self.clustering.labels_)

    def compute_self_similarity(self):
        """
        Calculate the self similarity of all encodings
        """
        encodings = self.encodings  # (num_chunks, emb_dim)
        self_similarity = np.matmul(encodings, encodings.T)  # (num_chunks, num_chunks)
        self.ssm = self_similarity

    def plot_colored_pr(self, filetype="png", savepath=None):
        """
        Save the result as a colored pianoroll
        Args:
            filetype: 'png' or 'mid'
                if 'png', apply colors to the pianoroll, and plot the pianoroll
                if 'mid', save the clusters as a new track of the midi file (TODO)
            savepath: path to save the file
        """
        self.clustering.labels_
        if filetype == "png": # svg is too large
            if savepath is None:
                src_name = os.path.splitext(os.path.basename(self.input_path))[0]
                savepath = "vis_" + src_name + "_colored_pr.png"
            assert os.path.splitext(savepath)[1] == ".png"
            data_chunks = self.dl.dataset.data_chunks
            for i in range(len(data_chunks)):
                data_chunks[i] = data_chunks[i].numpy()
            data_chunks = np.concatenate(data_chunks, axis=0)  # (num_chunks, 84)
            pr_all = data_chunks.T  # (84, num_chunks)
            pr_all = pr_all[::-1, :]  # reverse the order of the rows
            # apply coefficients to pianoroll, so that we have different colors
            for i in range(len(self.clustering.labels_)):
                if self.clustering.labels_[i] != -1:
                    chunk_grid_start = int(i * self.chunk_size / self.grid)
                    chunk_grid_end = int((i + 1) * self.chunk_size / self.grid)
                    pr_all[:, chunk_grid_start:chunk_grid_end] = pr_all[
                        :, chunk_grid_start:chunk_grid_end
                    ] * (self.clustering.labels_[i] + 2)
            width = len(self.clustering.labels_) * self.chunk_size / self.grid / 84 * 10
            plt.figure(figsize=(width, 10))  # (width, height)
            # set the color palette: this is so fun
            my_palette = ["#b2d2e8", "#eeeeee", "#eedd82", "#fa8072", "#dda0dd", "#4728a9", "#bfeeef", "#228b22"]
            cmap = sns.color_palette(my_palette, n_colors=len(set(self.clustering.labels_)) + 1)
            hm = sns.heatmap(
                pr_all, cmap=cmap, xticklabels=False, yticklabels=False, cbar=True
            )
            plt.savefig(savepath, dpi=200)

    def plot_hm(self, savepath=None):
        """
        Plot the motive tendencies as a heatmap
        """
        src_name = os.path.splitext(os.path.basename(self.input_path))[0]
        if savepath is None:
            savepath = "vis_" + src_name + "_hm.svg"

        # first, get the cluster centers
        n_clusters = len(set(self.clustering.labels_)) - (
            1 if -1 in self.clustering.labels_ else 0
        )
        cluster_center_embs = np.zeros((n_clusters, self.encodings.shape[1]))
        for i in range(n_clusters):
            cluster_center_embs[i] = np.mean(
                self.encodings[self.clustering.labels_ == i], axis=0
            )
        
        # then, compute the distance between each chunk and each cluster center
        hm_mtx = np.zeros((2 * n_clusters - 1, self.encodings.shape[0]))

        for i in range(n_clusters):
            for j in range(self.encodings.shape[0]):
                hm_mtx[2 * i, j] = np.linalg.norm(
                    self.encodings[j] - cluster_center_embs[i]
                )
        
        # change the zeros to the largest distance
        hm_mtx[hm_mtx == 0] = np.max(hm_mtx)

        height = int(hm_mtx.shape[0] / 2)
        width = int(hm_mtx.shape[1] / 8)

        plt.figure(figsize=(width, height))
        hm = sns.heatmap(hm_mtx, cmap="Blues_r", cbar=False)
        x_major_locator = MultipleLocator(10)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.set_xticklabels([int(i) for i in ax.get_xticks()], fontsize=16)
        plt.yticks([])
        plt.xlabel("Chunk Index", fontsize=16)
        plt.ylabel("Motif Index", fontsize=16)

        plt.savefig(savepath, bbox_inches="tight")

    def plot_lines(self, savepath=None):
        """
        Plot the motive tendencies as lines
        """
        src_name = os.path.splitext(os.path.basename(self.input_path))[0]
        if savepath is None:
            savepath = "vis_" + src_name + "_lines.svg"

        # first, get the cluster centers
        n_clusters = len(set(self.clustering.labels_)) - (
            1 if -1 in self.clustering.labels_ else 0
        )
        cluster_center_embs = np.zeros((n_clusters, self.encodings.shape[1]))
        for i in range(n_clusters):
            cluster_center_embs[i] = np.mean(
                self.encodings[self.clustering.labels_ == i], axis=0
            )

        # then, compute the distance between each chunk and each cluster center
        lines = [[] for i in range(n_clusters)]
        for i in range(n_clusters):
            for j in range(self.encodings.shape[0]):
                lines[i].append(np.linalg.norm(self.encodings[j] - cluster_center_embs[i]))
            # negative exponential
            lines[i] = np.exp(-np.array(lines[i]) / 10)

        plt.figure(figsize=(10, 3))
        for i, line in enumerate(lines):
            # need to smooth the lines
            line_smooth = make_interp_spline(np.arange(len(line)), line)(np.linspace(0, len(line) - 1, 1000))
            plt.plot(line_smooth, label="Motif " + str(i + 1), linewidth=2)

        plt.yticks([])
        plt.xticks([])
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Motif Presence", fontsize=20)
        # put legend outside the plot on the right, and remove the frame border
        plt.legend(fontsize=20, loc="upper right", framealpha=0, bbox_to_anchor=(1.25, 0.9))

        plt.savefig(savepath, bbox_inches="tight")

    def plot_scatters(self, savepath=None):
        """
        Plot the encodings as scatters
        """
        src_name = os.path.splitext(os.path.basename(self.input_path))[0]
        if savepath is None:
            savepath = "vis_" + src_name + "_scatters.svg"

        tsne = TSNE(n_components=2, n_iter=2000)
        encodings_tsne = tsne.fit_transform(self.encodings)

        # if two encodings have identical values, change one of them a little bit to make them distinguishable
        for i in range(encodings_tsne.shape[0]):
            for j in range(i + 1, encodings_tsne.shape[0]):
                if np.allclose(encodings_tsne[i], encodings_tsne[j]):
                    encodings_tsne[j] += np.random.rand(encodings_tsne.shape[1]) * 50

        # edit legend
        hue = self.clustering.labels_.tolist()
        for i in range(len(hue)):
            if hue[i] == -1:
                hue[i] = "Noise"
            else:
                hue[i] = "Motif " + str(hue[i])
        
        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            x=encodings_tsne[:, 0],
            y=encodings_tsne[:, 1],
            hue=hue,
            palette=sns.color_palette("hls", len(set(self.clustering.labels_))),
            legend="full",
            s=150
        )

        plt.xticks([])
        plt.yticks([])
        # the legend shouldn't be transparent
        plt.legend(fontsize=22, loc="upper right", framealpha=1)
        plt.savefig(savepath, bbox_inches="tight")
