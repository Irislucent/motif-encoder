This is the codebase of the paper titled [**Motif-Centric Representation Learning for Symbolic Music**](https://arxiv.org/abs/2309.10597).

***Abstract**: Music motif, as a conceptual building block of composition, is crucial for music structure analysis and automatic composition. While human listeners can identify motifs easily, existing computational models fall short in representing motifs and their developments. The reason is that the nature of motifs is implicit, and the diversity of motif variations extends beyond simple repetitions and modulations. In this study, we aim to learn the implicit relationship between motifs and their variations via representation learning, using the Siamese network architecture and a pretraining and fine-tuning pipeline. A regularization-based method, VICReg, is adopted for pretraining, while contrastive learning is used for fine-tuning. Experimental results on a retrieval-based task show that these two methods complement each other, yielding an improvement of 12.6% in the area under the precision-recall curve. Lastly, we visualize the acquired motif representations, offering an intuitive comprehension of the overall structure of a music piece. As far as we know, this work marks a noteworthy step forward in computational modeling of music motifs. We believe that this work lays the foundations for future applications of motifs in automatic music composition and music information retrieval.*

# How to synthesize data
```
cd dataset
python preprocess.py --data_dir "your dataset directory, default is pop909" --save_dir "directory to save preprocessed data"
python metaphor_by_rules.py --data_dir "directory of preprocessed data" --save_dir "directory to save metaphorized data" --n_metaphors "# data views"
python split_train_val.py --data_dir "directory of metaphorized data" --save_dir "directory to save train/val datasets"
```

# How to generate the real dataset from labels
```
cd dataset
python label_to_real_data.py --data_dir "directory of original data" --labels_dir "../data" --chunks_dir "directory of preprocessed data" --output_dir "directory to save relabeled data"
python split_train_val.py --data_dir "directory of relabeled data" --save_dir "directory to save train/val datasets"
```

# How to train a model
```
python run_training.py --config "your-config-file.yaml"
```
Two example .yaml files are provided, respectively corresponding to "contrastive/" and "regularized/".

# How to do motif-based music visualization
Enter model checkpoint path at the entry "active_checkpoint" in "your-config-file.yaml".
```
python run_visualization.py --config "your-config-file.yaml --input_path "path to target .mid file"
```
