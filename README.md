# circ2DGNN: circRNA-disease Association Prediction via Transformer-based Graph Neural Network

## Requirements
use anaconda to create the environment: conda install xxx
- python==3.8.16
- pytorch==1.12.1
- cudatoolkit==10.2.89
- dgl==1.1.1.cu102
- scikit-learn==1.2.2
- numpy==1.24.3
- pandas==1.5.3

## Data Description
Since the whole graph data is too large to store on GitHub, 
after pulling the code, run:
- circ_seq_similarity.py
- disease_semantic_similarity.py
- data_split.py

in src directory to generate the whole data.

Finally, open train.py and follow the comments to run the code. ^_^
