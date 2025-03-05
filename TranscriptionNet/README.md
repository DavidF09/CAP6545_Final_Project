## 🖋️ Introduction
**TranscriptionNet** is an attention-based deep learning algorithm that integrates various large-scale gene function network information to predict changes in induced gene expression (GECs) by perturbing each gene in the genome.<br>

An overview of TranscriptionNet can be seen below.
![image](TranscriptionNet.jpg)

**TranscriptionNet** is composed of two networks, **FunDNN** (functional network-based deep neural network) and **GenSAN** (genetic perturbation-based self-attention network), which load the genome-wide functional connection knowledge among genes and complementary information between different types of genetic interference manners on same genes, respectively.<br>
* **FunDNN** uses adjacency matrices for multiplex gene interaction networks as input. Each network passes through a three-layer Graph Attention Network (GAT) to generate network-specific gene features, which are then combined into the integrated features. The integrated features are propagated through multiple dense layers to fit pre-GECs induced by each types of genetic perturbation (RNAi, OE, and CRISPR).<br>
* **GenSAN** takes GECs for all three types of genetic perturbations as input. takes GECs for all three types of genetic perturbations as input. The pre-GECs of RNAi and the real GECs of CRISPR and OE are processed together through multi-layer transformer encoding blocks (with axial self-attention in the dual-tower architecture) to capture complementary gene expression information between them. Moreover, the self-attention block is reinforced by feeding the processed GECs recursively into the same modules (named “recycling”). Finally, the predicted GECs for RNAi can be extracted from the output GECs matrix. Moreover, the self-attention block is reinforced by feeding the processed GECs recursively into the same modules (named “recycling”). Finally, the predicted GECs for RNAi can be extracted from the output GEC matrix.<br>

## :gear: Installation
This package requires [Python 3.8](https://www.python.org/downloads/) with the following libraries:
```python
torch==2.0.0
numpy==1.26.0
pandas==2.1.4
scipy==1.11.4
scikit-learn==1.3.2
matplotlib==3.8.2
```

You can install these libraries by running the command 

```
pip install -r requirements.txt
``` 

from this project's root directory.

## 📁 Example input data

`/example/raw_data/` Raw data, including network integration features, gene expression change (GECs) data of 978 landmark genes in three types of RNAi, OE, and CRISPR. The GECs files for RNAi, OE and CRISPR are transcription profile data of 978 landmark genes induced by 4449, 3518 and 5139 perturbation genes, respectively. The row names are the Entrez IDs of the perturbation genes, and the column names are the Entrez IDs of the 978 landmark genes.  

`/example/gene networks/` The TranscriptionNet model takes as input seven distinct types of gene functional networks.  

`/example/gene meta/` This includes expression data for 12,328 genes induced by 100 perturbation genes of the RNAi perturbation type, along with meta-information for 23,985 genes.  

`/example/drug gene association/` This includes expression data for 12,328 genes induced by 100 drug perturbations, along with up-regulation and down-regulation feature labels for 100 genes.  

`data_process.py` The raw data processing process, dividing the training set, the validation set, and the test set.

`config_parser.py` All parameters of the Transcription model.

`example_run.ipynb` An example for running example data.
