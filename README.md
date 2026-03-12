# SjoMap
Code for manuscript titled "Spatial multi-omics reveals distinct stromal signatures of Sjögren’s Disease by anti-SSA antibody profile in Minor Salivary Gland tissues". 

The analysis code consists of `cosmx_xenium`, `merfish`, `seqfish`, `IF`, and `coculture` sections. Each folder contains the analysis code for the corresponding datasets.

## cosmx_xenium

`01_qcComparison.ipynb` calculates basic quanlity attribute comparisons between CosMx and Xenium data from the manuscript such as total transcripts detected per cells. Further, it performs patch effect normalization of common markers between CosMx and Xenium data. `annotation_v4.ipynb` performs cell type annotations for both Xenium and CosMx data based on the analysis output from the instrument run. `immuneCellSubtypes.ipynb` performs clustering analysis of each immune cell type to identify subtypes. `04_visualization.ipynb` generates tissue-level visualizations in napari.

### GATStromalNeighborhood

`GATStromalNeighborhood` folder contains code for graph attention network embedding of stromal neighborhoods. `combined_models.py` and `models.py` constructs necessary deep learning models. `subgraphSampling.ipynb` generates kNN graph across all cells in the same section, and samples cells within 2-hop distance from the stromal cells to form stromal neighborhoods. `graphClassifierTraining.ipynb` trains the GAT embedding and classification model using stromal neighborhoods from Xenium, and output basic model evaluation metrics and GAT projection. `graphEmbeddingAnalysisCommon.ipynb` performs downstream analysis based on the latent space embedding of stromal neighborhoods from Xenium data. `IntegrationClassification.ipynb` trains the domain adversarial neural network (DANN) to integrate stromal neighborhoods from Xenium and CosMx data, and calculate the latent space embedding. `DANNEmbeddingAnalysis.ipynb` performs downstream analysis based on the DANN integrated latent space. `pathwayFigures.ipynb` utilizes the output from Enrichr analysis to generate visualization of enriched pathways in selected cells. "04_visualization.ipynb" generates large-image visualization of stromal neighborhoods.

### duplicateVerification

`xenium_set1` and `xenium_set2` folders contain `combined_models.py`, `models.py`, `graphClassifierTraining.ipynb`, and `graphEmbeddingAnalysisCommon.ipynb`. They train the GAT model using one of the duplicates of Xenium data, and performs downstream analysis of the latent space. `cosmx_set1` and `cosmx_set2` folders contain `combined_models.py`, `models.py`, "integrationClassification.ipynb", and `DANNEmbeddingAnalysis.ipynb`. They train teh DANN model using one of the duplicates on CosMx data and performs downstream analysis of the latent space.

### tunableNeighborhoods

Both `3hop` and `4hop` folders contain `combined_models.py`, `models.py`, `graphClassifierTraining.ipynb`, `graphEmbeddingAnalysisCommon.ipynb`, and `subgraphSampling.ipynb`. `subgraphSampling.ipynb` generates stromal neighborhoods of appropriate sizes. `graphClassifierTraining.ipynb` trains the GAT model. `graphEmbeddingAnalysisCommon.ipynb` performs downstream anslysis on the latent space. `neighborhoodSize.ipynb` quantifies the average size of 2-hop, 3-hop, and 4-hop neighborhoods.

## merfish

`merfish` folder contains code for the analysis of MERFISH data of 140 custom gene panel and 815 immune-oncology panel. 

### 140GenePanel

`140GenePanel` folder contains code for analysis of 140 custom gene panel MERFISH data. `00_annotation.ipynb` performs cell type annotation. `01_neighborhoodSampling.ipynb` extracts stromal neighborhood. `02_DANNIntegration.ipynb` trains the DANN for data integration. `03_integrationAnalysis.ipynb` performs downstream analysis on the latent space. `04_pathwayFigures.ipynb` generates pathway enrichment visualization from Enrichr analysis results. `visualization.ipynb` generates napari visualization. The outlier cell filtering was performed witht he assistance of napari visualization.

### 815IOPanel

`815IOPanel` contains code for analysis of 815 immune oncology panel MERFISH data. `01_annotation.ipynb` performs cell-type annotation. `02_outlierFiltering.ipynb` uses the napari software to help filtering outlier cells isolated from the tissue section. `03_neighborhoodSampling.ipynb` finds stromal neighborhoods. `04_DANN.ipynb` trains the DANN for data integration. `05_integrationAnalysis.ipynb` performs downstream anlaysis of DANN integration. `06_pathway.ipynb` performs pathway enrichment analysis based on the elevated markers of selected cell groups.

# Code running environments

Different analysis needs to be run in diferent conda environments. Code for training deep learning models can be run under the environment specified by `torchEnv.txt`. Napari visualization codes can be run under `naparienv.txt`. All ohter code can be run under `scanpy.txt`.

The envionrments can be created using the following command: `conda create --name [envName] --file [environment].txt`
