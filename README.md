# SjoMap
Code for manuscript titled "Spatial multi-omics reveals distinct stromal signatures of Sjögren’s Disease by anti-SSA antibody profile in Minor Salivary Gland tissues". 

The analysis code consists of "cosmx_xenium", "merfish", "seqfish", "IF", and "coculture" sections. Each folder contains the analysis code for the corresponding datasets.

## cosmx_xenium

"01_qcComparison.ipynb" calculates basic quanlity attribute comparisons between CosMx and Xenium data from the manuscript such as total transcripts detected per cells. Further, it performs patch effect normalization of common markers between CosMx and Xenium data. "annotation_v4.ipynb" performs cell type annotations for both Xenium and CosMx data based on the analysis output from the instrument run. "immuneCellSubtypes.ipynb" performs clustering analysis of each immune cell type to identify subtypes. "04_visualization.ipynb" generates tissue-level visualizations in napari.

### GATStromalNeighborhood

"GATStromalNeighborhood" folder contains code for graph attention network embedding of stromal neighborhoods. "combined_models.py" and "models.py" constructs necessary deep learning models. "subgraphSampling.ipynb" generates kNN graph across all cells in the same section, and samples cells within 2-hop distance from the stromal cells to form stromal neighborhoods. "graphClassifierTraining.ipynb" trains the GAT embedding and classification model using stromal neighborhoods from Xenium, and output basic model evaluation metrics and GAT projection. "graphEmbeddingAnalysisCommon.ipynb" performs downstream analysis based on the latent space embedding of stromal neighborhoods from Xenium data. "IntegrationClassification.ipynb" trains the domain adversarial neural network (DANN) to integrate stromal neighborhoods from Xenium and CosMx data, and calculate the latent space embedding. "DANNEmbeddingAnalysis.ipynb" performs downstream analysis based on the DANN integrated latent space. "pathwayFigures.ipynb" utilizes the output from Enrichr analysis to generate visualization of enriched pathways in selected cells. "04_visualization.ipynb" generates large-image visualization of stromal neighborhoods.

### duplicateVerification

"xenium_set1" and "xenium_set2" folders contain "combined_models.py", "models.py", "graphClassifierTraining.ipynb", and "graphEmbeddingAnalysisCommon.ipynb". They train the GAT model using one of the duplicates of Xenium data, and performs downstream analysis of the latent space. "cosmx_set1" and "cosmx_set2" folders contain "combined_models.py", "models.py", "integrationClassification.ipynb", and "DANNEmbeddingAnalysis.ipynb". They train teh DANN model using one of the duplicates on CosMx data and performs downstream analysis of the latent space.

### tunableNeighborhoods
