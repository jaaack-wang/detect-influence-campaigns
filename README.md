### Description

This repo contains code for the paper: [Clustering Document Parts: Detecting and Characterizing Influence Campaigns from Documents](https://arxiv.org/abs/2402.17151). The paper is accepted by the [6th Workshop on Natural Language Processing and Computational Social Science (NLP+CSS)](https://sites.google.com/site/nlpandcss/nlp-css-at-naacl-2024) and will appear in the proceedings soon. 

The ``scripts`` folder has code mostly for data preprocessing, belief-tagging, and clustering. The code outside the folder was used for doing the experiments. The file ``main.sh`` logs the experimental procedure. 

The belief tagging system we use comes from this [generative-belief](https://github.com/yurpl/generative-belief) repo. We use the [Linguistic Feature Extractor](https://github.com/jaaack-wang/ling_feature_extractor) to extract the 96 general linguistic features (mostly lexical frequency counts). We modified the LFE program for the purpose of our paper and place the code in the ``LFE`` folder.



### Data

We use data collected during a large research program with [a DARPA INCAS project](https://www.darpa.mil/program/influence-campaign-awareness-and-sensemaking). We expect the data to be made public after the end of the research program. We use this dataset as we are not aware of any other datasets that have expert-verified annotations indicating if a collection of documents contain influence campaigns. 



### Clustering to classification

``sent-level-clustering2classification.ipynb`` is a conceptual demonstrtaion of how we can transform results obtained from clustering into classifying high-influence documents, with or without aggregation, provided that we know what clusters are high-influence clusters. In this demonstration based on results from sentence-level clustering, we use the ground truth label from our data and show that how the approach performns given varying thresholds of the ``alpha``, the percentage of documents that reflect an influence campaign inside a cluster in order for the cluster to be considered as a high-influence cluster. In our paper, we train cluster-level classifiers to detect high-influence clusters. 



### Citation

``````
@misc{wang2024clustering,
      title={Clustering Document Parts: Detecting and Characterizing Influence Campaigns From Documents}, 
      author={Zhengxiang Wang and Owen Rambow},
      year={2024},
      eprint={2402.17151},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
``````
