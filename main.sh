#!/bin/bash

# =========== process data ===========
python process_data.py --raw_data_fp=data/phase1b_fr.csv --save_dir=results/processed_data --cuda_device_num=3 --ckpt_dir=../checkpoints --batch_size=50


# =========== split train and test set docIDs ===========
mkdir experiments
python split_train_test_docIDs.py --data_fp=results/processed_data/phase1b_fr_preprocessed.csv --save_dir=experiments


# =========== create train and test sets (should be disjoint) ===========
python create_train_test_sets.py --raw_data_fp=data/phase1b_fr.csv --preprocessed_fp=results/processed_data/phase1b_fr_preprocessed.csv --text_span_fp=results/processed_data/phase1b_fr_text_spans.csv --save_dir=experiments


# =========== doc-level classification ===========
# some code here: use non-lexical features only 

python perform_clusterings.py --save_dir=experiments/doc-level/results_train --data_fp=experiments/doc-level/train.csv --text_col=text --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True


python perform_clusterings.py --save_dir=experiments/doc-level/results_test --data_fp=experiments/doc-level/test.csv --text_col=text --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True




# =========== sentence-level classification with clusterings ===========
# +++++++++++ clustering on train set +++++++++++ 
python perform_clusterings.py --save_dir=experiments/sentence-level/results_train --data_fp=experiments/sentence-level/train.csv --text_col=text --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True



# +++++++++++ clustering on test set +++++++++++
python perform_clusterings.py --save_dir=experiments/sentence-level/results_test --data_fp=experiments/sentence-level/test.csv --text_col=text --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True












# =========== target-level classification with clusterings ===========

# +++++++++++ clustering on train set +++++++++++
python perform_clusterings.py --save_dir=experiments/target-level/results_train --data_fp=experiments/target-level/train.csv --text_col=span --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True



# +++++++++++ clustering on test set +++++++++++
python perform_clusterings.py --save_dir=experiments/target-level/results_test --data_fp=experiments/target-level/test.csv --text_col=span --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True




# =========== author-true target-level classification with clusterings ===========

python perform_clusterings.py --save_dir=experiments/target-level/author_true/results_train --data_fp=experiments/target-level/author_true/train.csv --text_col=span --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True


python perform_clusterings.py --save_dir=experiments/target-level/author_true/results_test --data_fp=experiments/target-level/author_true/test.csv --text_col=span --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True








# ====== english test ======


python perform_clusterings.py --save_dir=test_en/sentence-level/results_test --data_fp=test_en/sentence-level/test.csv --text_col=text --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True 



python perform_clusterings.py --save_dir=test_en/target-level/results_test --data_fp=test_en/target-level/test.csv --text_col=text --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True



python perform_clusterings.py --save_dir=test_en/target-level/author_true/results_test --data_fp=test_en/target-level/author_true/test.csv --text_col=text --sbert_model_name_or_path=../../sbert_models/all-mpnet-base-v2 --sbert_batch_size=50 --cuda_device_num=3 --num_of_runs=3 --cluster_method=kmeans,hdbscan --n_clusters=10,20,30,40,50,60,70,80,90,100,150,200,250,300,500 --min_cluster_size=10,20,40,80,100,150,200,300,400,500 --reduced_dim=10,30,50 --perplexity=40 --other_fields=mediaType,label --visualize_result=False --re_do_embd=True




# deployed trained systems

python deploy_fnn.py --save_dir=test_en/sentence-level --model_dir=experiments/sentence-level/classification/ --cuda_device_num=0 --run_num=5


python deploy_fnn.py --save_dir=test_en/target-level --model_dir=experiments/target-level/classification/ --cuda_device_num=0 --run_num=5

python deploy_fnn.py --save_dir=test_en/target-level/author_true --model_dir=experiments/target-level/author_true/classification/ --cuda_device_num=0 --run_num=5
