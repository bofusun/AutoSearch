
# index_file=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/RL-Factory/corpus/e5_Flat.index
# corpus_file=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/RL-Factory/corpus/wiki-18.jsonl
# retriever_name=e5
# retriever_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/RL-Factory/model/e5-base-v2

# python /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/Search-R1-new/search_r1/search/retrieval_server.py --index_path $index_file \
#                                             --corpus_path $corpus_file \
#                                             --topk 3 \
#                                             --retriever_name $retriever_name \
#                                             --retriever_model $retriever_path \
#                                             --faiss_gpu

index_file=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/chongwenyue/RL-Factory/corpus/e5_Flat.index
corpus_file=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/chongwenyue/RL-Factory/corpus/wiki-18.jsonl
retriever_name=e5
retriever_path=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/chongwenyue/RL-Factory/model/e5-base-v2

python /mnt/dolphinfs/ssd_pool/docker/user/hadoop-xt-ai-search/ai-search/chongwenyue/Search-R1-new/search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
