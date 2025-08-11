# utils/mutation_utils.py
import torch

def random_mutation(embedding, mutation_rate=0.5, mutation_scale=0.1):
    """
    对输入的嵌入进行随机突变
    :param embedding: 输入的蛋白质嵌入张量 [batch_size, embeddings_dim, sequence_length]
    :param mutation_rate: 突变率，即每个元素发生突变的概率
    :param mutation_scale: 突变的尺度，即突变的幅度
    :return: 突变后的嵌入张量
    """
    mask = torch.rand_like(embedding) < mutation_rate
    noise = torch.randn_like(embedding) * mutation_scale
    mutated_embedding = embedding.clone()
    mutated_embedding[mask] += noise[mask]
    return mutated_embedding