import os
import torch.multiprocessing as mp
from src.utils import get_thread_logger


"""
Generate all passages for IQR-IP algorithm.
"""

# You may change num_parallel according to your GPU configuration.
num_parallel = 4


def generator(rank, k_p_sets):

    k_string, p_string = k_p_sets[rank]
    k_value = int(k_string.replace('k', ''))
    p_value = float('0.' + p_string.replace('p', ''))
    cuda_id = rank % num_parallel

    command = f"python generate_sample.py --is_en_gpt2_xl --length 200 " \
              f"--topp {p_value} --topk {k_value} --n_fraction 100 --iqr_ip_weighting " \
              f"--save_path_sub_dir gpt2_xl_en_{k_string}{p_string}n100_iqr " \
              f"--articles_per_title 5000 --cuda_id {cuda_id}"
    logger = get_thread_logger(f'RANK {rank}')
    logger.info(command)
    os.system(command)
    logger.info('DONE')


if __name__ == '__main__':
    k_choices = ['k200', 'k640', 'k2000', 'k6400']

    p_choices = ['p50', 'p60', 'p70', 'p80', 'p90', 'p95', 'p99', 'p999']

    k_p_sets = []
    for k in k_choices:
        for p in p_choices:
            k_p_sets.append([k, p])

    logger = get_thread_logger(f'MAIN')

    assert len(k_p_sets) % num_parallel == 0

    for ind_group in range(len(k_p_sets) // num_parallel):

        k_p_subsets = k_p_sets[ind_group * num_parallel: (ind_group + 1) * num_parallel]
        logger.info(f'RUNNING SUB GROUP {ind_group}: {k_p_subsets}')
        mp.spawn(generator,
                 args=(k_p_subsets,),
                 nprocs=len(k_p_subsets),
                 join=True)
        logger.info(f'SUB GROUP {ind_group} FINISHED')

