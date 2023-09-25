import os
import torch.multiprocessing as mp
from src.utils import get_thread_logger


"""
Generate all passages for temperature sampling.
"""

# You may change num_parallel according to your GPU configuration.
num_parallel = 4


def generator(rank, subsets):
    t_string = subsets[rank]

    t_value = int(t_string.replace('t', '')) / 100.0

    cuda_id = rank % len(subsets) + 6

    logger = get_thread_logger(f'RANK {rank}')

    num_artical = 5000
    command = f"python generate_sample.py --is_en_gpt2_xl --length 200 " \
              f"--topp 1.0 --topk 0 --temperature {t_value} --n_fraction 0 " \
              f"--save_path_sub_dir gpt2_xl_en_{t_string} " \
              f"--articles_per_title {num_artical} --cuda_id {cuda_id}"

    logger.info(command)

    try:
        os.system(command)
        logger.info(f'RANK {rank} DONE, temperature={t_string}')
    except:
        logger.info(f'RANK {rank} FAILED, temperature={t_string}')


if __name__ == '__main__':

    all_sets = ['t50', 't60', 't70', 't80', 't90', 't100', 't110', 't120', 't130', 't140', 't150', 't160']

    logger = get_thread_logger(f'MAIN')

    assert len(all_sets) % num_parallel == 0

    for ind_group in range(len(all_sets) // num_parallel):

        subsets = all_sets[ind_group * num_parallel: (ind_group + 1) * num_parallel]
        logger.info(f'RUNNING SUB GROUP {ind_group} / {len(all_sets) // num_parallel}: {subsets}')

        mp.spawn(generator,
                 args=(subsets,),
                 nprocs=len(subsets),
                 join=True)
        logger.info(f'SUB GROUP {ind_group} FINISHED')


