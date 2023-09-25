import torch
import subprocess
from pathlib import Path
import logging


en_gpt2_save_dir = 'models/en_gpt2_pretrained_models/'
en_gpt2_xl_save_dir = 'models/en_gpt2_xl_pretrained_models/'


def get_available_device(memory_thresh=7000, logger=None):
    """
    :param memory_thresh: occupied memory thresh
    :return: cuda device with memory occupied LESS THAN memory thresh
    """
    if torch.cuda.is_available():
        gpu_memory_map = get_gpu_memory_map()
        if logger is not None:
            logger.info(f'GPU MEMORIES: {gpu_memory_map}')
        else:
            print(f'>>> GPU MEMORIES: {gpu_memory_map}')
        for gpu_id in range(len(gpu_memory_map.keys())):
            if gpu_memory_map[gpu_id] < memory_thresh:
                if logger is not None:
                    logger.info(f"DEVICE {gpu_id} IS AVAILABLE.")
                else:
                    print(f">>> DEVICE {gpu_id} IS AVAILABLE.")
                return torch.device('cuda:{}'.format(gpu_id))
        else:
            if logger is not None:
                logger.info('CPU IS AVAILABLE.')
            else:
                print(">>> CPU IS AVAILABLE.")
            return torch.device('cpu')
    else:
        return torch.device('cpu')


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def create_params(parser):

    parser.add_argument('--memory_thresh', default=1000, type=int, required=False, help='gpu memory thresh')
    parser.add_argument('--cuda_id', default=-1, type=int, required=False, help='gpu id')

    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--override', action='store_true', default=False, help='override mode')
    parser.add_argument('--is_en_gpt2', action='store_true', default=False, help='GPT-2 Small')
    parser.add_argument('--is_en_gpt2_xl', action='store_true', default=False, help='GPT-2 XL')

    parser.add_argument('--length', default=-1, type=int, required=False, help='generation length')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='sampling temperature')
    parser.add_argument('--topk', default=0, type=int, required=False, help='k in top-k')
    parser.add_argument('--topp', default=0.95, type=float, required=False, help='p in top-p')
    parser.add_argument('--n_fraction', default=500, type=int, required=False, help='n in top1ctrl')
    parser.add_argument('--iqr_coef', default=1.50, type=float, required=False, help='iqr coefficient')
    parser.add_argument('--gumbel_scale', default=1.0, type=float, required=False, help='scale gumbel sampling (EXPERIMENTAL)')
    parser.add_argument('--sampling_strategy', default='rand', type=str, required=False, help='random sampling or minimized sampling (EXPERIMENTAL)')
    parser.add_argument('--model_path', default='', type=str, required=False, help='path to load model')
    parser.add_argument('--save_path', default='generated_samples', type=str, required=False, help='path to save samples')
    parser.add_argument('--save_path_sub_dir', default='', type=str, required=False, help='path for each sampling method')
    parser.add_argument('--articles_per_title', default=50, type=int, required=False, help='number of generation per each sampling parameter')
    parser.add_argument('--titles', default='She walks in beauty', type=str, required=False, help='input context')
    parser.add_argument('--beam_search', action='store_true', default=False, help='use beamsearch')
    parser.add_argument('--iqr_ip_weighting', action='store_true', default=False, help='use IQR-IP sampling')
    parser.add_argument('--show_tqdm_bar', action='store_true', default=False, help='show sampling progress')
    parser.add_argument('--seed', default='0', type=int, required=False, help='initial seed for sampling')


def get_latest_model(model_save_dir, epoch_num=None, logger=None):
    """
    :param model_save_dir:
    :return:
    """
    if logger is None:
        logger = get_thread_logger('MAIN')

    if type(model_save_dir) == str:
        model_save_dir = Path(model_save_dir)
    dir_name_list = list(directory.name for directory in model_save_dir.glob('*/') if directory.is_dir())

    if epoch_num is not None:
        target_dir = f'model_epoch{epoch_num}'
    else:
        target_dir = 'final_model'

    if target_dir in dir_name_list:
        result = model_save_dir / target_dir
        logger.info(f'CHOOSE DESIGNATED MODEL: {result}')
        return model_save_dir / target_dir
    else:
        dir_name_list = sorted(dir_name_list, key=lambda x: int(x.split('_')[1].strip('epoch')), reverse=True)
        if dir_name_list:
            result = model_save_dir / dir_name_list[0]
            logger.info(f'{target_dir} NOT FOUND. CHOOSE LATEST MODEL: {result}')

            return result
        else:
            logger.info(f'WARNING. NO RESULTS FOUND IN {model_save_dir}')
            return []


def get_absolute_path():
    prefix = Path(__file__).resolve().parent.parent.absolute()
    return prefix


def get_thread_logger(name='Main'):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not logger.handlers:
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)
    return logger


def split_token_list_with_windows(tokens, n_ctx, stride):
    start_point = 0
    samples = []
    while start_point < len(tokens) - n_ctx:
        samples.append(tokens[start_point: start_point + n_ctx])
        start_point += stride

    if start_point < len(tokens):
        samples.append(tokens[len(tokens) - n_ctx:])
    return samples

