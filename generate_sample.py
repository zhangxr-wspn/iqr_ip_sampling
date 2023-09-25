import torch
import numpy as np
import argparse
from pytorch_transformers import GPT2Tokenizer
from pathlib import Path
from src import utils
from src.generation import get_model, prepare_context
from tqdm import tqdm
from src.sampling import sample_sequence


"""
Generate function with access to many algorithm configurations.

Part of the code come from:
https://github.com/minimaxir/gpt-2-simple

"""


def main():
    parser = argparse.ArgumentParser()
    utils.create_params(parser)
    args = parser.parse_args()

    if args.cuda_id == -1:
        logger = utils.get_thread_logger()
        device = utils.get_available_device(memory_thresh=args.memory_thresh, logger=logger)
    else:
        device = f'cuda:{args.cuda_id}'
        logger = utils.get_thread_logger(device)
        logger.info(f'USING DESIGNATED DEVICE: {device}')

    """
    REPLACE THE FOLLOWING 9 LINES OF CODES IF YOU LOAD MODEL AUTOMATICALLY WITH:
        
    from pytorch_transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    model = GPT2Model.from_pretrained('gpt2-xl')
    """
    if args.is_en_gpt2_xl:
        args.model_path = Path(utils.en_gpt2_xl_save_dir) / 'final_model'
    else:
        raise NotImplementedError('WRONG MODEL SELECTION')
    if args.is_en_gpt2_xl:
        tokenizer = GPT2Tokenizer.from_pretrained(utils.en_gpt2_save_dir + 'final_model/')
    else:
        raise NotImplementedError('WRONG MODEL SELECTION')
    model = get_model(args, logger=logger)
    model.to(device)
    model.eval()

    args.save_path = Path(args.save_path) / Path(args.save_path_sub_dir) / args.model_path.parent.name
    args.save_path.mkdir(exist_ok=True, parents=True)
    logger.info(f'SAVE PATH: {args.save_path}')
    args.titles = ['She walks in beauty']
    n_ctx = model.config.n_ctx

    if args.length == -1:
        args.length = model.config.n_ctx

    np.random.seed(args.seed)
    seed_array = np.random.randint(1, 100000000, args.articles_per_title)

    for index, title in tqdm(enumerate(args.titles)):

        context_tokens = prepare_context(args, tokenizer, title)

        for j in tqdm(range(args.articles_per_title)):

            torch.manual_seed(seed_array[j])
            np.random.seed(seed_array[j])

            file = args.save_path / Path('sample' + str(j + 1) + '.txt')

            out, log, info = sample_sequence(
                n_ctx=n_ctx,
                model=model, length=args.length,
                context=context_tokens,
                temperature=args.temperature,
                top_k=args.topk, top_p=args.topp, n_fraction=args.n_fraction,
                device=device,
                show_tqdm_bar=args.show_tqdm_bar,
                use_ip_weighting=args.iqr_ip_weighting,
                fast_mode=True,
                coe_iqr=args.iqr_coef,
            )

            out = out.tolist()[0]

            text = []
            for ind_word, word in enumerate(out):
                token = tokenizer.convert_ids_to_tokens(word)
                text.append(token)

            delimiter = '/'
            text = delimiter.join(text).strip()

            with file.open(encoding='utf-8', mode='w') as f:
                f.write(text + '\n')
            logger.info(f'WRITE SAMPLE TO {file}')

            text = tokenizer.convert_tokens_to_string(text.split('/'))
            logger.info(
                f'\n====================SAMPLE{j}====================\n{text}'
                f'\n========================================'
            )


if __name__ == '__main__':
    with torch.no_grad():
        main()
