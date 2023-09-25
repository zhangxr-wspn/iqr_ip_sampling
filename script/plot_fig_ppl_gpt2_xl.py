import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from src import utils
from tqdm import tqdm
from pytorch_transformers import GPT2Tokenizer
from generate_sample import get_model
from src.reporter import Reporter, palette, marker


"""
Plot the variation of perplexity.
"""


sns.set(
    rc={
        'figure.figsize': (5, 3),
        'savefig.transparent': True
    },
    font_scale=2
)

sns.set_style("whitegrid")
sns.set_context("paper")
color_code = '#4169E1'


output_dir = 'figures/iclr/'
Path(output_dir).mkdir(exist_ok=True, parents=True)
output_fig = Path(output_dir) / f'fig_ppl_gpt2_xl.svg'

override = False

load_model = False


def main():
    parser = argparse.ArgumentParser()
    utils.create_params(parser)
    args = parser.parse_args()
    args.is_en_gpt2_xl = True
    logger = utils.get_thread_logger()
    device = utils.get_available_device(memory_thresh=args.memory_thresh, logger=logger)

    """
    REPLACE THE FOLLOWING 8 LINES OF CODES IF YOU LOAD MODEL AUTOMATICALLY WITH:

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

    if load_model:
        model = get_model(args, logger=logger)
        model.to(device)
        model.eval()
    else:
        model = None

    k_choices = ['k200', 'k640', 'k2000', 'k6400']
    p_choices = ['p50', 'p60', 'p70', 'p80', 'p90', 'p95', 'p99', 'p999']
    reporter_list_iqr = []
    for k in k_choices:
        for p in p_choices:
            reporter = Reporter(k=k, p=p, n='n100', t='', is_xl_model=True, iqr=True, metric='ppl', override=override)
            reporter.calculate_ppl(tokenizer, model, device, logger)
            reporter_list_iqr.append(reporter)

    reporter_list_k = []
    for k in k_choices:
        reporter = Reporter(k=k, p='', n='', t='', is_xl_model=True, iqr=False, metric='ppl', override=override)
        reporter.calculate_ppl(tokenizer, model, device, logger)
        reporter_list_k.append(reporter)

    reporter_list_p = []
    for p in p_choices:
        reporter = Reporter(k='', p=p, n='', t='', is_xl_model=True, iqr=False, metric='ppl', override=override)
        reporter.calculate_ppl(tokenizer, model, device, logger)
        reporter_list_p.append(reporter)

    t_choices = [
        't50', 't60', 't70', 't80', 't90', 't100', 't110', 't120',
    ]
    reporter_list_t = []
    for t in t_choices:
        reporter = Reporter(k='', p='', n='', t=t, is_xl_model=True, iqr=False, metric='ppl', override=override)
        reporter.calculate_ppl(tokenizer, model, device, logger)
        reporter_list_t.append(reporter)

    columns = ['PPL', '$p$', 'Method']
    fig, ax = plt.subplots()
    data_iqr = pd.DataFrame(columns=columns)
    for reporter in tqdm(reporter_list_iqr):
        data_iqr = data_iqr.append(
            pd.DataFrame(
                [
                    [
                        np.mean(reporter.ppl_list),
                        reporter.method_name,
                        f'IQR-IP, {reporter.k}'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(ax=ax,
                 data=data_iqr, x=columns[1], y=columns[0], hue=columns[2], style=columns[2], markers=True,
                 palette=palette.iqr, linewidth=1.5)
    data_nucleus = pd.DataFrame(columns=columns)
    for reporter in tqdm(reporter_list_p):
        data_nucleus = data_nucleus.append(
            pd.DataFrame(
                [
                    [
                        np.mean(reporter.ppl_list),
                        reporter.method_name,
                        f'Nucleus'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(ax=ax,
                 data=data_nucleus, x=columns[1], y=columns[0], hue=columns[2], style=columns[2],
                 markers=marker.nucleus,
                 palette=palette.nucleus, linewidth=1.5)

    data_top_k = pd.DataFrame(columns=columns)
    for reporter in tqdm(reporter_list_k):
        data_top_k = data_top_k.append(
            pd.DataFrame(
                [
                    [
                        np.mean(reporter.ppl_list),
                        'Top-k',
                        f'Top-k'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(ax=ax,
                 data=data_top_k, x=columns[1], y=columns[0], hue=columns[2], style=columns[2],
                 markers=marker.top_k,
                 palette=palette.top_k, linewidth=1.5, estimator=None)
    data_t = pd.DataFrame(columns=columns)
    for reporter in tqdm(reporter_list_t):
        data_t = data_t.append(
            pd.DataFrame(
                [
                    [
                        np.mean(reporter.ppl_list),
                        'Temp.',
                        f'Temperature'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(ax=ax,
                 data=data_t, x=columns[1], y=columns[0], hue=columns[2], style=columns[2],
                 markers=marker.t,
                 palette=palette.t, linewidth=1.5, estimator=None)

    start = 0.84
    for reporter in tqdm(reporter_list_k):
        plt.annotate(reporter.k, xy=(start + 6.4, np.mean(reporter.ppl_list)),
                     xycoords='data', size=6)

    start = 0.94
    for reporter in tqdm(reporter_list_t):
        plt.annotate(reporter.t + '0', xy=(start + 7.4, np.mean(reporter.ppl_list)),
                     xycoords='data', size=6)

    plt.axhline(y=18.34, ls="--", color='black', linewidth=1)
    plt.annotate('Human, 18.34', xy=(4.1, 13),
                 xycoords='data', size=6)
    ax.set_yscale('log')
    plt.tight_layout()

    plt.savefig(output_fig)
    logger.info(f'save to {output_fig}')


if __name__ == '__main__':
    main()
