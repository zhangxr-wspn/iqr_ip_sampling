import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from src import utils
from generate_sample import get_model
from src.reporter import Reporter, palette, marker, config
from matplotlib.patches import Ellipse
from pytorch_transformers import GPT2Tokenizer


"""
Plot the trade-off curve for 3-gram entropy against PPL.
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
output_fig = Path(output_dir) / f'fig_q_d_hrep_3_gpt2_xl.svg'

override = False

load_model = False

is_xl_model = True

metric_d = 'hrep_3'
metric_q = 'ppl'


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
    ppl_reporter_list_iqr = []
    for k in k_choices:
        for p in p_choices:
            reporter = Reporter(k=k, p=p, n='n100', t='', is_xl_model=is_xl_model, iqr=True, metric=metric_q, override=override)
            reporter.calculate_ppl(tokenizer, model, device, logger)
            ppl_reporter_list_iqr.append(reporter)

    ppl_reporter_list_k = []
    for k in k_choices:
        reporter = Reporter(k=k, p='', n='', t='', is_xl_model=is_xl_model, iqr=False, metric=metric_q, override=override)
        reporter.calculate_ppl(tokenizer, model, device, logger)
        ppl_reporter_list_k.append(reporter)

    ppl_reporter_list_p = []
    for p in p_choices:
        reporter = Reporter(k='', p=p, n='', t='', is_xl_model=is_xl_model, iqr=False, metric=metric_q, override=override)
        reporter.calculate_ppl(tokenizer, model, device, logger)
        ppl_reporter_list_p.append(reporter)

    t_choices = [
        't50', 't60', 't70', 't80', 't90', 't100', 't110', 't120',
    ]
    ppl_reporter_list_t = []
    for t in t_choices:
        reporter = Reporter(k='', p='', n='', t=t, is_xl_model=is_xl_model, iqr=False, metric=metric_q, override=override)
        reporter.calculate_ppl(tokenizer, model, device, logger)
        ppl_reporter_list_t.append(reporter)

    d_reporter_list_iqr = []
    for k in k_choices:
        for p in p_choices:
            reporter = Reporter(k=k, p=p, n='n100', t='', is_xl_model=is_xl_model, iqr=True, metric=metric_d, override=override)
            reporter.calculate_hrep(logger)
            d_reporter_list_iqr.append(reporter)

    d_reporter_list_k = []
    for k in k_choices:
        reporter = Reporter(k=k, p='', n='', t='', is_xl_model=is_xl_model, iqr=False, metric=metric_d, override=override)
        reporter.calculate_hrep(logger)
        d_reporter_list_k.append(reporter)

    d_reporter_list_p = []
    for p in p_choices:
        reporter = Reporter(k='', p=p, n='', t='', is_xl_model=is_xl_model, iqr=False, metric=metric_d, override=override)
        reporter.calculate_hrep(logger)
        d_reporter_list_p.append(reporter)

    d_reporter_list_t = []
    for t in t_choices:
        reporter = Reporter(k='', p='', n='', t=t, is_xl_model=is_xl_model, iqr=False, metric=metric_d,
                            override=override)
        reporter.calculate_hrep(logger)
        d_reporter_list_t.append(reporter)

    columns = ['Log PPL', 'Entropy', 'Method']
    fig, ax = plt.subplots()
    data_iqr = pd.DataFrame(columns=columns)
    for ppl_reporter, d_reporter in zip(ppl_reporter_list_iqr, d_reporter_list_iqr):

        data_iqr = data_iqr.append(
            pd.DataFrame(
                [
                    [
                        np.log(np.mean(ppl_reporter.ppl_list)),
                        np.mean(d_reporter.rr_list),
                        f'IQR-IP, {ppl_reporter.k}'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(data=data_iqr, x=columns[0], y=columns[1], hue=columns[2], style=columns[2], markers=True,
                 palette=palette.iqr, linewidth=config.linewidth, markersize=config.markersize, alpha=config.alpha)

    data_nucleus = pd.DataFrame(columns=columns)
    for ppl_reporter, d_reporter in zip(ppl_reporter_list_p, d_reporter_list_p):

        data_nucleus = data_nucleus.append(
            pd.DataFrame(
                [
                    [
                        np.log(np.mean(ppl_reporter.ppl_list)),
                        np.mean(d_reporter.rr_list),
                        f'Nucleus'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(data=data_nucleus, x=columns[0], y=columns[1], hue=columns[2], style=columns[2],
                 markers=marker.nucleus,
                 palette=palette.nucleus, linewidth=config.linewidth, markersize=config.markersize, alpha=config.alpha)

    data_top_k = pd.DataFrame(columns=columns)
    for ppl_reporter, d_reporter in zip(ppl_reporter_list_k, d_reporter_list_k):

        data_top_k = data_top_k.append(
            pd.DataFrame(
                [
                    [
                        np.log(np.mean(ppl_reporter.ppl_list)),
                        np.mean(d_reporter.rr_list),
                        f'Top-k'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(data=data_top_k, x=columns[0], y=columns[1], hue=columns[2], style=columns[2], markers=marker.top_k,
                 palette=palette.top_k, linewidth=config.linewidth, markersize=config.markersize, alpha=config.alpha)

    data_t = pd.DataFrame(columns=columns)
    for ppl_reporter, d_reporter in zip(ppl_reporter_list_t, d_reporter_list_t):

        data_t = data_t.append(
            pd.DataFrame(
                [
                    [
                        np.log(np.mean(ppl_reporter.ppl_list)),
                        np.mean(d_reporter.rr_list),
                        f'Temperature'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(data=data_t, x=columns[0], y=columns[1], hue=columns[2], style=columns[2], markers=marker.t,
                 palette=palette.t, linewidth=config.linewidth, markersize=config.markersize, alpha=config.alpha)

    data_human = pd.DataFrame(columns=columns)
    data_human = data_human.append(
        pd.DataFrame(
            [
                [
                    np.log(18.34),
                    5.22,
                    f'Human'
                ],
            ],
            columns=columns)
    )
    sns.lineplot(data=data_human, x=columns[0], y=columns[1], hue=columns[2], style=columns[2], markers=marker.human,
                 palette=palette.human, linewidth=config.linewidth, markersize=config.markersize, alpha=config.alpha)

    plt.xlabel('log PPL')
    plt.ylabel('3-gram Entropy')

    ax.add_artist(Ellipse((3.2, 5.20), 0.2, 0.3, facecolor='none', edgecolor='red'))
    plt.annotate('All methods are on par', xy=(3.35, 5.10),
                 xycoords='data', size=6, color='red', )

    plt.annotate('Gold', xy=(np.log(18.34) - 0.2, 5.10),
                 xycoords='data', size=6, color=sns.color_palette(palette.human, 1)[0])

    handles, labels = ax.get_legend_handles_labels()
    legend1 = plt.legend(handles=handles)
    plt.gca().add_artist(legend1)
    plt.tight_layout()

    plt.savefig(output_fig)
    logger.info(f'save to {output_fig}')


if __name__ == '__main__':
    main()
