import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from src.utils import get_thread_logger
import numpy as np
from pathlib import Path
from src.sampling import report_mean_value_of_metric
from tqdm import tqdm
from src.reporter import Reporter, palette, marker


"""
Plot the variation of self-BLEU 4/5.
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
output_fig_4 = Path(output_dir) / f'fig_bleu_4_gpt2_xl.svg'
output_fig_5 = Path(output_dir) / f'fig_bleu_5_gpt2_xl.svg'

override = False


def main():
    logger = get_thread_logger()

    weights = {
        '4gram': (1 / 4., 1 / 4., 1 / 4., 1 / 4.),
        '5gram': (1 / 5., 1 / 5., 1 / 5., 1 / 5., 1 / 5.)}

    k_choices = ['k200', 'k640', 'k2000', 'k6400']
    p_choices = ['p50', 'p60', 'p70', 'p80', 'p90', 'p95', 'p99', 'p999']

    bleu_type = ['4gram', '5gram']

    columns1 = ['Self-BLEU 4', '$p$', 'Method']
    columns2 = ['Self-BLEU 5', '$p$', 'Method']
    x_labels = ['BLEU-4', 'BLEU-5']

    reporter_list_iqr = []
    for k in k_choices:
        for p in p_choices:
            reporter = Reporter(k=k, p=p, n='n100', t='', is_xl_model=True, iqr=True, metric='bleu', override=override)
            reporter.calculate_bleu(weights, logger)
            reporter_list_iqr.append(reporter)

    reporter_list_k = []
    for k in k_choices:
        reporter = Reporter(k=k, p='', n='', t='', is_xl_model=True, iqr=False, metric='bleu', override=override)
        reporter.calculate_bleu(weights, logger)
        reporter_list_k.append(reporter)

    reporter_list_p = []
    for p in p_choices:
        reporter = Reporter(k='', p=p, n='', t='', is_xl_model=True, iqr=False, metric='bleu', override=override)
        reporter.calculate_bleu(weights, logger)
        reporter_list_p.append(reporter)

    t_choices = [
        't50', 't60', 't70', 't80', 't90', 't100', 't110', 't120',
    ]
    reporter_list_t = []
    for t in t_choices:
        reporter = Reporter(k='', p='', n='', t=t, is_xl_model=True, iqr=False, metric='bleu', override=override)
        reporter.calculate_bleu(weights, logger)
        reporter_list_t.append(reporter)

    data_iqr_1 = pd.DataFrame(columns=columns1)
    data_iqr_2 = pd.DataFrame(columns=columns2)

    data_nucleus_1 = pd.DataFrame(columns=columns1)
    data_nucleus_2 = pd.DataFrame(columns=columns2)

    data_top_k_1 = pd.DataFrame(columns=columns1)
    data_top_k_2 = pd.DataFrame(columns=columns2)

    data_t_1 = pd.DataFrame(columns=columns1)
    data_t_2 = pd.DataFrame(columns=columns2)

    data_iqr = [data_iqr_1, data_iqr_2]
    data_nucleus = [data_nucleus_1, data_nucleus_2]
    data_top_k = [data_top_k_1, data_top_k_2]
    data_t = [data_t_1, data_t_2]

    columns = [columns1, columns2]
    for i in tqdm(range(len(bleu_type))):
        key = bleu_type[i]
        for reporter in tqdm(reporter_list_iqr):
            data_iqr[i] = data_iqr[i].append(
                pd.DataFrame(
                    [
                        [
                         np.mean(reporter.bleu[key]),
                         reporter.method_name,
                         f'IQR-IP, {reporter.k}'],
                    ],
                    columns=columns[i])
            )

        for reporter in tqdm(reporter_list_p):
            data_nucleus[i] = data_nucleus[i].append(
                pd.DataFrame(
                    [
                        [
                         np.mean(reporter.bleu[key]),
                         reporter.method_name,
                         'Nucleus'],
                    ],
                    columns=columns[i])
            )

        for reporter in tqdm(reporter_list_k):
            data_top_k[i] = data_top_k[i].append(
                pd.DataFrame(
                    [
                        [
                            np.mean(reporter.bleu[key]),
                            'Top-k',
                            'Top-k',
                        ],
                    ],
                    columns=columns[i])
            )
            report_mean_value_of_metric(
                reporter.bleu[key],
                f'{reporter.method_name}, {x_labels[i]}'
            )

        for reporter in tqdm(reporter_list_t):
            data_t[i] = data_t[i].append(
                pd.DataFrame(
                    [
                        [
                            np.mean(reporter.bleu[key]),
                            'Temp.',
                            f'Temperature'
                        ],
                    ],
                    columns=columns[i])
            )

    fig, ax = plt.subplots()
    sns.lineplot(ax=ax,
                 data=data_iqr[0], x=columns[0][1], y=columns[0][0], hue=columns[0][2], style=columns[0][2],
                 markers=True,
                 palette=palette.iqr, linewidth=1.5)

    sns.lineplot(ax=ax,
                 data=data_nucleus[0], x=columns[0][1], y=columns[0][0], hue=columns[0][2], style=columns[0][2],
                 markers=marker.nucleus,
                 palette=palette.nucleus, linewidth=1.5, estimator=None)

    sns.lineplot(ax=ax,
                 data=data_top_k[0], x=columns[0][1], y=columns[0][0], hue=columns[0][2], style=columns[0][2],
                 markers=marker.top_k,
                 palette=palette.top_k, linewidth=1.5, estimator=None)

    sns.lineplot(ax=ax,
                 data=data_t[0], x=columns[0][1], y=columns[0][0], hue=columns[0][2], style=columns[0][2],
                 markers=marker.t,
                 palette=palette.t, linewidth=1.5, estimator=None)

    start = 0.84
    for reporter in tqdm(reporter_list_k):
        plt.annotate(reporter.k, xy=(start + 6.4, np.mean(reporter.bleu[bleu_type[0]])),
                     xycoords='data', size=6)

    start = 0.94
    for reporter in tqdm(reporter_list_t):
        plt.annotate(reporter.t + '0', xy=(start + 7.4, np.mean(reporter.bleu[bleu_type[0]])),
                     xycoords='data', size=6)

    plt.axhline(y=0.31, ls="--", color='black', linewidth=1)
    plt.annotate('Human, 0.31', xy=(4.6, 0.28),
                 xycoords='data', size=6)
    plt.tight_layout()

    plt.savefig(output_fig_4)
    logger.info(f'save to {output_fig_4}')

    fig, ax = plt.subplots()
    sns.lineplot(ax=ax,
                 data=data_iqr[1], x=columns[1][1], y=columns[1][0], hue=columns[1][2], style=columns[1][2],
                 markers=True,
                 palette=palette.iqr, linewidth=1.5)

    sns.lineplot(ax=ax,
                 data=data_nucleus[1], x=columns[1][1], y=columns[1][0], hue=columns[1][2], style=columns[1][2],
                 markers=marker.nucleus,
                 palette=palette.nucleus, linewidth=1.5, estimator=None)

    sns.lineplot(ax=ax,
                 data=data_top_k[1], x=columns[1][1], y=columns[1][0], hue=columns[1][2], style=columns[1][2],
                 markers=marker.top_k,
                 palette=palette.top_k, linewidth=1.5, estimator=None)

    sns.lineplot(ax=ax,
                 data=data_t[1], x=columns[1][1], y=columns[1][0], hue=columns[1][2], style=columns[1][2],
                 markers=marker.t,
                 palette=palette.t, linewidth=1.5, estimator=None)

    start = 0.84
    for reporter in tqdm(reporter_list_k):

        plt.annotate(reporter.k, xy=(start + 6.4, np.mean(reporter.bleu[bleu_type[1]])),
                     xycoords='data', size=6)

    start = 0.94
    for reporter in tqdm(reporter_list_t):
        plt.annotate(reporter.t + '0', xy=(start + 7.4, np.mean(reporter.bleu[bleu_type[1]])),
                     xycoords='data', size=6)

    plt.axhline(y=0.17, ls="--", color='black', linewidth=1)
    plt.annotate('Human, 0.17', xy=(5.3, 0.14),
                 xycoords='data', size=6)

    plt.legend(loc='upper center', ncol=2)
    plt.tight_layout()

    plt.savefig(output_fig_5)
    logger.info(f'save to {output_fig_5}')


if __name__ == '__main__':
    main()
