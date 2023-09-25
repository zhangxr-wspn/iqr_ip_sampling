import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from src.utils import get_thread_logger
from src.reporter import Reporter, palette, marker


"""
Plot the variation of Zipf coefficient.
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
output_fig = Path(output_dir) / f'fig_zipf_gpt2_xl.svg'

override = False


def main():

    logger = get_thread_logger()

    k_choices = ['k200', 'k640', 'k2000', 'k6400']
    p_choices = ['p50', 'p60', 'p70', 'p80', 'p90', 'p95', 'p99', 'p999']
    reporter_list_iqr = []
    for k in k_choices:
        for p in p_choices:
            reporter = Reporter(k=k, p=p, n='n100', t='', is_xl_model=True, iqr=True, metric='zipf', override=override)
            reporter.calculate_zipf(logger)
            reporter_list_iqr.append(reporter)

    reporter_list_k = []
    for k in k_choices:
        reporter = Reporter(k=k, p='', n='', t='', is_xl_model=True, iqr=False, metric='zipf', override=override)
        reporter.calculate_zipf(logger)
        reporter_list_k.append(reporter)

    reporter_list_p = []
    for p in p_choices:
        reporter = Reporter(k='', p=p, n='', t='', is_xl_model=True, iqr=False, metric='zipf', override=override)
        reporter.calculate_zipf(logger)
        reporter_list_p.append(reporter)

    t_choices = [
        't50', 't60', 't70', 't80', 't90', 't100', 't110', 't120',
    ]
    reporter_list_t = []
    for t in t_choices:
        reporter = Reporter(k='', p='', n='', t=t, is_xl_model=True, iqr=False, metric='zipf', override=override)
        reporter.calculate_zipf(logger)
        reporter_list_t.append(reporter)

    columns = ['Zipf Coefficient', '$p$', 'Method']
    fig, ax = plt.subplots()
    data_iqr = pd.DataFrame(columns=columns)
    for reporter in tqdm(reporter_list_iqr):
        data_iqr = data_iqr.append(
            pd.DataFrame(
                [
                    [
                        reporter.zipf_coef['a'],
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
                        reporter.zipf_coef['a'],
                        reporter.method_name,
                        f'Nucleus'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(ax=ax,
                 data=data_nucleus, x=columns[1], y=columns[0], hue=columns[2], style=columns[2], markers=marker.nucleus,
                 palette=palette.nucleus, linewidth=1.5)

    data_top_k = pd.DataFrame(columns=columns)
    for reporter in tqdm(reporter_list_k):
        data_top_k = data_top_k.append(
            pd.DataFrame(
                [
                    [
                        reporter.zipf_coef['a'],
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
                        reporter.zipf_coef['a'],
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
    i = 0
    y_start = reporter_list_k[0].zipf_coef['a'] - 0.09
    for reporter in tqdm(reporter_list_k):
        plt.annotate(reporter.k, xy=(start + 6.4, y_start + 0.07 * i),
                     xycoords='data', size=6)
        i = i + 1

    start = 0.94
    for reporter in tqdm(reporter_list_t):
        plt.annotate(reporter.t + '0', xy=(start + 7.4, reporter.zipf_coef['a']),
                     xycoords='data', size=6)

    plt.axhline(y=0.93, ls="--", color='black', linewidth=1)
    plt.annotate('Human, 0.93', xy=(3, 0.84),
                 xycoords='data', size=6)
    plt.legend(loc='upper left', ncol=2)
    plt.tight_layout()

    plt.savefig(output_fig)
    logger.info(f'save to {output_fig}')


if __name__ == '__main__':
    main()
