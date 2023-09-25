import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.utils import get_thread_logger
from src.reporter import Reporter, palette, marker, get_human_hrep


"""
Plot the variation of 3-gram entropy.
"""

sns.set(
    rc={
        'figure.figsize': (5, 3),
        'savefig.transparent': True
    },
    font_scale=2
)

sns.set_style("whitegrid", {
    "font.sans-serif": ['Arial'],
                            })
sns.set_context("paper")
color_code = '#4169E1'

output_dir = 'figures/iclr/'
Path(output_dir).mkdir(exist_ok=True, parents=True)
output_fig = Path(output_dir) / f'fig_hrep_3_gpt2_xl.svg'

override = False

is_xl_model = True

metric = 'hrep_3'


def main():

    logger = get_thread_logger()

    valid_ent = get_human_hrep(3, logger, override=override)
    human_ent = np.mean(valid_ent)

    k_choices = ['k200', 'k640', 'k2000', 'k6400']
    p_choices = ['p50', 'p60', 'p70', 'p80', 'p90', 'p95', 'p99', 'p999']
    reporter_list_iqr = []
    for k in k_choices:
        for p in p_choices:
            reporter = Reporter(k=k, p=p, n='n100', t='', is_xl_model=is_xl_model, iqr=True, metric=metric, override=override)
            reporter.calculate_hrep(logger)
            reporter_list_iqr.append(reporter)

    reporter_list_k = []
    for k in k_choices:
        reporter = Reporter(k=k, p='', n='', t='', is_xl_model=is_xl_model, iqr=False, metric=metric, override=override)
        reporter.calculate_hrep(logger)
        reporter_list_k.append(reporter)

    reporter_list_p = []
    for p in p_choices:
        reporter = Reporter(k='', p=p, n='', t='', is_xl_model=is_xl_model, iqr=False, metric=metric, override=override)
        reporter.calculate_hrep(logger)
        reporter_list_p.append(reporter)

    t_choices = [
        't50', 't60', 't70', 't80', 't90', 't100', 't110', 't120',
    ]
    reporter_list_t = []
    for t in t_choices:
        reporter = Reporter(k='', p='', n='', t=t, is_xl_model=is_xl_model, iqr=False, metric=metric, override=override)
        reporter.calculate_hrep(logger)
        reporter_list_t.append(reporter)

    columns = ['3-gram Entropy', '$p$', 'Method']
    fig, ax = plt.subplots()
    data_iqr = pd.DataFrame(columns=columns)

    for reporter in tqdm(reporter_list_iqr):
        data_iqr = data_iqr.append(
            pd.DataFrame(
                [
                    [
                        np.mean(reporter.rr_list),
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
                        np.mean(reporter.rr_list),
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
                        np.mean(reporter.rr_list),
                        'Top-k',
                        f'Top-k'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(ax=ax,
                 data=data_top_k, x=columns[1], y=columns[0], hue=columns[2], style=columns[2], markers=marker.top_k,
                 palette=palette.top_k, linewidth=1.5, estimator=None)
    data_t = pd.DataFrame(columns=columns)
    for reporter in tqdm(reporter_list_t):
        data_t = data_t.append(
            pd.DataFrame(
                [
                    [
                        np.mean(reporter.rr_list),
                        'Temp.',
                        f'Temperature'
                    ],
                ],
                columns=columns)
        )
    sns.lineplot(ax=ax,
                 data=data_t, x=columns[1], y=columns[0], hue=columns[2], style=columns[2], markers=marker.t,
                 palette=palette.t, linewidth=1.5, estimator=None)

    start = 0.84
    i = 0
    y_start = np.mean(reporter_list_k[0].rr_list) - 0.12
    for reporter in tqdm(reporter_list_k):
        plt.annotate(reporter.k, xy=(start + 6.4, y_start + 0.06 * i),
                     xycoords='data', size=6)
        i = i + 1

    start = 0.94
    for reporter in tqdm(reporter_list_t):
        if reporter.t in ['t=1.0', 't=1.2']:
            plt.annotate(reporter.t + '0', xy=(start + 8.2, np.mean(reporter.rr_list)),
                         xycoords='data', size=6)
        elif reporter.t in ['t=0.9']:
            plt.annotate(reporter.t + '0', xy=(start + 7.4, np.mean(reporter.rr_list) - 0.03),
                         xycoords='data', size=6)
        else:
            plt.annotate(reporter.t + '0', xy=(start + 7.4, np.mean(reporter.rr_list)),
                         xycoords='data', size=6)

    plt.axhline(y=human_ent, ls="--", color='black', linewidth=1)
    plt.annotate(f'Human, {human_ent}', xy=(1.8, 5.24),
                 xycoords='data', size=6)

    plt.tight_layout()
    plt.savefig(output_fig)

    logger.info(f'save to {output_fig}')


if __name__ == '__main__':
    main()
