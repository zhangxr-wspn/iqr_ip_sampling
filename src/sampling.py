import torch
from torch.distributions.gumbel import Gumbel
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import statistics


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, n_fraction=0.0,
                          filter_value=-float('Inf'), gumbel_scale=0.0, temperature=1.0, device='cpu'):
    """
    This is based on the original top-p/top-k implementation. We add top1ctrl filtering from Eq.3 in our paper.
    n_fraction: n in top1ctrl filtering

    Original implementation:

    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check

    assert temperature > 0

    if gumbel_scale != 0:
        assert gumbel_scale > 0
        sampler = Gumbel(loc=logits,
                         scale=torch.tensor([gumbel_scale], dtype=torch.float64, device=device, requires_grad=False))
        logits = sampler.sample()

    logits = logits / temperature

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    elif top_p > 0.0 and n_fraction == 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    elif n_fraction > 0.0 and top_p == 0.0:
        probability = F.softmax(logits, dim=-1)
        max_p = probability[probability.nonzero()].max()
        indices_to_remove = probability < max_p / n_fraction
        logits[indices_to_remove] = filter_value

    else:
        raise ValueError('Contradicting Params. top_k={}, top_p={}, n_fraction={}.'
                         .format(top_k, top_p, n_fraction))

    return logits


def ip_rescale(p_list):
    """
    permutation on the distribution with inverse probability (Eq.5)
    :param p_list:
    :return:
    """
    if len(p_list) == 0 or len(p_list) == 1:
        return p_list
    """Eq. 5"""
    p_sum = sum(p_list)

    p_inverse = [1 / float(p) for p in p_list]
    p_inverse_sum = sum(p_inverse)
    p_inverse = [p / p_inverse_sum for p in p_inverse]
    assert abs(sum(p_inverse) - 1) < 1e-8

    new_p_list = [p * p_sum for p in p_inverse]

    assert abs(sum(new_p_list) - p_sum) < 1e-8

    return new_p_list


def sample_sequence(
        model, context, length, n_ctx,
        temperature=1.0, top_k=30, top_p=0.0, n_fraction=0.0, gumbel_scale=0.0, device='cpu',
        sampling_strategy='rand',
        seed=None,
        show_tqdm_bar=True,
        use_ip_weighting=False, protected_special_tokens=None,
        iqr_type='iqr',
        fast_mode=False,
        coe_iqr=1.5
):
    """
    Sampling function
    """
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
        generated = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

        if show_tqdm_bar:
            iterator = trange(length - len(context))
        else:
            iterator = range(length - len(context))

        past = None

        for _ in iterator:
            if fast_mode:
                if past is None:
                    inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0),
                              'past': past}
                else:
                    inputs = {'input_ids': generated[0][-1].unsqueeze(0).unsqueeze(0),
                              'past': past}
            else:
                inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0),}

            outputs = model(**inputs)
            next_token_logits = outputs[0][0, -1, :]

            if fast_mode:
                past = outputs[1]
                if past[0].size()[-2] >= n_ctx:
                    past = None

            if not use_ip_weighting:
                if top_p != 0.0 and n_fraction == 0.0 and top_k == 0.0:
                    filtered_logits = top_k_top_p_filtering(next_token_logits,
                                                            top_k=0, top_p=top_p, n_fraction=0,
                                                            gumbel_scale=gumbel_scale,
                                                            temperature=temperature,
                                                            device=device
                                                            )
                    probability = F.softmax(filtered_logits, dim=-1)

                elif top_p == 0.0 and n_fraction != 0.0 and top_k == 0.0:
                    filtered_logits = top_k_top_p_filtering(next_token_logits,
                                                            top_k=0, top_p=0, n_fraction=n_fraction,
                                                            gumbel_scale=gumbel_scale,
                                                            temperature=temperature,
                                                            device=device
                                                            )
                    probability = F.softmax(filtered_logits, dim=-1)

                elif top_p == 0.0 and n_fraction == 0.0 and top_k != 0.0:
                    filtered_logits = top_k_top_p_filtering(next_token_logits,
                                                            top_k=top_k, top_p=0, n_fraction=0,
                                                            gumbel_scale=gumbel_scale,
                                                            temperature=temperature,
                                                            device=device
                                                            )
                    probability = F.softmax(filtered_logits, dim=-1)

                elif top_p != 0.0 and n_fraction != 0.0 and top_k == 0.0:
                    # Check top_p length
                    filtered_logits_top_p = top_k_top_p_filtering(next_token_logits,
                                                                  top_k=0, top_p=top_p, n_fraction=0,
                                                                  gumbel_scale=gumbel_scale,
                                                                  temperature=temperature,
                                                                  device=device
                                                                  )
                    probability_top_p = F.softmax(filtered_logits_top_p, dim=-1)
                    candidate_length_top_p = len(probability_top_p.nonzero().tolist())

                    # Check n_fraction length
                    filtered_logits_n_fraction = top_k_top_p_filtering(next_token_logits,
                                                                       top_k=0, top_p=0, n_fraction=n_fraction,
                                                                       gumbel_scale=gumbel_scale,
                                                                       temperature=temperature,
                                                                       device=device
                                                                       )
                    probability_n_fraction = F.softmax(filtered_logits_n_fraction, dim=-1)
                    candidate_length_n_fraction = len(probability_n_fraction.nonzero().tolist())

                    if candidate_length_top_p <= candidate_length_n_fraction:
                        filtered_logits = filtered_logits_top_p
                    else:
                        filtered_logits = filtered_logits_n_fraction

                    probability = F.softmax(filtered_logits, dim=-1)

                elif top_p != 0.0 and n_fraction == 0.0 and top_k != 0.0:
                    # Check top_p length
                    filtered_logits_top_p = top_k_top_p_filtering(next_token_logits,
                                                                  top_k=0, top_p=top_p, n_fraction=0,
                                                                  gumbel_scale=gumbel_scale,
                                                                  temperature=temperature,
                                                                  device=device
                                                                  )
                    probability_top_p = F.softmax(filtered_logits_top_p, dim=-1)
                    candidate_length_top_p = len(probability_top_p.nonzero().tolist())

                    # Check top_k length
                    filtered_logits_top_k = top_k_top_p_filtering(next_token_logits,
                                                                  top_k=top_k, top_p=0, n_fraction=0,
                                                                  gumbel_scale=gumbel_scale,
                                                                  temperature=temperature,
                                                                  device=device
                                                                  )
                    probability_top_k = F.softmax(filtered_logits_top_k, dim=-1)
                    candidate_length_top_k = len(probability_top_k.nonzero().tolist())

                    if candidate_length_top_p <= candidate_length_top_k:
                        filtered_logits = filtered_logits_top_p
                    else:
                        filtered_logits = filtered_logits_top_k

                    probability = F.softmax(filtered_logits, dim=-1)

                elif top_p != 0.0 and n_fraction != 0.0 and top_k != 0.0:
                    probability, iqr_info = iqr_transform(
                        next_token_logits,
                        device=device,
                        iqr_type=iqr_type,
                        p=top_p,
                        k=top_k,
                        n=int(n_fraction),
                        do_inverse_ip=False,
                        coe_iqr=coe_iqr
                    )
                else:
                    raise ValueError(f'WRONG SAMPLING PARAMETERS: p={top_p}，k={top_k}，n={n_fraction}')

            else:
                probability, iqr_info = iqr_transform(
                    next_token_logits,
                    device=device,
                    iqr_type=iqr_type,
                    p=top_p,
                    k=top_k,
                    n=int(n_fraction),
                    protected_special_tokens=protected_special_tokens,
                    coe_iqr=coe_iqr
                )

            if sampling_strategy == 'min':
                next_token = (probability == probability[probability.nonzero()].min()).nonzero()[0]
            elif sampling_strategy == 'rand':
                next_token = torch.multinomial(probability, num_samples=1, replacement=True)
            else:
                raise AttributeError(f'WRONG SAMPLING STRATEGY: {sampling_strategy}')

            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

    return generated, _, _


def split_iqr_from_sorted_probability(sorted_indices, sorted_probability, iqr_type, coe_iqr=1.5):
    """
    split 5 set to identify the head (Eq.2)
    :param sorted_indices:
    :param sorted_probability:
    :param iqr_type:
    :return:
    """
    if iqr_type == 'iqr':
        """Eq.2"""
        q75, q25 = np.percentile(sorted_probability.tolist(), [75, 25])
        iqr = q75 - q25

        thresh = [q75 + coe_iqr * iqr,
                  q75,
                  q25,
                  q25 - coe_iqr * iqr]
        metric1 = q75
        metric2 = q25
    elif iqr_type == 'm_std':
        """EXPERIMENTAL, NOT MENTIONED IN OUR PAPER"""
        m = np.mean(sorted_probability.tolist())
        std = np.std(sorted_probability.tolist())

        thresh = [m + 3 * std,
                  m + 1 * std,
                  m - 1 * std,
                  m - 3 * std]
        metric1 = m
        metric2 = std
    else:
        raise ValueError('WRONG IQR SPLIT STRATEGY')

    set_length = [len((sorted_probability >= thresh[0]).nonzero()),
                  len(((sorted_probability >= thresh[1]) & (sorted_probability < thresh[0])).nonzero()),
                  len(((sorted_probability >= thresh[2]) & (sorted_probability < thresh[1])).nonzero()),
                  len(((sorted_probability >= thresh[3]) & (sorted_probability < thresh[2])).nonzero()),
                  len((sorted_probability < thresh[3]).nonzero())
                  ]
    assert sum(set_length) == sorted_probability.size()[0], 'SANITY CHECK FAILS'

    set_length = list(np.cumsum(set_length))

    result = []

    result.append((sorted_indices[0: set_length[0]].tolist(),
                   sorted_probability[0: set_length[0]].tolist()))
    result.append((sorted_indices[set_length[0]: set_length[1]].tolist(),
                   sorted_probability[set_length[0]: set_length[1]].tolist()))
    result.append((sorted_indices[set_length[1]: set_length[2]].tolist(),
                   sorted_probability[set_length[1]: set_length[2]].tolist()))
    result.append((sorted_indices[set_length[2]: set_length[3]].tolist(),
                   sorted_probability[set_length[2]: set_length[3]].tolist()))
    result.append((sorted_indices[set_length[3]: set_length[4]].tolist(),
                   sorted_probability[set_length[3]: set_length[4]].tolist()))

    return result, thresh, metric1, metric2, set_length


def split_iqr_set(next_token_logits, p=0.9, k=200, n=100, device='cpu',
                  iqr_type='iqr', coe_iqr=1.5):
    """
    determine filtered vocabulary size according to Eq.1 to 4.
    :param n: top1ctrl, n_fraction
    :param p: top p
    :param k: top k
    :return:
    """
    info = dict()

    # Check top_p length
    assert not (k == 0 and p == 0), f"WRONG VALUE: p={p},k={k}"
    if p > 0:
        filtered_logits_top_p = top_k_top_p_filtering(next_token_logits,
                                                      top_k=0, top_p=p, n_fraction=0,
                                                      device=device
                                                      )
        probability_top_p = F.softmax(filtered_logits_top_p, dim=-1)
        candidate_length_top_p = len(probability_top_p.nonzero().tolist())
    else:
        candidate_length_top_p = 1e6

    # Check top_k length
    if k > 0:
        filtered_logits_top_k = top_k_top_p_filtering(next_token_logits,
                                                      top_k=k, top_p=0, n_fraction=0,
                                                      device=device
                                                      )
        probability_top_k = F.softmax(filtered_logits_top_k, dim=-1)
        candidate_length_top_k = len(probability_top_k.nonzero().tolist())
    else:
        candidate_length_top_k = 1e6

    """Eq.1"""
    if candidate_length_top_p <= candidate_length_top_k:
        k0_candidate_length = candidate_length_top_p
    else:
        k0_candidate_length = candidate_length_top_k

    info['k0_top_k'] = k0_candidate_length

    probability = F.softmax(next_token_logits, dim=-1)

    sorted_probability, sorted_indices = torch.sort(probability, descending=True)
    k0_probability = sorted_probability[:k0_candidate_length].contiguous()
    k0_indices = sorted_indices[:k0_candidate_length].contiguous()

    """Eq.2"""
    k0_result, k0_thresh, k0_metric1, k0_metric2, k0_set_length = \
        split_iqr_from_sorted_probability(k0_indices, k0_probability, iqr_type, coe_iqr=coe_iqr)

    k1_result = None

    """Eq.3"""
    top1ctrl_thresh = probability.max().tolist() / int(n)

    if k0_result[0][1] and top1ctrl_thresh >= k0_thresh[0] > 0:
        top1ctrl_iqr_set = 'Very High'
    elif k0_result[1][1] and top1ctrl_thresh >= k0_thresh[1] > 0:
        top1ctrl_iqr_set = 'High'
    elif k0_result[2][1] and top1ctrl_thresh >= k0_thresh[2] > 0:
        top1ctrl_iqr_set = 'Medium'
    elif k0_result[3][1] and top1ctrl_thresh >= k0_thresh[3] > 0:
        top1ctrl_iqr_set = 'Low'
    elif k0_result[4][1] and top1ctrl_thresh >= k0_result[4][1][-1] > 0:
        top1ctrl_iqr_set = 'Very Low'
    else:
        top1ctrl_iqr_set = 'Outside'

    if top1ctrl_iqr_set in ['Very High', 'High']:
        k1_candidate_length = len(k0_result[0][1]) + len(k0_result[1][1])

    else:
        k1_candidate_length = len((probability >= top1ctrl_thresh).nonzero())

    if k1_candidate_length > k0_candidate_length:
        info['k1_top_k'] = k0_candidate_length

    else:
        info['k1_top_k'] = k1_candidate_length

    return k0_result, k1_result, info


def report_mean_value_of_metric(metrics, key, logger=None):
    if isinstance(metrics[0], str):
        samples = set(metrics)
        dist = [metrics.count(sample) / len(metrics) for sample in samples]

        if logger is not None:
            logger.info(f'{key} SAMPLIES={samples}, DIST={dist}, TOTAL NUM:{len(metrics)}')
        else:
            print(f'>>> {key} SAMPLIES={samples}, DIST={dist}, TOTAL NUM:{len(metrics)}')
    else:
        m = statistics.mean(metrics)
        if logger is not None:
            logger.info(f'{key} MEAN={m:.2f}, TOTAL NUM:{len(metrics)}')
        else:
            print(f'>>> {key} MEAN={m:.2f}, TOTAL NUM:{len(metrics)}')


def iqr_transform(next_token_logits, device,
                  iqr_type='iqr',
                  p=0.9, k=200, n=100, do_inverse_ip=True,
                  protected_special_tokens=None,
                  coe_iqr=1.5):
    """
    iqr ip transformation
    :param protected_special_tokens: list of symbols that are skipped by permutation
    :param do_inverse_ip: do permutation
    :param iqr_type:
        ='iqr': use iqr for division
        ='m_std':use mean/std for division (EXPERIMENTAL)
    :param next_token_logits: predicted logits
    :return: rescaled distribution to sample on
    """
    iqr_result = split_iqr_set(next_token_logits,
                               p=p,
                               k=k,
                               n=n,
                               device=device,
                               iqr_type=iqr_type,
                               coe_iqr=coe_iqr)

    k0_results, k1_results, info = iqr_result[0:3]

    probability = F.softmax(next_token_logits, dim=-1)

    sorted_probability, sorted_indices = torch.sort(probability, descending=True)

    """Eq. 4"""
    if info['k1_top_k'] < info['k0_top_k']:
        remaining_length = info['k1_top_k']
        new_k0_results = []
        for indice, p in k0_results:
            if len(indice) < remaining_length:
                new_k0_results.append((indice, p))
                remaining_length = remaining_length - len(indice)
            else:
                new_k0_results.append((indice[:remaining_length], p[:remaining_length]))
                remaining_length = 0
        k0_results = new_k0_results
        keep_length = info['k1_top_k']
    else:
        keep_length = info['k0_top_k']

    probability[sorted_indices[keep_length:]] = 0
    probability = probability / probability.sum()

    if do_inverse_ip:
        """head (very high, vh)"""
        vh_indice, _ = k0_results[0]
        if protected_special_tokens:
            ignored_indices_in_vh_indice = [vh_indice.index(token) for token in protected_special_tokens if token in vh_indice]
            ignored_vh_indice = [vh_indice[i] for i in ignored_indices_in_vh_indice]
            ignored_vh_p = probability[ignored_vh_indice].tolist()

            manipulate_indices_in_vh_indice = [i for i in range(len(vh_indice)) if i not in ignored_indices_in_vh_indice]
            manipulate_vh_indice = [vh_indice[i] for i in manipulate_indices_in_vh_indice]
            manipulate_vh_p = ip_rescale(probability[manipulate_vh_indice].tolist())

            new_vh_p = []
            for i in range(len(vh_indice)):
                if i in ignored_indices_in_vh_indice:
                    p = ignored_vh_p.pop(0)
                else:
                    p = manipulate_vh_p.pop(0)
                new_vh_p.append(p)

        else:
            new_vh_p = ip_rescale(probability[vh_indice].tolist())

        """rescale head"""
        for ind_p, new_p in zip(vh_indice, new_vh_p):
            probability[ind_p] = new_p

    return probability, info

