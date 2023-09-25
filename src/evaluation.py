from collections import defaultdict
import numpy as np
from scipy.optimize import curve_fit
import torch
import torch.nn.functional as F


def ngrams(tokens, n):
    ngram = []
    for token in tokens:
        if len(ngram) < n:
            ngram.append(token)
        else:
            yield ngram
            ngram.pop(0)
            ngram.append(token)
    if len(ngram) == n:
        yield ngram


def count_ngrams(tokens, n):
    counts = defaultdict(int)
    for ngram in ngrams(tokens, n):
        counts[' '.join(ngram)] += 1
    return counts


def zipf(x, a, b, k):
    return k / np.power(x + b, a)


def fit_zipf(ref_list):
    popt, pcov = curve_fit(zipf, np.arange((len(ref_list))) + 1, ref_list, maxfev=1000000, p0=[1, 0, 0.2])
    return {'a': popt[0], 'b': popt[1], 'k': popt[2]}


def get_token_dist(ref_list):
    count = defaultdict(int)
    for token in ref_list:
        count[token] += 1
    sorted_dist = sorted(count.values(), reverse=True)
    sorted_dist = np.array(sorted_dist) / sum(sorted_dist)
    return fit_zipf(sorted_dist)


def calculate_ppl(token_list, model, device, logger, n_ctx=128):
    with torch.no_grad():
        stride = int(n_ctx / 2)
        minus_log_p_list = []
        generated = torch.tensor(token_list, dtype=torch.long, device=device).unsqueeze(0)

        for ind_ctx in range(0, len(token_list), stride):
            start = max(ind_ctx + stride - n_ctx, 0)
            end = ind_ctx + stride
            if end > len(token_list):
                break

            inputs = generated[0][start:end].unsqueeze(0)
            outputs = model.forward(input_ids=inputs)
            logits = outputs[0]
            shift_logits = logits[..., -stride:-1, :].contiguous()
            shift_labels = inputs[..., 1 - stride:].contiguous()
            cross_entropy = F.cross_entropy(input=shift_logits.view(-1, shift_logits.size(-1)),
                                            target=shift_labels.view(-1), ignore_index=-1, reduction='mean')
            minus_log_p_list.append(cross_entropy.tolist())
        if minus_log_p_list:
            ppl = np.exp(np.mean(minus_log_p_list))
        else:
            ppl = 0
            logger.info('RETURN 0 PPL')

        return ppl


def entropy_of_distribution(token_list):
    count = defaultdict(int)
    for token in token_list:
        count[token] += 1

    frequency = list(count.values())
    distribution = np.array(frequency) / sum(frequency)
    entropy = sum(-d * np.log(d) for d in distribution)
    return entropy


def get_predicted_distribution(token_list, model, device, n_ctx=128):
    with torch.no_grad():
        assert len(token_list) == n_ctx
        inputs = torch.tensor(token_list, dtype=torch.long, device=device).unsqueeze(0)
        outputs = model.forward(input_ids=inputs)
        logits = outputs[0].squeeze()

        predicted_distribution = F.softmax(logits, dim=-1)
        sorted_distribution, _ = torch.sort(predicted_distribution, descending=True, dim=-1)
        sorted_distribution = sorted_distribution.tolist()
        return sorted_distribution


def get_iqr_from_list(token_list):
    if not token_list:
        return 0
    token_list = sorted(token_list)
    q75, q25 = np.percentile(token_list, [75, 25])
    iqr = q75 - q25
    return iqr


def calculate_loss_iqr(token_list, model, device, logger, n_ctx=128):
    with torch.no_grad():
        window = n_ctx
        loss_iqr_list = []
        generated = torch.tensor(token_list, dtype=torch.long, device=device).unsqueeze(0)

        for ind_ctx in range(0, len(token_list), window):
            start = ind_ctx
            end = ind_ctx + window
            stride_length = window
            if end > len(token_list):
                if start == 0:
                    end = len(token_list)
                    stride_length = len(token_list)
                else:
                    break

            inputs = generated[0][start:end].unsqueeze(0)

            outputs = model.forward(input_ids=inputs)
            logits = outputs[0]
            shift_logits = logits[..., -stride_length:-1, :].contiguous()
            shift_labels = inputs[..., 1 - stride_length:].contiguous()
            cross_entropy = F.cross_entropy(input=shift_logits.view(-1, shift_logits.size(-1)),
                                            target=shift_labels.view(-1), ignore_index=-1, reduction='none')

            loss_iqr_list.append(get_iqr_from_list(cross_entropy.tolist()))
        if loss_iqr_list:
            loss_iqr = np.mean(loss_iqr_list)
        else:
            loss_iqr = 0

        if loss_iqr == 0:
            logger.info('RETURN 0 LOSS IQR')

        return loss_iqr

