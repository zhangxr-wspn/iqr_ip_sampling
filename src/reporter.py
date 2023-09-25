from pathlib import Path
import json
from src.evaluation import calculate_ppl, get_token_dist
from src.sampling import report_mean_value_of_metric
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from generate_sample import get_model
import argparse
from src import utils
from pytorch_transformers import GPT2Tokenizer
from fast_bleu import SelfBLEU


class palette:
    iqr = 'viridis'
    nucleus = 'copper'
    top_k = 'pink'
    t = 'gray'
    human = 'Wistia'


class marker:
    nucleus = 'v'
    top_k = '^'
    t = 'p'
    human = 'D'


class config:
    linewidth = 0.5
    markersize = 6
    alpha = 0.9


def entropy_of_distribution(token_list):
    count = defaultdict(int)
    for token in token_list:
        count[token] += 1

    frequency = list(count.values())
    distribution = np.array(frequency) / sum(frequency)
    entropy = sum(-d * np.log(d) for d in distribution)
    return entropy


def entropy_of_distribution_ngram(token_list, n):
    ngram_list = []
    for i in range(len(token_list) - n + 1):
        ngram_list.append(''.join(token_list[i: i + n]))

    count = defaultdict(int)
    for token in ngram_list:
        count[token] += 1

    frequency = list(count.values())
    distribution = np.array(frequency) / sum(frequency)
    entropy = sum(-d * np.log(d) for d in distribution)
    return entropy


def dump_json_to_file(save_file: Path, json_data, logger):
    save_file.parent.mkdir(exist_ok=True, parents=True)
    with save_file.open(encoding='utf-8', mode='w') as f:
        json.dump(json_data, f)
        f.close()
    logger.info(f'SAVE RESULT TO {save_file}')


def load_json_from_file(save_file: Path, logger, mute=False):

    with save_file.open(encoding='utf-8', mode='r') as f:
        json_data = json.load(f)
        f.close()
    if not mute:
        logger.info(f'LOAD RESULT FROM {save_file}')
    return json_data


class Reporter(object):
    def __init__(self, k, p, n, iqr, t, is_xl_model, metric, override=False):

        if is_xl_model:
            model_dir = 'en_gpt2_xl_pretrained_models'
        else:
            model_dir = 'en_gpt2_pretrained_models'

        parent_dir = 'generated_samples'

        if is_xl_model:
            self.sub_dir = 'gpt2_xl_en_' + k + p + n + t
        else:
            self.sub_dir = 'gpt2_en_' + k + p + n + t

        if iqr:
            self.sub_dir += '_iqr'

        self.text_dir = Path(parent_dir) / self.sub_dir / model_dir
        self.ppl_list = []

        if metric == 'ppl':
            self.save_file = Path('save') / Path(self.text_dir) / 'ppl.json'
        elif metric == 'bleu':
            self.save_file = Path('save') / Path(self.text_dir) / 'self_bleu.json'
        elif metric == 'zipf':
            self.save_file = Path('save') / Path(self.text_dir) / 'zipf.json'
        elif metric == 'hrep':
            self.save_file = Path('save') / Path(self.text_dir) / 'rep_entropy.json'
        elif metric == 'hrep_3':
            self.save_file = Path('save') / Path(self.text_dir) / 'rep_entropy_3.json'
        else:
            raise ValueError(metric)

        self.metric = metric

        self.save_file.parent.mkdir(exist_ok=True, parents=True)
        self.override = override
        self.k = k.replace('k', 'k=')
        self.p = p.replace('p', 'p=0.')
        self.n = n
        self.iqr = iqr

        if t:
            self.t = 't=' + str(int(t.replace('t', '')) / 100.0)
        else:
            self.t = ''

        if k and not p:
            self.method_name = int(self.k.replace('k=', ''))
        else:
            self.method_name = self.p.replace('p=', '')

    def calculate_ppl(self, tokenizer, model, device, logger):
        if self.save_file.exists() and not self.override:
            logger.info(f'LOAD RESULT FROM {self.save_file}')
            with self.save_file.open(encoding='utf-8', mode='r') as f:
                self.ppl_list = json.load(f)
                f.close()

        else:
            logger.info(f'CALCULATING FROM {self.text_dir}')

            for ind_text in tqdm(range(0, 5000)):
                base_name = 'sample' + str(ind_text + 1) + '.txt'
                compare_text = Path(self.text_dir) / base_name
                token_list = self.read_tokens(compare_text, tokenizer)
                self.ppl_list.append(calculate_ppl(token_list, model, device, logger))

            with self.save_file.open(encoding='utf-8', mode='w') as f:
                json.dump(self.ppl_list, f)
                f.close()
            logger.info(f'SAVE RESULT TO {self.save_file}')

        report_mean_value_of_metric(self.ppl_list, f'{self.sub_dir} PPL')

    def calculate_ppl_and_return(self, tokenizer, model, device, logger):

        logger.info(f'CALCULATING FROM {self.text_dir}')
        results = []

        for ind_text in tqdm(range(0, 5000)):
            base_name = 'sample' + str(ind_text + 1) + '.txt'
            compare_text = Path(self.text_dir) / base_name
            token_list = self.read_tokens(compare_text, tokenizer)
            text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token_list))
            ppl = calculate_ppl(token_list, model, device, logger)
            results.append(
                {
                    'base_name': base_name,
                    'text': text,
                    'tokens': token_list,
                    'ppl': ppl,
                }
            )

        return results

    @staticmethod
    def read_tokens(file: Path, tokenizer):
        with file.open(encoding='utf-8', mode='r') as f:
            content = f.read().strip().split('/')
            f.close()
        content = content[:200]
        tokens = tokenizer.convert_tokens_to_ids(content)
        return tokens

    def calculate_bleu(self, weights, logger):
        if self.save_file.exists() and not self.override:
            logger.info(f'LOAD RESULT FROM {self.save_file}')
            with self.save_file.open(encoding='utf-8', mode='r') as f:
                self.bleu = json.load(f)
                f.close()

        else:
            logger.info(f'CALCULATING FROM {self.text_dir}')

            ref_list = []
            for ind_text in tqdm(range(0, 5000)):
                base_name = 'sample' + str(ind_text + 1) + '.txt'
                compare_text = Path(self.text_dir) / base_name
                ref_list.append(self.read_text(compare_text))

            logger.info(f'CALCULATING BLEU')
            self.bleu = SelfBLEU(ref_list, weights).get_score()

            with self.save_file.open(encoding='utf-8', mode='w') as f:
                json.dump(self.bleu, f)
                f.close()
            logger.info(f'SAVE RESULT TO {self.save_file}')

    @staticmethod
    def read_text(file: Path):
        with file.open(encoding='utf-8', mode='r') as f:
            content = f.read().strip().split('/')
            f.close()
        content = content[:200]
        return content

    def calculate_zipf(self, logger):
        if self.save_file.exists() and not self.override:
            logger.info(f'LOAD RESULT FROM {self.save_file}')
            with self.save_file.open(encoding='utf-8', mode='r') as f:
                self.zipf_coef = json.load(f)
                f.close()

        else:
            logger.info(f'CALCULATING FROM {self.text_dir}')
            ref_list = []
            for ind_text in tqdm(range(0, 5000)):
                base_name = 'sample' + str(ind_text + 1) + '.txt'
                compare_text = Path(self.text_dir) / base_name
                ref_list.extend(self.read_text(compare_text))

            self.zipf_coef = get_token_dist(ref_list)

            with self.save_file.open(encoding='utf-8', mode='w') as f:
                json.dump(self.zipf_coef, f)
                f.close()
            logger.info(f'SAVE RESULT TO {self.save_file}')

        logger.info(f'{self.sub_dir}: {self.zipf_coef}')

    def calculate_hrep(self, logger):
        if self.save_file.exists() and not self.override:
            logger.info(f'LOAD RESULT FROM {self.save_file}')
            with self.save_file.open(encoding='utf-8', mode='r') as f:
                self.rr_list = json.load(f)
                f.close()
        else:
            logger.info(f'CALCULATING FROM {self.text_dir}')
            self.rr_list = []
            for ind_text in tqdm(range(0, 5000)):
                base_name = 'sample' + str(ind_text + 1) + '.txt'
                compare_text = Path(self.text_dir) / base_name
                token_list = self.read_text(compare_text)
                for start in range(0, len(token_list), 200):
                    end = start + 200
                    if end > len(token_list):
                        break
                    if self.metric == 'hrep':
                        rr_result = entropy_of_distribution(token_list[start:end])
                    elif self.metric == 'hrep_3':
                        rr_result = entropy_of_distribution_ngram(token_list[start:end], 3)
                    else:
                        raise NotImplementedError(self.metric)
                    if rr_result is not None:
                        self.rr_list.append(rr_result)

            with self.save_file.open(encoding='utf-8', mode='w') as f:
                json.dump(self.rr_list, f)
                f.close()
            logger.info(f'SAVE RESULT TO {self.save_file}')

        logger.info(f'{self.sub_dir}: {np.mean(self.rr_list)}')


def get_human_hrep(n, logger, override=False):
    """
    compute the n-gram entropy on wikitext-103
    """

    save_file = Path('generated_samples/lyrics_ground_truth/wikitext103_test_rep_entropy_3.json')

    if save_file.exists() and not override:
        logger.info(f'LOAD RESULT FROM {save_file}')
        with save_file.open(encoding='utf-8', mode='r') as f:
            rr_list = json.load(f)
            f.close()
    else:
        logger.info(f'CALCULATING 3-GRAM ENTROPY ON WIKITEXT103.')
        file = Path('dataset/wikitext-103/wiki.train.tokens')
        with file.open(encoding='utf-8', mode='r') as f:
            texts = f.read()
            f.close()

        parser = argparse.ArgumentParser()
        utils.create_params(parser)
        args = parser.parse_args()
        args.is_en_gpt2 = True

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
        model.eval()

        rr_list = []
        rr_num = 0

        texts = texts.split('\n')

        for text in tqdm(texts):
            tokens = tokenizer.tokenize(text)
            if len(tokens) >= 200:
                for start in range(0, len(tokens), 200):
                    end = start + 200
                    if end > len(tokens):
                        break
                    rr_result = entropy_of_distribution_ngram(tokens[start:end], n)
                    if rr_result is not None:
                        rr_list.append(rr_result)
                        rr_num += 1
                    if rr_num >= 5000:
                        break
            if rr_num >= 5000:
                break

        with save_file.open(encoding='utf-8', mode='w') as f:
            json.dump(rr_list, f)
            f.close()
        logger.info(f'SAVE RESULT TO {save_file}')

    logger.info(f'VALID RR: {np.mean(rr_list)}')
    return rr_list
