from pytorch_transformers import GPT2LMHeadModel


def is_en_model(args):
    return args.is_en_gpt2 or args.is_en_gpt2_xl


def get_model(args, logger):

    logger.info('LOADING MODEL')
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    logger.info(f'PRETRAINED GPT2LMHeadModel MODEL LOADED FROM {args.model_path}')

    return model


def prepare_context(args, tokenizer, title):
    if args.is_en_gpt2 or args.is_en_gpt2_xl:
        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title))
    else:
        raise NotImplementedError('WRONG MODEL SELECTION')

    return context_tokens

