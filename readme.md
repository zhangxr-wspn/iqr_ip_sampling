## Code for the paper "Improve the Diversity and Novelty for Open-ended Neural Text Generation via Inverse Probability Weighting" (NLPCC 2023)

#### Corresponding equations in our paper are marked in `src/sampling.py`

1. Make env and download pre-trained models
    ```bash
    conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
    pip install -r requirements.txt
    ```
    Our implementation is based on [HuggingFace Transformer](https://github.com/huggingface/transformers). Part of the implementation borrows the generation loops in [gpt2-simple](https://github.com/minimaxir/gpt-2-simple).    
    
    Download `config.json`, `merges.txt`, `pytorch_model.bin`, `vocab.json` from https://huggingface.co/gpt2-xl/tree/main, and put them in `models/en_gpt2_xl_pretrained_models/final_model`
    
    (optional) You can also load these models automatically with the following code. If you prefer this, you must replace the model/tokenizer loading part in our code with the following code.
    
    ```base
    from pytorch_transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
    model = GPT2Model.from_pretrained('gpt2-xl')
    ```

2. Download the [wikitext-103 dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), unzip the training token file to `dataset/wikitext-103/wiki.train.tokens`.

3. Generate one sample with IQR-IP sampling algorithm and GPT-2 XL model, print it in the console, and save it locally.
    ```bash
    PYTHONPATH=. python generate_sample.py --is_en_gpt2_xl --topp 0.8 --topk 640 --n_fraction 100 --iqr_ip_weighting --save_path_sub_dir gpt2_en_iqr_test --articles_per_title 1 --cuda_id 0 --length 200 --show_tqdm_bar
    ```

4. Generate 5,000 samples per sampling parameter for our method and baseline methods for evaluation.
    ```bash    
    PYTHONPATH=. python script/generate_iqr_gpt2_xl_en.py
    PYTHONPATH=. python script/generate_top_p_k_gpt2_xl_en.py
    PYTHONPATH=. python script/generate_temperature_gpt2_xl_en.py
    ```
    Note: it may take more than 50 hours on 8 NVIDIA 2080 ti GPUs to finish all generation processes. We recommend [downloading](https://drive.google.com/file/d/13jf4HDZhFgckqe8R6AuGxydUvNDZ2jDs/view?usp=sharing) all generated samples evaluated in our paper, unzip and put them in the root dir for evaluation. If so, you may skip the generation and go straight to step 5.

5. plot figures for the metric variation. 
    ```bash
    PYTHONPATH=. python script/plot_fig_ppl_gpt2_xl.py
    PYTHONPATH=. python script/plot_fig_bleu_gpt2_xl.py
    PYTHONPATH=. python script/plot_fig_ent_gpt2_xl.py
    PYTHONPATH=. python script/plot_fig_zipf_gpt2_xl.py
    ```
   
6. plot figures for the metric trade-off. 
    ```bash    
    PYTHONPATH=. python script/plot_fig_q_d_bleu4_gpt2_xl.py
    PYTHONPATH=. python script/plot_fig_q_d_bleu5_gpt2_xl.py
    PYTHONPATH=. python script/plot_fig_q_d_ent_gpt2_xl.py
    PYTHONPATH=. python script/plot_fig_q_d_zipf_gpt2_xl.py
    ```