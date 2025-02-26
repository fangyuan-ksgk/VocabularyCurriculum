<div align="center">
<h2 align="center">
   <b>Scale better with vocabulary curriculum</b>
</h2>

<div>
  <a target="_blank" href="https://scholar.google.com.sg/citations?user=GqZfs_IAAAAJ&hl=en">Fangyuan&nbsp;Yu</a><sup>1</sup>
</div>

<br />
<sup>1</sup>Temus&nbsp;&nbsp;&nbsp;
<br />
<div align="center">
    <a href="xxx" target="_blank">
</div>
</div>

![better-scale-vocab-curriculum-1](https://github.com/user-attachments/assets/85616d2b-c882-4aa8-a36d-4f988011ed59)

Modern language model pre-training relies on a static
vocabulary, fixed before training and detached from the model’s learning dynam-
ics—unlike human language acquisition. We introduce a vocabulary curriculum,
an adaptive approach that enhances pre-training efficiency and improves bits per
character (BPC) by a factor that scales log-linearly with vocabulary size. Our
method alternates between vocabulary adaptation and language model optimiza-
tion, allowing the model to learn representations across multiple token granularities,
leading to significantly improved compression. 

![scale-better-incre-vocab](https://github.com/user-attachments/assets/7ef0598c-adeb-428b-b357-f657322c0dfa)


## :new: Updates
- [02/2025] [arXiv Preprint] https://arxiv.org/pdf/2502.17910

Environment setup: 
```
bash set.sh
```

Key to our approach is a vocabulary which update according to modeling entropy, as vocabulary update includes tokenization update as well as model update, we provide a 'Magicab' package to streamline the joint vocabulary update process. Refer to 'demo.ipynb' to for a demo of the joint tokenization and model update process. 

```python
from magicab.magicab import Magicab

magicab = Magicab(model, tokenizer)

magicab.cache_vocab_change(text=texts)
magicab.update_vocab(max_size_change=1000)
```
Above code will update the vocabulary according to the modeling entropy of the text. 

To replicate our experiments, follow the steps below: 

1. Prepare cleaned enwiki9 dataset 
```
bash data/enwiki/download_data.sh
python data/enwiki/prepare_data.py
```

2. Run vocabulary curriculum training 
```
bash run_circular.sh
```
Note that we include circular training which increase and decrease vocabulary size in this run. 

3. Run compute matching experiment 
```
bash run_compute_match.sh
```

This codebase heavily borrows from the following repos: 
- nano-gpt: https://github.com/karpathy/nanoGPT
- minBPE: https://github.com/jxnl/minbpe
