<<<<<<< HEAD
# VocabularyCurriculum
Scale better with vocabulary curriculum
=======
Scale better with vocabulary curriculum 


arXiv: 

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

>>>>>>> f12ae99 (Initial commit)
