# PMI Masking for HuggingFace (pytorch)
This repository provides an implementation for the [PMI Masking (Levine et al., 2020)](https://arxiv.org/abs/2010.01825)
It uses the files from [AI21Labs/pmi-masking](https://github.com/AI21Labs/pmi-masking) repository.

## Citation
The code in this repository was written for the paper 
[Scaling Laws Under the Microscope: Predicting Transformer Performance from Small Scale Experiments (Ivgi et al., 2022)](https://arxiv.org/abs/2202.06387).

If using the code here, please cite both the above paper and the original PMI paper

```bibtex
@article{Ivgi2022ScalingLU,
  title={Scaling Laws Under the Microscope: Predicting Transformer Performance from Small Scale Experiments},
  author={Maor Ivgi and Yair Carmon and Jonathan Berant},
  journal={ArXiv},
  year={2022},
  volume={abs/2202.06387}
}
```

```bibtex
BibTex
MLA
APA
Chicago
@article{Levine2021PMIMaskingPM,
  title={PMI-Masking: Principled masking of correlated spans},
  author={Yoav Levine and Barak Lenz and Opher Lieber and Omri Abend and Kevin Leyton-Brown and Moshe Tennenholtz and Yoav Shoham},
  journal={ArXiv},
  year={2021},
  volume={abs/2010.01825}
}
```

## Usage
To use the code, download the vocabulary file and the DataCollatorForPMIMasking module, and in your pretraining script replace the data_collator initialization with the below:

```python
import DataCollatorForPMIMasking
with open(ngrams_vocab_path) as f:
    ngrams_vocab_set = set(f.read().split('\n'))
data_collator = DataCollatorForPMIMasking(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability,
                                          ngrams_vocab_set=ngrams_vocab_set)
```
