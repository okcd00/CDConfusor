## Confusor

> A module for generating the confusion-set for each word (1 or more tokens).



### Overview

**Confusor is a retrieval-based module for the word-level confusion-set generation.** This module retrieves the required confusion-set for a target word from more than 8 million Chinese words. 

Confusor applies the two-step retrieval: **pinyin retrieval** and **word retrieval**. In the pinyin retrieval step, confusor retrieves most similar pinyin sequences from a large pinyin sequence corpus by the **refined edit distance (RED) algorithm**, which uses two manually-set transition matrices to model the effects of typical errors such as fuzzy tones and keyboard mistouch. Then the set of candidate words can be easily obtained. In the word retrieval step, confusor further computes a weighted score to retrieve words in a small confusion-set from the large set of candidate words.

![image-20210806161749207](./docs/overview.png)However, the pinyin retrieval step is time-consuming. Three retrieval methods are proposed to address this issue: **two-stage retrieval, DCC retrieval, beam search retrieval.**

See `/docs/confset.pdf` for more details.


### Datasets
Datasets come form this [repo](https://github.com/okcd00/realworld_chinese_typos), will be published soon.
+ `.tsv` for datasets with pair-wise sentences: 
  + `<error_sentence>\t<correct_sentence>`
+ `.dcn.txt` for datasets with DCN-form: 
  + train: `(pid=test-01)\t<error_sentence>\t<correct_sentence>\t<attention_ids>\t<pinyin_ids_for_err>`
  + test: `(pid=test-01)\t<error_sentence>`


### Usage

#### 1. Initialization

```
conf = Confusor(method='beam', pinyin_sample_mode='special', token_sample_mode='sort', weight=[1, 0.5, 1])
```

**Parameters:**

| Param                | Note                              | Option                                 |
| -------------------- | --------------------------------- | -------------------------------------- |
| `method`             | pinyin sequence retrieval method. | 'baseline', 'two-stage', 'dcc', 'beam' |
| `pinyin_sample_mode` | pinyin sequence sampling mode.    | 'sort', 'random', 'special'            |
| `token_sample_mode`  | word sampling mode.               | 'sort', 'random'                       |

Use `weight` to adjust the weight of the RED score, cosine similarity score and frequency score for a word respectively.

See `/bbcm/data/loaders/confusor.py` for more details.



#### 2. Call the Confusor!

```
conf('世界')
```

Then you can get the confusion-set of size 10 as follows:

```
['实际', '视界', '世纪', '世届', '十届', '十界', '视觉', '事界', '四届', '时节']
```

