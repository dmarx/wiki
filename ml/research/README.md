# Recent ML Research

Created: `2021-05-25`

I've been collecting articles to read -- or that I have read and want to hold on to -- as github stars, browser favorites, saved reddit posts.... 

I used to collect them in a table in a github gist, and then was using arxiv-sanity.org...

Let's try this and see what happens.


# To Review

## Architecture/theory

### Misc

* ["More Data Can Hurt for Linear Regression: Sample-wise Double Descent" - 2019 - Preetum Nakkiran -](https://arxiv.org/abs/1912.07242)
  * [Gradient Double Descent](../topics/gradient-double-descent.md) in linear regression
* NADE - Neural autoregressive density estimation - 2016 - https://arxiv.org/abs/1605.02226

### Model Compression / Compressed learning

* [What Do Compressed Deep Neural Networks Forget?](https://arxiv.org/pdf/1911.05248v2.pdf) - 2019
  * We find that models with radically different numbers of weights have comparable top-line performance metrics but diverge considerably in behavior on a narrow subset of the dataset. This small subset of data points, which we term Pruning Identified Exemplars (PIEs) are systematically more impacted by the introduction of sparsity. Compression disproportionately impacts model performance on the underrepresented long-tail of the data distribution. PIEs over-index on atypical or noisy images that are far more challenging for both humans and algorithms to classify. 

* https://github.com/microsoft/fnl_paper/tree/main/deficient-efficient

### Transformers

* Performers - 2020 - [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)
  * Google/Deepmind
  * more efficient/sparser linear attention that approximates full-rank softmax transformer attention

* https://hyunjik11.github.io/talks/Attention_the_Analogue_of_Kernels_in_Deep_Learning.pdf

* FNet - ["FNet: Mixing Tokens with Fourier Transforms" - 2021 - James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon](https://arxiv.org/abs/2105.03824  )
  * Google Research
  * Yannic Kilcher overview: https://www.youtube.com/watch?v=JJR3pBl78zw&t=1297s 
  * Fourier transforms replacing attention in transformers
  * Q: Could this be used to accelerate pretraining?  

* ExpireSpan - ["Not All Memories are Created Equal: Learning to Forget by Expiring"](https://arxiv.org/abs/2105.06548)
  * Facebook AI
  * https://github.com/facebookresearch/transformer-sequential
  * Yannic Kilcher: https://www.youtube.com/watch?v=2PYLNHqxd5A 

## Causal Inference


<!-- TAGS
-->
