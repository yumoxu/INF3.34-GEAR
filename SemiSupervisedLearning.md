# Semi-supervised Learning

*Written by Yumo Xu*

**Paper Name**: Semi-Supervised Learning with Deep Generative Models, NIPS 2014

**Link**: https://arxiv.org/pdf/1406.5298.pdf

**Contribution**: 提出了三种利用unlabeled data的方法：

1. Latent-feature discriminative model (M1);
2. Generative semi-supervised model (M2);
3. Stacked generative semi-supervised model (M1 + M2).

####Latent-feature discriminative model (M1)

**Motivation**: Too naive to have a motivation...

**Factorization**: $p(\mathbf{x}, \mathbf{z}) = p( \mathbf{x}|\mathbf{z})p(\mathbf{z})$

**Intractable posterior**: $p(\mathbf z | \mathbf x)$

**Pipeline**

1. 在labeled data和unlabeled data上learn $p(\mathbf{z} | \mathbf{x})$;
3. Infer $p(\mathbf{z} | \mathbf{x})$ from labeled data as a **feature extractor**;
3. 把$\mathbf z$喂到某个分类器中, e.g., Transductive SVM.

**Note**: 此处 $p(\mathbf{z} | \mathbf{x})$只作为feature extractor, 所以该方法被称为discriminative model.

####Generative semi-supervised model (M2)

**Motivation**: 在unlabeled data上，$y$也可以被作为latent variable. $y$与$\mathbf{z}$ marginally independent. 这种independence有实际意义。比如在数字生成问题上，$y$为class specification, 而$\mathbf z$捕捉的是所有图片共有的writing style.

**Factorization**: $p(\mathbf{x}, y, \mathbf{z}) = p(\mathbf{x} | y, \mathbf{z})p(y)p(\mathbf{z})$ where

$p(y) \sim Multinomial(y|\pi),\;p(\mathbf{z}) \sim \mathcal N(\mathbf 0 | \mathbf I), \;p_{\theta}(\mathbf x | y, \mathbf z) = f(\mathbf x; y, \mathbf z, \theta)$

**Intractable posterior**: $p(y|\mathbf{x}), \;p(\mathbf{z}|\mathbf{x})$

**Pipeline**

1. Learn $p(y|\mathbf{x})$ and $p(\mathbf{z}|\mathbf{x})$ from both labeled & unlabeled data;
2. Predictions for unlabeled data as inference using $p(y|\mathbf{x})$.

Note: $p(\mathbf{x}, y, \mathbf{z}) $的factorization中，$p(y)$ 为离散分布， $p(\mathbf{z})$为连续分布。因此可以该模型可以看做*hybrid continuous-discrete mixture model*. 

####Stacked generative semi-supervised model (M1 + M2)

**Motivation**: 用M1中extract的$\mathbf{z}_1$取代M2中的$\mathbf{x}$, 相当于先做了预处理，作为input更好的表示。

**Factorization**: $p_{\theta}(\mathbf x, y, \mathbf z_1, \mathbf z_2) = p(y)p(\mathbf z_2)p(\mathbf z_2 | y, \mathbf z_1)p(\mathbf x | \mathbf z_1)$
其中$\mathbf{z}_2$为M2中的隐变量。$p(y)p(\mathbf z_2)p(\mathbf z_2 | y, \mathbf z_1)$相当于M2, 只是生成了$\mathbf z_2$. 接着M1学出来的$p(\mathbf x | \mathbf z_1)$生成$\mathbf x$.

**Pipeline**

1. M1: 在unlabeled data上学习$p(\mathbf z_1 | \mathbf x)$;
2. M2: 在labeled data和unlabeled data上先用$p(\mathbf z_1 | \mathbf x)$infer $\mathbf z_1$, 再学习$p(y), \; p(\mathbf{z}_2|\mathbf{z_1}, y)$.

Note: Pipeline-style stacking.