# NVI Basics

*Written by Yumo Xu*

##Terminology

- **Learning**: learn (parameters in) distributions, e.g., $p(\mathbf z| \mathbf x)$;
- **Inference**: infer variables, e.g., infer $\mathbf z$ from $\mathbf x$ with $p(\mathbf z| \mathbf x)$.

##NVI Basics
以$\mathbf{z} \rightarrow \mathbf{x}​$为例。Posterior distribution是贝叶斯统计的核心成分。**Learning**的一个重要目标即是从data中学习posterior distribution $p(\mathbf z | \mathbf x)​$, 来更新对$\mathbf z|\mathbf x​$的belief. 

最直观的方法是从joint probability中marginalize $\mathbf z$:

$$p(\mathbf z | \mathbf x) = \frac{p(\mathbf x, \mathbf z)}{\int_{\mathbf z} p(\mathbf x, \mathbf z)}$$

然而，绝大多数model都无法计算出以上normalization constant中的积分。

我们只能换一种思路去解 $p(\mathbf z | \mathbf x)$. 但我们不了解$p(\mathbf z | \mathbf x)$的任何特征，例如分布族，有的只有 $\mathbf x$. 为了让它可解，基于approximation inference的思想，我们可以先构造一个简单好算的分布 $q$，然后让 $q$和 $p(\mathbf z | \mathbf x)$尽可能接近。如此，$q$就是个posterior approximator, 充当着代理的角色(proxy). VI也将Bayesian inference转化成了一个优化问题。

所以我们的任务就由两个部分组成:

1. 基于$\mathbf x$ parametrize $q$;
2. Minimize $q$与 $p(\mathbf z | \mathbf x)$的差异。

NVI在(1)中选择neural approximation, 用NN去parametrize $q$. 这里使用了*reparameterization trick*使得采样的过程可微，在此不再详述。

如果在(2)中使用KL divergence, 即是$\min KL[q || p(\mathbf z | \mathbf x)]$. 计算KL时，为避免直接处理无法计算的$p(\mathbf z | \mathbf x)$, 我们进一步利用graph factorization来break down $p(\mathbf{z} | \mathbf x)$:
$$
\begin{align}
KL[q || p(\mathbf z | \mathbf x)] =& KL[q || \frac{p(\mathbf{x} | \mathbf z) p(\mathbf z)}{p(\mathbf{x})}]\\=&E_q[\log q - \log p(\mathbf{x} | \mathbf z) - \log p(\mathbf z)] + \log p(\mathbf{x}) \\=& E_q[- \log p(\mathbf{x} | \mathbf z) + \log q - \log p(\mathbf z)] + \log p(\mathbf{x}) \\=& - E_q[\log p(\mathbf{x} | \mathbf z)] + KL[q || p(\mathbf z)] + \log p(\mathbf{x})
\end{align}
$$

这样一来，variable relation也被利用起来，encode到了对$q$的approximation中。由于左式is minimized with respect to $q$, 右式中的$\log p(\mathbf x)$即为常数。故等价于优化：

$$\min \mathcal L = - E_q[\log p(\mathbf{x} | \mathbf z)] + KL[q || p(\mathbf z)]$$

现在我们知道一个优化目标$\min \mathcal L$，也知道它的组成，故可以设计出一个NN+BP的架构去解它：首先需要一个组件paramterize $q$, 之后需要一个组件基于$\mathbf z \sim q$计算出第一部分的likelihood loss. 这个liklihood loss可以类比为discriminative model中的reconstruction loss.

如此，通过这样一个NN的架构，我们可以从data中learn出最合适的$q$作为posterior approximator. Learning完成后，给定$\mathbf x$便可infer $\mathbf z$. 故该NN被称作*inference network*.

由于优化目标$\min \mathcal L$等价于KL, 我们可以从3个角度去思考如何优化generative model:

1. 找到更富表达力的approxmator $q$;
2. 找到更准确的$p(\mathbf z| \mathbf x)$, 即更好的factorization去捕捉变量之间的关系;
3. 找到更好的metric去用$q$逼近$p(\mathbf z| \mathbf x)$.
