# 5.8
## TODO
### EXP
1. task: ours 0.5 0.75 0.1 all 0.75
2. main: ours pems08-调参 pems03 stgode pems08
3. lambda: pems04 pems03
4. sparsity: pems04 pems03
5. ablation
### New Exp
1. task: ours 0.5 0.75 0.1 | stgode 0.05 pems03| 04: lstm stgcn | 08: stgcn | 03: stgcn
2. ablation: w/o sim loss 08/03 | w/o aug | original graph
3. hyper: pems03 all 
4. main: pems03 stgode stgcn(ours) | pems08 stgcn(ours) seed 2
### Writing
1. lambda better illustration
2. 强调limited nodes
3. 给一个city的例子除了covid






# 5.4 
## TODO
### parameter
1. alpha, o
2. frame: 2 * 3
3. one dataset: PEMS03
### task setting
1. 0.05 0.1 0.25 0.5 0.75 1
2. model: ours, stgcn, lstm, stgode
3. dataset: 3
4. frame: 3 * 3
### ablation
1. only graph
2. only augmentation
3. only manifold
4. only mix-up
5. joint training
### efficiency



# 5.3
## TODO
### TranGTR
1. best need to change
2. make sure no specific name in string

## dependencies
Wandb
Torch 
Geometric
Pandas
matplotlib
tensorboardX
torchdiffeq
optuna
networkx


# 5.1
## TODO
### Baseline
1. SVR?
2. MLP + our method




# 4.29
## TODO 
### Main Exp Performance better
1. 不同参数设置效果差很大
   1. 是否需要batch same graph？inference的时候不好解释
2. batch一个图的效果也很差
   1. 固定节点mixup
   2. vae loss 加大？variance. vae, graph.
   





# 4.28
## Problem 
### TransGTR
1. validation cancel
   1. train足够多。
   2. val足够少。
   3. 

## Writing
### Augmentation
1. check equation of VAE
2. time window or series
### Structure learning
1. 完全一样的
2. 为什么要sparse？
3. graph batch?



# 4.27 
## Problem
### TranGTR
1. MAE validation set protection。
2. exp setting:
   1. 3 days 1 day w/o validation 差的合理。主表放什么？都放。
   2. short validation (could be even shorter if performance worse)
      1. how short ?




# 4.26
## Baseline
### Traditional 3
1. VAR
2. LSTM [v]
3. MLP [x]
4. ARIMA
5. HI [v]
### Classical 1
1. STGCN [v]
2. (DCRNN) [v]
### Common 1
1. STFGNN [x]
2. ASTGCN [x] contain parameters dimension as node number
3. GraphWaveNet [x] learn node embedding
4. STSGCN [x] learn node embedding
5. STGODE
6. STAEFormer [x] contain parameters dimension as node number
7. AGCRN [x] learn node embedding
8. STWave [x]
9. STNorm [x]
10. Fogs
### Fine-tune 1
1. TransGTR 
2. GTS
3. TransGTR baselines

## TODO
### STGODE
1. add our method
### TRANSGTR
1. add our task in training
2. custom fine-tune length
### COVID data

## Problem
1. GPU
   1. PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python main.py --c configs/STGCN_ST/PEMS03.yml




# 4.25
## Baseline
### Traditional 3
1. VAR
2. LSTM [v]
3. MLP [x]
4. ARIMA
5. HI [v]
### Classical 1
1. STGCN [v]
2. (DCRNN) [v]
### Common 1
1. STFGNN [x]
2. ASTGCN [x] contain parameters dimension as node number
3. GraphWaveNet [x] learn node embedding
4. STSGCN [x] learn node embedding
5. STGODE
6. STAEFormer [x] contain parameters dimension as node number
7. AGCRN [x] learn node embedding
8. STWave
9. STNorm
10. Fogs
### Fine-tune 1
1. TransGTR 
2. GTS
3. TransGTR baselines

## TODO
### STGODE
1. 调整threshold
2. 跑通
### TRANSGTR





# 4.24
## TODO
### Intitial
1. check results, 我跑了什么? 调参，就是选择可能会有影响的跑跑。选择最好的参数
2. 跑baseline实验结果，lstm，stgcn，hi，写一个bash，调参
   1. lr，dp
3. 03的dataset

## Thinking 
### What to do
1. 做了什么？部分结果。这个结果可用吗？可以。有调参吗？调参很重要。
   1. 手动记录数据太繁琐，需要今天设计出来一个不是手动记录的



# 4.23
## Analysis
### Increased variance
1. the variance smaller the better -> more fidelity
### Dynamic mixture
1. dynamic better than non-dynamic
   1. don't break fidelity
### VAE loss
1. bigger, regular more, worse -> fidelity



# 4.22
## Method Imp
### Increased variance
### VAE loss 
1. bad performance?
   1. vae can increase diversity but still overfit
### Graph sparse based on regularization
### Rational augmentation




# 4.20
## Method Imp
### Increased variance
### VAE loss 
### Graph sparse based on regularization
### Rational augmentation

## Story
### Augmentation
1. Other domain: manifold mixiup
2. Time series domain: time series mixup in time / frequency 
3. time series domain
   1. AUGMENT ON MANIFOLD: MIXUP REGULARIZATION WITH UMAP



# 4.19
## Exp
### Interesting ablation
1. WD on the VAE - mostly will decrease the performance
2. Number of ablated nodes

## Analysis
### Augmentation, diverse mixup improve the results
### Augmentation using contrastive loss will make aug loss huge, and unstable, indicating the similarity among nodes
### Different datasets need different hyperparameters, PEMS07 need 0.01 learning rate

## TODO
### Two baselines
### Check the problem of pems07
### Table Draft
### Methodology

# 4.18
## Baseline
### Traditional 3
1. VAR
2. LSTM [v]
3. MLP [v]
4. ARIMA
5. HI [v]
### Classical 1
1. STGCN [v]
2. (DCRNN) [v]
### Common 1
1. STFGNN [x]
2. ASTGCN [x] contain parameters dimension as node number
3. GraphWaveNet [x] learn node embedding
4. STSGCN [x] learn node embedding
5. STGODE
6. STAEFormer [x] contain parameters dimension as node number
7. AGCRN [x] learn node embedding
8. STWave
9. STNorm
10. Fogs
### Fine-tune 1
1. TransGTR 
2. GTS
3. TransGTR baselines


## Exp Design
### Main
1. Horizon 3 组： 3， 6， 12
2. MAE，RMSE，MAPE
3. 排版
### Compare with Fine-tune
1. a new table as main table
### Compare under different training node ratios
1. 0.05, 0.1, 0.2, 0.5, 1
2. choose only 3 baselines
3. plot
### ablation study on a dataset
1. no learnable graph
2. dynamic graph
3. no augmentation
4. no loss 1
5. no loss 2
6. no masknode
### visualization

# 4.17

## Idea Summary
### Training with Augmentation
1. initiate with random augmentation
   1. bad performance in theory. 
      1. 第一步就开始学augmentation，warm-up (前几步不augment)
         1. 什么时候知道warm-up好了呢？
2. alternative training
   1. train augmentation seperately 
      1. have to get a learned model to train augmentation
      2. augmentation has to be used during training
3. the rationale is to learn augmentation as close as original time series
   1. why not directly learn some nodes --> they are not dynamic through every time slots, can't provide enough diversity
### learn a structure with gumbel-softmax
1. do I write the right code?
2. why it performs so good?
3. why do I have to design like that? what advantage?
4. why learn a structure can overcome overfit?
   1. structural overfit: original topologies vary differently among regions. : might exp with teaser. **https://arxiv.org/pdf/2309.04332.pdf**
5. why learn a graph
   1. static not optimal
### masked loss
1. how does it work? improving the diversity.
   1. training the models on different time series sets, thus not overfit to a fixed local minima
   2. similar to node dropout



# 4.15

## Analysis
### No connection between augmented nodes
1. learn a space where all nodes are addable at the beginning
2. add a regularization term. edge should be there.
3. make the augmentaed nodes as similar as true nodes
### Neighbor augmentation is useless
1. only calculate the loss of certain nodes of original graph will overfit 
2. learning a structure will decrease the overfitting -> **we could make an experiment to demonstrate the effectiveness of learning a graph**


## Method
### Learn graph with Gumbel-softmax
1. why gumbel-softmax
   1. consider noise in graph
   2. no need for weight
2. why learn a graph
   1. dynamic structure
### Training time augmentation with VAE
1. why augmentation
   1. diverse
2. why training time?
   1. only augment training data. so the validation dataset have a little domain shift as training data
### Node loss mask
1. incorporate more diversity


# 4.14

## TODO
### How to learn a graph
1. make sure the graph is sparse. ()
2. make sure the topology varies. 
### How to augment the graph
1. automatically learn the best augmentation during training
   1. how to define best. ground truth?
      1. any loss? 
      2. weighted of each augmentation?
      3. need to be similar to training samples. how to ?
         1. simply purturbation
         2. learn
   2. For each sample, the augmentation policy be the same or not?
      1. same, so the best could be chose.




# 4.8 

## Analysis
### How to augment time series data as close as true data
1. augment only for close enough nodes
2. augment with trend and seasonal perturbation (decomposition)
3. gramian augmentation
4. AAFT surrogates, 多个从中选取和graph nodes最相关的
5. flipping, down sampling (smooth)
6. phase or magnitude pertubation (STFT) we could have a larger time frame as input
7. time steps / frequency masking
8. rank-based augmentation 
9. automatically choosing the best augmentation

### Node dropout
1. weighted node dropout
2. weighted loss

### batch norm the training loss worse
1. different parameter affecting
2. reason
   1. regularization
   2. skew

### why sparse random graph works
1. de-weight. pattern 分布在train和test不一致。




## TODO
1. decompostion
   1. statsmodel






# 4.7

## Analysis
### How to augment time series data as close as true data
1. augment only for close enough nodes
2. augment with trend and seasonal perturbation (decomposition)
3. gramian augmentation
4. AAFT surrogates, 多个从中选取和graph nodes最相关的
5. flipping, down sampling (smooth)
6. phase or magnitude pertubation (STFT) we could have a larger time frame as input
7. time steps / frequency masking

### Node dropout



# 4.6

## Analyse
### random graph is better than learned graph?
1. different level of sparsity / randomness 
2. reason
    1. even learn, two nodes might easily be neighbors. So even learning a graph, this two might mutual enhance, make the model learn their pattern. 
        1. heteoro neighbors
        2. independent edge graphs (not applicable to inductive setting)
        3. random graph model
### why is there still overfit?
1. any paper related? overfit and ood of time series. why?
2. not related to temporal dimension, since validation set has good performance.
3. reason
    1. still learn some pattern?
        1. can we randomly mask loss of some nodes?
### Normalize might be a cause of good result?
1. maybe the distribution of graph make it
### How is sparsity working?
1. more sparse, result better.
    1. oversmooth
    2. act like a noise, but the noise has to be restricted
    3. has to be applied on every node, each node should be equally noisilized
    4. a fixed graph or a learned graph have less noise
### Why we need graph
1. sparse graph is better than no graph
    1. kind of noise
    2. neighborhood information learnt
### How is batch normalize and layer normalize
1. The result converge slower
    1. act like a smooth of noise. stabilize bad.
2. Training loss is high
### Dropout performs bad
1. Training loss is super high
    1. It might add some noise not related to pattern




## TODO
### Add more datasets
1. datasets show your works importance
### Temporal novelty



# 4.5

## TODO
1. VAE coding
2. Training paradigm coding
    1. first train VAE only
    2. then train all others without VAE
3. Mixup coding
4. Graph learning coding





<!-- Idea
1. 现在的问题是数据稀缺性？在空间上的数据稀缺。会导致空间上的过拟合--》很自然的思路就是生成新的数据
2. 新的数据生成--很多paper有
3. 但是如何应用在inductive的设定上面？
    1. 如何把新的节点连接到所有节点上面？
    2. 如何使这些新augmentation的time series能等同于不同节点的信息
    3. 如何使得这些新的time series有效
4. 问题
    1. 没有足够的边能够学到这个inductive bias. 相似的节点，应该相关 --》 为什么？
    2. 抽样得到的train数据集，有可能不包含边，如果原始的边就是sparse的话。
        1. 我们需要学习一些边，使得我们在train的时候能利用到graph的归纳偏置。
        2. 但是如果train的节点不具有足够的相似性，在train的时候却学到了较高的相似的embedding。比如说a,b,c都不算邻居，但是a和b更接近，在train的时候b就会分给a较大的权重。但是在test的时候很可能有更多大量的相似的节点，而这些节点不能够被很好的区分开。
            1. 避免饱和的相似性
            2. augmentation足够相似的节点
    3. augmentation 真的能解决问题吗？比较promising，因为有类似的数据很少的工作使用augmentation的方法。
    4. 用augmentation还能叫做inductive吗？
5. method
    1. interpolate / exptrapolate augmentation based on VAE space
    2. training-time augmentation / end-to-end
    3. full time series augmentation -->
<!-- # Paper
1. augmentation
    1. 扩散模型。生成足够真实的数据
    2. barycentric averaging -->

    