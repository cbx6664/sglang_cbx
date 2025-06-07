## **Introduction**

1. **Prefill 阶段会并行处理输入 prompt 的所有 token，因此很小的 batch size 就会打满 GPU utilization。**比如说，13B 的 LLaMA 输入一条 512 tokens 的 prompt 做 prefill 就会打满单卡 A6000。**反过来，（开启 [KV Cache](https://zhida.zhihu.com/search?content_id=247844397&content_type=Article&match_order=1&q=KV+Cache&zhida_source=entity)）的 decode 阶段每个自回归阶段仅仅会生成一个 token，因此 GPU utilization 很低。**
2. Decode 时，单个 token 的开销显著大于 prefill 阶段；增大 batch size，prefill 的单个 token 开销几乎是不变的，而 decode 的开销显著下降。可见，prefill 在 batch size 很小时就已经占满了 GPU 效率，而 decode 阶段在 batch size 很大时才会占满。实际上后文会说明，prefill 阶段的输入 L 很长，因此计算开销大；而 decode 阶段输入的 L 一直是 1，但是需要反复读读 KV Cache，故而 IO 开销很大。
3. 需要把 decode 阶段的 batch size 开的非常大才有可能占满 GPU utilization，但是开这么大的 batch size 会因为 KV Cache 读写开销太大而变得不现实，所以 docode 阶段的 GPU utilization 是很难占满的。因此，在实际情况下，decode 仍旧是 memory bounded 而非 compute bounded 的。
4. 一条 prompt 会被 prefill 一次，但是会 decode 多次，直到 decode 满足终结条件，譬如 end token 或者长度限制。
5. [Tensor Parallize](https://zhida.zhihu.com/search?content_id=247844397&content_type=Article&match_order=1&q=Tensor+Parallize&zhida_source=entity) 卡间通讯需求大，而 Pipeline Paralize 需要不断优化 pipeline bubble。
6. Chunked Prefill 做出了两步优化。首先将长短不一的 prompts 拆分为长短一致的 chunks 进行 prefill；其次这些 chunks 间的气泡可以插入/捎带（piggyback）其他完成了 prefill 的 prompts 的 decode 需求。

![img](https://pic3.zhimg.com/v2-cc9a3f288b3e2dcec132bfb0e9a45dca_1440w.jpg)

## **[Transformer Architecture](https://zhida.zhihu.com/search?content_id=247844397&content_type=Article&match_order=1&q=Transformer+Architecture&zhida_source=entity)**

1. Transformer decoder block 在计算上可以看做六个操作的总和：pre-proj，attn，post-proj，ffn_ln1，ffn_ln2，others（比如说 layer normalization，activation functions，residual connection..）
2. Transformer 的输出可以视为一个 tensor X of shape [B, L, H]。其中 B 是 batch size，L 是 input tokens length，H 是模型的 embedding size。
3. Prefill 的第一步做 pre-proj。从数学上，就是简单的线性运算，分别用三个大小为 [H, H] 的矩阵 W^Q,W^K,W^V 和 X 做乘积。从计算上，就是输入 X 和一个大小为 [H, 3H] 的矩阵相乘。
4. attn 操作在计算上的输入是 Q, K, V，而输出 Y 仍旧是大小为 [B, L, H] 的 tensor。post-proj 采用大小为 [H, H] 的 W_0 矩阵和 Y 相乘，输出结果 Z 的大小仍旧为 [B, L, H]。
5. ffn_ln1 和 ffn_ln2 在计算上的输入是 Z。ffn_ln1 中，Z 和大小为 [H, H’] 的矩阵相乘，得到大小为 [B, L, H’] 的 tensor，接着和大小为 [H’, H] 的矩阵相乘，再投影回去得到一个大小仍旧为 [B, L, H] 的 tensor。上式中，H’ 为模型的 second hidden dimension。
6. Decode 阶段和 Prefill 阶段的操作完全一致，不过每次只会生成一个 token 并且输入给下个阶段。采用 KV Cache 后，实质上的输入是上次生成的那一个 token，输入的 tensor 大小是 [B, 1, H]（input tokens 的长度为 1）。
7. 每个 token 的 KV Cache 大小均为 [1, H]。

## **多 GPU 推理**

1. tensor parallelism 在单机多卡且卡间通讯强时较优；而 pipeline parallism 主要在多机情况下用于 serve 大到没法在单机上运行的模型。
2. pipeline parallism 是卡间通讯不好时唯一可行的 model parallelism 方法。
3. PP 将模型按照 layer 分开，每个 GPU 负责一部分 layer；而 TP shards 【TODO】每个 layer 到所有 GPU 上。PP 比起 TP 具有更好的 compute-communication ratio，因而不要求昂贵的卡间高速通讯。

## **Motivation**

1. 当前的推理框架存在两个低效原因，首先是 decode 阶段的 memory boundary，第二是 pipeline parallelism 带来的 pipeline bubble。
2. 在 transformer block 中，除开五个主要部分之外的 others 占据的开销不超过 5%。
3. inference 只会做 forward passes 而没有 training 中的 backward。
4. 基于如上的分析可见 prefill 和 decode 具有很不一样的优化目标，因此 chunked prefill 进行了两点优化：对 prefill 进行数学上等价的切分；在 prefill 切成的这些 chunks 的气泡处捎带其他 request 的 decode pass。

## **具体实现**

1. 随着模型 hidden size 增大，更小的 chunk size 就会打满 gpu。
2. 实际上模型部署时有着非常长的 system prompts，因此 chunk 是可行的。
3. 如果 chunk size 开的非常小，那么 prefill 的效率会因为 GPU 利用率变低而降低。
4. Chunked prefill 需要特殊处理 attention mask。

![img](https://picx.zhimg.com/v2-c080a42a297ee9790ceadb615f008855_1440w.jpg)

1. 做了 chunked prefill 后，prefill 的开销会略微增大。因为计算后续 chunk 的 KV 时需要不断地从 GPU memory 中里读出当前 chunk 的 KV 到 kernal 里面；而不做 chunked prefill 时，最开端的那些 KV Cache 可以不用反复从 GPU memory 中反复读取进入 kernal，毕竟他们一直在 kernal 里面。
2. 即便如此，我们仍旧要做 chunked prefill，因为做了 chunk 之后，可以在 chunk 的 bubble 处捎带 decode 请求。这么做是有利于 decode 的，因为 decode 的 memory 开销除了要从 GPU memory 中 fetch KV Cache 之外，还有一部分开销是要 fetch 模型参数。采用 piggyback 的方式捎带 decode 到 chunk 的 bubble 后（称为 decode-maximal batching），可以直接 reuse prefill 阶段 fetch 的 模型参数。如此操作，几乎可以让 decode 从一个 memory bound 操作转换为一个 compute bound 操作。经过测试，通过捎带的 decode 的耗时会显著降低到原本的 10%。
3. 由此可见【TODO】，更小的 chunk size 可以捎带更多的 decode 请求，但是降低了 prefill 效率；chunk size 也是一个 trade off 罢。
4. sequence length 整除 GPU tile size 时，GPU 的效率最高；反过来，仅仅增加 1 个 token length 都可能显著降低 GPU 效率。
5. 可见，一句话总结：chunked prefill 一大意义在于利用 model parameters resue 来降低 decode 的开销。此外，也减小了 pipeline bubble 的影响。