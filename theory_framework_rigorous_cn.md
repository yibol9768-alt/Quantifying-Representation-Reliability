# 多模型预训练特征融合的理论框架（严谨版）

> 适用场景：冻结多个预训练视觉编码器，只在目标任务上训练线性探针、MLP、门控器或轻量路由器。  
> 写作原则：本文明确区分四类内容：
> 1. 严格可证命题  
> 2. 带条件成立的命题  
> 3. 经验性背景  
> 4. 工程代理量  
>
> 这样写的目的是避免把经验现象写成普遍定理，也避免把 CKA、LogME 这类代理量直接写成互信息本身。

## 1. 问题设定

设预训练模型池为
\[
\mathcal M = \{m_1,\dots,m_M\}.
\]
每个模型 \(m\) 对输入 \(x\in\mathcal X\) 输出冻结特征
\[
f_m(x)\in\mathbb R^{d_m}.
\]

对任意子集 \(S\subseteq \mathcal M\)，定义拼接特征
\[
F_S(x)=\bigoplus_{m\in S} f_m(x)\in\mathbb R^{d_S},\qquad d_S=\sum_{m\in S} d_m.
\]

如果进一步使用门控器/路由器，则可写成
\[
Z_{S,\theta}(x)=\Psi_\theta\big(\{f_m(x)\}_{m\in S}\big),
\]
其中 \(\theta\) 是在目标任务上学习的融合参数。

目标任务标签记为随机变量 \(Y\)。目标训练集为
\[
s=\{(x_i,y_i)\}_{i=1}^n\sim P^{\otimes n}.
\]

本文要回答两个问题：

1. 为什么“更多模型”在总体上似乎不会坏，但在 few-shot 实验里却经常变差？
2. 为什么模型选择不能只看多样性，还必须看任务相关性？

---

## 2. 总体层面哪些结论是严格成立的

### 命题 1：Bayes 最优风险对特征扩充单调不增

定义子集 \(S\) 上的 Bayes 风险
\[
R^\star(S)=\inf_g \Pr[g(F_S)\neq Y],
\]
其中 \(g\) 在所有可测分类器上取下确界。

若 \(S\subseteq T\)，则
\[
R^\star(T)\le R^\star(S).
\]

**证明。** 设 \(\pi_S\) 是从 \(F_T\) 投影到 \(F_S\) 的坐标投影。对任意定义在 \(F_S\) 上的分类器 \(g\)，都可构造定义在 \(F_T\) 上的分类器
\[
\tilde g(F_T)=g(\pi_S(F_T)).
\]
因此 \(T\) 上可用的分类器集合包含了“忽略新增特征”的所有做法，于是
\[
\inf_h \Pr[h(F_T)\neq Y] \le \inf_g \Pr[g(F_S)\neq Y].
\]
命题得证。 \(\square\)

**解释。** 如果有无限样本、无限计算、并且能学到 Bayes 最优决策，那么增加特征不会让最优风险变差。  
因此实验里出现的“加模型反而掉点”不可能是这个命题的反例；它只能来自有限样本、有限假设类或优化误差。

### 命题 2：互信息对特征扩充单调不减

若 \(S\subseteq T\)，则
\[
I(F_T;Y)=I(F_S;Y)+I(F_{T\setminus S};Y\mid F_S)\ge I(F_S;Y).
\]

**证明。** 直接由互信息链式法则与条件互信息非负性得到。 \(\square\)

**解释。** 在总体意义上，更多模型不会减少“可用的标签信息”。但“可用信息变多”并不等于“有限样本下的测试精度必然变好”。

### 命题 3：边际信息增益的精确分解

对候选模型 \(m\notin S\)，其相对当前子集 \(S\) 的边际标签信息为
\[
\Delta_{\mathrm{MI}}(m\mid S)\triangleq I(F_m;Y\mid F_S).
\]

它满足精确恒等式
\[
I(F_m;Y\mid F_S)
= I(F_m;Y)-I(F_m;F_S)+I(F_m;F_S\mid Y).
\]

**证明。** 对联合量 \((Y,F_S)\) 分别按两种顺序展开：
\[
I(F_m;Y,F_S)=I(F_m;F_S)+I(F_m;Y\mid F_S),
\]
\[
I(F_m;Y,F_S)=I(F_m;Y)+I(F_m;F_S\mid Y).
\]
两式相等，移项即可得
\[
I(F_m;Y\mid F_S)
= I(F_m;Y)-I(F_m;F_S)+I(F_m;F_S\mid Y).
\]
\(\square\)

这个分解里三项分别对应：

- \(I(F_m;Y)\)：任务相关性  
- \(I(F_m;F_S)\)：冗余  
- \(I(F_m;F_S\mid Y)\)：类条件互补性/协同项

**重要说明。**

1. 这是严格恒等式，不是启发式。
2. 不能把第三项直接删掉写成“边际收益 = 相关性 - 冗余”，除非你额外假设类条件协同项较小。
3. 因而“低冗余”本身不够，“高相关 + 低冗余”才是正确方向。

这正好解释了你后面观察到的现象：  
**MAE 可能很“新颖”，但如果它对当前任务本身不强，边际收益仍然很小。**

---

## 3. 为什么有限样本下会出现“先升后降”

你原稿里最不稳的一句是把性能直接写成
\[
\mathrm{Perf}(S,n)\le I(F_S;Y)-\Phi(d_S,n).
\]
这个式子太强，也不标准。更稳妥的写法是从**风险分解**出发。

### 定理 4：测试风险变化可被精确分解为“Bayes 收益”与“额外代价”的竞争

设 \(h_{S,n}\) 是用 \(n\) 个目标样本在子集 \(S\) 上学到的实际分类器，记其真实风险为
\[
R_n(S)\triangleq \Pr[h_{S,n}(F_S)\neq Y].
\]
定义相对于 Bayes 风险的 excess risk
\[
E_n(S)\triangleq R_n(S)-R^\star(S)\ge 0.
\]

考虑一条嵌套选择路径
\[
S_0\subset S_1\subset \cdots \subset S_K,
\]
其中每一步只新增一个模型。则对第 \(k\) 步有
\[
R_n(S_k)-R_n(S_{k-1})
= \underbrace{E_n(S_k)-E_n(S_{k-1})}_{\text{有限样本/优化新增代价}}
-\underbrace{\big(R^\star(S_{k-1})-R^\star(S_k)\big)}_{\text{Bayes 收益}}.
\]

**证明。** 直接加减 \(R^\star(S_k)\) 与 \(R^\star(S_{k-1})\)：
\[
R_n(S_k)-R_n(S_{k-1})
= [R_n(S_k)-R^\star(S_k)]-[R_n(S_{k-1})-R^\star(S_{k-1})]
\]
\[
\qquad - [R^\star(S_{k-1})-R^\star(S_k)].
\]
即得结论。 \(\square\)

因此，第 \(k\) 个模型让测试风险变差，当且仅当
\[
E_n(S_k)-E_n(S_{k-1}) > R^\star(S_{k-1})-R^\star(S_k).
\]

这句话非常关键。它告诉我们：

- 在总体层面，新增模型的 Bayes 收益永远非负；
- 但在有限样本下，excess risk 的增长可以更快；
- 一旦新增模型的统计/优化代价超过 Bayes 收益，测试精度就会下降。

这个定理已经足以支撑“更多模型不一定更好”：

> **更多模型会使测试性能下降，当且仅当它带来的有限样本学习代价超过它带来的 Bayes 收益。**

换句话说，“加模型掉点”不是因为总体信息减少，而是因为学习代价增得更快。

### 推论 4.1：单步性能退化的充要条件

沿着任意嵌套路径 \(S_{k-1}\subset S_k\)，定义
\[
\delta_k^\star \triangleq R^\star(S_{k-1})-R^\star(S_k)\ge 0,
\qquad
c_k \triangleq E_n(S_k)-E_n(S_{k-1}).
\]
则
\[
R_n(S_k)>R_n(S_{k-1})
\quad\Longleftrightarrow\quad
c_k>\delta_k^\star.
\]

**证明。** 直接由定理 4 的恒等式移项得到。 \(\square\)

这个推论的意义在于：它把“为什么会掉点”精确地归结为一个量纲一致的比较，而不需要引入任何额外近似。

### 推论 4.2：一旦新增代价持续超过 Bayes 收益，后续模型会持续有害

如果存在某个 \(K\)，使得对所有 \(k>K\) 都有
\[
c_k>\delta_k^\star,
\]
则对所有 \(k>K\)，都有
\[
R_n(S_k)>R_n(S_{k-1}).
\]

**证明。** 由推论 4.1 逐步应用即可。 \(\square\)

这个推论比“更多模型通常不好”更准确。  
它说的不是所有新增模型都坏，而是：

> **一旦系统进入“新增代价主导”区间，后续继续加模型就会持续拉高测试风险。**

### 命题 5：在“边际收益递减 + 边际代价递增”下，必然存在最优模型数

记
\[
\delta_k^\star \triangleq R^\star(S_{k-1})-R^\star(S_k)\ge 0
\]
为第 \(k\) 个模型带来的 Bayes 收益，
\[
c_k \triangleq E_n(S_k)-E_n(S_{k-1})
\]
为第 \(k\) 个模型带来的新增代价。

若满足：

1. \(\delta_1^\star \ge \delta_2^\star \ge \cdots\)（边际收益递减）；
2. \(c_1 \le c_2 \le \cdots\)（边际代价递增）；

则存在阈值 \(K^\star\)，使得在 \(k\le K^\star\) 时新增模型有利，而在 \(k>K^\star\) 时新增模型有害。

**证明。** 定义
\[
a_k \triangleq c_k-\delta_k^\star.
\]
由假设 1 和 2 可知 \(a_k\) 单调不减。若存在某个 \(k\) 使 \(a_k\le 0\)，则定义
\[
K^\star=\max\{k:\ a_k\le 0\}.
\]
于是对所有 \(k\le K^\star\)，有
\[
R_n(S_k)-R_n(S_{k-1})=a_k\le 0,
\]
即测试风险不升；对所有 \(k>K^\star\)，因单调性有 \(a_k>0\)，故
\[
R_n(S_k)-R_n(S_{k-1})>0.
\]
命题得证。 \(\square\)

这就是“先升后降”的严格版本。  
它不依赖把性能强行写成某个互信息减罚项，而是直接从风险分解出发。

### 3.1 为什么新增代价会随着模型数上升

上面的定理和推论是严格的，但它们本身还没有说明 \(c_k\) 为什么会越来越大。  
对你的设定，一个很自然的机制是：**特征维度和可训练参数量随模型数快速增长，而样本量不变。**

在固定编码器、轻量头部训练的场景里，excess risk 来自三类来源：

1. **统计估计代价**：维度升高后，有限样本下估计方差上升；
2. **优化代价**：更高维、更大参数空间使训练更不稳定；
3. **表示复杂度代价**：新增模型带来更多与标签条件无关但仍被保留的输入信息。

因此，虽然严格数学上 \(c_k\) 的精确形式依赖具体学习算法，但在 few-shot 场景中，它随模型数上升是完全符合统计学习直觉的。

若进一步考虑你当前的 3 层 MLP 头部，则每增加一个 768 维模型，第一层就新增
\[
768\times 512 = 393{,}216
\]
个权重。  
也就是说，模型数按线性增加，但头部第一层参数量也在线性攀升；在样本数固定很小的时候，这会直接放大 \(c_k\)。

### 3.2 这个主线如何对应你的实验

你当前实验中，每个模型特征维度基本是 768。于是：

- 2 个模型：1536 维
- 4 个模型：3072 维
- 6 个模型：4608 维

若分类器第一层是 \(4608\to 512\)，仅第一层参数就有
\[
4608\times 512 = 2{,}359{,}296
\]
个权重。

在 10-shot 设置下：

- STL10 / SVHN 约 100 个标注样本
- GTSRB 约 430 个标注样本

这时 \(c_k\) 会非常大，因为：

1. 特征维度线性膨胀  
2. 分类器参数量急剧增加  
3. few-shot 下估计方差和优化不稳定性都很强

而在 full-data 设置下，\(n\) 从几千到几万不等，\(c_k\) 明显下降，于是保留全部信息的 Concat 开始占优。

这与本地实验完全一致：

- **SVHN**：10-shot 下 Gated \(33.01\%\) 明显优于 Concat \(22.47\%\)；但 full-data 下 Concat \(83.05\%\) 反超 Gated \(77.97\%\)。
- **GTSRB**：10-shot 下 MoE \(69.18\%\) 优于 Concat \(60.93\%\)；但 full-data 下 Concat \(91.18\%\) 反超 MoE \(90.98\%\)。
- **STL10**：few-shot 峰值在 4 模型，full-data 则基本持续上升到 6 模型。

因此更稳的结论不是“更多模型本质上一定有害”，而是：

> **在固定预训练编码器的多模型融合中，新增模型同时带来 Bayes 收益与有限样本代价。few-shot 下后者常常增长得更快，于是出现最优模型数；full-data 下该代价下降，最优模型数会右移。**

### 3.3 IB 在这条证明主线里的位置

在这条主线下，Information Bottleneck 的最佳角色不是“直接证明更多模型会变差”，而是解释**为什么新增代价会变大**。

更具体地说：

- 新增模型可能提高 \(I(Y;Z)\)，即对任务有用的信息；
- 但也可能显著提高 \(I(X;Z\mid Y)\)，即给定标签后仍被表示保留的输入相关复杂性；
- 在 few-shot 下，这部分复杂性会更强地反映到 generalization gap 中。

因此，IB 提供的是对 \(c_k\) 增长机制的解释，而不是替代定理 4 的证明本身。

这也是本文更稳的逻辑顺序：

1. 先用 Bayes 风险与 excess risk 的精确分解证明“为什么会掉点”；
2. 再用 IB 解释“为什么新增代价在 few-shot 下更容易主导”。

---

## 4. 原论文真正能为你提供什么理论支撑

### 4.1 Information Bottleneck：固定表示时，泛化差与 \(I(X;Z\mid Y)\) 有关

[Kawaguchi et al., ICML 2023](https://proceedings.mlr.press/v202/kawaguchi23a.html) 的关键贡献，不是简单地说“压缩有用”，而是给出了更严谨的结论：

当表示 \(Z_l=\phi_l(X)\) **独立于当前目标训练集**时，泛化差的主导项可由
\[
I(X;Z_l\mid Y)
\]
控制，而不是单纯的 \(I(X;Z_l)\)。

文中的固定编码器定理可概括为：
\[
\Delta(s)\ \lesssim\ \sqrt{\frac{I(X;Z_l\mid Y)}{n}}+\frac{1}{\sqrt n}+\frac{1}{n},
\]
其中 \(\Delta(s)\) 是 generalization gap，省略了与层、损失和有界性有关的常数项。

这件事对你尤其重要，因为你的主设定正是**冻结预训练编码器 + 目标任务上训练轻量头部**，属于它明确讨论的 fixed-feature 场景。

**为什么是 \(I(X;Z\mid Y)\) 而不是 \(I(X;Z)\)？**

因为
\[
I(X;Z)=I(X;Z\mid Y)+I(Y;Z).
\]
其中：

- \(I(Y;Z)\) 是任务相关信息，应尽量保留；
- \(I(X;Z\mid Y)\) 是在给定标签后仍然保留的剩余输入信息，可理解为“标签无关但被表示保留下来的信息”。

对融合问题而言，这意味着：

1. 新增模型可能确实带来一点 \(I(Y;Z)\)；
2. 但也可能同时大幅增加 \(I(X;Z\mid Y)\)；
3. 在 few-shot 下，后者会通过泛化差显著放大。

这恰好解释了为什么：

- full-data 下 Concat 往往恢复优势；
- few-shot 下 Gated / MoE 这类“隐式压缩”方法有时更好。

### 4.2 Learned representation：如果表示函数本身也在目标集上学，额外代价还要再加一项

同一篇论文的主定理还给出：

\[
\Delta(s)\ \lesssim\ \sqrt{\frac{I(X;Z_l^s\mid Y)+I(\phi_l^S;S)}{n}}+\cdots
\]

这里多出来的
\[
I(\phi_l^S;S)
\]
衡量的是**表示函数对训练集 \(S\) 的依赖程度**。

这个结论对你的门控器/路由器分析非常有用：

- **Concat**：主要是固定表示 + 学一个分类头，额外表示依赖项较小；
- **Gated / MoE / Attention Router**：表示函数本身是目标任务上学出来的，因而可能付出更大的 \(I(\phi_l^S;S)\) 代价；
- 数据极少时，这个代价会变得明显；
- 数据足够多时，路由器的统计代价下降，但若压缩损失了有用信息，Concat 仍可能赢。

因此，路由器在 few-shot 中的收益，并不应该被写成“路由更聪明”，更稳的表述是：

> **路由器在小样本下通过压缩表示来降低 excess risk，但它自身也会引入额外的数据依赖复杂度；当样本足够多时，保留完整信息的简单拼接往往重新占优。**

### 4.3 一个必须避开的坑：不能把“最小化 \(I(X;Z)\)”直接写成充分条件

Kawaguchi 一文在导论里专门指出：如果编码器也是用目标训练集学出来的，那么只看 \(I(X;Z)\) 是不够的。  
原因是编码器可以对训练集发生记忆，却仍然让 \(I(X;Z)\) 看起来不大。

因此，你的论文里最好不要写：

> “只要压低 \(I(X;Z)\)，就能保证泛化。”

更稳的写法是：

> “在固定表示或额外控制表示函数复杂度的条件下，information bottleneck 为泛化提供了一个有效但非唯一的控制路径。”

---

## 5. 为什么 CKA 是合理的冗余代理量，但不是互信息本身

### 5.1 Kornblith 等人的关键定理：高维表示下，过强的不变性会导致度量退化

[Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html) 证明了一个和你场景高度相关的事实：

> 当表示维度大于样本数时，任何对可逆线性变换不变的相似度统计量，都无法有意义地区分两个满秩表示。

这也是为什么 CCA 类指标在高维深度表示比较中容易失效。

其补充材料给出的证明可以几乎原样改写如下。

### 命题 6（改写自 Kornblith et al. Supplement, Theorem 1）

令 \(X,Y\in\mathbb R^{n\times p}\)，且 \(\mathrm{rank}(X)=\mathrm{rank}(Y)=n\)。  
若相似度统计量 \(s(\cdot,\cdot)\) 对第一自变量满足可逆线性变换不变性，即
\[
s(X,Z)=s(XA,Z)
\]
对任意满秩 \(A\in\mathbb R^{p\times p}\) 都成立，那么对任意 \(Z\) 都有
\[
s(X,Z)=s(Y,Z).
\]

**证明。** 取 \(K_X\) 为 \(X\) 的行空间零空间的一组基，\(K_Y\) 类似。构造
\[
X'=\begin{bmatrix} X \\ K_X\end{bmatrix},\qquad
Y'=\begin{bmatrix} Y \\ K_Y\end{bmatrix}.
\]
由于 \(\mathrm{rank}(X)=\mathrm{rank}(Y)=n\)，而 \(K_X,K_Y\) 分别补足了剩余 \(p-n\) 个方向，所以 \(X'\) 与 \(Y'\) 都是可逆的 \(p\times p\) 矩阵。令
\[
A=(X')^{-1}Y'.
\]
则 \(A\) 可逆，且由构造可知
\[
X A = Y.
\]
于是由可逆线性变换不变性，
\[
s(X,Z)=s(XA,Z)=s(Y,Z).
\]
命题得证。 \(\square\)

**含义。**  
如果一个表示相似度度量对可逆线性变换“太不敏感”，那么在高维小样本场景下，它会把太多本应不同的表示视为一样。对你的问题，这意味着：

- 不能简单依赖 CCA 一类度量；
- 需要使用在高维条件下仍可工作的度量。

### 5.2 CKA 的优势来自更合适的不变性

Kornblith 一文指出，CKA 保留的是：

- 对正交变换不变；
- 对各向同性缩放不变；

而不是对任意可逆线性变换都不变。

在线性核下，CKA 可写为
\[
\mathrm{CKA}(X,Y)=
\frac{\|Y^\top X\|_F^2}
{\|X^\top X\|_F\,\|Y^\top Y\|_F}.
\]

它度量的是两个表示诱导出的样本间相似性结构是否一致。

这对你的设定是合理的，因为：

1. 不同预训练模型的特征空间往往只差旋转、重缩放和坐标重排；
2. 你关注的是“这些模型是否在样本关系上表达了相近结构”；
3. 你使用 patch token 后先 PCA 到 256 维，也进一步缓解了原始高维退化与数值不稳定。

### 5.3 但 CKA 只能是冗余代理量，不能直接写成互信息

必须明确：

- HSIC / CKA 不是 mutual information estimator；
- 低 CKA 表示“线性/核相似结构不强”，不等于“信息上独立”；
- 高 CKA 表示“相似结构较强”，但不一定意味着对标签完全冗余。

因此，在论文里更稳的说法是：

> CKA 被用作**表征冗余的代理量**，而不是 \(I(F_m;F_S)\) 的无偏估计。

### 5.4 关于第三项：set-level 很难，pairwise 低阶近似更现实

如果目标是补充分析
\[
I(F_m;F_S\mid Y),
\]
那么最大的困难在于：这里的 \(F_S\) 是**整个已选集合的拼接表示**。  
一旦 \(S\) 包含多个模型，\(F_S\) 的维度会迅速增长，因此直接估计 set-level 条件互信息通常不现实。

更稳的做法是采用**低阶近似**：不用 whole-set 的
\[
I(F_m;F_S\mid Y),
\]
而改用 pairwise 条件项
\[
\frac{1}{|S|}\sum_{j\in S} I(F_m;F_j\mid Y).
\]

这条路线有文献依据。Brown et al.（JMLR 2012）把精确目标在低阶近似下写成
\[
I(X_k;Y)-\sum_{j\in S}I(X_k;X_j)+\sum_{j\in S}I(X_k;X_j\mid Y),
\]
这正是 CIFE 一类准则背后的形式。

对本文而言，更稳的表述是：

> 我们不直接估计 set-level 的 \(I(F_m;F_S\mid Y)\)，而是采用 Brown et al. 风格的 pairwise low-order approximation，用 \(\frac{1}{|S|}\sum_{j\in S} I(F_m;F_j\mid Y)\) 近似第三项的类条件修正作用。

在实现层面，本文进一步采用：

1. 先对每个模型特征做 PCA 降维；
2. 假设类条件高斯分布；
3. 估计
   \[
   I(F_m;F_j\mid Y)=\sum_y p(y) I(F_m;F_j\mid Y=y).
   \]

这仍然是近似，而不是严格一致估计，但它比直接估 set-level 条件互信息更可落地，也比单纯的标签条件相似性诊断更贴近理论式本身。

另一方面，如果只额外计算
\[
\widehat D_{\mathrm{cc}}(m,S)
= \sum_y \hat p(y)\frac{1}{|S|}\sum_{j\in S}\mathrm{CKA}(F_m^{(y)},F_j^{(y)}),
\]
那么这个量仍然只能被看作 class-conditional similarity diagnostic，不能替代上面的 pairwise conditional MI 近似。

---

## 6. 什么时候可以谈贪心选择，什么时候不能

### 6.1 经典贪心保证需要次模性，这不是自动成立的

若集合函数 \(G(S)\) 是**单调次模**的，则经典结果表明，大小受限的贪心选择满足
\[
G(S_{\mathrm{greedy}})\ge \left(1-\frac{1}{e}\right) G(S_{\mathrm{opt}}).
\]

[Krause, Singh, Guestrin, JMLR 2008](https://jmlr.org/papers/v9/krause08a.html) 通过互信息目标的次模性给出了相应保证，并重述了 Nemhauser 的经典结论。

### 6.2 但你的 \(G(S)=I(F_S;Y)\) 不能无条件写成“普遍次模”

这是你原稿里另一个容易被抓住的问题。更稳的写法应该是：

1. **总体上**，\(I(F_S;Y)\) 对 \(S\) 单调不减，这个是严格成立的；
2. **是否次模**，需要额外结构条件，例如高斯假设、近似条件独立、或可验证的 diminishing returns 经验规律；
3. 因而贪心的 \((1-1/e)\) 保证只能作为**带条件命题**使用，不能无条件宣称。

更安全的表述模板是：

> 在把模型子集效用近似为单调次模函数的条件下，贪心策略可获得经典的 \((1-1/e)\) 近似保证；本文将此作为算法设计动机，而不把它当作对所有深度特征分布都无条件成立的定理。

---

## 7. 从精确目标到可实现算法：哪些是定理，哪些是代理

### 7.1 真正的精确目标

从命题 3 出发，理想的第 \(k\) 步应选择
\[
m_k^\star=\arg\max_{m\notin S_{k-1}} I(F_m;Y\mid F_{S_{k-1}}).
\]

同时，是否继续加入模型，原则上应比较：

- Bayes 收益是否还大；
- finite-sample 新增代价是否已超过它。

这对应第 3 节的风险分解。

### 7.2 为什么不能直接估计这个量

在你的设定里，单模型通常是 768 维，6 模型拼接到 4608 维；而用于 CKA 的 patch token 原始维度更高。  
直接估计高维条件互信息 \(I(F_m;Y\mid F_S)\) 在样本量有限时非常不稳定，且计算代价高。

所以工程上必须转向 proxy。

### 7.3 更稳的写法：两项 tractable 主项 + 一个 low-order 条件修正 + 复杂度项

可以把当前方法写成四层：

1. **相关性代理** \(\widehat R(m,T)\)  
   可以用单模型验证精度，或 [LogME](https://proceedings.mlr.press/v139/you21b.html) 这类 transferability score。  
   但应写明：LogME 衡量的是最大证据（maximum evidence），是转移性能代理，不是互信息本身。

2. **冗余代理** \(\widehat D(m,S)\)  
   例如
   \[
   \widehat D(m,S)=\frac{1}{|S|}\sum_{j\in S}\mathrm{CKA}(F_m,F_j).
   \]
   它是 \(I(F_m;F_S)\) 的代理，而非等价量。

3. **第三项的 low-order 条件修正**  
   精确目标包含
   \[
   I(F_m;F_S\mid Y),
   \]
   但其 set-level 形式太难直接估计。  
   因而更现实的做法是采用 Brown et al. 风格的 pairwise 低阶近似：
   \[
   I(F_m;Y\mid F_S)
   =
   \underbrace{I(F_m;Y)-I(F_m;F_S)}_{\text{两项主项}}
   +
   \underbrace{I(F_m;F_S\mid Y)}_{\text{set-level 条件修正}}
   \]
   \[
   \approx
   I(F_m;Y)-\frac{1}{|S|}\sum_{j\in S}I(F_m;F_j)
   +\frac{1}{|S|}\sum_{j\in S}I(F_m;F_j\mid Y).
   \]
   在实现上，本文用 PCA + 类条件高斯估计器去近似最后这一项。

4. **复杂度/样本惩罚** \(\widehat C(m,S;n)\)  
   可以用新增输入维度、或新增分类器参数量表示。  
   在你的 3 层 MLP 里，每增加一个 768 维模型，第一层就新增
   \[
   768\times 512 = 393{,}216
   \]
   个权重，这个量在 few-shot 下绝对不能忽略。

### 7.4 推荐的论文写法

把“排序”和“停止”分开，会比把一切塞进一个伪定理里更稳。

**两项基线排序分数：**
\[
U_{\mathrm{order}}(m\mid S,T)
= \widehat R(m,T)^\alpha \cdot \big(1-\widehat D(m,S)\big)^\beta.
\]

这里乘法形式的意义不是“它精确等于条件互信息”，而是：

- 相关性是必要条件；
- 新颖性也是必要条件；
- 任何一项接近 0，都不应得到高排序。

这恰好符合你的经验观察：

- MAE 可能很新颖，但相关性弱，不能优先选；
- DINO 与 CLIP 虽然不是最不相似的一对，但二者都强，且互补，所以边际收益高。

**当前实现说明。**  
你现有的 MUMS / joint selection 主分支实际只用了前两项代理：

- \(\widehat R(m,T)\)：单模型 accuracy 或 LogME；
- \(\widehat D(m,S)\)：平均 CKA 冗余。

也就是说，原始 MUMS 分支并**没有**建模
\[
I(F_m;F_S\mid Y).
\]
因此更稳的表述不是“原始 MUMS 近似了完整的三项目标”，而是：

> **原始 MUMS 优化的是一个两项 surrogate，用它去近似真实边际收益中的主导部分。**

而本文新增的三项分支则采用
\[
\widehat C(m,S)=\frac{1}{|S|}\sum_{j\in S}\widehat I(F_m;F_j\mid Y)
\]
作为 low-order conditional correction，并得到一个 CIFE-style 评分：
\[
\mathrm{Score}_{3\text{-term}}(m\mid S)
= \widehat R(m)-\lambda\,\widehat D(m,S)+\eta\,\widehat C(m,S).
\]

这里仍然要强调：

- 这不是对精确条件互信息的无偏估计；
- 它是一个文献支持的低阶近似实现；
- 它的价值需要通过实验而不是纯理论宣称来检验。

**停止准则：**
\[
\text{仅当}\quad
U_{\mathrm{order}}(m_k\mid S_{k-1},T) > \tau(n,\Delta\mathrm{Comp})
\quad\text{时继续加入。}
\]

这里 \(\tau\) 可通过验证集确定，或设计为随样本量变小、复杂度变大而升高的阈值。

这个写法的优点是：

1. 与第 3 节的“收益-代价竞争”完全一致；
2. 不会把代理量误写成精确互信息；
3. 能自然解释为什么最优模型数依赖于数据量。

---

## 8. 用这套框架解释你已经观察到的实验事实

### 8.1 为什么 10-shot 下普遍在 4-5 个模型达峰

你当前汇总结果显示：

- STL10：4 模型达峰 \(95.71\%\)
- GTSRB：4 模型达峰 \(75.00\%\)
- SVHN：4 模型达峰 \(35.42\%\)
- Pets / EuroSAT / DTD：5 模型附近达峰
- Country211：1 模型 CLIP 即最优 \(29.62\%\)

这可以统一解释为：

- 前几步的 \(\delta_k^\star\) 大，特别是加入 DINO 时；
- 到第 4、第 5 个模型后，\(\delta_k^\star\) 快速变小；
- few-shot 下 \(c_k\) 继续上升，于是发生拐点。

### 8.2 为什么 CLIP + DINO 是最有价值的一步

你的实验里，加入 DINO 几乎总能带来最大的单步提升。  
这说明它满足两件事：

1. 单模型本身强，\(\widehat R\) 高；
2. 相对于 CLIP 并非完全冗余，\(\widehat D\) 不高到抵消收益。

如果用训练目标来理解：

- CLIP 更偏语言对齐语义；
- DINOv2 更偏自监督视觉结构；

那么二者的边际收益大，正是命题 3 里的“高相关 + 可接受冗余”的体现。

### 8.3 为什么 Country211 会出现“融合有害”

Country211 上，单模型 CLIP 最优，而加入别的模型普遍下降。  
这说明在这个任务上：

- CLIP 的任务相关性已经很高；
- 新增模型提供的 \(\delta_k^\star\) 很小；
- 但增加了显著的维度和估计代价。

换句话说，Country211 是“新增模型边际 Bayes 收益过小”的极端案例。

### 8.4 为什么 CKA-only 失败而 relevance + diversity 成功

你已观察到：

- 纯 CKA 选择在 35 次比较里大多输给原始顺序；
- 但 joint relevance + novelty 的 MUMS 风格选择明显更稳。

这与命题 3 完全一致。  
纯 CKA 只在近似
\[
I(F_m;F_S)
\]
这一项，却忽略了
\[
I(F_m;Y)
\]
这一项。

所以“最不相似”不等于“最有用”。  
低冗余只是必要条件，不是充分条件。

---

## 9. 论文里哪些话可以直接说，哪些最好不要说

### 9.1 可以直接说

1. **总体结论**  
   在总体层面，增加冻结编码器不会提高 Bayes 最优风险，也不会降低可用标签信息。

2. **有限样本结论**  
   在有限样本下，新增模型会同时带来 Bayes 收益与 excess risk 代价；当后者超过前者时，测试性能下降。

3. **模型选择原则**  
   候选模型的边际价值由任务相关性、与已选模型的冗余，以及类条件协同共同决定。

4. **CKA 的角色**  
   CKA 是表征冗余的合理代理量，尤其适合高维表示比较；但它不是互信息估计器。

5. **第三项的实现边界**  
   若实现第三项，更稳的路线是 pairwise low-order approximation，而不是直接估 set-level 的 \(I(F_m;F_S\mid Y)\)。  
   class-conditional CKA 至多是标签条件相似性的探索性诊断，不能替代 pairwise conditional MI 近似。

6. **few-shot / full-data 交叉现象**  
   交叉现象支持“有限样本代价是关键因素之一”的解释。

### 9.2 最好不要直接说

1. 不要写
   \[
   \mathrm{Perf}(S,n)=I(F_S;Y)-\Phi(d_S,n)
   \]
   或任何过强的等式。

2. 不要无条件写
   \[
   G(S)=I(F_S;Y)
   \]
   是次模函数。

3. 不要写 “CKA \(\approx\) mutual information”。

4. 不要写 “LogME 与 mutual information 成正比” 除非你明确写出所用的高斯/线性假设，并把它降级为近似动机。

5. 不要把 Platonic Representation Hypothesis 写成严格推导的理论前提。  
   它更适合作为**现象学背景**，不是本文主定理的逻辑支柱。

---

## 10. 建议你在论文里采用的最终主张

下面这段话可以直接作为论文的理论主张骨架：

> 对于冻结预训练编码器的多模型特征融合，增加模型在总体层面不会降低 Bayes 可达性能；然而在有限样本下，新增模型带来的统计估计代价与优化代价可能增长得快于其 Bayes 收益，从而导致测试性能出现先升后降。对任一候选模型，其边际标签信息可被精确分解为任务相关性、与已选模型的冗余以及类条件协同三项。由于高维 set-level 条件互信息难以直接估计，本文将其实现分成两层：一是“相关性 + 冗余”的 tractable 主项排序；二是 Brown et al. 风格的 pairwise low-order conditional correction。该框架解释了三个事实：其一，few-shot 下普遍存在最优模型数；其二，full-data 下简单拼接往往重新占优；其三，纯多样性选择会失败，因为低冗余并不等于高任务价值。 

如果你想把 MUMS 保留下来，推荐这样落笔：

> 原始 MUMS 不是条件互信息的闭式表达，而是其“高相关、低冗余”结构的工程近似实现；其作用是做两项主项排序，不是宣称对真实条件互信息进行精确估计。

如果你想把这次新增分析也写进去，建议补一句：

> 对于分解中的类条件项，本文不直接估计 set-level 的 \(I(F_m;F_S\mid Y)\)，而采用 Brown et al. 风格的 pairwise low-order approximation，并用 PCA + 类条件高斯估计器实现 \(\frac{1}{|S|}\sum_{j\in S} I(F_m;F_j\mid Y)\)；此外，class-conditional CKA 仅作为探索性的标签条件相似性诊断。

---

## 参考文献

1. Huh, M., Cheung, B., Wang, T., Isola, P. **Position: The Platonic Representation Hypothesis.** ICML 2024.  
   https://proceedings.mlr.press/v235/huh24a.html

2. Kawaguchi, K., Deng, Z., Ji, X., Huang, J. **How Does Information Bottleneck Help Deep Learning?** ICML 2023.  
   https://proceedings.mlr.press/v202/kawaguchi23a.html

3. Kornblith, S., Norouzi, M., Lee, H., Hinton, G. **Similarity of Neural Network Representations Revisited.** ICML 2019.  
   https://proceedings.mlr.press/v97/kornblith19a.html

4. Krause, A., Singh, A., Guestrin, C. **Near-Optimal Sensor Placements in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies.** JMLR 2008.  
   https://jmlr.org/papers/v9/krause08a.html

5. You, K., Liu, Y., Wang, J., Long, M. **LogME: Practical Assessment of Pre-trained Models for Transfer Learning.** ICML 2021.  
   https://proceedings.mlr.press/v139/you21b.html

6. Wood, D., Mu, T., Webb, A. M., Reeve, H. W. J., Lujan, M., Brown, G. **A Unified Theory of Diversity in Ensemble Learning.** JMLR 2023.  
   https://jmlr.org/papers/v24/23-0041.html

7. Radford, A. et al. **Learning Transferable Visual Models From Natural Language Supervision.** ICML 2021.  
   https://proceedings.mlr.press/v139/radford21a.html

8. Vimal K. B., Bachu, S., Garg, T., Narasimhan, N. L., Konuru, R., Balasubramanian, V. N. **Building a Winning Team: Selecting Source Model Ensembles using a Submodular Transferability Estimation Approach.** ICCV 2023.  
   https://openaccess.thecvf.com/content/ICCV2023/html/B_Building_a_Winning_Team_Selecting_Source_Model_Ensembles_using_a_ICCV_2023_paper.html

9. Luo, X. et al. **Less is More: On the Feature Redundancy of Pretrained Models When Transferring to Few-shot Tasks.** arXiv 2023.  
   https://arxiv.org/abs/2310.03843

10. Nanda, V. et al. **Diffused Redundancy in Pre-trained Representations.** NeurIPS 2023.  
   https://proceedings.neurips.cc/paper_files/paper/2023/hash/0c86142265c5e2c900613dd1d031cb90-Abstract-Conference.html

11. Brown, G., Pocock, A., Zhao, M.-J., Lujan, M. **Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection.** JMLR 2012.  
   https://www.jmlr.org/papers/v13/brown12a.html
