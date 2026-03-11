# 算法图论复习 LaTeX 项目

## 项目结构

```
算法图论/
├── CLAUDE.md          # 本文件
├── main.tex           # 主文件
├── chapters/          # 章节目录
│   ├── basics.tex     # 图论基础
│   ├── traversal.tex  # 图的遍历
│   ├── shortest.tex   # 最短路径
│   ├── mst.tex        # 最小生成树
│   └── ...
└── figures/           # 图片目录
```

## 编译说明

**每次修改完成后，必须执行以下命令编译 LaTeX：**

```bash
cd /Users/liuyibo/Desktop/d/算法图论 && xelatex main.tex
```

如需生成目录和交叉引用，需编译两次。

## 例题完成规则

**重要：** 如果某个概念或定理有对应的例题（example），必须确保例题完整：
- 例题应包含题目描述、图示（用 TikZ 绘制）、解答过程
- 如果发现例题只有题目没有解答，需要补充完整的解答
- 例题解答要清晰、步骤完整

## 绘图指南

本项目使用 **TikZ** 绘制图论相关图形。

### 常用 TikZ 示例

#### 1. 简单无向图
```latex
\begin{tikzpicture}
  \node[circle, draw] (1) at (0,0) {1};
  \node[circle, draw] (2) at (2,0) {2};
  \node[circle, draw] (3) at (1,1.5) {3};
  \draw (1) -- (2) -- (3) -- (1);
\end{tikzpicture}
```

#### 2. 带权图
```latex
\begin{tikzpicture}
  \node[circle, draw] (A) at (0,0) {A};
  \node[circle, draw] (B) at (3,0) {B};
  \node[circle, draw] (C) at (1.5,2) {C};
  \draw (A) -- node[below] {5} (B);
  \draw (A) -- node[left] {3} (C);
  \draw (B) -- node[right] {2} (C);
\end{tikzpicture}
```

#### 3. 有向图
```latex
\begin{tikzpicture}[->, >=stealth]
  \node[circle, draw] (1) at (0,0) {1};
  \node[circle, draw] (2) at (2,0) {2};
  \node[circle, draw] (3) at (1,1.5) {3};
  \draw (1) -> (2);
  \draw (2) -> (3);
  \draw (3) -> (1);
\end{tikzpicture}
```

#### 4. 树结构
```latex
\begin{tikzpicture}
  \node[circle, draw] (root) at (2,3) {R};
  \node[circle, draw] (l) at (1,2) {L};
  \node[circle, draw] (r) at (3,2) {R};
  \node[circle, draw] (ll) at (0.5,1) {LL};
  \node[circle, draw] (lr) at (1.5,1) {LR};
  \draw (root) -- (l) -- (ll);
  \draw (l) -- (lr);
  \draw (root) -- (r);
\end{tikzpicture}
```

### 需要的宏包

```latex
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, calc}
```

## 章节内容规划

1. **图论基础** - 图的定义、表示方法（邻接矩阵、邻接表）
2. **图的遍历** - DFS、BFS 及其应用
3. **最短路径** - Dijkstra、Bellman-Ford、Floyd-Warshall
4. **最小生成树** - Prim、Kruskal 算法
5. **拓扑排序** - DAG、AOV 网
6. **连通性** - 强连通分量、割点、桥
7. **网络流** - 最大流、最小割

## 注意事项

- 使用 XeLaTeX 编译以支持中文
- 图片统一放在 `figures/` 目录
- 每个章节独立文件，便于维护
