# 最优传输驱动的高斯致密化研究图谱

一个围绕“以最优传输规范 3D Gaussian Splatting 自适应致密化”的中文研究网站。它不是论文结论页，而是一份可持续更新的研究判断、数学框架、实验路线和文献台账。

## 本地预览

这是无构建依赖的静态站点：

```bash
cd dist
python3 -m http.server 8080
```

浏览器访问 `http://localhost:8080`。

## 更新内容

- 页面结构与研究叙述：`dist/index.html`
- 视觉样式：`dist/styles.css`
- 交互逻辑：`dist/app.js`
- 文献、代码和标签：`dist/data/papers.js`

新增论文时，在 `papers.js` 的 `PAPERS` 数组中追加一项即可；筛选器和文献计数会自动更新。

## 研究口径

核心命题是把经验性的 clone / split / prune 改写成带模型复杂度的变阶测度优化：

`min_(N,θ) D_OT(ν, μ_(N,θ)) + β L_render(θ) + λN`

网站刻意区分四件事：

1. 目标测度 `ν` 定义“当前表示缺少什么”；
2. Wasserstein/GMM 距离度量全局表示失真；
3. 边际变分收益决定何时 split、merge、relocate 或 prune；
4. 动态 OT 与连续性方程负责时间一致性，具体 PDE 属于额外物理层。

渲染不透明度不是物理质量；增加一个 Gaussian primitive 只是增加离散自由度，不等于产生物质。

## GHAP 的继承边界

GHAP 已经完成并验证的是：把训练后冗余的静态 3DGS 视为高阶 GMM，在 KD-tree 分块内最小化 composite transportation divergence，约简到预先指定的保留率，再微调颜色和 opacity。其最近代表元硬分配来自 CTD-GMR 的化简，不需要为了“更像 OT”而强行替换成 Sinkhorn。

本项目不重新验证上述结论。新增研究命题是：

1. 将 GHAP 作为周期性投影，规整经验 densification 产生的候选；
2. 用 CTD 失真与模型复杂度自动选择局部阶数，替代固定保留率；
3. 在动态 3DGS 中加入跨时间的 assignment、barycenter 与质量一致性。

因此，官方 GHAP 结果的复现只用于锁定基线；真正需要验证的是自适应、训练期和动态扩展。

## 主要资料

页面中的外部链接均指向论文主页、出版社页面或作者官方代码仓库。关键入口包括原始 3DGS、3DGS-MCMC、SteepGS、GHAP、MMGS、Gaussian-mixture Wasserstein 距离、Benamou–Brenier 动态 OT、JKO 梯度流、unbalanced OT、LagrangianSplats 与保守粒子 split/merge。
