# 动态高斯 OT 致密化研究图谱

一个围绕“从 GHAP 出发，以最优传输规范动态 3D Gaussian Splatting 致密化”的中文研究网站。核心关注 Gaussian 在时间中的出生、合并、迁移、消失和身份连续性，而不是单独研究静态压缩。

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

核心命题是把逐帧、经验性的 clone / split / prune 改写成带时间一致性和模型复杂度的时空变阶测度优化：

`min Σ_t [D_CTD(ν_t, μ_t) + β L_render,t + λ N(t)] + γ R_time`

网站刻意区分四件事：

1. GHAP/CTD 提供静态分布级约简基础；
2. 动态组件代价同时比较几何、速度、deformation 或轨迹；
3. 时间窗口内的边际收益决定何时 split、merge、relocate 或 prune；
4. continuity / unbalanced OT 只在质量语义明确时引入，不能把 primitive 生灭直接当作物理源汇。

渲染不透明度不是物理质量；增加一个 Gaussian primitive 只是增加离散自由度，不等于产生物质。

## GHAP 的继承边界

GHAP 已经完成并验证的是：把训练后冗余的静态 3DGS 视为高阶 GMM，在 KD-tree 分块内最小化 composite transportation divergence，约简到预先指定的保留率，再微调颜色和 opacity。其最近代表元硬分配来自 CTD-GMR 的化简，不需要为了“更像 OT”而强行替换成 Sinkhorn。

本项目不重新验证上述静态结论。新增研究命题全部围绕动态场景：

1. 将 GHAP 从单个静态 GMM 推广到时间相关的 GMM 序列；
2. 让 assignment、barycenter 和 Gaussian identity 在滑动窗口内保持一致；
3. 用窗口累计 CTD 收益决定动态 split / merge，并自适应选择每个时刻的局部阶数；
4. 区分暂时遮挡、拓扑变化、表示容量生灭和真实物理质量源汇。

因此，官方 GHAP 结果的复现只用于锁定静态极限；真正需要验证的是逐帧 GHAP 为什么不够，以及窗口一致、轨迹级变阶能否改善动态重建。

## 主要资料

页面中的外部链接均指向论文主页、出版社页面或作者官方代码仓库。关键入口包括原始 3DGS、3DGS-MCMC、SteepGS、GHAP、MMGS、Gaussian-mixture Wasserstein 距离、Benamou–Brenier 动态 OT、JKO 梯度流、unbalanced OT、LagrangianSplats 与保守粒子 split/merge。
