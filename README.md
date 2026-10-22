# 动态高斯约简研究图谱

一个围绕“物理守恒的动态 3D Gaussian Splatting 约简”的中文研究网站。它不是论文结论页，而是一份可持续更新的研究判断、数学框架、实验路线和文献台账。

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

核心命题是让高斯约简算子 `R` 与动力学演化 `Φ` 近似可交换：

`R(ΦΔt(Gt)) ≈ ΦΔt(R(Gt))`

网站刻意区分三件事：

1. Bures/Wasserstein 几何负责相似度与匹配；
2. 质量、动量和二阶矩约束负责保守代表元；
3. 含时 PDE 与交换损失负责动态一致性。

渲染不透明度不是物理质量；若要谈守恒，需单独定义物理通道。

## 主要资料

页面中的外部链接均指向论文主页、出版社页面或作者官方代码仓库。关键入口包括 GHAP、LagrangianSplats、PIDG、TC3DGS、PINF、FluidGS、GaussFluids、MEGA、PhysGaussian、CTD-GMR 与 Benamou–Brenier 动态最优输运。

