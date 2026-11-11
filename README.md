# 动态高斯 OT 致密化研究图谱

一个围绕“从 GHAP 出发，以最优传输规范动态 3D Gaussian Splatting 致密化”的中文研究网站。核心关注 Gaussian 在时间中的出生、合并、迁移、消失和身份连续性，而不是单独研究静态压缩。

## 本地预览

这是无构建依赖的静态站点：

```bash
cd dist
python3 -m http.server 8080
```

浏览器访问 `http://localhost:8080`。

## GitHub Pages 访问

发布地址：[动态高斯 OT 致密化研究图谱](https://zehnkoto.github.io/dynamic-gaussian-ot-densification-research/)。首次启用时，打开仓库 **Settings → Pages → Build and deployment → Source**，选择 **GitHub Actions**。

仓库已提供 `.github/workflows/pages.yml`，直接将静态站点 `dist/` 发布到 GitHub Pages，无需安装构建依赖。首次启用后，在 **Actions → Deploy research website to GitHub Pages → Run workflow** 选择 `main` 执行；也可重新运行首次失败的部署。以后 `main` 上的网页文件变更在 push 后自动部署。部署状态见 [Actions](https://github.com/Zehnkoto/dynamic-gaussian-ot-densification-research/actions/workflows/pages.yml)，首次部署完成前访问地址可能返回404。

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

## 新服务器实验入口

实验准备与执行说明见 [experiments/README.md](experiments/README.md)。环境、数据、缓存和输出均限定在 `/data3/store1/<用户名>/`；已完成新服务器基线、固定预算 CTD 对照、K=10000 的共享外观微调试验及离线 split 原型的预算／训练顺序稳定性检查；后续已完成预设 split 事件、评分与分辨率对照；E13 加入有界共享位置优化和固定分配 CTD 联合约束；E14 补均匀 L2 对照；E15 仅用拟合集校准偏离预算，4%预算匹配时CTD与L2的PSNR接近，指标仍有取舍；1%预算存在预设容差失败，尚未证明CTD的全面优势。E16 回到固定10000共享分组，仅改变代表协方差构造：矩匹配使PSNR提高约0.821dB、时间残差降低约20%，但LPIPS恶化约26.5%，不作为全面升级或直接叠加模块的依据。最新 E17／E17b 对冻结模型的20视角区域和教师对齐指标复算：矩匹配在教师相对像素误差下降时，教师与GT相对LPIPS都变差，指向当前训练目标与感知指标的取舍。E18在相同分组、形状与训练预算下，只给76个拟合时刻的教师损失加入预先固定的0.1 LPIPS项；三个外观训练顺序中两种形状的GT LPIPS都改善约2.0%／6.0%，但SSIM、时间残差及教师留出时刻像素损失略变差，矩匹配的GT LPIPS仍明显劣于平均协方差。继续保留诊断分支，详见实验记录。E19已在独立的 D-NeRF jumpingjacks 场景复验形状与感知训练项：矩匹配仍提高PSNR／降低时间残差却恶化LPIPS；固定感知项仍降低LPIPS，却小幅损害PSNR、SSIM和时间残差。单新底模／单外观顺序，仍依赖教师。旧服务器结果未导入或验证，E20开始检验拟合窗口内成员更新：同K=9276、每窗口最多2%切换，完整几何与仅位置距离均使PSNR小幅提高约0.046～0.048dB，但仅位置对照略好；当前未归一化完整代价里协方差项仅约0.064%，跨窗口时间残差约升2%。仍只是离线教师依赖诊断，完整 4DGS 自适应致密化尚未实现。

E21只读尺度审计显示：旧代理中协方差Frobenius项仅占固定分组成本的0.0644%，而量纲一致的Gaussian W2/Bures协方差项占21.08%；E20窗口换组降低位置代价约27%，却让Bures形状代价升约5%。这还不是W2驱动分配，也不是GHAP完整CTD或训练期自适应split。见 [E21实验记录](experiments/README.md)。

E22首次在官方4DGS的canonical高斯与deformation可训练的路径执行同预算split小试：晚期完成模型共同预热后，对照不split、梯度top64与跨时间W2失真top64。W2选点比梯度选点少损失，但两个split组均未胜过不split；这只是原始Adam状态缺失的单场景晚期继续训练，不是完整训练期动态OT致密化结果。有效v1c及无效尝试说明见 [E22实验记录](experiments/README.md)。

当前主线已明确为**完整动态教师到预算受限、可独立运动的学生**：先让学生不依赖教师逐时刻几何完成时间序列渲染，再以拆分前后的教师—学生分布和渲染收益决定是否增加容量，并与简单选点按相同预算比较。原生4DGS从头致密化作为后续扩展；E22的W2压缩失真选点不等于GHAP CTD拆分收益。路线修订与既有负结果见 [实验记录](experiments/README.md)。
