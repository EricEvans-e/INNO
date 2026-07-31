# 面向 MoE 大模型推理的 3D 异构 CIM 资源调度优化

本目录包含赛道二/赛题二设计报告的独立 LaTeX 源码。正文使用中性双栏学术版式，视觉上参考提供的计算机学报模板，但不包含期刊卷期、DOI、基金、收稿日期或投稿作者信息。

## Overleaf

1. 将整个 docs/paper 目录上传为一个 Overleaf 项目。
2. 在 Menu 中将 Compiler 设为 XeLaTeX。
3. 将 main.tex 设为主文件并编译。

## 本地构建

在本目录依次执行命令：xelatex -interaction=nonstopmode -output-directory=output main.tex；biber output/main；再执行两次 xelatex -interaction=nonstopmode -output-directory=output main.tex。

最终 PDF 为 output/cim_3d_scheduler_design_report.pdf，共 8 页。如需渲染检查，执行：pdftoppm -png -r 150 output/cim_3d_scheduler_design_report.pdf output/rendered/page

## 结果来源

- V5 primary：原工作区的 outputs/submission_v5_primary/comparison_metrics.json。
- V5.1 candidate：原工作区的 outputs/submission_v5_1_candidate/comparison_metrics.json。
- 大规模验证：原工作区的 outputs/large_scale_v5/run_strict/comparison_metrics.json。

论文首页按学术论文格式编排，不放置队名、指导教师和竞赛完成日期。首页包含题目、作者、单位和通讯邮箱；作者为祝铭堃，单位为南开大学，邮箱为 2312309@mail.nankai.edu.cn。

## 参考文献

参考文献已经过原始页面或 DOI/arXiv 入口核验，核验记录见 `REFERENCE_VERIFICATION.md`。赛事题面作为本地资料列出，其他学术条目均保留可检索 URL。
