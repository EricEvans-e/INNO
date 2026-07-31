# 没头绪｜赛道二 / 赛题二提交说明

## 参赛信息

- 队名：没头绪
- 成员：祝铭堃
- 学校：南开大学
- 指导教师：无
- 提交内容：源码、项目介绍文档、论文、PPT
- 视频：本包暂不包含

## 本包的推荐提交内容

1. `source/`：完整源码、样例模型、样例轨迹、测试、配置和依赖说明。
2. `formal_result/`：V5 primary 正式推荐结果、`solution.json`、运行清单和严格验证报告。
3. `paper/`：设计报告 PDF 和 LaTeX 源码。
4. `presentation/`：最终答辩 PPTX 文件。
5. `docs/`：项目介绍、技术复现指南和 PPT 制作方案。
5. `evidence/`：测试和验证摘要。

## 重要口径

本包的正式推荐结果为 V5 primary：在项目自定义 `trace_balanced` 轨迹上，模拟延迟从 539,729 cycles 降到 234,436 cycles，降低 56.56%，严格容量比例 1.9862，`mapping_rate=1.0`，验证器通过。

V5.1 的 231,672 cycles 是候选结果，不能在没有官方轨迹复验的情况下作为正式最佳结果宣传。

当前结果是性能模型模拟，不是具体 CIM 芯片实测。官方激活轨迹到达后，应重新运行解析器、映射器、模拟器和验证器，并替换报告中的对应数据。

## 关于视频

根据赛道交流群沟通，本阶段可暂不提交视频，因此本包不包含 MP4。官方赛道页面曾列出演示视频要求，提交时应保留群聊确认记录；若报名系统出现强制上传项，需要向组织方再次确认，或补充 PPT 录屏作为备用材料。

## 快速入口

- 源码说明：`docs/REPRODUCTION_AND_HANDOFF_GUIDE.md`
- 项目介绍：`docs/PROJECT_INTRODUCTION.md`
- PPT 方案：`docs/PPT制作方案.md`
- 最终 PPT：`presentation/没头绪-赛道二-赛题二-项目介绍PPT.pptx`
- 论文 PDF：`paper/cim_3d_scheduler_design_report.pdf`
- 正式结果：`formal_result/solution.json`
- 验证报告：`formal_result/validation_report.json`
