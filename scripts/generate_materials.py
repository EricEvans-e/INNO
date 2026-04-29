from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_json


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def generate_materials(output_dir: Path, docs_dir: Path) -> None:
    comparison = _load(output_dir / "comparison_metrics.json")
    profile = _load(output_dir / "optimized_profile.json")

    baseline = comparison.get("baseline", {})
    optimized = comparison.get("optimized", {})
    improve = comparison.get("improvement", {})

    latency_imp = float(improve.get("latency", 0.0)) * 100.0
    temporal_imp = float(improve.get("temporal_utilization", 0.0)) * 100.0

    ppt_outline = f"""# 路演PPT提纲（自动生成）

## 1. 赛题与目标
- 赛道二：3D异构CIM资源调度优化
- 目标：在保证映射率的同时降低时延并提升利用率

## 2. 方法总览
- 热子图精确分配（近似ILP子问题）
- swap/2-opt局部搜索 + 并行多起点
- 模拟退火全局搜索
- 自适应副本预算控制
- 分块量化 + 稀疏压缩
- 异构算力/带宽时序仿真

## 3. 核心结果
- baseline latency: {baseline.get('latency', 0):.2f}
- optimized latency: {optimized.get('latency', 0):.2f}
- latency improvement: {latency_imp:.2f}%
- temporal utilization improvement: {temporal_imp:.2f}%

## 4. 可解释分析
- expert调用频次、延迟分布、sub-cube争用、带宽利用率
- 冲突分数对比与消融

## 5. 提交价值
- 可复现脚本完整
- 参数可调、可扩展到更大模型
- 输出材料完备（图表 + JSON + 报告）
"""

    demo_script = f"""# 2分钟演示稿（自动生成）

大家好，我们的方案面向赛道二3D异构CIM调度优化。

第一，我们把MoE共现热点抽成热子图，做精确分配，先解决最关键冲突。
第二，在全局上采用并行多起点的swap/2-opt局部搜索，并叠加模拟退火，避免陷入局部最优。
第三，引入自适应副本预算与分块量化稀疏压缩，降低传输开销与冲突。
第四，时序仿真中显式建模子立方体异构算力和带宽，指标更贴近真实硬件。

在当前样例中，端到端时延从 {baseline.get('latency', 0):.0f} cycles 降到 {optimized.get('latency', 0):.0f} cycles，降低 {latency_imp:.2f}%。
同时时间利用率提升 {temporal_imp:.2f}%。

我们还输出了完整可解释剖析，包括每次推理时延分布和sub-cube争用热度，便于评审快速判断方案稳定性与可扩展性。
谢谢大家。
"""

    submission_checklist = f"""# 提交检查清单（自动生成）

## 代码与环境
- [ ] environment.yml 可创建环境
- [ ] requirements.txt 依赖完整
- [ ] main.py 一键跑通并生成 outputs/

## 实验与结果
- [ ] comparison_metrics.json 已生成
- [ ] optimized_profile.json 已生成
- [ ] 关键图表齐全（heatmap / mapping / gantt / latency / contention）

## 结果摘要
- baseline latency: {baseline.get('latency', 0):.2f}
- optimized latency: {optimized.get('latency', 0):.2f}
- latency improvement: {latency_imp:.2f}%
- temporal utilization improvement: {temporal_imp:.2f}%

## 演示材料
- [ ] PPT提纲完成
- [ ] 2分钟讲稿完成
- [ ] 提交包结构核对完成

## 备注
- 优化剖析样本数: {len(profile.get('latency_by_inference', []))}
- 记录的expert数量: {len(profile.get('expert_call_frequency', {}))}
"""

    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "PPT_OUTLINE.md").write_text(ppt_outline, encoding="utf-8")
    (docs_dir / "DEMO_SCRIPT_2MIN.md").write_text(demo_script, encoding="utf-8")
    (docs_dir / "SUBMISSION_CHECKLIST.md").write_text(submission_checklist, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate competition materials from output artifacts")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()

    generate_materials(args.output, args.docs)
    print("Generated docs/PPT_OUTLINE.md, docs/DEMO_SCRIPT_2MIN.md, docs/SUBMISSION_CHECKLIST.md")


if __name__ == "__main__":
    main()
