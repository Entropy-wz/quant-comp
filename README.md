# quant-comp

2026 量化交易研究大赛 — GBDT 主线工作区（LightGBM / 时序 API 策略包）。

本仓库只包含代码、配置与文档；**不含**官方大数据包 `public_release_20260630/` 与训练产物。

## 目录

| 路径 | 说明 |
|------|------|
| `workspace/` | 训练脚本、特征、策略包、测试 |
| `docs/superpowers/` | 设计与分阶段计划 |
| `competition_description.md` | 赛题说明摘录 |

## 本地使用

1. 自行准备官方 `public_release` 数据，并在 `workspace/configs/paths.yaml` 指向数据根目录  
2. 安装依赖：`pip install -r workspace/requirements-dev.txt`  
3. 在 `workspace/` 下设置 `PYTHONPATH=src` 后运行训练 / 导出 / runner gate  

详见 `workspace/README.md` 与 `docs/superpowers/plans/`。

## 说明

- `.gitignore` 已排除官方数据、模型二进制、公榜 CSV 与本地日志  
- 私榜提交需自行 `export_strategy_v1` 生成 `strategy_v1/model/`  
