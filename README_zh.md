# GraphSkillAA

GraphSkillAA 是一个面向冻结语言模型的图结构外部技能优化框架。统一的技能图支持语义激活、失败归因、定向编辑、受影响样本重测和回滚。

![GraphSkillAA 框架](assets/graphskillaa-framework.png)

## 配置

| 项目 | 固定值 |
|---|---|
| 方法 | 完整版 GraphSkillAA |
| 教师 | `gpt-5.6-sol` |
| 学生 | `gpt-5.6-sol` |
| 基准 | SearchQA、LiveMathematicianBench、DocVQA |
| 优化 | 3 个 epoch，之后进行 held-out 评估 |

## 协议

对于每个基准，更新池与 held-out 测试集彼此独立。更新池被划分为由四个更新样本组成的固定分组。Held-out 测试 ID 单独存储在 `test_ids.json` 中；它们不会出现在任何更新分组中，也不会用于归因、patch 合成、Local Gate 或 Big Gate。

| 基准 | 更新样本 | Held-out 测试样本 | 指标 |
|---|---:|---:|---|
| SearchQA | 800 | 200 | 精确匹配准确率 |
| LiveMathematicianBench | 468 | 117 | 精确匹配准确率 |
| DocVQA | 800 | 200 | 硬准确率（`ANLS = 1`） |

## 仓库结构

```text
configs/                  实验配置
data/                     划分元数据和准备好的基准数据载荷
graphopt/                 图表示、归因、编辑和 Gates
scripts/prepare_data.py  基准数据准备
scripts/run_graphskillaa.py    实验启动器
scripts/summarize_results.py
scripts/verify_release.py
tests/                    项目测试
```

模型客户端、基准适配器、数据加载器、评估器、图优化器和 Gate 实现均已包含在本仓库中。运行时不需要单独安装其他技能优化框架。

## 安装

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

在 `.env` 中设置兼容 OpenAI 接口的 OpenLux 凭据：

```dotenv
OPENLUX_API_KEY=your_key_here
```

教师和学生均使用固定的 `gpt-5.6-sol:nitro` 路由及 medium reasoning effort。请勿提交 `.env`。

## 数据准备

本项目不重新分发原始基准文件。以下辅助脚本会下载固定版本的官方快照，并生成可运行的参考数据划分：

```bash
python scripts/prepare_data.py --dataset searchqa
python scripts/prepare_data.py --dataset livemathematicianbench
python scripts/prepare_data.py --dataset docvqa
```

你也可以按照文档手动准备文件结构，详见 [`data/README.md`](data/README.md)。在发起付费请求前，请检查数据载荷是否可读、完整、互不重叠，并与更新分组兼容：

```bash
python scripts/run_graphskillaa.py --dataset searchqa --check-only
python scripts/run_graphskillaa.py --dataset livemathematicianbench --check-only
python scripts/run_graphskillaa.py --dataset docvqa --check-only
```

下载原始数据之前，可以先检查 ID 元数据：

```bash
python scripts/run_graphskillaa.py --dataset searchqa --check-manifests
python scripts/run_graphskillaa.py --dataset livemathematicianbench --check-manifests
python scripts/run_graphskillaa.py --dataset docvqa --check-manifests
```

## 运行 GraphSkillAA

运行单个基准：

```bash
python scripts/run_graphskillaa.py --dataset searchqa
python scripts/run_graphskillaa.py --dataset livemathematicianbench
python scripts/run_graphskillaa.py --dataset docvqa
```

可以续跑部分完成的实验，而不会覆盖已经提交的状态：

```bash
python scripts/run_graphskillaa.py --dataset searchqa --resume
```

在不发起 API 调用的情况下查看锁定的命令：

```bash
python scripts/run_graphskillaa.py --dataset searchqa --print-command
```

默认输出路径为：

```text
outputs/graphskillaa/<dataset>/<run>/
```

重要产物包括 `config.json`、`trainer_state.json`、已提交的图快照、Gate 记录、逐样本 rollout 和 `final_test.json`。Held-out 结果仅用于评估，绝不会控制图的提交。

## 结果汇总

实验套件运行完成后：

```bash
python scripts/summarize_results.py --output-root outputs/graphskillaa
```

该脚本会汇总已有的 `final_test.json` 结果并输出汇总表。

## 本地验证

```bash
python scripts/verify_release.py
python -m compileall -q graphopt scripts
python -m unittest discover -s tests -v
```

这些检查会验证模型配置、仅包含更新样本的四元组 manifest、导入和命令构造。它们不会发起模型请求。

## 实验说明

- 只有图状态会发生变化；教师和学生的模型参数保持冻结。
- 初始图和所有 prompt 模板均包含在 `graphopt/envs/<benchmark>/` 下。
- Local Gate 和 Big Gate 在更新池上使用基准的硬评分。
- 每个最终图只会读取一次测试数据以进行最终评估；测试数据绝不会用于归因、编辑、Gate 决策或图选择。
- API 托管模型的行为可能随时间变化。请保存每次运行的原始响应、时间戳、解析后的部署字符串和图哈希。

有关基准数据的说明，请参阅 [`NOTICE.md`](NOTICE.md)。
