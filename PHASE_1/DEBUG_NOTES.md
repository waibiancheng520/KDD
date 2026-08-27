# DataAgent 调试笔记（task_11 及基线调优）

> 记录从零跑通 baseline、修复工程缺陷、到开始调优的完整过程。

## 一、环境搭建

| 项 | 结论 |
|---|---|
| 模型 | Google AI Studio 免费额度，OpenAI 兼容端点 `https://generativelanguage.googleapis.com/v1beta/openai/` |
| 关键坑 | Gemini 按**请求出口 IP** 判断地区。国内 IP → `User location is not supported`，必须用美国/日本/新加坡等支持地区的代理节点 |
| 代理坑 | `all_proxy=socks://...` SDK 不支持，需强制 `http://` 协议 |

## 二、修复的 6 个 baseline 工程缺陷

这些都是通用问题，对所有任务有效。

| # | 文件 | 缺陷 | 症状 | 修复 |
|---|---|---|---|---|
| 1 | react.py | JSON 严格解析 | 模型写多行代码全部报错 | `JSONDecoder(strict=False)` |
| 2 | react.py | **把模型自编的假观察喂回历史** | **编造数据（幻觉）** | `_build_messages` 只回喂真实动作 |
| 3 | react.py | 上下文无限膨胀 | 撞爆 token 配额（429） | 老步骤截断，最近 3 步保留完整 |
| 4 | model.py | 每次新建 API 客户端 | 连接泄漏→CLOSE-WAIT 卡死 10 分钟 | 客户端建一次复用 + 超时 |
| 5 | model.py | 空响应/429 直接崩任务 | 全部步骤白跑 | 空响应转可恢复错误；429 按 retryDelay 退避重试 |
| 6 | runner.py | **multiprocessing 队列死锁** | 数据>64KB 时永久卡死 | 先 `queue.get(timeout)` 再 join |

**最有价值的两个**：#2（幻觉，让答案全假）和 #6（死锁，让程序永久卡住），都不是显而易见的 bug。

## 三、幻觉 bug 深度解析（核心一课）

Gemini 在**一次回复里自导自演**了整段假对话：假装看到不存在的 `data.sqlite`、假装跑 SQL、编出 `P001 John Doe` 等玩具数据。baseline 的 `_build_messages` 把这段含假观察的原始输出原样喂回历史，模型便信以为真、越编越顺。

> **教训**：ReAct 里"模型思考"和"工具执行"必须严格分离。绝不能让模型自己写"工具返回了什么"，也不能把它写的假结果喂回上下文。

## 四、调优实验记录（--limit 5）

| 版本 | 改动 | 准确率 | F1 | 结论 |
|---|---|---|---|---|
| baseline | 修完 6 个 bug | 60% | 0.638 | 起点 |
| v2/v3 | 加规则14"先枚举所有表再选" | 40% | 0.40/0.42 | ❌ 无效，且耗步数。多看≠会选 |
| v4+ | 规则14改"给判断依据"（实体表vs事实表）+ 规则16输出格式 | 待测 | — | 进行中 |

### 关键方法论
1. **一次只改一处**，用新 run_id，跑完立即打分存档
2. `--limit 5` 波动大（1题=20%），确认效果要跑 20+ 题
3. **给判断依据 > 要求额外动作**：规则该告诉它"怎么选"，而非"多做一步"
4. 提示词要**跨领域通用**，别举具体领域的例子（否则只覆盖几道题）

## 五、错题的三类病因（都不是"编数据"或"格式错"）

| 题 | 病因 | 类型 |
|---|---|---|
| task_11 | SEX/Diagnosis 该从 Patient 主表取；该用 INNER JOIN（只留能关联上的） | 关联语义 |
| task_25 | cost 关联路径选错（经 budget 关联到 event） | 关联语义 |
| task_19 | 答案对，但"full name"给1列，gold要2列 | 输出格式 |

> agent 每步查询都真实，但**关联策略/取数路径选错**，结果就错。这是数据 agent 最核心的难点。

## 六、工具

- `scripts/score.py`：对比 prediction.csv 与 gold.csv 打分。列名不比、行序不比、数字按值比、空值归一化。指标：准确率（主）+ 平均行级 F1（看趋势）。
- `data/public/output/task_<id>/gold.csv`：50 道题的标准答案，可批量评分。

## 七、下一步方向（按优先级）

1. 验证通用版规则 14-16 的效果（跑 --limit 5，再 --limit 20 确认）
2. 增强工具：`read_json/read_csv` 直接返回列名+行数+样例，省探索步数
3. 加 reflection：answer 后让模型自查一次再结束（对付"自信答错"）
4. 稳定后跑全量 50 题拿可靠基线

---

# 附录：问题 → 解决 → 文件 完整清单

## A. 环境类（改配置）

| 问题 | 解决 | 文件/位置 |
|---|---|---|
| 模型名 `gemini-2.0-flash` 已下线 | 改用 `gemini-3.7-flash` | configs/react_baseline.local.yaml |
| 代理 `socks://` 协议 SDK 不支持 | 运行时强制 `http_proxy/https_proxy/all_proxy=http://127.0.0.1:7897` | 运行命令 |
| Gemini 报 "User location is not supported" | 代理切美国节点，需全局模式让 DNS/流量都走美国 | Clash Verge 设置 |
| run_id 目录冲突、超时不够 | run_id 每次换新；task_timeout_seconds 调大 | configs/react_baseline.local.yaml |

## B. 工程缺陷类（改代码）

| # | 问题 | 症状 | 解决 | 文件 |
|---|---|---|---|---|
| 1 | JSON 严格解析，不容忍字符串内换行 | 模型多行代码全报错 | `JSONDecoder(strict=False)` | react.py:42 |
| 2 | 把模型自编的假观察喂回历史 | 编造数据（幻觉） | `_build_messages` 只回喂真实动作 | react.py:100 |
| 3 | 上下文无限膨胀 | 撞爆 token 配额(429) | 老步骤截断700字符，最近3步保留4000 | react.py:20 + prompt.py:77 |
| 4 | 每次调用新建 API 客户端 | 连接泄漏→CLOSE-WAIT 卡死 | 客户端建一次复用 + timeout=60,max_retries=2 | model.py:58 |
| 5a | 空响应直接抛异常 | 整任务崩溃 | 空响应转可恢复的 `__empty_response__` | model.py:110 |
| 5b | 429 限流直接崩任务 | 全部白跑 | 按服务器 retryDelay 退避重试(最多3次) | model.py:15 |
| 6 | multiprocessing 队列死锁 | 数据>64KB 永久卡死 | 先 queue.get(timeout) 再 join | runner.py:145 |

## C. 调优类（改提示词/工具）

| 问题 | 解决 | 文件/位置 |
|---|---|---|
| 关联路径/取数选错(task_11,25) | 通用规则14-15：属性从主表取、默认 INNER JOIN | prompt.py 规则14-15 |
| 输出列拆分与 gold 不同(task_19) | 规则16：一属性一列、按提问顺序、不合并不拆分 | prompt.py 规则16 |
| 步数预算不透明 | 每次观察附带 Step X/N, 剩 M 步 | prompt.py:85 |
| 无法量化效果 | 自动评分脚本 | scripts/score.py（新增） |
| 调试看不见进度 | 实时日志(每步+每次API耗时) | model.py:83 + react.py:136 |

## 改动/新增文件汇总（7 个）

| 文件 | 类型 |
|---|---|
| src/data_agent_baseline/agents/react.py | 改（bug 1,2,3 + 日志） |
| src/data_agent_baseline/agents/model.py | 改（bug 4,5 + 日志） |
| src/data_agent_baseline/agents/prompt.py | 改（bug 3 截断 + 规则14-16 + 步数预算） |
| src/data_agent_baseline/run/runner.py | 改（bug 6 死锁） |
| scripts/score.py | 新增（评分工具） |
| DEBUG_NOTES.md | 新增（本文档） |
| configs/react_baseline.local.yaml | 新增（本地配置） |
