# DataAgent 调试笔记

> 从跑通 baseline 到 70.8% 的完整记录：修了什么、为什么、以及**哪些尝试失败了**。
> 失败记录和成功记录同样重要——它们标出了这个模型的能力边界，避免重复投入。

---

## 〇、调试时间线（从"跑 50 个 task"到 70.8%）

| # | 遇到的问题 | 处理 | 结果 |
|---|---|---|---|
| 1 | 首次跑 50 题，**全部失败**：`User location is not supported` | Gemini 按出口 IP 判地区；实测挂代理后节点仍在不支持地区 | 阻塞 |
| 2 | 代理切换后仍不通 | **换 DeepSeek**（国内直连，无需代理） | 跑通，v7 = 44% |
| 3 | 11 题"未产出答案" | 查 trace 发现 20 步全是 `__error__`：**解析器吃不下 JSON 前的自然语言前言** | 50%，未产出 11→6 |
| 4 | 姓名被拼成一列、数值被四舍五入 | 加拆列规则 11 + 全精度规则 12 | v8 = 54%（但 prompt 变长导致新增超时，净赚有限） |
| 5 | 仍有题空转 | 查 trace：**模型陷入逐字重复的死循环** | v9 = 60%，加反重复守卫 |
| 6 | 探索耗步数多、手写 join 易错 | 新增 `profile_context`（一步画像）+ `query_files`（DuckDB 跨源 SQL） | v12 = 62% |
| 7 | 想根除格式问题 | 上原生 Function Calling | **v13 = 58%，退步**→ 查明丢了链式推理，回退为开关 |
| 8 | 系统性找瓶颈 | 派 2 个 subagent 深挖代码与失败 trace + 跨版本方差分析 | 定位 3 个机制缺陷 |
| 9 | 落地机制修复 | 文档分页 + 头尾截断 + 强制收尾轮 + 兜底打捞 + 反重复硬拒绝 + 问题重述 | **v14 = 70.8%，未产出归零** |
| 10 | 两题怎么调都错 | 用数据核对，确认**文档与 gold 互相矛盾** | 排除并建立双口径评分 |

---

## 一、成绩演进

全部 50 题（除最后一行外），DeepSeek `deepseek-chat`，temperature 0。

| 版本 | 关键改动 | 正确 | 准确率 | F1 | 未产出 |
|---|---|---:|---:|---:|---:|
| v7_generic | DeepSeek 基线 | 22/50 | 44.0% | 0.490 | 11 |
| *(未存档)* | **修 JSON 解析器**（容忍前言文本） | 25/50 | 50.0% | — | 6 |
| v8_promptfix | 拆列规则 + 全精度规则 | 27/50 | 54.0% | 0.549 | 10 |
| v9_loopguard | 反重复守卫 + 强制收尾提示 | 30/50 | 60.0% | 0.631 | 5 |
| v12_duckdb | `profile_context` + `query_files`(DuckDB) | 31/50 | 62.0% | 0.671 | 4 |
| v13_funccall | 原生 Function Calling | 29/50 | 58.0% | 0.607 | 7 |
| **v14_ctxfix** | **文档分页 + 头尾截断 + 强制收尾轮 + 反重复硬拒绝** | **34/48** | **70.8%** | **0.739** | **0** |

> v13 是**退步**，已回退（原因见第五节 5.1）。
> v14 排除了 2 道"文档与 gold 矛盾"的题（见第六节）。同口径对比：v12 = 31/48 = 64.6% → v14 = 34/48 = **70.8%**。

**关键里程碑：未产出答案 11 → 0。**

---

## 二、环境与配置

| 项 | 当前状态 |
|---|---|
| 模型 | `deepseek-chat`（服务端实际路由到 `deepseek-v4-flash`） |
| 端点 | `https://api.deepseek.com/v1`，**国内直连，不需要代理** |
| max_steps | 15（v14 实测所有题都在 15 步内完成） |
| max_workers | 1（串行） |

### 环境坑（本次调试中踩到的）

- **Gemini 地区限制**：按请求**出口 IP** 判断地区，国内 IP 报 `User location is not supported`。实测挂 7897 代理后端口通、但节点落在不支持地区，仍然 400。**最终换 DeepSeek 解决**（国内直连）。
- **socks 代理不兼容**：`all_proxy=socks://...` 会让 OpenAI SDK 初始化即崩，报错却显示为 `Invalid value for run.run_id`（极具误导性）。DeepSeek 直连，运行时用 `env -u all_proxy -u ALL_PROXY -u http_proxy ...` 清掉代理变量即可。

### 运行命令

```bash
V=v15_next && sed -i "s/^  run_id: .*/  run_id: $V/" configs/react_baseline.local.yaml && rm -rf artifacts/runs/$V \
  && env -u all_proxy -u ALL_PROXY -u http_proxy -u HTTP_PROXY -u https_proxy -u HTTPS_PROXY \
     uv run dabench run-benchmark --config configs/react_baseline.local.yaml \
  && uv run python scripts/score.py artifacts/runs/$V --verbose --json artifacts/reports/$V.json
```

---

## 三、前期已修的 6 个 baseline 工程缺陷

> 这些是本轮调试**开始之前**就已修复的，与具体任务无关的通用缺陷，列此存档。

这些都是通用问题，与具体任务无关。

| # | 文件 | 缺陷 | 症状 | 修复 |
|---|---|---|---|---|
| 1 | react.py | JSON 严格解析 | 模型写多行代码全报错 | `JSONDecoder(strict=False)` |
| 2 | react.py | **把模型自编的假观察喂回历史** | **幻觉：编造数据** | `_build_messages` 只回喂真实动作 |
| 3 | react.py | 上下文无限膨胀 | 撞爆 token 配额(429) | 老步骤截断，最近 3 步保留完整 |
| 4 | model.py | 每次新建 API 客户端 | 连接泄漏→CLOSE-WAIT 卡死 | 客户端建一次复用 + 超时 |
| 5 | model.py | 空响应/429 直接崩任务 | 全部步骤白跑 | 空响应转可恢复错误；429 按 retryDelay 退避 |
| 6 | runner.py | **multiprocessing 队列死锁** | 数据 >64KB 永久卡死 | 先 `queue.get(timeout)` 再 join |

### 幻觉 bug 深度解析（核心一课）

模型在**一次回复里自导自演**整段假对话：假装看到不存在的 `data.sqlite`、假装跑了 SQL、编出 `P001 John Doe` 之类玩具数据。baseline 把这段含假观察的原始输出原样喂回历史，模型便信以为真、越编越顺。

> **教训**：ReAct 里"模型思考"与"工具执行"必须严格分离。绝不能让模型自己写"工具返回了什么"，更不能把它写的假结果喂回上下文。

---

## 四、本轮调优的改动（v7 → v14）

### 4.1 JSON 解析器容忍前言文本 → +6pt，未产出 11→6

**根因**：DeepSeek 习惯在 JSON 动作前写一句自然语言（`I need to explore...\n\n{"thought":...}`），而 `_load_single_json_object` 用 `raw_decode` **从第 0 个字符**解析，遇到前言立刻抛 `Expecting value: line 1 column 1`。结果 20 步全部 `__error__` 空转。

**修复**（`react.py`）：向后扫描第一个能解析成 JSON 对象的 `{` 再解析，忽略前后自然语言。某个 `{` 解析失败（如 SQL 里的花括号）就跳到下一个。

> 这类"格式不兼容"是最隐蔽的分数杀手：看起来像模型不会做，实际是根本没执行。

### 4.2 提示词规则

| 规则 | 内容 | 针对 |
|---|---|---|
| 0 | 每步必须写非空 `thought` | 推理缺失 |
| 11 | **禁止拼接不同源列**（姓名分两列，`['1','1']` 不是 `'1-1'`），文档里读到的值也适用 | task_19/27/330/355 |
| 12 | 不许四舍五入/改格式，保留全精度 | task_249/303/408 |
| 13 | 提交前自检：行数 / 列 / 数量级 | 整表倒出 |
| 10 | **最小投影**：只输出问题要的属性，过滤/分组用的实体和顺带算的 count 都不算 | task_38/180/379 ⚠️效果有限 |

### 4.3 新增工具

| 工具 | 作用 |
|---|---|
| `profile_context` | **一次调用**画像整个 context：CSV 每列的类型/唯一值/空值/样例/数值范围，JSON 的记录数与字段，文档预览，sqlite 表结构。替代 5-10 步的 list+read |
| `query_files` | **DuckDB 跨源统一 SQL**：csv + json + sqlite 全部注册成表，可互相 JOIN。取代易错的手写 python 循环 |

> DuckDB 的 sqlite 扩展**本机已装，离线可用**，不需要下载。
> 实测 `query_files` 成为使用最多的工具（v12 用 102 次、v13 用 209 次）。

### 4.4 循环与收尾机制（v14 的主要来源）

| 机制 | 实现 | 效果 |
|---|---|---|
| **反重复硬拒绝** | 同一动作第 3 次起**直接不执行**，返回 REFUSED | 之前只警告，实测 task_396 能无视 6 次警告 |
| **强制收尾轮** | 步数耗尽后追加一轮，**只开放 `answer` 工具** | task_420 靠它产出 |
| **兜底打捞** | 仍无答案则倒查最后一个表格型 observation 自动提交 | task_379 从 0 分变成 7 行全对 |
| **warning 外提** | `repeat_warning` 移到截断范围之外 | 之前埋在正文里，正好被截断切掉 |

### 4.5 上下文管理（v14 的另一主要来源）

| 问题 | 修复 |
|---|---|
| **`read_doc` 默认截 4000，observation 再截 4000** —— 而 **50/50 个 `knowledge.md` 都超过 4000 字符**，"Ambiguity Resolution"章节永远在末尾 → **结构性看不见** | 窗口提到 8000 + 新增 `offset` 分页（返回 `next_offset`）；observation 预算提到 9000（必须 ≥ 文档窗口） |
| 截断只留头部，而**结果永远打印在最后** | 改为头+尾保留（2/3 头 + 1/3 尾） |
| 问题只在第 2 条消息出现一次，第 15 步时已在 4 万字符之外 | 在最新 observation 后重述 `Question:` |

> 这是**单点收益最大的修复**：50/50 的任务都受影响。

### 4.6 评测基建

- `configs/excluded_tasks.json`：排除"文档与 gold 矛盾"的题，**每条带证据和验证日期**
- `scripts/score.py`：**双口径报告**——主口径排除、同时显示全量口径，避免把问题藏起来
- `run-benchmark` 自动跳过排除项（50 → 48 题），顺带省 API 费用

---

## 五、⚠️ 失败的尝试（重要：不要重复投入）

### 5.1 原生 Function Calling —— 净退步 4pt，已回退

改用 OpenAI tools API 后，格式问题确实根除了，但：

| | 步数 | 空 thought | 平均推理长度 |
|---|---:|---:|---:|
| v12 文本模式 | 484 | 0 (0%) | **344 字符** |
| v13 function calling | 493 | **493 (100%)** | **0** |

**`tool_choice="required"` 下 DeepSeek 每一步都零推理直接调工具，丢掉了链式推理。**
补救尝试（把 `thought` 设为 schema 必填 + 加 prompt 规则 0）**都无效**——DeepSeek 不严格执行 schema 的 `required`，只在最后 `answer` 那步才写推理。

**处理**：做成开关 `agent.native_tools`（默认 `false`），代码保留。给推理能力更强的模型（如 GPT-4 系）可以打开。

### 5.2 强制列举证 —— 无效，已撤销

给 `answer` 加必填 `columns_asked_for`（每列必须引用问题原话），想解决多带列。
**失败原因**：模型总能编出合理理由——
- task_379 的 `count` 理由是 **"Tally"**（tally 确实有计数义）
- task_180 的 `CustomerID` 理由是 **"all the people"**

**问题不在模型偷懒，而在问题本身有歧义，gold 总取最小投影。** 强制举证只是让它把合理化写出来。

### 5.3 提示词管列纪律 —— 基本无效

即使把 task_379、task_38 的原句**当作反例明写进 prompt**，模型照样多带列。

> **结论：这个模型的"列投影"纪律，靠提示词管不住。** 需要程序化提交通道（见第七节）。

### 5.4 方法论教训

| 教训 | 依据 |
|---|---|
| **机制 > 劝导** | 反重复从"警告"改成"硬拒绝"立刻见效；纯 prompt 规则反复失败 |
| 警告要放在**不会被截断**的位置 | 埋在 observation 正文里 = 等于没有 |
| 加规则会**变长→挤占步数** | v8 加规则救回 5 题，却因 prompt 变长新增 4 个超时，净赚只有 2 |
| **一次只改一处**，用新 run_id | v13 若与别的改动混在一起，就无法定位是 FC 导致退步 |
| 子 agent 的结论**必须自己核验** | 有报告基于过期 run 得出"新工具零使用"，实测是使用最多的工具 |

---

## 六、已知不可解 / 有争议的题

### 6.1 已排除（`configs/excluded_tasks.json`）

| task | 矛盾 |
|---|---|
| **task_89** | knowledge.md 明写 "Use `positionOrder` for final race rankings"，照做得 `+14.925`；gold 是 `+16.445`，用的是 `rank` 列。**已用数据核对确认** |
| **task_169** | knowledge.md 定义 "Average Monthly Consumption = **Total** Annual Consumption / 12"(SUM/12 → 82,027,220)；gold 列名直接写着 `AVG(T2.Consumption) / 12` = 459.956 |

> ⚠️ **隐藏测试集很可能也有这类题。** 排除后的准确率是**开发信号**（用于干净比较版本改动），**不等于排行榜预期得分**。score.py 会同时显示全量口径。

### 6.2 待确认

- **task_38**：context 里的 `results.txt` 是**整行 10 列**的竖线分隔格式，而 gold 只要 `trans_id` 一列。规则 7 恰好鼓励模型参考 context 里的示例——疑似同类矛盾，**尚未最终确认**。

---

## 七、当前瓶颈与下一步

v14 剩余 14 个错题，按模式归类：

### 瓶颈 A：多带列（3 题，行数与主键值全对，纯死在列上）

| task | gold | pred |
|---|---|---|
| task_38 | 140行**1列** | 140行**10列**（`SELECT *`） |
| task_180 | 9行**1列** | 9行**2列**（多 group key） |
| task_379 | 7行**1列** | 7行**2列**（多 count） |

**修掉这一个模式 = +3 题 → 77%。** 提示词已证明无效（5.2/5.3），需要**程序化提交通道**：
- `answer_sql {sql}`：直接把 DuckDB 结果集作为答案，SELECT 列表即答案列
- 或给 `execute_python` 注入 `submit(df)` 内建函数

### 瓶颈 B：该聚合没聚合（task_199 交了 87 行 vs gold 6 行）

### 瓶颈 C：真·推理/题意错（task_163 分组键、task_200、task_344、task_396）

### 瓶颈 D：方差（约 8pt）

三版本并集 = 35/50 = 70%，单版本最佳 62%（v12 时的统计）——说明**有一批题模型有能力做对但不稳定**。
省钱方案：**选择性 self-consistency**——只对"可疑"的题（行数异常多、走了兜底打捞）重跑投票，而非全部重跑。

---

## 附录 A：改动文件清单

| 文件 | 本次改动 |
|---|---|
| `agents/react.py` | 解析器容忍前言；反重复硬拒绝；强制收尾轮 `_forced_answer_turn`；兜底打捞 `_salvage_answer`；观察预算 9000/1200 |
| `agents/prompt.py` | 规则重编号 0-13；最小投影/不拼接/全精度/自检；头+尾截断；warning 外提；问题重述；`TEXT_MODE_FORMAT_RULES` 按模式条件注入 |
| `agents/model.py` | function calling 支持（默认关，可回退）；tool_call → 规范 JSON；端点不支持自动降级 |
| `tools/profile.py` | **新增**：一次画像整个 context |
| `tools/duckdb_files.py` | **新增**：DuckDB 跨源 SQL（csv+json+sqlite） |
| `tools/registry.py` | 注册新工具；JSON Schema；`read_doc/read_json` 加 `offset`；`DOC_WINDOW_CHARS=8000` |
| `tools/filesystem.py` | `_window()` 分页读取，返回 `next_offset` |
| `run/runner.py` | 接入排除清单 `load_excluded_task_ids()` |
| `config.py` | 新增 `agent.native_tools` 开关 |
| `configs/excluded_tasks.json` | **新增**：排除清单（带证据） |
| `scripts/score.py` | 双口径报告 |

## 附录 B：调优方法论

1. **一次只改一处**，用新 run_id，跑完立即打分存档到 `artifacts/reports/`
2. 小样本（`--limit 5`）波动极大（1 题 = 20%），确认效果必须跑全量
3. **给判断依据 > 要求额外动作**：规则该告诉它"怎么选"，而非"多做一步"
4. **能用机制强制的，就不要用提示词劝导**
5. 提示词要跨领域通用，别举具体领域的例子
6. 改动后先在**目标失败 task 上单独验证**，再跑全量（省钱省时间）
