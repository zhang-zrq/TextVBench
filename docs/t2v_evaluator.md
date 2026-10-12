# TextVBench-T2V Evaluator 实现说明

> 目标：基于公开的 `data/T2V/benchmark_300.jsonl` 与模型生成视频，实现自动化 Visual-Text Evaluator。
>
> 本文档描述流程、接口语义、配置和必须遵守的工程约束；具体代码组织、类设计和并发实现由 coding agent 按现有仓库风格完成。

## 1. 与当前仓库的关系

Evaluator 按子 benchmark 采用并列目录组织：`src/textvbench/evaluators/t2v/`
承载本文的 T2V 实现，未来 Ref2V 等 evaluator 放在同级目录，不把不同
benchmark 的 metric 或 prompt 混在一个实现包中。

Evaluator 是独立评测流程，直接消费公开 T2V JSONL 与对应的模型生成视频。
公开记录提供 carrier / slot、`text_en` / `text_zh` 以及平行的生成 prompt，
不需要数据构造 pipeline 或中间 stage 产物。

沿用仓库既有规范：
- JSONL 作为批量输入输出；
- YAML 管理 evaluator 配置；
- JSON schema 校验结构化结果；
- system prompt 独立从 Markdown 文件加载；
- 使用现有 OpenAI-compatible VLM client、并发、retry、checkpoint、failed/passed 机制；
- 所有阈值必须来自配置，不在代码和 prompt 中硬编码第二套值。

Evaluator 不修改公开 benchmark 数据。

### 1.1 语言必须显式指定

每条样本同时包含 `text_en` / `text_zh` 时，Evaluator 不自行判断语言，也不
从记录中的 `generation_language`、`prompt_language` 或配置默认值推断。调用
方必须为整个 run 显式指定一个语言：

```bash
textvbench evaluator t2v ... --language en
textvbench evaluator t2v ... --language zh
```

一个 run 只能评一种语言。EN 视频和 ZH 视频需要分别运行，分别绑定对应的
target text、prompt context 和 review artifact 目录；缺少 `--language` 应直接
在 CLI 参数解析阶段失败。

### 1.2 视频映射与输出模式

批量评测使用 `--video-dir`，严格按 `<sample_id>.<suffix>` 查找视频；缺少任意
待评测视频会在 OCR/VLM 前直接报错。视频文件名不一致时使用
`--video-manifest` JSONL，每行包含 `sample_id` 和 `video_path`。单条测试使用
`--sample-id` + `--video`。这样 prompt JSONL 与生成视频的绑定是显式且可审计的，
不会依赖一个目录中模糊的顺序匹配。

默认输出为 metrics-only：passed JSONL 包含 audit 和 metrics。为便于人工
judge/review，传 `--debug-dir`（`--review-dir` 为兼容别名）后，每条记录按
`<sample_id>_<language>/` 保存输入记录、解析配置、抽样原图、OCR 与候选过滤
结果、carrier/matching marker、每次 VLM retry 的最终 system prompt、user payload、
返回 JSON、校验错误、逐帧 observation 和最终 result。detail bundle 用于测试产物
和 feature review，不改变正式 `passed/failed` 输出契约。

## 2. 总体流程

```text
Video + GT Structured Spec
(target / slot / carrier / position)
                │
                ↓
① Sparse Temporal Sampling
   3 fps + deterministic jitter
   min=1 / max=30
   + adaptive resampling
                │
                ↓
② PaddleOCR per sampled frame
   bbox + ocr_text + confidence
                │
                ↓
③ Candidate Filtering
   confidence threshold
   duplicate compression
   max candidate limit
                │
                ↓
④ Carrier Consistency Judge
   carrier 是否出现在当前帧？
   哪些 OCR candidates 属于该 carrier？
                │
                ↓
⑤ Program Fast Matching
   carrier 内 target ↔ candidate
   NED similarity >= threshold
                │
                ↓
⑥ VLM Target–Candidate Matching
   unresolved targets only
   target batch <= 3
                │
                ↓
⑦ frame_match result
                │
                ↓
⑧ 按 slot_id 聚合 sampled frames
                │
        ┌───────┴───────┐
        ↓               ↓
 Spatial Fidelity   Temporal Fidelity
        └───────┬───────┘
                ↓
           TextVScore

Diagnostic:
Extra Text Ratio
```

核心原则：每个 sampled frame 独立完成 carrier / target matching；跨帧不做 OCR tracking，只通过 benchmark 已知的 `slot_id` 聚合。

## 3. Temporal Sampling

### 3.1 初始采样

默认：

```yaml
sampling:
  fps: 3
  min_frames: 1
  max_frames: 30
  seed: <fixed integer>
```

按视频 timestamp 采样，不依赖裸 frame index。

规则：
1. 根据视频 duration 按 `3 fps` 生成目标 timestamps；
2. 至少采 `min_frames=1`；
3. 最多采 `max_frames=30`；
4. 首尾附近必须有覆盖；
5. jitter 使用固定 seed，保证同一视频、同一配置下完全可复现。

建议随机数由：

```text
global_seed + stable(sample_id)
```

确定，使不同视频 jitter 不同，但重复评测得到相同采样点。

### 3.2 Hidden Temporal Jitter

每个基础 timestamp 在一个小范围内做随机偏移。

- jitter range 配置化；
- random seed 固定；
- 最终实际 timestamps 写入 evaluator audit；
- leaderboard 不需要公开具体 jitter sequence。

目的：避免模型针对固定采样帧投机优化。

### 3.3 Adaptive Sampling

第一轮 sparse evaluation 完成后，对相邻 sampled observations 检查：

1. target match state 发生变化，例如 `matched ↔ missing`；
2. 同一 target 的 NED score 出现明显 jump。

满足条件时，在该 interval 中补采 timestamp，再执行同样的 frame evaluator。

约束：
- 仍不能超过 `max_frames`；
- jump threshold 配置化；
- 第一版不增加更多复杂触发规则。

## 4. PaddleOCR

使用运行环境中已有的 PaddleOCR，不绑定 v5/v6。

每个 sampled frame 获取：

```text
candidate_id
bbox / polygon
ocr_text
confidence
```

PaddleOCR 是 evaluator 中唯一的 OCR 文本来源。

发送给 VLM 的 `ocr_candidates` 使用紧凑字段 `id`、`ocr_text` 和
`confidence`，不再重复发送 `bbox/polygon`。空间对应关系只通过 marker image
中的彩色 bbox 和同编号 ID tag 建立；原始 bbox 仍保留在本地 observation/debug
artifact 中，供 review 和后续指标计算使用。

VLM 不允许：
- 自由 OCR；
- 修正 OCR 字符；
- 补全文字；
- 根据 target 猜文字；
- 翻译 OCR 内容。

## 5. Marker Image

VLM 只输入 marker image，不额外输入原始 image。

Marker image 保留完整原视频帧，并叠加：
- 每个 OCR candidate 一个不同颜色 bbox；
- candidate ID tag；
- tag 默认放在 bbox 左上外侧；
- 空间不足时放左上内侧；
- tag 使用白色背景；
- tag 字体颜色与 bbox 颜色一致。

Marker image 必须与当前送入 VLM 的 candidate list 完全一致。

因此：每完成一个 target batch 并消费 candidate 后，下一批 VLM 调用需要重新生成 marker image，删除已消费 candidate 的 bbox / ID。

## 6. Candidate Filtering

所有阈值配置化，例如：

```yaml
ocr:
  confidence_threshold: 0.80
  max_candidates: 20

matching:
  fast_match_threshold: 0.95
  target_batch_size: 3
```

### 6.1 Confidence Filter

默认：

```text
confidence < 0.80
→ 丢弃
```

低于阈值的 candidate 不参与：
- carrier judge；
- target matching；
- Extra Text Ratio。

### 6.2 Duplicate Compression

对 confidence filter 后的 candidate 做 normalized exact-text duplicate 检查。

Normalization 使用与 metric 一致的规则：
- 去除所有空格；
- 去除标点；
- 保留大小写。

如果多个 candidate 的 normalized OCR text 完全相同：
- 默认保留 confidence 最高的 candidate 进入 active pool；
- 其余进入 `extra_candidate_pool`。

例外：如果 GT 中存在多个不同 slots，且这些 slots 的 normalized target text 本身相同，则不要因为字符串相同而把所需实例全局压成一个。

不做 fuzzy duplicate compression。

### 6.3 Max Candidate Limit

如果 duplicate compression 后：

```text
num_candidates > max_candidates
```

则排序：
1. normalized OCR text 字符数更多优先；
2. 字符数相同时 confidence 更高优先。

保留前 `max_candidates` 进入 active VLM pool。

其余进入 `extra_candidate_pool`。

目的：避免 dense frame 中过多 bbox 导致 VLM context 退化。

## 7. Carrier Consistency Judge

在任何 target textual matching 之前，先独立调用 Carrier Judge。

不要把 carrier consistency 和 target–candidate matching 混在同一个 prompt 中。

输入：
- marker image；
- 当前帧全部 unique GT carriers；
- 每个 carrier 的 `carrier_id / type / position`；
- active OCR candidates；
- scene / generation prompt 可作为简短视觉语境。

输出：

```json
{
  "carriers": [
    {
      "carrier_id": "C1",
      "present": true,
      "candidate_ids": ["P003", "P008"]
    }
  ]
}
```

含义：
- `present=true`：该 carrier 在当前帧中可见/可辨认；
- `candidate_ids`：视觉上属于该 carrier 的 OCR candidate IDs；
- carrier 可以 `present=true` 但 `candidate_ids=[]`，表示 carrier 存在但 PaddleOCR 没有识别出其上的文字；
- `present=false` 时 `candidate_ids=[]`。

Carrier Judge 不判断：
- OCR 字符是否正确；
- candidate 对应哪个 target slot；
- target 是否 missing。

一个 OCR candidate 默认最多归属一个 GT carrier。

## 8. Carrier Presence 的计分语义

不依赖 Stage 3/5 的 `behavior` 判断当前帧是否必须出现文字。

当前帧是否评估某 target，直接由其 carrier 是否存在决定。

### Carrier 当前帧不存在

`present=false` 时：
- 该 target 在当前帧不参与 Spatial score；
- 该 timestamp 不进入该 target 的 Temporal observation sequence；
- 不将其判为 text missing。

### Carrier 当前帧存在

则该 carrier 下所有 targets 都应该进入 matching：
- 找到 candidate → matched；
- 找不到 candidate → missing。

### 整个视频所有 sampled frames 中 carrier 从未出现

该 carrier 下 targets 不能因为一直被 mask 而获得高分：

```text
target Spatial = 0
target Temporal = 0
```

这样允许正常出画/遮挡，同时防止模型通过根本不生成 carrier 绕过评测。

## 9. Program Fast Matching

Carrier Judge 完成后，只在对应 carrier 的 `candidate_ids` 内进行程序匹配。

对 target 和单个 OCR candidate：

```text
sim = 1 - NED(normalized_target, normalized_ocr_text)
```

Normalization：
- 删除空格；
- 删除标点；
- 保留英文字母大小写；
- 使用 character-level NED。

如果：

```text
sim >= fast_match_threshold
```

则可以直接建立 provisional match。

第一版 Fast Match 只处理一个 target ↔ 一个 OCR candidate 的高置信度匹配。

以下情况继续交给 VLM：
- 没有 candidate 达到 threshold；
- 一条 target 可能被拆成多个 OCR boxes；
- 多个 candidates 都有较强语义/布局歧义；
- 需要结合位置和 layout 才能确定 intended instance。

Fast Match 后被消费的 candidate 从 active pool 删除。

## 10. VLM Target–Candidate Matching

只处理 unresolved targets。

沿用图像 visual-text evaluator 已验证的核心原则：Candidate selection，而不是 free OCR。

输入：
- 重新生成后的 marker image；
- unresolved targets，每次最多 3 条；
- 每个 target：
  - `slot_id`
  - exact `target_text`
  - `carrier`
  - `position`
- 当前剩余 OCR candidates：
  - `id`
  - `ocr_text`
  - `confidence`

不再输入 `behavior`。

### 10.1 Batch Processing

如果 unresolved targets > 3：

```text
T1,T2,T3 → batch 1
T4,T5,T6 → batch 2
...
```

每一批完成后：
1. 收集确定的 `selected_ids`；
2. 从 candidate pool 删除这些 IDs；
3. 重新生成 marker image；
4. 下一批只看到仍可用 candidates。

不额外实现低频 global conflict resolver。

### 10.2 Multiple OCR Boxes

一个 target 可以选择：

```json
["P003", "P007", "P010"]
```

System Prompt 明确要求：`selected_ids` 必须按照视觉阅读顺序输出。

程序严格按照该顺序拼接 OCR texts，再做 normalization / NED。

不额外增加复杂 OCR grouping module。

## 11. Frame-level Result

每个 sampled frame 最终形成一个逻辑上的：

```json
{
  "timestamp": 2.67,
  "carriers": {
    "C1": true,
    "C2": false
  },
  "matches": [
    {
      "slot_id": "T1",
      "selected_ids": ["P003", "P007"],
      "pred_text": "MOONLIGT CAFE"
    },
    {
      "slot_id": "T2",
      "selected_ids": [],
      "pred_text": ""
    }
  ],
  "extra_ids": ["P011", "P014"]
}
```

`pred_text / extra_ids` 应尽量由程序根据 IDs 和 candidate table 推导，而不是要求 VLM 输出。

不要求为每一帧创建独立磁盘 JSON 文件；按仓库既有 JSONL / record schema 保存即可。

## 12. Extra Text

Extra Text 不进入 TextVScore，但必须作为独立 diagnostic metric。

一帧中以下 candidate 进入 extra pool：
1. duplicate compression 中被淘汰的高置信重复 candidate；
2. max-candidate cap 中未进入 VLM active pool 的 candidate；
3. Carrier Judge 后不属于任何 required carrier 的 active candidate；
4. target matching 全部完成后仍未被任何 target 消费的 candidate；
5. 同一 target 被额外重复生成的其他实例；
6. 内容正确但放在错误 carrier / subtitle / overlay 上的 candidate。

confidence filter 阶段被丢弃的 candidate 不计入 Extra Text。

### 12.1 Extra Text Ratio

对 OCR text 使用相同 normalization 后统计字符数：

```text
ExtraTextRatio
=
extra OCR characters
/
(matched OCR characters + extra OCR characters)
```

如果分母为 0，则该帧 ratio 定义为 0。

最终对视频聚合得到：

```text
Extra Text Ratio ↓
```

用途：
- 分析模型是否 hallucinate visual text；
- 分析模型是否重复生成 target 以提高 OCR 命中率；
- 发现把所有 target 到处都写一遍的 reward hacking。

该指标不进入 TextVScore。

## 13. 防止错误区域 / 字幕刷分

Carrier Judge 是硬边界。

例如 target 要求：

```text
"SALE" → poster
```

而模型生成：

```text
"SALE" → bottom subtitle
```

即使 OCR 完全正确：
- subtitle candidate 不属于目标 poster；
- 不允许作为 target match；
- target 应被判 missing；
- subtitle candidate 进入 Extra Text。

同理，把所有 target 同时写在每个 carrier 上，只能消费 intended instances，其余重复文字进入 Extra Text。

## 14. Spatial Fidelity

只评价：当 target carrier 出现在当前帧时，该 slot 的文字内容是否正确。

对 target `t` 在 carrier-present frame `f`：

```text
matched:
s_tf = 1 - NED(target_t, pred_tf)

missing:
s_tf = 0
```

carrier absent frame 不计。

每个 target：

```text
S_t = mean(s_tf over carrier-present sampled frames)
```

若整个视频该 target 的 carrier 从未在 sampled frames 中出现：

```text
S_t = 0
```

## 15. Temporal Fidelity

v1 只评估 persistent visual text。

以下类型不要进入 Main Track：
- scrolling / 跑马灯；
- state-changing display；
- progressive typing / handwriting；
- countdown / scoreboard update；
- 其它正确内容本应随时间变化的 visual text。

对 target `t`，按 timestamp 收集所有 carrier-present observations：

```text
P1, P2, ..., Pk
```

其中 missing 仍作为一个 observation 保留。

相邻两个 observation：

```text
两者均 matched:
r_i = 1 - NED(pred_i, pred_{i+1})

任一个 missing:
r_i = 0
```

然后：

```text
T_t = mean(r_i)
```

如果只有一个 carrier-present observation：

```text
T_t = S_t
```

避免单帧可见 target 无条件获得 Temporal=1。

如果 carrier 在整个视频从未出现：

```text
T_t = 0
```

## 16. Target-length Weighted Video Score

Spatial 和 Temporal 都先得到 target-level score，再聚合到 video-level。

权重：

```text
w_t = normalized target 字符数
```

Normalization：
- 删除空格；
- 删除标点；
- 保留大小写。

视频级：

```text
S_video = sum(w_t * S_t) / sum(w_t)

T_video = sum(w_t * T_t) / sum(w_t)
```

## 17. TextVScore

最终正式综合指标：

```text
TextVScore
=
100 * 2 * S_video * T_video
      ----------------------
      S_video + T_video
```

即 Spatial / Temporal 的 harmonic mean。

当两者均为 0 时：

```text
TextVScore = 0
```

主结果建议：

```text
TextVScore ↑
Spatial Fidelity ↑
Temporal Fidelity ↑
Extra Text Ratio ↓   # diagnostic only
```

不再向 leaderboard 平铺更多内部指标。

## 18. Adaptive Sampling 所需的中间量

初始 sparse pass 后，可用：

```text
slot match state
slot NED
```

检测 interval：

### Trigger A

```text
matched ↔ missing
```

### Trigger B

```text
abs(NED_i - NED_j) >= configured_jump_threshold
```

触发后补采 interval 内 timestamp。

Carrier absent 本身不作为 text corruption；但 `carrier present ↔ absent` 可以作为可选 adaptive hint，配置控制。

## 19. Cache

第一版直接实现 cache，至少缓存：

```text
video metadata
sampled timestamps / frame indices
sampled frames
PaddleOCR raw results
filtered candidate tables
marker images
carrier judge outputs
target matching outputs
```

目的：
- 调整 metric 不需要重新 OCR / VLM；
- 调整 VLM prompt 时可复用 OCR；
- 调整 sampling 配置时只重跑受影响部分。

Cache key 必须包含影响结果的关键配置 / prompt version，避免错误复用旧结果。

## 20. Deterministic VLM

Evaluator 使用确定性配置：

```text
temperature ≈ 0
strict JSON output
schema validation
candidate ID validation
retry on malformed / illegal output
```

沿用仓库已有：
- bounded concurrency；
- record-level retry；
- checkpoint；
- passed / failed；
- audit error message。

不保留自由文本 `reason` 字段。

## 21. 建议配置项

至少包含：

```yaml
sampling:
  fps: 3
  min_frames: 1
  max_frames: 30
  seed: 42
  jitter_sec: ...
  adaptive: true
  ned_jump_threshold: ...

ocr:
  confidence_threshold: 0.80
  max_candidates: 20

matching:
  fast_match_threshold: 0.95
  target_batch_size: 3

vlm:
  carrier_prompt_path: ...
  matching_prompt_path: ...
```

具体配置层级可以按仓库现有 YAML 组织调整。

## 22. 不做的事情

v1 evaluator 明确不实现：

```text
OCR box tracking
复杂 video text spotting
VLM free OCR fallback
复杂 visibility state machine
基于 behavior 的 frame mask
复杂 OCR paragraph grouping
低频 VLM/PaddleOCR 偶发错误修补
dynamic-text relation evaluator
```

这些不应为了少量 case 增加 Main Track 复杂度。

## 23. Coding Agent 实现完成后的最低检查

实现后至少：
1. 为 sampling / normalization / NED / candidate filtering / duplicate compression / target-length weighted score 写单元测试；
2. 构造一个多 carrier、多 target、多 bbox split、重复文字、错误 subtitle 的小 fixture；
3. 验证 target batch >3 时会重新生成 marker image，并从后续 candidate list 中删除已消费 IDs；
4. 验证 carrier absent frame 不参与该 target frame score，但 carrier 全视频从未出现时 target score=0；
5. 验证 Extra Text Ratio 不进入 TextVScore；
6. 使用仓库约定测试命令跑全量 tests；
7. 若新增 CLI / config / schema / prompt 路径，按仓库规范同步 AGENTS.md；若改变 benchmark 设计，再同步 overview 文档。
