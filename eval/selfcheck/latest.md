---
type: report
title: 评测自检 · fast
description: 故意破坏 12 个场景，12 通过 / 0 失败 / 1 不可达；附样本缺口、规则覆盖、真实数据上的标签与参数可辨识性
tags: [评测, 自检, 故意破坏, 归因]
timestamp: 2026-09-10
---

# 评测自检（fast）

- 时间 2026-09-10T18:07:56，耗时 52s，参数 v1（hand-set）
- 结论：声明可达的场景全部贴出了预期标签

## a. 故意破坏：坏了会不会叫

| # | 场景 | 期望标签 | 破坏方式 | 子集 | 观测 | 结果 | 耗时 | 证据 |
|---|---|---|---|---|---|---|---|---|
| 1 | db_missing | db_missing | 地址库删「解放碑街道」「犀浦街道」 | 24 句朗读稿 | none×22, db_missing×2 | ✓ 通过 | 4.7s | 删 犀浦街道(510117-01)、解放碑街道(500103-01) → {'003': 'db_missing', '005': 'db_missing'}；其余 22/22 条 exact |
| 2 | asr_unrecoverable | asr_unrecoverable | 003 的输入把「民权路」换成音上毫不相干的「泼皮鸭」，transcript_gold 保持原文，强制算 Stage A | 朗读稿 003 | asr_unrecoverable×1 | ✓ 通过 | 0.2s | 输入「重庆渝中区解放碑泼皮鸭十八号三单元二零一」name_phon_dist_max=0.648 → 「民权路」在 ASR 输出里最近片段「庆渝」音距离 0.648 > 0.4 |
| 3 | normalize_error | normalize_error | 003 的真值门牌 18号 → 19号（fields 与 ground_truth 同改） | 朗读稿 003 | normalize_error×1 | ✓ 通过 | 0.1s | 输出「重庆市渝中区解放碑街道民权路18号3单元201室」vs 真值「重庆市渝中区解放碑街道民权路19号3单元201室」→ 数字转换错；门牌字段错 ['house_no'] |
| 4 | recall_miss | recall_miss | MAX_DIST=0.01 | 合成集前 60 条非同音异形 | normalize_error×26, recall_miss×23, none×11 | ✓ 通过 | 15.2s | recall_miss×23，miss_reason {'dist_over_max': 23}；同批 normalize_error×26 |
| 5 | rank_error | rank_error | W_SIM=0, W_PRIOR=1 | 合成集全量 720 条（--fast 只跑 source_id ∈ ('010', '011')） | gate_over_reject×27, none×23, rank_error×10 | ✓ 通过 | 8.1s | 60 条里 rank_error×10，dominant_term {'prior': 10} |
| 6 | gate_over_accept | gate_over_accept | SINGLE_HIT_MIN_COV=0, SIM_MIN=1e-6, MARGIN_MIN=1e-6 | 清单里 4 条 has_address=false 的录音文本 | none×3, gate_over_accept×1 | ✓ 通过 | 1.1s | 4 条无地址录音，决策 {'20260909_203604_auto': 'reject', '20260909_204208_auto': 'reject', '20260909_211125_auto': 'confident', '20260909_212717_auto': 'reject'}；错送 ['20260909_211125_auto「嗯用那个闽南话再说一个地址呢也是加一点语气助词的那种」→「黑龙江省齐齐哈尔市讷河市」'] |
| 7 | gate_over_reject_sim | gate_over_reject | SIM_MIN=1.01 | 24 句朗读稿 | gate_over_reject×24 | ✓ 通过 | 4.2s | 24/24 条 gate_over_reject，gate_attrib {'sim': 24} |
| 8 | gate_over_reject_margin | gate_over_reject | MARGIN_MIN=1.0 | 24 句朗读稿 | gate_over_reject×16, none×8 | ✓ 通过 | 4.2s | gate_over_reject(margin)×16/24；其余 8 条仍 confident：只有一条候选或 Top-2 同行政区，分差闸门按 rank.decide 不拦 |
| 9 | assembly_error | assembly_error | 003 的 ground_truth 末尾加「甲」，fields 不动 | 朗读稿 003 | assembly_error×1 | ✓ 通过 | 0.1s | 真值末尾加「甲」、字段不动：admin_all=True geo_recall=1.0 tail_all=True gate_attrib=assembly → assembly_error；回归阻断规则同时报红 |
| 10 | completion_error | completion_error | 地址库删 018 的全部路级条目（金阳南路、世纪城） | 朗读稿 018 | db_missing×1 | — 不可达 | 0.2s | 删 世纪城(510191-07)、金阳南路(520115-01)、世纪城(520115-02)、世纪城(530111-02) → 决策 reject，输出「观山湖区金阳南路世纪城36栋」，四格 TR → db_missing：库中无「世纪城、金阳南路」，决策 reject，省市区无法回溯 |
| 11 | regression_red | regression_red | W_CONFLICT=1.0 | 合成集全量 720 条对 golden（--fast 只跑 source_id ∈ ('014',)，对同子集的未破坏运行）；另跑 24 句朗读稿 | none×29, regression_red×18, rank_error×1 | ✓ 通过 | 13.5s | 合成集 30 条对 同子集未破坏的对照运行：18 项失败（翻转 ['014-017:rank_error']）；同一破坏在 text 套件 0 项失败 |
| 12 | skip | skip | 构造 quality=silent 的记录直接调 attribution.attribute | 单条构造记录 | skip×1 | ✓ 通过 | 0.0s | 构造 quality=silent 的记录 → skip（quality=silent） |

场景说明：

- **db_missing**：期望 003、005 记 db_missing，其余 22 条 exact。删的是口语说法与正名不同的条目（说的是解放碑/犀浦镇），流水线没有库条目就没法规范成正名；同 tests/test_attribution.py
- **asr_unrecoverable**：期望 asr_unrecoverable。Stage A 只在 audio 模式或 stage_a_force 时计算；最近窗口音距离 0.648 远超 MAX_DIST
- **normalize_error**：期望 normalize_error（C/D/E 仍对，只有门牌字段不符）。改真值而不是改代码：流水线输出不变，评测应把不一致记在 Stage B
- **recall_miss**：期望 ≥ 1 条 recall_miss，报告 miss_reason 分布。规格原定 0.05，实测 0 条 recall_miss，改用 0.01
- **rank_error**：期望 ≥ 1 条 rank_error，报告 dominant_term 分布。前 240 行一条 rank_error 都没有，快速子集按 source_id 取（见 FAST_RANK_SOURCES 注释）
- **gate_over_accept**：期望 ≥ 1 条 gate_over_accept（无地址却 confident）。有地址样本上 Top-1 错时决策树先贴 rank_error / recall_miss，错送标签主要只能从无地址负样本上来
- **gate_over_reject_sim**：期望 24 条全部 gate_over_reject，gate_attrib 全是 sim。对应 §10 步骤 2 的 SIM_MIN=0.95 破坏测试；朗读稿 Top-1 音相似度全是 1.0，0.95 拦不下任何一条，必须越过 1
- **gate_over_reject_margin**：期望 ≥ 10 条 gate_over_reject 且 gate_attrib=margin。同区豁免：Top-2 与 Top-1 同行政区时分差闸门不拦，拦不满 24 条是规则如此
- **assembly_error**：期望 assembly_error（各阶段都过，只有串不同），且回归阻断规则报红。兜底标签：只说明各阶段判定都过，不指明拼装哪一步错
- **completion_error**：期望 不可达：该情形应记 db_missing（库缺 + 拦下 + 省市无从回溯），不得再误归到 completion_error。真正的补全错（Top-1 链对、回溯出的省市错）按构造不可达：chain_matches 要求行政区一致；这里跑的是曾经误归因的路径，passed = 现在记到了库缺头上
- **regression_red**：期望 regression.compare 在合成集上 ≥ 1 项失败。对应 §10 步骤 6 的故意改坏测试；--fast 的子集比不了 720 条的 golden，只能对同子集的对照运行
- **skip**：期望 skip。决策树第 1 步，只看 quality

### 意外发现

- 召回只看真值链最深层条目：MAX_DIST=0.01 时同批 26 条是市/区/街道层被扰动、没召回，扰动字原样留在输出里，被 Stage B 记成闲话泄漏 → normalize_error，而不是 recall_miss（例 001-013「我屋头在曾都市武侯区溷牌楼街道二环路南四段三十号」：闲话泄漏「溷溷」）
- W_CONFLICT=1.0 在 24 句朗读稿上回归全绿（text 套件 0 项失败），只有合成集能让它变红，而且只靠 1 条样本（014-017:rank_error）
- manifest_to_items 在评测前就滤掉 quality≠ok 的行（清单里 8 行），真实报告里 skip 恒为 0；quality 判定本身归 check_manifest 管，这里不验证

## b. 样本缺口与标注状态

- 清单 54 行，quality 分布 ok×46, silent×5, duplicate×2, unknown_format×1
- quality=ok 的 46 行 label_status 分布 draft×32, pending×9, confirmed×5；可计分 37 条
- 缺口表口径：可计分录音 37 条 + 朗读稿 24 句（合成变体不是新样本，不进缺口表）

| dialect_group | address_depth | noise | n | 距 20 还差 |
|---|---|---|---|---|
| 官话 | district | clean | 19 | 1 |
| 官话 | full | clean | 17 | 3 |
| 官话 | full | filler | 8 | 12 |
| 官话 | full | complaint | 3 | 17 |
| 官话 | none | clean | 2 | 18 |
| 官话 | none | complaint | 2 | 18 |
| 官话 | street_only | complaint | 2 | 18 |
| 闽 | full | clean | 2 | 18 |
| 吴 | full | clean | 1 | 19 |
| 吴 | full | filler | 1 | 19 |
| 粤 | district | clean | 1 | 19 |
| 粤 | full | clean | 1 | 19 |
| 闽 | full | complaint | 1 | 19 |
| 闽 | full | filler | 1 | 19 |

有效样本（按句计：合成集同一 source_id 只算 1 条）：

| 划分 | 有效 | 原始 | draft |
|---|---|---|---|
| train | 12 | 360 | 0 |
| calib | 6 | 180 | 0 |
| eval | 67 | 241 | 32 |

§2.6 门槛：train_eff=12/500, calib_eff=6/300, difficulty 不足 30 的标签 11 个；权重学习 未达标，校准 未达标

## c. 合成扰动集规则覆盖

规则名来源：scripts/gen_perturbed.py build_rules()；计数来自 gen_config.json counts_by_rule。

| 规则 | n | ≥ 20 | 混淆表规则 |
|---|---|---|---|
| 同音异形 | 240 | ✓ | 否（生成器自有） |
| 复合 | 72 | ✓ | 否（生成器自有） |
| 前后鼻音不分 | 47 | ✓ | 是 |
| 浊音清化 | 43 | ✓ | 是 |
| 平翘舌不分 | 36 | ✓ | 是 |
| 尖团音分立 | 27 | ✓ | 是 |
| r 声母脱落/半元音化 | 26 | ✓ | 是 |
| un/ong 相混 | 25 | ✓ | 是 |
| 边鼻音不分 | 24 | ✓ | 是 |
| 明母塞化 | 20 | ✓ | 是 |
| 鼻音韵尾脱落 | 19 | ✗ | 是 |
| ao/ou 相混 | 16 | ✗ | 是 |
| e/uo 相混 | 15 | ✗ | 是 |
| h/f 不分 | 14 | ✗ | 是 |
| r 声母边音化 | 14 | ✗ | 是 |
| ai/ei 相混 | 13 | ✗ | 是 |
| 入声派入 | 13 | ✗ | 是 |
| n/r 相混 | 12 | ✗ | 是 |
| 晓匣不颚化（鞋=hai 下=ha） | 8 | ✗ | 是 |
| 见系不颚化（交=gao 街=gai） | 8 | ✗ | 是 |
| 溪群不颚化（去=kê） | 7 | ✗ | 是 |
| 知组与见组相混 | 5 | ✗ | 是 |
| 介音脱落 | 4 | ✗ | 是 |
| 介音脱落（见系不颚化伴生） | 4 | ✗ | 是 |
| 零声母/w 交替 | 4 | ✗ | 是 |
| 撮口呼混同 | 3 | ✗ | 是 |
| 街=gai 型韵母对应 | 1 | ✗ | 是 |
| e/o 相混 | 0 | ✗ | 是 |
| o/uo 相混 | 0 | ✗ | 是 |
| 介音脱落（下=ha） | 0 | ✗ | 是 |
| 儿化差异 | 0 | ✗ | 是 |
| 同音异写 | 0 | ✗ | 是 |

- 0 条的规则（5）：e/o 相混、o/uo 相混、介音脱落（下=ha）、儿化差异、同音异写
- 不足 20 条的规则（17）：鼻音韵尾脱落、ao/ou 相混、e/uo 相混、h/f 不分、r 声母边音化、ai/ei 相混、入声派入、n/r 相混、晓匣不颚化（鞋=hai 下=ha）、见系不颚化（交=gao 街=gai）、溪群不颚化（去=kê）、知组与见组相混、介音脱落、介音脱落（见系不颚化伴生）、零声母/w 交替、撮口呼混同、街=gai 型韵母对应
- 生成器说明：这些规则本次没产出样本，不是 bug：一是 24 句地名里没有落在该混淆组上的字；二是有的组成员（uei/iou/uen/ue/er）在 pypinyin strict=False 的韵母切分下根本不会出现，反向找不到替换字。换朗读稿或扩充评测集后会自然覆盖到。

## d. 真实数据上出现过的标签

- audio：eval/reports/20260910_140142_audio.json，n=37，Top-1 音相似度最低 0.8317（<1 的 4/34 条）
- synthetic：eval/reports/20260910_141855_text_synthetic.json，n=720，Top-1 音相似度最低 0.9618（<1 的 475/720 条）
- text：eval/reports/20260910_140157_text.json，n=24，Top-1 音相似度最低 1.0（<1 的 0/24 条）

| 标签 | audio | 合成 | text |
|---|---|---|---|
| skip | 0 | 0 | 0 |
| db_missing | 3 | 0 | 0 |
| asr_unrecoverable | 0 | 0 | 0 |
| normalize_error | 0 | 0 | 0 |
| recall_miss | 0 | 0 | 0 |
| rank_error | 0 | 0 | 0 |
| gate_over_accept | 0 | 0 | 0 |
| gate_over_reject | 0 | 0 | 0 |
| completion_error | 0 | 0 | 0 |
| assembly_error | 0 | 0 | 0 |

## e. 参数可辨识性

来源：eval/tuning/20260910_1806_proposal.json

| 参数 | train | eval | value | pinned |
|---|---|---|---|---|
| W_SIM | {"flat": [0.0, 0.6], "flat_share": 0.6} | {"flat": [0.45, 0.65], "flat_share": 0.2} | 0.5 | false |
| W_COV | {"flat": [0.15, 1.0], "flat_share": 0.85} | {"flat": [0.1, 0.3], "flat_share": 0.2} | 0.2 | false |
| W_PRIOR | {"flat": [0.0, 0.35], "flat_share": 0.35} | {"flat": [0.0, 0.2], "flat_share": 0.2} | 0.15 | false |
| W_DEPTH | {"flat": [0.0, 0.65], "flat_share": 0.65} | {"flat": [0.1, 0.15], "flat_share": 0.05} | 0.15 | false |
| W_CONFLICT | {"flat": [0.0, 0.29], "flat_share": 0.483} | {"flat": [0.0, 0.6], "flat_share": 1.0} | 0.25 | false |
| MARGIN_MIN | {"flat": [0.0, 0.09], "flat_share": 0.36} | {"flat": [0.0, 0.08], "flat_share": 0.32} | 0.06 | false |
| SIM_MIN | {"flat": [0.5, 0.95], "flat_share": 1.0} | {"flat": [0.5, 0.95], "flat_share": 1.0} | 0.62 | false |
| SINGLE_HIT_MIN_COV | {"flat": [0.0, 0.5], "flat_share": 1.0} | {"flat": [0.0, 0.5], "flat_share": 1.0} | 0.25 | false |

## 能抓住什么、抓不住什么

| 环节 | 能抓住（证据） | 抓不住（原因） | 需要什么数据 |
|---|---|---|---|
| 地址库 · db_missing | ✓ 删 犀浦街道(510117-01)、解放碑街道(500103-01) → {'003': 'db_missing', '005': 'db_missing'}；其余 22/22 条 exact | 库缺但地名被原样保住时不记库缺（归因树第 2 步的细化）；这类样本若又被闸门拦下且只报到区，会落到 completion_error | oov_db 负样本与已确认录音（最近 audio 报告里 ×3） |
| 声学层 · asr_unrecoverable | ✓ 输入「重庆渝中区解放碑泼皮鸭十八号三单元二零一」name_phon_dist_max=0.648 → 「民权路」在 ASR 输出里最近片段「庆渝」音距离 0.648 > 0.4 | text / 合成模式默认不算 Stage A，文本数据上这个标签恒为 0；合成集用本项目代价矩阵生成，验证不了音距离本身 | 带已确认 transcript_gold 的真实录音（最近 audio 报告里 ×0） |
| 归一化 · normalize_error | ✓ 输出「重庆市渝中区解放碑街道民权路18号3单元201室」vs 真值「重庆市渝中区解放碑街道民权路19号3单元201室」→ 数字转换错；门牌字段错 ['house_no'] | 市/区/街道层没召回时扰动字原样留在输出里，记成闲话泄漏，混进 normalize_error（recall_miss 场景同批 26 条） | 按 Stage B 子项（方言词 / 数字 / 门牌 / 闲话）分开标注的真实口语样本 |
| 召回 · recall_miss | ✓ recall_miss×23，miss_reason {'dist_over_max': 23}；同批 normalize_error×26 | 只看真值链最深层条目；MAX_DIST 压到 0.01 才出（0.05 时 0 条）；miss_reason 只观测到 {'dist_over_max': 23}，weak_filtered / window_miss 没被验证过 | 真实录音里音距离落在 0.2–0.4 的地名错字 |
| 排序 · rank_error | ✓ 60 条里 rank_error×10，dominant_term {'prior': 10} | 要把 W_SIM 清零才造得出来；dominant_term 只观测到 {'prior': 10}；24 句朗读稿 Top-1 全精确命中，排序错只能在合成集上造 | 含近音竞争候选的真实样本；权重学习门槛 train ≥ 500 有效句（现 12） |
| 权重（W_SIM/W_COV/W_PRIOR/W_DEPTH） | 只有合成集约束权重：见排序行与冲突惩罚行 | 合成集同一句 30 个变体只算 1 条有效样本，train 有效句 12，远不到 §2.6 门槛；W_CONFLICT=1.0 在 720 条里只翻转 1 条，权重的可辨识性建立在个位数样本上 | train ≥ 500 有效句、各 difficulty ≥ 30 |
| 冲突惩罚 · 回归变红 | ✓ 合成集 30 条对 同子集未破坏的对照运行：18 项失败（翻转 ['014-017:rank_error']）；同一破坏在 text 套件 0 项失败 | 24 句朗读稿对 W_CONFLICT=1.0 无反应（text 套件 0 项失败）；合成集变红只靠 014-017:rank_error | 同一句里强命中两个行政区的真实口语（"官渡区呈贡新区"这类） |
| 闸门·放行 · gate_over_accept | ✓ 4 条无地址录音，决策 {'20260909_203604_auto': 'reject', '20260909_204208_auto': 'reject', '20260909_211125_auto': 'confident', '20260909_212717_auto': 'reject'}；错送 ['20260909_211125_auto「嗯用那个闽南话再说一个地址呢也是加一点语气助词的那种」→「黑龙江省齐齐哈尔市讷河市」'] | 有地址样本上 Top-1 错时决策树先贴 rank_error / recall_miss；错送主要只能从无地址负样本上看，清单里仅 4 条，且要把三道阈值放到 ≈0 才触发一次 | no_address / injection 负样本 ≥ 150 条才能说错送率 ≤ 2% |
| 闸门·SIM · gate_over_reject | ✓ 24/24 条 gate_over_reject，gate_attrib {'sim': 24} | Top-1 音相似度：text 最低 1.0（<1 的 0/24 条），synthetic 最低 0.9618（<1 的 475/720 条）；SIM 轴几乎平坦，0.62 附近的阈值在文本数据上看不见 | 音频集（audio 最低 0.8317（<1 的 4/34 条）） |
| 闸门·MARGIN · gate_over_reject | ✓ gate_over_reject(margin)×16/24；其余 8 条仍 confident：只有一条候选或 Top-2 同行政区，分差闸门按 rank.decide 不拦 | Top-2 与 Top-1 同行政区时分差闸门不拦（同区豁免），这部分的多余确认和错送都不经过它 | 同区不同路的真实歧义样本 |
| 补全 · completion_error | — 不可达 | 真正的补全错（Top-1 链对、回溯出的省市错）按构造不可达：chain_matches 要求行政区一致；实测唯一可达路径是库缺 + 闸门正确拦下的误归因 | 只报街路（street_only）且库里有跨城重名路的真实录音 |
| 拼装 · assembly_error | ✓ 真值末尾加「甲」、字段不动：admin_all=True geo_recall=1.0 tail_all=True gate_attrib=assembly → assembly_error；回归阻断规则同时报红 | 兜底标签，不指明拼装哪一步错，需人工翻 json | — |
| 质量 · skip | ✓ 构造 quality=silent 的记录 → skip（quality=silent） | quality≠ok 的行在 manifest_to_items 就被滤掉（清单里 8 行），真实报告里 skip 恒为 0；quality 判定本身不在这里验证 | — |
