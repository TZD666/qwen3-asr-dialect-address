---
type: reference
title: data 目录说明
description: 逐目录标注地址库与测试数据的来源、性质与使用注意
tags: [数据说明, 地址库, 评测集, 语音数据]
timestamp: 2026-09-10
---

# data 目录说明

本目录混放了三类性质完全不同的数据，用途和可复用性差别很大，逐个说明。

| 路径 | 类别 | 体积 | 来源 |
|---|---|---|---|
| `addresses/` | 地址库 | 0.8 MB | 公开行政区划数据 |
| `params/` | 后处理参数与调参配置 | 4 KB | `rank_params.json` v1 手设；`eval/tune.py --apply` 写新版本 |
| `eval/xinan_guanhua.json`、`eval/录音指南.md` | 评测集 | 24 KB | 本项目编写 |
| `eval/manifest.jsonl`、`eval/splits.json` | 录音清单与划分 | 30 KB | 脚本生成 + 人工草稿 |
| `eval/negatives/` | 负样本定义 | 4 KB | 本项目编写 |
| `eval/synthetic/` | 合成文本扰动集 | 0.45 MB | `scripts/gen_perturbed.py` 生成 |
| `eval/audio/_smoke/` | 合成语音 | 0.4 MB | TTS 生成 |
| `eval/audio/recordings/` | **真人录音** | 10.7 MB | 作者本人录制 |

---

## addresses/ 地址库

`cn_subset.json` 是运行时实际加载的库，3561 条：

| 层级 | 条数 | 说明 |
|---|---|---|
| 省 | 34 | 全国全量 |
| 市 | 337 | 全国全量 |
| 区县 | 2843 | 全国全量 |
| 路 / 地标 | 347 | **精选，非全量** |

省市区三级依据 GB/T 2260 行政区划代码，原始表 `addresses/raw/adcodes.csv` 取自 [cpca](https://github.com/DQinYuan/chinese_city_adcode) 项目，含经纬度。

路级只有 347 条，是为跑通流程而手工精选的样本，**远非全国全量**。全国道路与小区是百万量级且持续变化，离线穷举不可行，生产环境须接高德或百度 POI 接口（`src/dialect_addr/address_db.py` 已留 `POIProvider` 协议）。

这个不完整性有实际后果，仓库文档里记录了一个真实案例：贵阳「中华中路」被系统改成了「中华北路」，因为库里只有北路和南路。排序算法解决不了正确答案不在候选集里的问题。

**行政区划代码会变。** 民政部自 2026 年起不再单独公布区划代码，撤县设区之类的变动需要对照国家地名信息库自行核对。

## eval/xinan_guanhua.json 评测集

24 句西南官话地址，覆盖 11 个难点维度（同音异形、方言词、口语数字、语序颠倒、地址残缺、生僻字、n-l 不分、平翘舌、h-f 不分、前后鼻音、层级归属陷阱）。

每条含朗读稿（`spoken`）与结构化真值（`ground_truth`）。**地址全部为公开地标或商业地址**，人工编写，不涉及任何真实个人信息。

评测体系 v2 给每条加了分片字段：`dialect_group`（音系）、`sub_dialect`、`address_depth`（说到哪一层：full / district / street_only）、`noise`（clean / filler / complaint）、`orthography_expected`、`gold_in_db`（真值最深层条目是否在库里，**脚本自动算，不手填**）、`negative_type`、`split`。`scripts/extend_eval_items.py --check` 校验这些派生字段与重算一致。

`录音指南.md` 是配套的朗读稿与录音规范，别人可以照着录自己的一套音频。

## eval/manifest.jsonl 录音清单

`recordings/` 下每个文件一行（53 行）。`quality` 由 `scripts/cache_recordings.py` 判定：45 条 ok、5 条静音、2 条浏览器原始格式（已转 wav）、1 条未知格式。

`transcript_gold`（逐字，含口头禅）和 `address_gold` / `fields_gold` 按规矩要由**说话人本人**写，不能事后凭听力标——听录音标注的人会被 ASR 输出带偏。目前的状态用 `label_status` 标明：

| label_status | 条数 | 含义 |
|---|---|---|
| confirmed | 5 | TTS 冒烟音频，合成时的文本已知 |
| draft | 32 | 由 ASR 输出 + 文档记载预填的草稿，**未经说话人确认** |
| pending | 8 | 听不出可靠地址，真值留空，跑但不计分 |
| n/a | 8 | 静音 / 重复 / 未知格式 |

草稿由 `scripts/build_manifest.py` 里的 `DRAFTS` 表生成；说话人确认后把行里的 `label_status` 改成 `confirmed`，重跑脚本不会覆盖 confirmed 的行。`eval/check_manifest.py --strict` 要求全部 confirmed，对外报数前用。

`splits.json` 把 eval / calib / train 物理分开：calib（保序回归）和 train（权重学习）在样本量达到 300 / 500 前为空。

## eval/negatives/ 负样本

- `oov_drop.json`：评测时从地址库临时删掉的 10 条路级条目，选的都是库里有近音邻居的（同名异城的世纪城 / 北京路，音距离 0.25 的天府一街 / 中华南路）。测的是过度纠正：正确答案不在库里时，系统会不会把它"纠"成邻居。
- `injection.json`：`reports/context_injection.json` 六种上下文注入条件的固化版，`known_flip: true` 标记已知会被带偏的两条。回归时翻转数不得增加。

## eval/synthetic/ 合成扰动集

`perturbed.jsonl` 720 条，由 `scripts/gen_perturbed.py` 用 `pinyin_dialect.py` 的混淆表反向生成：对 24 句朗读稿的地名片段按规则换同音 / 近音字，每条记下用的规则和替换前后的拼音，`--verify` 可逐条回放。

**适用边界：它是用本项目自己的代价矩阵生成的，只能测排序、闸门、门牌解析、补全这些下游环节，不能用来验证代价矩阵本身**——那是循环论证。

## eval/audio/_smoke/ 合成语音

2 条 TTS 合成样例，普通话与粤语各一，用于最小冒烟测试与上下文注入实验（`eval/reports/context_injection.json` 用的就是 `tts_mandarin.wav`）。合成语音，无隐私问题。

另有 `tts.m4a`、`tts.webm` 两个格式转换测试文件。

## eval/audio/recordings/ 真人录音

**这是作者本人的真实录音，52 个文件，10.7 MB。**

由测试界面自动落盘产生，`demo/server.py` 每次调用 `/api/transcribe` 都会把音频存到这里。文件名格式 `日期_时间_方言标签.wav`。

需要知道的几点：

构成是 49 个 `.wav`，外加 1 个 `.webm`、1 个 `.m4a`、1 个 `.bin`（后三个是格式转换路径的测试残留，`.bin` 是未知格式的原始 blob）。49 个 wav 里 5 条是静音，44 条有信号，其中 43 条能跑出完整结果。

需要知道的几点：

- **内容是自发口语，不是朗读稿。** 说话人按平时说话的方式报地址，带口头禅、抱怨、语气词。这正是它的价值所在，评测集的朗读稿测不出这些。
- **包含作者本人的声纹。**
- **未经逐条回听确认。** 这些是测试过程中自动落盘的，不是精心录制的数据集。
- 5 条静音是录音时输入设备选到了虚拟声卡造成的，保留下来是因为它们正好是「静音诊断」这个功能的测试样本。
- 这批录音的标注见上面的 `manifest.jsonl` 一节：目前是草稿，说话人确认前只能看趋势。要出裸模型与改造后的正式对照指标，还需按 `录音指南.md` 把 24 句评测集逐条录齐，并按报告第 1 节的样本缺口表补录。

这批数据以原样提供，供研究与复现使用。如果你要在自己的项目里用，建议只当作方言口语地址的样本参考。

## 复现

不下载任何模型权重就能跑通评测集：

```bash
pip install pypinyin numpy
python eval/run_eval.py --mode text
```

应得到 EM 24/24。详见根目录 [README](../README.md)。
