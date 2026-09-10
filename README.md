# 方言地址识别 — 把方言语音里说的地址，认成能送到的地址

用四川话、天津话、粤语说一句地址，语音识别经常把字听错。「劝业场」变成「全叶厂」，「交通巷」变成「高通巷」，字全错了，快递也就送不到了。

这个项目在语音识别之后加一层纠正。它拿转写结果去比对全国行政区划库，在**读音**而不是汉字的空间里找回正确地名，补全说话人没提的省市区，把门牌号按规则抽出来，最后给一个结构化地址。拿不准的时候它会说拿不准，而不是编一个。

底层用 Qwen3-ASR-1.7B，**不需要微调，不需要训练数据**。

## 效果

真人录音，左边是语音识别原始输出，右边是本项目输出。

| 说的话（识别原文） | 系统输出 |
|---|---|
| 天津和平区滨江道一百六十八号**全叶厂**十四楼 | 天津市和平区滨江道168号**劝业场**14楼 |
| 成都市花牌坊**高通巷**三十八号碧园公寓二栋四单元六零三 | 四川省成都市金牛区花牌坊街**交通巷**38号碧园公寓2栋4单元603室 |
| 哎呀真的烦，你能不能给我寄到那个花牌坊高通巷三十八号碧园公寓 | **四川省成都市金牛区**花牌坊街交通巷38号碧园公寓 |
| 喏侬听好，上海市静安区南京西路**八百一百八**号 | 上海市静安区南京西路八百一百八号（门牌字段留空，整段标未核验） |

第三行说话人全程没提省市区，是靠行政区划树回溯补出来的。第四行「八百一百八」不是合法的中文数字，识别听错了，系统拒绝把它算成 908 号，门牌字段留空、原文原样交出来。

结构化输出长这样：

结构化输出（`POST /api/text` 的真实返回，省略了空字段和中间结果）：

```json
{
  "address": "四川省成都市金牛区花牌坊街交通巷38号碧园公寓2栋4单元603室",
  "decision": "confident",
  "reason": "分差 0.351",
  "fields": {
    "province": "四川省",
    "city": "成都市",
    "district": "金牛区",
    "road": "花牌坊街",
    "community": "交通巷",
    "house_no": "38号",
    "building": "2栋",
    "unit": "4单元",
    "room": "603室",
    "unverified": "碧园公寓"
  }
}
```

`decision` 有四档。`confident` 可直接用，`ambiguous` 会在 `nbest` 里给候选列表让人选，`partial` 表示只匹配到省市级，`reject` 表示不采信、保留原文。`unverified` 装的是地址库里没有、但原样保留下来的片段，不会被悄悄吞掉。

## 三分钟先跑起来（不用下模型）

后处理这一层完全不依赖模型权重。装两个纯 Python 包就能跑通 24 句评测集：

```bash
pip install pypinyin numpy
python eval/run_eval.py --mode text
```

预期 `归一化完全匹配（EM） 100.0%`，24 条全部 confident。再跑一下门牌解析的回归用例：

```bash
python tests/test_normalize.py
```

跑通了再决定要不要下那 4.4 GB 权重。

## 完整安装

| 项 | 要求 |
|---|---|
| Python | 3.10 ~ 3.12，推荐 3.12。3.13+ 未验证 |
| 磁盘 | 权重 4.4 GB，加依赖约 6 GB |
| 内存 | 显存 4 GB 以上，或纯 CPU 时约 8 GB 内存 |
| ffmpeg | 可选，只有在网页上录 webm/m4a 时需要 |

**macOS（Apple Silicon）**

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

自动走 MPS，单句推理 0.4~2.2 秒。

**Windows**

torch 要单独装，PyPI 上的默认轮子是纯 CPU 版，有 N 卡必须指定 CUDA 源。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# 有 NVIDIA 显卡（cu124 换成你的 CUDA 版本）
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 没有独显
pip install torch

pip install -r requirements.txt
```

Windows 上有两个坑要提前知道。一是控制台编码，脚本会打印 `✓` 和中文，重定向到文件时中文系统默认 GBK 会报 `UnicodeEncodeError`，先执行 `chcp 65001` 或设 `set PYTHONIOENCODING=utf-8`。二是纯 CPU 跑 1.7B 模型单句要几十秒、吃 7 GB 内存，只想看后处理效果就用上面的文本模式，别下权重。

**Linux**

和 Windows 一样单独装 torch，其余相同。

## 下载模型权重

```bash
pip install huggingface_hub
hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir models/Qwen3-ASR-1.7B-hf
```

国内网络慢就走镜像：

```bash
# macOS / Linux
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir models/Qwen3-ASR-1.7B-hf
```

```powershell
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"; hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir models/Qwen3-ASR-1.7B-hf
```

权重不想放仓库里就下到别处，然后设 `DIALECT_ADDR_MODEL_DIR` 指过去。

<details>
<summary>已经有非 -hf 版权重时的转换办法</summary>

官方发了两套。`Qwen3-ASR-1.7B` 是 qwen-asr 包和 vLLM 用的 `thinker.*` 布局，transformers 原生类加载不了；`-hf` 版是原生布局，张量内容相同。已经下了非 `-hf` 版可以本地转换，省一次 4.7 GB 下载：

```bash
python scripts/verify_model.py     # 校验大小 + safetensors 头部 + sha256
python scripts/convert_to_hf.py    # 纯键名重命名，708/708 键 + 形状双校验，约 1 秒
```

两个前提。转换脚本需要 `models/_hf_ref/` 下的 4 个配置文件，用 `hf download Qwen/Qwen3-ASR-1.7B-hf --include "*.json" "*.jinja" --local-dir models/_hf_ref` 取。另外转换不删源，磁盘要占两份约 10 GB，直接下 `-hf` 版只要 4.4 GB。

`verify_model.py` 里的 sha256 是 2026-09-09 的上游快照，官方若重新上传会校验失败，那时用文件大小核对即可。
</details>

## 网页测试界面

```bash
python demo/server.py                              # → http://127.0.0.1:8850
python demo/server.py --no-model                   # 不加载权重，只开文本接口
python demo/server.py --port 8851 --host 0.0.0.0   # 换端口 / 允许局域网访问
```

界面可以直接录音、上传音频、或者输入文本直测。顶部两栏对照，左边是语音识别的原始输出，右边是本项目输出，一眼看出后处理改了什么。默认只绑 `127.0.0.1`。

## 命令行评测

```bash
python eval/run_eval.py --mode text                      # 朗读稿当完美识别，只测后处理，不需要权重
python eval/run_eval.py --mode text --negatives oov      # 库缺负样本：临时删 10 条库条目，看会不会过度纠正
python eval/run_eval.py --mode text --oracle all         # 库完备 / 排序完美 / 闸门完美 各能推到多少
python eval/run_eval.py --mode text --eval data/eval/synthetic/perturbed.jsonl --tag synthetic   # 720 条合成扰动
python eval/run_eval.py --mode audio                     # 按 data/eval/manifest.jsonl 跑全部录音，配对 baseline
python eval/regression.py                                # 无模型回归：与 eval/golden 逐分片比，退化即失败
python eval/selfcheck.py --fast                          # 评测自检：故意破坏，看每个归因标签抓不抓得住
python eval/features.py && python eval/tune.py           # 调参闭环：检索一次、重放搜权重阈值 → 提案（不落地）
python eval/tune.py --apply eval/tuning/<提案>.json       # 人看过提案后才写 data/params/rank_params.json
```

后处理的参数（五个权重、三道阈值、四个检索常量）不在代码里，在 `data/params/rank_params.json`，带版本、来源（手设 / 学出）和数据快照指纹；`/api/status` 和每份报告头都报当前版本与代码提交号。新录音进来的路径是 `scripts/intake.py`：转写进缓存 → 补清单行 → 说话人填真值 → 校验 → 特征 dump → 提案。提案里最要紧的是**可辨识表**：每个参数在当前数据上被钉住了多少。数据钉不住的参数，闭环不会动它。

评测不只报一个总分。每条样本按六个阶段分别打分（裸 ASR、归一化、召回、排序、闸门、补全），端到端错的样本用决策树贴**一个**归因标签（库缺 / 声学不可恢复 / 归一化错 / 召回漏 / 排错 / 闸门放行 / 闸门误拦 / 补全错），报告按方言、地址深度、口语噪声、难点、负样本类型分片，百分比一律带 Wilson 区间，n < 20 只报 k/n。三类负样本（库缺、ASR 已对、注入带偏）和三个 oracle 开关分别回答"改坏率多少"和"每个环节的天花板多高"。完整设计见 [评测体系设计.md](评测体系设计.md)，实施与调优结果见 [docs/方案推演.md](docs/方案推演.md) 第 16、17 节。

`201室` 错成 `202室` 只差一个字符但快递彻底送错门，所以端到端主指标仍是字段级精确匹配，另有与损失对齐的决策四格：错送率（confident 却错）对自动通过率。

当前数据上的结果（2026-09-10，全部可用 `eval/regression.py` 与 `eval/run_eval.py --mode audio` 复现）：

| 集合 | n | 结果 |
|---|---|---|
| 朗读稿 text | 24 | EM 24/24 |
| 合成扰动 | 720 | EM 720/720（调优前 690） |
| 库缺负样本 | 10 | 过度纠正 0（调优前 2），原文保住 10 |
| 真人录音（草稿真值） | 33 | EM 30/33，错送 1，多余确认 0，裸 ASR 已对的 21 条改坏 0 |
| 调参提案 | — | 训练片 12 句有效样本，参数全部"平"，REPORT_ONLY |

录音那一行的真值未经说话人确认，只能看趋势。剩下的 3 条录音错例全是地址库里没有那条路，接 POI 才解决得了。调优过程见 [docs/方案推演.md](docs/方案推演.md) 第 16 节，闭环与自检见第 17 节。

## 工作原理

一句话：**语音识别只负责听音，地址库负责定字。**

方言误识有个可利用的性质，错的是字，音基本还在。「全叶厂」和「劝业场」拼音完全相同，「高通巷」和「交通巷」只差一个声母。所以匹配在读音空间做，不在汉字空间做。

```
音频
 ↓  Qwen3-ASR 第 1 遍：裸转写 + 方言标签
 ↓  方言词归一（喺→在、屋头→家）
 ↓  口语数字转阿拉伯数字，剥离门牌/楼栋/单元/室
 ↓  按方言选音系空间（普通话拼音 / 粤拼 / 台罗）
 ↓  滑窗对齐找命中 → 沿父链回溯补全 → 打分 → 决策
 ↓  Qwen3-ASR 第 2 遍：把候选写进 system prompt 重新解码（可关）
规范地址 + 结构化字段 + N-best
```

四个设计要点：

- **按音系路由，不是一套拼音打天下。** 模型的 22 个方言标签是按省份开的，粤语有普通话没有的入声韵尾，拿普通话拼音去量等于用错尺子。
- **加权编辑距离。** n/l 不分、见系不颚化这类系统性音变的替换代价压到 0.1~0.2，随机混淆仍是 1，共 20 条声母规则、26 条韵母规则。
- **层级是硬约束。** 候选链一律由「路→区→市→省」回溯生成，把假设空间从 113 亿压到 3561 条，「成都市+江北区」这种链根本构造不出来。
- **三道阈值防自信答错。** Top1/Top2 分差不足且指向不同行政区就吐候选让人确认；音相似度过低就保留原文；门牌以下绝不从库里猜。

完整推导过程见 [docs/方案推演.md](docs/方案推演.md)。

## 已知边界

- **路级地址库只有 347 条**，是为跑通流程精选的样本。省市区三级是全国全量 3214 条，但街道、道路、小区是开放集，百万量级且持续变化，生产环境必须接高德或百度 POI 接口（代码已留接口）。实测吃过亏：贵阳「中华中路」被系统改成了「中华北路」，因为库里只有北路和南路。正确答案不在候选集里的时候，排序算法救不了。
- **粤拼路径没有端到端验证。** 依赖可选包 `pycantonese`，作者机器上因网络问题装不上，粤语实际走的是「普通话拼音·粤语近似」降级路径。降级机制本身可用，`/api/status` 会报出原因。台罗（闽南语）已装并验证。
- **吴语、浙江、湘、赣没有成熟罗马化库**，只能用普通话近似，信号较弱。
- **真实录音的标注还是草稿。** 46 条可用录音已进 `data/eval/manifest.jsonl`，但其中 32 条的逐字稿和规范地址是按 ASR 输出加文档记载预填的草稿，9 条听不清的留空，只有 5 条 TTS 是确认过的。说话人本人过目、把 `label_status` 改成 `confirmed` 之前，音频模式的数字只能看趋势，不能对外报。EM 100% 那个数字来自文本模式，用朗读稿当完美识别输入。
- **样本量撑不起置信区间。** 24 句朗读稿 + 33 条有标注录音，任何分片都不到 150 条，报告里每个百分比旁边的 Wilson 区间都很宽；样本缺口表（报告第 1 节）列了哪个格子还差多少。
- **参数仍是手设值，调参闭环跑得通但落不了地。** 五个权重、三道阈值现在是 `data/params/rank_params.json` v1（hand-set）。闭环（`eval/tune.py`）在今天的数据上给出的结论是：训练片只有 12 句有效样本（门槛 500），目标函数是平的，8 个可重放参数没有一个被数据钉住，提案与手设一致、verdict 为 REPORT_ONLY。这是如实的结果，不是闭环没跑。保序回归校准同样只报不存。
- **评测自检列出了抓不住的东西。** `eval/selfcheck/latest.md` 收尾那张表写明：朗读稿对冲突惩罚权重毫无反应，只有合成集的 1 条样本能让回归变红；文本数据上音相似度全为 1.0，SIM 闸门只有音频集能测；市区街道层没召回时会被记成闲话泄漏，是归因树的已知盲区。

## 常见问题

**录音全是静音，模型返回空串。** 装了 BlackHole、VB-Cable 这类虚拟声卡时系统默认输入常被它占用，录出来是全零音轨。界面会显示当前输入设备和实时电平并自动跳过虚拟设备，手动选一个真实麦克风即可。

**本地接口连不上或测试全红。** 开了代理软件的 TUN 模式时 `127.0.0.1` 可能被劫持，跑测试带上 `no_proxy=127.0.0.1,localhost`。

**上传 webm/m4a 报错要 ffmpeg。** 浏览器直传 wav 不需要，只有 MediaRecorder 回退路径要转码。macOS `brew install ffmpeg`，Windows `winget install Gyan.FFmpeg`。

**`.m4a` / `.aac` 评测文件读不出来。** librosa 新版去掉了 audioread，任何平台都解不了，先用 ffmpeg 转成 wav。

## 仓库结构

```
src/dialect_addr/
  pinyin_dialect.py   方言音变代价矩阵 + 音节级加权编辑距离
  romanize.py         多音系路由：22 方言 → 拼音/粤拼/台罗，库缺失自动降级
  dialect_lexicon.py  方言词汇归一化（封闭集查表）
  normalize.py        口语数字归一化 + 门牌/楼栋/单元/室抽取（纯规则）
  address_db.py       分层地址库 + 高德/百度 POI 接口
  rank.py             滑窗命中 → 父链回溯 → 打分 → 三道阈值决策
  asr.py              Qwen3-ASR 封装，两遍解码
  pipeline.py         端到端编排，保留每一步中间结果
demo/                 标准库 HTTP 服务 + 单文件 HTML 测试界面
eval/
  run_eval.py         分阶段评测：text / baseline / audio，负样本、oracle、阈值网格、配对统计
  stages.py           六个阶段的打分函数（纯函数）
  attribution.py      错误归因决策树
  regression.py       无模型回归套件，与 eval/golden 快照逐分片比
  selfcheck.py        评测自检：故意破坏 → 每个归因标签抓得住；"能抓住什么、抓不住什么"
  features.py         调参闭环第一步：检索一次，存全部候选链的五项分量
  rescore.py          任意 (权重, 阈值) 毫秒级重放三道闸门
  tune.py             搜权重阈值 → 可辨识表 → 约束 → 提案；--apply 写参数新版本
  calibrate.py        保序回归校准（只报不接闸门）
  splits.py           eval / calib / train 划分与有效样本门槛
  check_manifest.py   录音清单与评测集 schema 校验
  asr_cache.py        ASR 结果磁盘缓存（同一段音频只让模型跑一次）
data/params/          后处理参数（版本化 JSON）与调参目标配置
scripts/              权重校验、格式转换、地址库构建、清单生成、合成扰动集生成
data/                 地址库与测试数据，来源标注见 data/README.md
docs/方案推演.md       完整的方案推导过程与评测调优结果
评测体系设计.md        评测体系 v2 设计（分阶段、分片、负样本、oracle）
.github/workflows/    CI：库 / 代价矩阵 / 权重 / 拼装任一改动都跑无模型回归
```
