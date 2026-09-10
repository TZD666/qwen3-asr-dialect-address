#!/usr/bin/env python3
"""生成 / 合并 data/eval/manifest.jsonl（评测体系设计 §2.2）。

    .venv/bin/python scripts/cache_recordings.py      # 先跑，产出质量判定 + 裸 ASR 草稿
    .venv/bin/python scripts/build_manifest.py        # 再跑本脚本

每条录音一行。质量（silent / unknown_format / duplicate）由脚本判定；
`transcript_gold` / `address_gold` / `fields_gold` 按规矩必须由**说话人本人**写。
本脚本做的是把已知信息预填成**草稿**（label_status=draft）：
  * TTS 冒烟音频：文本已知（合成时的输入），直接算 confirmed
  * 文档里作者已明确写过真值的录音（劝业场 / 交通巷 / 中华中路 / 花牌坊抱怨）：按文档填
  * 其余：裸 ASR 输出放在 transcript_asr 里做参考，草稿由维护者按听感整理，等说话人过目
  * 听不出可靠地址的：label_status=pending，gold 留空

合并规则：已有 manifest 里 label_status=confirmed 的行**永不覆盖**；其余行用本脚本的草稿刷新。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr.address_db import AddressDB  # noqa: E402
from stages import address_depth_of, gold_chain  # noqa: E402

REC = ROOT / "data" / "eval" / "audio" / "recordings"
QUALITY = ROOT / "eval" / "cache" / "recordings_quality.json"
OUT = ROOT / "data" / "eval" / "manifest.jsonl"
EVAL = ROOT / "data" / "eval" / "xinan_guanhua.json"

F = ("province", "city", "district", "street", "road", "community", "house_no", "building", "unit", "room", "floor")


def fields(**kw) -> dict:
    return {k: kw.get(k, "") for k in F}


# TTS 冒烟音频（经测试页上传，重编码过，按时长+RMS 认）
TTS_MANDARIN = {
    "set": "tts", "item_id": "001", "speaker_id": "tts", "dialect_group": "官话", "sub_dialect": "普通话(TTS)",
    "transcript_gold": "我家在成都市武侯区红牌楼街道二环路南四段三十号",
    "address_gold": "四川省成都市武侯区红牌楼街道二环路南四段30号",
    "fields_gold": fields(province="四川省", city="成都市", district="武侯区", street="红牌楼街道", road="二环路南四段", house_no="30号"),
    "noise": "clean", "orthography_expected": "simplified", "label_status": "confirmed",
    "gold_source": "eval/reports/context_injection.json 真值",
}
TTS_CANTONESE = {
    "set": "tts", "item_id": None, "speaker_id": "tts", "dialect_group": "粤", "sub_dialect": "广州话(TTS)",
    "transcript_gold": "我住喺广州市天河区体育西路一百零三号",
    "address_gold": "广东省广州市天河区体育西路103号",
    "fields_gold": fields(province="广东省", city="广州市", district="天河区", road="体育西路", house_no="103号"),
    "noise": "clean", "orthography_expected": "cantonese_written", "label_status": "confirmed",
    "gold_source": "docs/方案推演.md §4",
}

HUAPAIFANG = dict(province="四川省", city="成都市", district="金牛区", street="花牌坊街", road="交通巷")

# 说话人本人的自发录音：草稿。transcript_gold 逐字（含口头禅），address_gold 规范地址。
DRAFTS: dict[str, dict] = {
    "20260909_200024_auto.wav": TTS_CANTONESE,
    "20260909_200530_auto.wav": TTS_MANDARIN,
    "20260909_200716_auto.wav": TTS_MANDARIN,
    "20260909_200717_auto.wav": TTS_MANDARIN,
    "20260909_200718_auto.wav": TTS_MANDARIN,
    "20260909_201002_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "成都话", "noise": "clean",
        "transcript_gold": "成都市花牌坊交通巷三十八号",
        "address_gold": "四川省成都市金牛区花牌坊街交通巷38号",
        "fields_gold": fields(**HUAPAIFANG, house_no="38号"), "gold_source": "docs/方案推演.md §6（交通巷）",
    },
    "20260909_201418_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "成都话", "noise": "clean",
        "transcript_gold": "成都市花牌坊交通巷三十八号",
        "address_gold": "四川省成都市金牛区花牌坊街交通巷38号",
        "fields_gold": fields(**HUAPAIFANG, house_no="38号"), "gold_source": "docs/方案推演.md §6（ASR 听成高通巷）",
    },
    "20260909_201452_auto.wav": {"dialect_group": "粤", "sub_dialect": "广州话", "orthography_expected": "traditional",
                                 "label_status": "pending", "notes": "ASR 输出「北京路二百三三七號越海養中匯」不可靠，需说话人补真值"},
    "20260909_201614_auto.wav": {"dialect_group": "粤", "sub_dialect": "广州话", "orthography_expected": "traditional",
                                 "label_status": "pending", "notes": "「京滬百納廣場」存疑（疑为京基百纳），需说话人补真值"},
    "20260909_201659_auto.wav": {"dialect_group": "吴", "sub_dialect": "上海话", "noise": "filler",
                                 "label_status": "pending", "notes": "「南京东路一弄四十三号，永安百货北路」尾段存疑"},
    "20260909_201851_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "东北话", "noise": "filler",
        "transcript_gold": "妥了听好辽宁省沈阳市沈河区中街路一百六十八号兴隆大家庭十一楼",
        "address_gold": "辽宁省沈阳市沈河区中街路168号兴隆大家庭11楼",
        "fields_gold": fields(province="辽宁省", city="沈阳市", district="沈河区", road="中街路", community="兴隆大家庭", house_no="168号", floor="11楼"),
    },
    "20260909_202057_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "成都话", "noise": "clean",
        "transcript_gold": "成都市花牌坊交通巷三十八号碧园公寓二栋四单元六零三",
        "address_gold": "四川省成都市金牛区花牌坊街交通巷38号碧园公寓2栋4单元603室",
        "fields_gold": fields(**HUAPAIFANG, community="碧园公寓", house_no="38号", building="2栋", unit="4单元", room="603室"),
        "gold_source": "docs/方案推演.md §6（ASR 听成高通巷）",
    },
    "20260909_203503_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "成都话", "noise": "clean",
        "transcript_gold": "成都市花牌坊交通巷三十八号",
        "address_gold": "四川省成都市金牛区花牌坊街交通巷38号",
        "fields_gold": fields(**HUAPAIFANG, house_no="38号"), "gold_source": "docs/方案推演.md §6",
    },
    "20260909_203544_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "贵阳话", "noise": "clean",
        "transcript_gold": "贵阳市云岩区中华中路一百二十号喷水池国贸广场二十三楼",
        "address_gold": "贵州省贵阳市云岩区中华中路120号喷水池国贸广场23楼",
        "fields_gold": fields(province="贵州省", city="贵阳市", district="云岩区", road="中华中路", community="喷水池国贸广场", house_no="120号", floor="23楼"),
        "gold_source": "docs/方案推演.md §10（模型输出完全正确，库里只有中华北/南路）",
    },
    "20260909_203604_auto.wav": {"has_address": False, "transcript_gold": "嗯", "negative_type": "no_address", "noise": "clean",
                                 "dialect_group": "官话", "notes": "1.8s 只有一个语气词，期望输出为空/reject"},
    "20260909_203614_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "昆明话", "noise": "clean",
        "transcript_gold": "昆明五华区正义路九十九号北盛购物中心十五楼",
        "address_gold": "云南省昆明市五华区正义路99号北盛购物中心15楼",
        "fields_gold": fields(province="云南省", city="昆明市", district="五华区", road="正义路", community="北盛购物中心", house_no="99号", floor="15楼"),
        "notes": "「北盛购物中心」按 ASR 听写，需说话人确认",
    },
    "20260909_203639_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "武汉话", "noise": "filler",
        "transcript_gold": "那我来句地道武汉腔的武汉市江汉区江汉路步行街一百零八号大洋百货十二楼",
        "address_gold": "湖北省武汉市江汉区江汉路步行街108号大洋百货12楼",
        "fields_gold": fields(province="湖北省", city="武汉市", district="江汉区", road="江汉路步行街", community="大洋百货", house_no="108号", floor="12楼"),
    },
    "20260909_204208_auto.wav": {"has_address": False, "transcript_gold": "来句河南话说一个地址", "negative_type": "no_address",
                                 "noise": "complaint", "dialect_group": "官话", "sub_dialect": "河南话", "notes": "整段无地址，期望 reject"},
    "20260909_204217_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "河南话", "noise": "filler",
        "transcript_gold": "好啊郑州市二七区德化街一百号亚细亚商场十七楼",
        "address_gold": "河南省郑州市二七区德化街100号亚细亚商场17楼",
        "fields_gold": fields(province="河南省", city="郑州市", district="二七区", road="德化街", community="亚细亚商场", house_no="100号", floor="17楼"),
    },
    "20260909_204305_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "南阳话", "noise": "filler",
        "transcript_gold": "用咱南阳河南话跟你说南阳市卧龙区人民路三十六号万德隆购物广场九楼",
        "address_gold": "河南省南阳市卧龙区人民路36号万德隆购物广场9楼",
        "fields_gold": fields(province="河南省", city="南阳市", district="卧龙区", road="人民路", community="万德隆购物广场", house_no="36号", floor="9楼"),
    },
    "20260909_205915_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "南阳话", "noise": "filler",
        "transcript_gold": "中就说咱南阳本地的淅川县淅川大道与人民路交叉口往南两百米盛世商贸城十二栋三号",
        "address_gold": "河南省南阳市淅川县淅川大道与人民路交叉口往南两百米盛世商贸城12栋3号",
        "fields_gold": fields(province="河南省", city="南阳市", district="淅川县", road="淅川大道", community="盛世商贸城", building="12栋", house_no="3号"),
        "notes": "ASR 把淅川听成西川；交叉口描述式地址，规范化口径待定",
    },
    "20260909_205957_auto.wav": {"dialect_group": "官话", "sub_dialect": "成都话", "label_status": "pending",
                                 "notes": "「春熙路东段一号IFS二九九」尾段存疑，需说话人补真值"},
    "20260909_210150_auto.wav": {
        "dialect_group": "粤", "sub_dialect": "澳门粤语", "orthography_expected": "traditional", "noise": "clean",
        "transcript_gold": "澳门大堂区殷皇子大马路三百三十三号新濠天地二十五楼",
        "address_gold": "澳门特别行政区大堂区殷皇子大马路333号新濠天地25楼",
        "fields_gold": fields(province="澳门特别行政区", city="澳门特别行政区", district="大堂区", road="殷皇子大马路", community="新濠天地", house_no="333号", floor="25楼"),
        "notes": "ASR 把殷皇子听成欣皇子；docs §14 的「澳门被判弱命中误匹配到吉林」案例",
    },
    "20260909_210550_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "天津话", "noise": "clean",
        "transcript_gold": "天津和平区滨江道一百六十八号劝业场十四楼",
        "address_gold": "天津市和平区滨江道168号劝业场14楼",
        "fields_gold": fields(province="天津市", city="天津市", district="和平区", road="滨江道", community="劝业场", house_no="168号", floor="14楼"),
        "gold_source": "docs/方案推演.md §6（全叶厂→劝业场）",
    },
    "20260909_210749_auto.wav": {
        "dialect_group": "闽", "sub_dialect": "泉州话", "noise": "clean",
        "transcript_gold": "泉州市鲤城区中山路二百一十七号泉州百和大楼九楼",
        "address_gold": "福建省泉州市鲤城区中山路217号泉州百和大楼9楼",
        "fields_gold": fields(province="福建省", city="泉州市", district="鲤城区", road="中山路", community="泉州百和大楼", house_no="217号", floor="9楼"),
        "notes": "「百和大楼」按 ASR 听写，需确认",
    },
    "20260909_210829_auto.wav": {"dialect_group": "闽", "sub_dialect": "台湾闽南语", "orthography_expected": "traditional",
                                 "label_status": "pending", "notes": "ASR 输出「民雄區甘蔗街到淡水線中大路」不可靠"},
    "20260909_211020_auto.wav": {
        "dialect_group": "闽", "sub_dialect": "厦门话", "noise": "clean",
        "transcript_gold": "厦门市思明区中山路一百九十三号",
        "address_gold": "福建省厦门市思明区中山路193号",
        "fields_gold": fields(province="福建省", city="厦门市", district="思明区", road="中山路", house_no="193号"),
    },
    "20260909_211055_auto.wav": {
        "dialect_group": "闽", "sub_dialect": "厦门话", "noise": "filler", "orthography_expected": "traditional",
        "transcript_gold": "哎你听好哦厦门市思明区中山路一百九十三号啦",
        "address_gold": "福建省厦门市思明区中山路193号",
        "fields_gold": fields(province="福建省", city="厦门市", district="思明区", road="中山路", house_no="193号"),
    },
    "20260909_211125_auto.wav": {"has_address": False, "transcript_gold": "嗯用那个闽南话再说一个地址呢也是加一点语气助词的那种",
                                 "negative_type": "no_address", "noise": "complaint", "dialect_group": "官话", "notes": "整段无地址，期望 reject"},
    "20260909_211137_auto.wav": {"dialect_group": "闽", "sub_dialect": "漳州话", "orthography_expected": "traditional", "noise": "filler",
                                 "label_status": "pending", "notes": "「洪德區土門街」存疑（漳州无洪德区），需说话人补真值"},
    "20260909_211216_auto.wav": {"dialect_group": "吴", "sub_dialect": "上海话", "noise": "filler", "label_status": "pending",
                                 "notes": "ASR 听成「八百一百八号」（非法数串，tests/test_normalize.py 的案例）；真实门牌需说话人补"},
    "20260909_212302_auto.wav": {
        "dialect_group": "吴", "sub_dialect": "上海话", "noise": "clean",
        "transcript_gold": "上海市静安区南京西路八百一十号",
        "address_gold": "上海市静安区南京西路810号",
        "fields_gold": fields(province="上海市", city="上海市", district="静安区", road="南京西路", house_no="810号"),
    },
    "20260909_212319_auto.wav": {
        "dialect_group": "吴", "sub_dialect": "上海话", "noise": "filler",
        "transcript_gold": "好侬听好上海市静安区南京西路八百一十八号",
        "address_gold": "上海市静安区南京西路818号",
        "fields_gold": fields(province="上海市", city="上海市", district="静安区", road="南京西路", house_no="818号"),
    },
    "20260909_212400_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "天津话", "noise": "filler",
        "transcript_gold": "嚯您听好喽天津市南开区古文化街七十七号嘛",
        "address_gold": "天津市南开区古文化街77号",
        "fields_gold": fields(province="天津市", city="天津市", district="南开区", road="古文化街", house_no="77号"),
    },
    "20260909_212600_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "兰州话", "noise": "complaint",
        "transcript_gold": "五十二号建兰市场门口老酿皮铺地道的美食",
        "address_gold": "甘肃省兰州市七里河区西站西路52号建兰市场门口老酿皮铺",
        "fields_gold": fields(province="甘肃省", city="兰州市", district="七里河区", road="西站西路", community="建兰市场", house_no="52号"),
        "notes": "只说了门牌和市场名（下一条的半截）；省市区路只能靠库补——库里没有建兰市场，预期 reject/partial",
    },
    "20260909_212623_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "兰州话", "noise": "complaint",
        "transcript_gold": "兰州市七里河区西站西路五十二号建兰市场门口老酿皮铺还是兰州腔给您唠的明明白白的",
        "address_gold": "甘肃省兰州市七里河区西站西路52号建兰市场门口老酿皮铺",
        "fields_gold": fields(province="甘肃省", city="兰州市", district="七里河区", road="西站西路", community="建兰市场", house_no="52号"),
    },
    "20260909_212717_auto.wav": {"has_address": False, "transcript_gold": "嗯", "negative_type": "no_address", "noise": "clean",
                                 "dialect_group": "官话", "notes": "0.9s 语气词，期望输出为空"},
    "20260909_212731_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "银川话", "noise": "filler",
        "transcript_gold": "用兰银官话银川宁夏本地腔说银川市兴庆区解放西街一百一十七号老新华书店",
        "address_gold": "宁夏回族自治区银川市兴庆区解放西街117号老新华书店",
        "fields_gold": fields(province="宁夏回族自治区", city="银川市", district="兴庆区", road="解放西街", community="老新华书店", house_no="117号"),
    },
    "20260909_212954_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "东北话", "noise": "filler",
        "transcript_gold": "那必须整地道东北大碴子味儿沈阳市沈河区中街路一百五十六号老北市对龙麻花铺",
        "address_gold": "辽宁省沈阳市沈河区中街路156号老北市对龙麻花铺",
        "fields_gold": fields(province="辽宁省", city="沈阳市", district="沈河区", road="中街路", community="老北市对龙麻花铺", house_no="156号"),
        "notes": "「对龙麻花铺」按 ASR 听写，需确认",
    },
    "20260909_213038_auto.wav": {
        "dialect_group": "闽", "sub_dialect": "厦门话", "noise": "complaint", "orthography_expected": "traditional",
        "transcript_gold": "好你听好啦厦门市思明区鼓浪屿龙头路八十九号够有闽南厝的烟火气",
        "address_gold": "福建省厦门市思明区鼓浪屿龙头路89号",
        "fields_gold": fields(province="福建省", city="厦门市", district="思明区", street="鼓浪屿", road="龙头路", house_no="89号"),
        "notes": "尾句「够有闽南厝的烟火气」是闲话；ASR 写成萬南厝嘅",
    },
    "20260909_213143_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "重庆话", "noise": "complaint",
        "transcript_gold": "要得用四川话给你扯起说重庆渝中区解放碑八一路好吃街幺二六号老火锅馆巴适的板",
        "address_gold": "重庆市渝中区解放碑街道八一路好吃街126号老火锅馆",
        "fields_gold": fields(province="重庆市", city="重庆市", district="渝中区", street="解放碑街道", road="八一路", community="好吃街", house_no="126号"),
        "gold_source": "docs/方案推演.md §14（巴适的板被带入地址串）",
    },
    "20260909_213221_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "兰州话", "noise": "complaint",
        "transcript_gold": "兰州腔跟你说兰州市城关区正宁路小吃街七十三号老牛奶鸡蛋醪糟摊地道甘肃味儿",
        "address_gold": "甘肃省兰州市城关区正宁路小吃街73号老牛奶鸡蛋醪糟摊",
        "fields_gold": fields(province="甘肃省", city="兰州市", district="城关区", road="正宁路小吃街", community="老牛奶鸡蛋醪糟摊", house_no="73号"),
    },
    "20260909_214344_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "成都话", "noise": "complaint",
        "transcript_gold": "哎呀真的烦求的要死你能不能给我寄到那个花牌坊交通巷三十八号碧园公寓",
        "address_gold": "四川省成都市金牛区花牌坊街交通巷38号碧园公寓",
        "fields_gold": fields(**HUAPAIFANG, community="碧园公寓", house_no="38号"),
        "gold_source": "评测体系设计.md §2.2 示例 / docs §9",
    },
    "20260909_231344_auto.wav": {
        "dialect_group": "官话", "sub_dialect": "成都话", "noise": "clean",
        "transcript_gold": "成都市花牌坊交通巷三十八号碧园公寓",
        "address_gold": "四川省成都市金牛区花牌坊街交通巷38号碧园公寓",
        "fields_gold": fields(**HUAPAIFANG, community="碧园公寓", house_no="38号"),
        "gold_source": "docs/方案推演.md §6（ASR 听成高通巷）",
    },
    "20260910_011958_auto.wav": {"dialect_group": "吴", "sub_dialect": "杭州话", "label_status": "pending",
                                 "notes": "「人才公寓二栋一单元二零」室号疑似被截断，需说话人补真值"},
}


def build(db: AddressDB) -> list[dict]:
    qual = json.loads(QUALITY.read_text(encoding="utf-8")) if QUALITY.exists() else {}
    items = {it["id"]: it for it in json.loads(EVAL.read_text(encoding="utf-8"))["items"]}
    rows = []
    for p in sorted(REC.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        q = qual.get(p.name, {"quality": "unknown_format"})
        row: dict = {
            "file": f"recordings/{p.name}", "set": "spontaneous", "item_id": None, "speaker_id": "spk01",
            "dialect_group": "", "sub_dialect": "", "transcript_gold": "", "address_gold": "", "fields_gold": {},
            "address_depth": "", "noise": "", "orthography_expected": "simplified", "gold_in_db": None,
            "negative_type": None, "split": "eval", "quality": q["quality"],
            "label_status": "pending", "has_address": True,
            "duration": q.get("dur"), "transcript_asr": q.get("asr_text", ""), "asr_language": q.get("asr_language"),
        }
        if q["quality"] == "duplicate":
            row["notes"] = f"浏览器原始格式，已转成 {q['of']}；评测跳过"
        d = DRAFTS.get(p.name)
        if d:
            row.update({k: v for k, v in d.items()})
            if "label_status" not in d:
                row["label_status"] = "draft"
        if row["item_id"] and row["item_id"] in items and not row["fields_gold"]:
            it = items[row["item_id"]]
            row["address_gold"], row["fields_gold"] = it["ground_truth"], it["fields"]
        if row["quality"] != "ok":
            row["label_status"] = "n/a"
        if not row["has_address"]:
            row["address_depth"] = "none"
        elif row["fields_gold"]:
            row["address_depth"] = row.get("address_depth") or address_depth_of(row["transcript_gold"], row["fields_gold"])
            row["gold_in_db"] = gold_chain(db, row["fields_gold"]).in_db
        rows.append(row)
    return rows


def merge(new: list[dict], old_path: Path) -> list[dict]:
    if not old_path.exists():
        return new
    old = {json.loads(l)["file"]: json.loads(l) for l in old_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    out = []
    for r in new:
        o = old.get(r["file"])
        if o and o.get("label_status") == "confirmed" and o.get("set") != "tts":
            # 说话人确认过的行只刷新派生量
            o["gold_in_db"] = r["gold_in_db"] if r["fields_gold"] == o.get("fields_gold") else o.get("gold_in_db")
            o["transcript_asr"], o["asr_language"], o["duration"] = r["transcript_asr"], r["asr_language"], r["duration"]
            out.append(o)
        else:
            out.append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    db = AddressDB.default()
    rows = merge(build(db), Path(a.out))
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    from collections import Counter
    print(f"{len(rows)} 行 → {a.out}")
    print("quality:", dict(Counter(r["quality"] for r in rows)))
    print("label_status:", dict(Counter(r["label_status"] for r in rows)))
    print("dialect_group:", dict(Counter(r["dialect_group"] for r in rows if r["quality"] == "ok")))


if __name__ == "__main__":
    main()
