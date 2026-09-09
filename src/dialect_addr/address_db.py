"""地址库：离线库打底 + 在线 POI API 可选升级。

为什么要分层检索而不是把地址库一次性全丢给模型：

Google CLAS 论文（Contextual LAS）披露了 contextual biasing 的规模特性——
候选在**数百到数千条**时增益最大，上万条时优势显著收窄。全国 POI 上亿条，
直接注入必然退化。

而地址天然是**严格树形**的：
    全国 → 省(34) → 市(~330) → 区(~2800) → 该区内道路/小区(几千条)
每一级用上一级的结果过滤下一级，最后一级的候选集正好落在那个甜区。

这就是本模块的核心方法：**逐级收窄**（progressive narrowing）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Protocol

from .pinyin_dialect import (
    DialectProfile,
    GENERIC,
    Syllable,
    pinyin_distance,
    to_syllables,
)

Level = Literal["province", "city", "district", "street", "road", "poi"]

LEVEL_ORDER: list[Level] = ["province", "city", "district", "street", "road", "poi"]


@dataclass
class AddressEntry:
    """地址库中的一个条目。

    adcode 为行政区划代码（GB/T 2260 体系）；road/poi 级没有官方代码，
    用 "父adcode-序号" 合成，只要求库内唯一。
    """

    name: str
    level: Level
    adcode: str
    parent: str | None = None
    aliases: tuple[str, ...] = ()
    # 先验热度 0~1：POI 热度/人口密度/历史下单频次的归一化值。
    # 打分函数里作为 P(A) 的一部分，用来在音近候选之间做区分。
    prior: float = 0.5
    lng: float | None = None
    lat: float | None = None

    _syl: tuple[Syllable, ...] | None = field(default=None, repr=False, compare=False)

    @property
    def syllables(self) -> tuple[Syllable, ...]:
        if self._syl is None:
            object.__setattr__(self, "_syl", to_syllables(self.name))
        return self._syl  # type: ignore[return-value]

    def all_names(self) -> tuple[str, ...]:
        """正名 + 别名。别名很重要：口语里说的往往不是全称。

        例："武侯区" 口语说 "武侯"；"红牌楼街道" 口语说 "红牌楼"。
        """
        return (self.name, *self.aliases)


class POIProvider(Protocol):
    """在线 POI 检索接口。离线库不够用时挂上去。"""

    def search(self, keyword: str, region: str | None = None, limit: int = 20) -> list[AddressEntry]:
        ...


class AddressDB:
    """分层地址库。"""

    def __init__(self, entries: Iterable[AddressEntry], provider: POIProvider | None = None):
        self.entries: list[AddressEntry] = list(entries)
        self.by_adcode: dict[str, AddressEntry] = {e.adcode: e for e in self.entries}
        self.by_level: dict[Level, list[AddressEntry]] = {lv: [] for lv in LEVEL_ORDER}
        self.children: dict[str, list[AddressEntry]] = {}
        for e in self.entries:
            self.by_level[e.level].append(e)
            if e.parent:
                self.children.setdefault(e.parent, []).append(e)
        self.provider = provider

    # ---------------- 构造 ----------------

    @classmethod
    def load(cls, path: str | Path, provider: POIProvider | None = None) -> AddressDB:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [
            AddressEntry(
                name=d["name"],
                level=d["level"],
                adcode=d["adcode"],
                parent=d.get("parent"),
                aliases=tuple(d.get("aliases", ())),
                prior=float(d.get("prior", 0.5)),
                lng=d.get("lng"),
                lat=d.get("lat"),
            )
            for d in data["entries"]
        ]
        return cls(entries, provider=provider)

    @classmethod
    def default(cls, provider: POIProvider | None = None) -> AddressDB:
        """加载离线库。环境变量 DIALECT_ADDR_DB 优先，便于换用自建的更大的库。"""
        env = os.environ.get("DIALECT_ADDR_DB")
        if env:
            return cls.load(Path(env).expanduser(), provider=provider)
        here = Path(__file__).resolve().parents[2]
        return cls.load(here / "data" / "addresses" / "cn_subset.json", provider=provider)

    # ---------------- 检索 ----------------

    def candidates_at(self, level: Level, parent: str | None = None) -> list[AddressEntry]:
        """取某一层级的候选，可选按父节点过滤——逐级收窄的基本操作。"""
        if parent is None:
            return list(self.by_level[level])
        return [e for e in self.children.get(parent, []) if e.level == level]

    def descendants_at(self, level: Level, root: str) -> list[AddressEntry]:
        """取 root 子树下某层级的全部节点（跨越中间层级）。

        必要性：口语地址经常跳级——"成都红牌楼"直接从市跳到街道，
        不说区。所以从"市"找"街道"必须能跨过"区"这一层。
        """
        out: list[AddressEntry] = []
        stack = [root]
        seen = {root}
        while stack:
            cur = stack.pop()
            for ch in self.children.get(cur, []):
                if ch.level == level:
                    out.append(ch)
                if ch.adcode not in seen:
                    seen.add(ch.adcode)
                    stack.append(ch.adcode)
        return out

    def match(
        self,
        query: str,
        level: Level,
        parent: str | None = None,
        profile: DialectProfile = GENERIC,
        topk: int = 5,
        max_distance: float = 0.55,
        subtree: bool = False,
    ) -> list[tuple[AddressEntry, float]]:
        """在指定层级按**方言加权拼音距离**检索。

        返回 [(条目, 距离)]，距离升序。max_distance 之外的直接丢弃——
        与其返回一个音都对不上的候选，不如返回空让上层走兜底。
        """
        if parent and subtree:
            pool = self.descendants_at(level, parent)
        else:
            pool = self.candidates_at(level, parent)

        scored: list[tuple[AddressEntry, float]] = []
        for e in pool:
            # 别名也参与匹配，取最小距离——口语说的可能是任一种叫法
            d = min(pinyin_distance(query, nm, profile) for nm in e.all_names())
            if d <= max_distance:
                scored.append((e, d))
        scored.sort(key=lambda t: (t[1], -t[0].prior))
        return scored[:topk]

    def search_online(
        self, keyword: str, region: str | None = None, limit: int = 20
    ) -> list[AddressEntry]:
        """走在线 POI（未配置 provider 时返回空，不报错）。"""
        if self.provider is None:
            return []
        try:
            return self.provider.search(keyword, region=region, limit=limit)
        except Exception:
            # 在线检索是增强项而非必需项：挂了就退回离线结果，不能拖垮主流程
            return []

    def context_lines(self, entries: Iterable[AddressEntry]) -> list[str]:
        """把候选拼成注入 ASR system prompt 的文本行。

        带上父级全名，让模型看到完整层级——只给"红牌楼"没有上下文，
        模型不知道它是成都的地名。
        """
        out = []
        for e in entries:
            out.append(self.full_name(e))
        return out

    def full_name(self, e: AddressEntry) -> str:
        """回溯父链拼出全称。"""
        parts = [e.name]
        cur = e.parent
        guard = 0
        while cur and guard < 10:
            p = self.by_adcode.get(cur)
            if p is None:
                break
            parts.append(p.name)
            cur = p.parent
            guard += 1
        return "".join(reversed(parts))

    def __len__(self) -> int:
        return len(self.entries)

    def stats(self) -> dict[str, int]:
        return {lv: len(self.by_level[lv]) for lv in LEVEL_ORDER}


# --------------------------------------------------------------------------
# 在线 POI Provider
# --------------------------------------------------------------------------


class AmapProvider:
    """高德 Web 服务 API。

    需要 key：https://lbs.amap.com/  控制台申请「Web服务」类型 key。
    环境变量 AMAP_KEY，或构造时传入。
    """

    ENDPOINT = "https://restapi.amap.com/v5/place/text"

    def __init__(self, key: str | None = None, timeout: int = 8):
        self.key = key or os.environ.get("AMAP_KEY", "")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.key)

    def search(self, keyword: str, region: str | None = None, limit: int = 20) -> list[AddressEntry]:
        if not self.key:
            return []
        import urllib.parse
        import urllib.request

        params = {
            "key": self.key,
            "keywords": keyword,
            "page_size": str(min(limit, 25)),
        }
        if region:
            params["region"] = region
            params["city_limit"] = "true"

        url = f"{self.ENDPOINT}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))

        if data.get("status") != "1":
            return []

        out: list[AddressEntry] = []
        for i, poi in enumerate(data.get("pois", [])):
            loc = (poi.get("location") or ",").split(",")
            try:
                lng, lat = float(loc[0]), float(loc[1])
            except (ValueError, IndexError):
                lng = lat = None
            out.append(
                AddressEntry(
                    name=poi.get("name", ""),
                    level="poi",
                    adcode=f"amap-{poi.get('id', i)}",
                    parent=poi.get("adcode"),
                    prior=0.6,
                    lng=lng,
                    lat=lat,
                )
            )
        return out


class BaiduProvider:
    """百度地图 Place API v2。环境变量 BAIDU_MAP_AK。"""

    ENDPOINT = "https://api.map.baidu.com/place/v2/search"

    def __init__(self, ak: str | None = None, timeout: int = 8):
        self.ak = ak or os.environ.get("BAIDU_MAP_AK", "")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.ak)

    def search(self, keyword: str, region: str | None = None, limit: int = 20) -> list[AddressEntry]:
        if not self.ak:
            return []
        import urllib.parse
        import urllib.request

        params = {
            "query": keyword,
            "region": region or "全国",
            "output": "json",
            "ak": self.ak,
            "page_size": str(min(limit, 20)),
        }
        url = f"{self.ENDPOINT}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))

        if data.get("status") != 0:
            return []

        out: list[AddressEntry] = []
        for i, poi in enumerate(data.get("results", [])):
            loc = poi.get("location") or {}
            out.append(
                AddressEntry(
                    name=poi.get("name", ""),
                    level="poi",
                    adcode=f"baidu-{poi.get('uid', i)}",
                    prior=0.6,
                    lng=loc.get("lng"),
                    lat=loc.get("lat"),
                )
            )
        return out


def auto_provider() -> POIProvider | None:
    """按环境变量自动挑一个可用的在线 provider；都没配则返回 None。"""
    for cls in (AmapProvider, BaiduProvider):
        p = cls()
        if p.available():  # type: ignore[attr-defined]
            return p
    return None
