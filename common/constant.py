from pathlib import Path
from typing import Literal


POOL = {
    "fund": {
        "510300": {
            "name": "沪深300ETF",
            "asset_category": "大盘宽基",
            "role": "A股核心资产",
        },
        "510500": {
            "name": "中证500ETF",
            "asset_category": "中盘宽基",
            "role": "增加中盘弹性",
        },
        "512100": {
            "name": "中证1000ETF",
            "asset_category": "小盘宽基",
            "role": "增加小盘暴露",
        },
        "159531": {
            "name": "中证2000ETF",
            "asset_category": "小盘/微盘宽基",
            "role": "增加更小市值暴露",
        },
        "518880": {
            "name": "黄金ETF",
            "asset_category": "黄金",
            "role": "分散风险",
        },
        "511010": {
            "name": "国债ETF",
            "asset_category": "国债",
            "role": "降低组合波动",
        },
        "511880": {
            "name": "银华日利",
            "asset_category": "货币",
            "role": "现金管理",
        },
    },
    "stock": {
        "300750": {
            "name": "宁德时代",
            "asset_category": "新能源/电池",
            "role": "成长制造龙头观察",
        },
        "600519": {
            "name": "贵州茅台",
            "asset_category": "白酒/消费",
            "role": "消费龙头观察",
        },
        "601318": {
            "name": "中国平安",
            "asset_category": "保险/金融",
            "role": "大金融代表观察",
        },
        "300308": {
            "name": "中际旭创",
            "asset_category": "AI算力/光模块",
            "role": "科技成长代表观察",
        },
        "601899": {
            "name": "紫金矿业",
            "asset_category": "有色/资源",
            "role": "资源周期代表观察",
        },
        "600036": {
            "name": "招商银行",
            "asset_category": "银行",
            "role": "优质银行代表观察",
        },
        "300502": {
            "name": "新易盛",
            "asset_category": "AI算力/光模块",
            "role": "科技高弹性代表观察",
        },
        "000333": {
            "name": "美的集团",
            "asset_category": "家电/制造",
            "role": "稳定制造代表观察",
        },
        "601166": {
            "name": "兴业银行",
            "asset_category": "银行",
            "role": "银行代表观察",
        },
        "600900": {
            "name": "长江电力",
            "asset_category": "公用事业/电力",
            "role": "高股息防御代表观察",
        },
    },
}


AssetType = Literal["fund", "stock"]
AdjustType = Literal["raw", "qfq", "hfq"]

ADJUST_TYPE_MAP: dict[AdjustType, str] = {
    "raw": "原始",
    "qfq": "前复权",
    "hfq": "后复权",
}


FUND_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "akshare" / "fund"
STOCK_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "tushare" / "stock"


# 行业轮动专用 ETF 池：对照 SPDR Select Sector (XL-)。
# 主集 7 只皆为 2013-2014 年成立的中证/上证一级行业 ETF，可支撑 10 年回测。
# 注释掉的 3 只是细分行业（成立较晚），需要时放开即可。
SECTOR_ETFS: dict[str, dict[str, str]] = {
    "510230": {"name": "金融ETF", "sector": "金融", "us_analog": "XLF"},
    "159939": {"name": "信息技术ETF", "sector": "信息技术", "us_analog": "XLK"},
    "159928": {"name": "消费ETF", "sector": "主要消费", "us_analog": "XLP/XLY"},
    "159930": {"name": "能源ETF", "sector": "能源", "us_analog": "XLE"},
    "159938": {"name": "医药ETF", "sector": "医药卫生", "us_analog": "XLV"},
    "159945": {"name": "工业ETF", "sector": "工业", "us_analog": "XLI"},
    "159944": {"name": "材料ETF", "sector": "材料", "us_analog": "XLB"},
    # Optional (shorter history):
    # "159952": {"name": "可选消费ETF", "sector": "可选消费", "us_analog": "XLY"},
    # "512000": {"name": "券商ETF",     "sector": "券商",     "us_analog": "XLF (sub)"},
    # "512760": {"name": "半导体ETF",   "sector": "半导体",   "us_analog": "XLK (sub)"},
}