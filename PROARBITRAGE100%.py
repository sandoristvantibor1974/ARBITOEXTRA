# -*- coding: utf-8 -*-
"""
arb_bot_ops_hardened_plus_improved.py
Ops-hardened STRICT 2-way arb scanner for The Odds API v4 + extras.
ÚJ KÉPESSÉGEK (opcionálisak, flag-ekkel kapcsolhatók):
- Trend-based scoring (EWMA, momentum/CUSUM) -> score boost/gate
- ML prediction layer (shadow->live), sklearn opcionális
- Napi automata kalibráció ledgerből (min_edge, slippage, stake profil finomhangolás)
- ÚJ: “csak változásra jelezzen” + pre-match emlékeztető rendszer (dedupe precision + reminder cache)
Alapértelmezés: MOCK mód (külső API/ML nélkül is fut).
Nem köt fogadást – csak elemzés/riasztás, ledger (CSV+JSONL) és Telegram (ha beállítod).
Parancssor:
  python arb_bot_ops_hardened_plus_improved.py --help

Javítások ebben a kiadásban:
- Típusok: Optional[Dict[...]] fixek (már nem None-értékű Dict mezők).
- MAD küszöb ENV-ből (MAD_Z_THR), combined filter pontosítás.
- Telegram 4096 karakter feletti üzenetek darabolása.
- Trend/ML lazy-init biztonságosabb megoldással.
- Kalibráció (_calibrate) befejezve + napi kapu és drawdown brake.
- Fő futtatási ciklus (_run), egyszeri kör (_service_once) és main() hozzáadva.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, getcontext
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Iterator

import cProfile  # Profilozás

_missing: List[str] = []
try:
    from pydantic import BaseModel as _PydBaseModel  # type: ignore
    from pydantic import Field as _PydField
    BaseModel = _PydBaseModel
    Field = _PydField
    _USING_PYDANTIC = True
except Exception:
    _missing.append("pydantic")
    _USING_PYDANTIC = False

    class BaseModel:  # minimál shim pydantic nélkül
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def Field(default=None, default_factory=None, **kwargs):  # type: ignore
        return default_factory() if default_factory is not None else default



try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore
    _missing.append("httpx")
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore
    _missing.append("PyYAML")
try:
    from tenacity import (  # type: ignore
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential_jitter,
    )
except Exception:
    retry = None  # type: ignore
    retry_if_exception_type = None  # type: ignore
    stop_after_attempt = None  # type: ignore
    wait_exponential_jitter = None  # type: ignore
    _missing.append("tenacity")
# ML (opcionális)
_SKLEARN_OK = True
try:
    import pickle  # always available

    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except Exception:
        _SKLEARN_OK = False
        _missing.append("scikit-learn")
except Exception:
    _SKLEARN_OK = False
    _missing.append("pickle?")
try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None  # type: ignore
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

# ───────────── constants / decimal ─────────────
ctx = getcontext()
ctx.prec = 18
ctx.rounding = ROUND_HALF_EVEN
EPS = Decimal("1e-18")
ODDS_MIN = Decimal("1.01")
ODDS_MAX = Decimal("100.0")


def _env_int(name: str, default: int, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    try:
        v = int(str(os.getenv(name, str(default))).strip())
    except Exception:
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v

DEDUPE_ODDS_PRECISION = _env_int("DEDUPE_ODDS_PRECISION", 3, lo=0, hi=8)  # alap: 3 tizedes

# ───────────── logging ─────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
SERVICE_LOG = LOG_DIR / "service.log"


def _setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("arb-bot")
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    def _gz_namer(default_name: str) -> str:
        return str(default_name) + ".gz"

    def _gz_rotator(source: str, dest: str) -> None:
        import gzip, shutil

        with open(source, "rb") as sf, gzip.open(dest, "wb", compresslevel=6) as df:
            shutil.copyfileobj(sf, df)
        try:
            os.remove(source)
        except Exception:
            pass

    handlers = [sh]
    try:
        fh = RotatingFileHandler(SERVICE_LOG, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        fh.namer = _gz_namer
        fh.rotator = _gz_rotator
        fh.setFormatter(fmt)
        handlers.append(fh)
    except Exception:
        pass

    logger.handlers.clear()
    for h in handlers:
        logger.addHandler(h)
    return logger


logger = _setup_logging()


class InstanceLockError(RuntimeError):
    pass


def healthcheck() -> int:
    checks = {
        "httpx": {"ok": httpx is not None},
        "pydantic": {"ok": _USING_PYDANTIC},
        "tenacity": {"ok": retry is not None},
        "logs_writable": {"ok": True},
    }
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        t = LOG_DIR / ".hc_write"
        t.write_text("ok", encoding="utf-8")
        t.unlink(missing_ok=True)
    except Exception as e:
        checks["logs_writable"] = {"ok": False, "error": repr(e)}
    ok = all(v.get("ok") for v in checks.values())
    print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False))
    return 0 if ok else 2


@contextmanager
def single_instance(lock_path: str) -> Iterator[None]:
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            try:
                f.write(b"\0")
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                raise InstanceLockError("Another instance appears to be running (Windows).")
        else:
            import fcntl

            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                raise InstanceLockError("Another instance appears to be running (Unix).")
        yield
    finally:
        try:
            if locked:
                if os.name == "nt":
                    import msvcrt

                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl

                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
        finally:
            try:
                f.close()
            except Exception:
                pass


# ───────────── utils ─────────────
def D(x: Any) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


def pct_to_decimal(v: Any) -> Decimal:
    if isinstance(v, Decimal):
        return v
    s = str(v).strip()
    if s.endswith("%"):
        return Decimal(s[:-1]) / 100
    return Decimal(s)


def clamp(x: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    return max(lo, min(hi, x))


def _now() -> int:
    return int(time.time())


def _mask(tok: Optional[str]) -> str:
    if not tok:
        return "*"
    t = str(tok)
    return t[:6] + "..." + t[-4:] if len(t) > 12 else "*"


def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _normalize_line(x: Any) -> Optional[Decimal]:
    try:
        return D(x).quantize(Decimal("0.01"))
    except Exception:
        return None


# ───────────── config ─────────────
def _normalize_regions(regions: str) -> str:
    s = (regions or "").strip().lower()
    return "au,eu,uk,us" if s in ("", "global") else s


def _lowercase_key_map_decimal(raw: Optional[Dict[str, Any]]) -> Dict[str, Decimal]:
    return {str(k).lower(): pct_to_decimal(v) for k, v in (raw or {}).items()}


def _lowercase_key_map_int(raw: Optional[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, v in (raw or {}).items():
        try:
            out[str(k).lower()] = int(v)
        except Exception:
            pass
    return out


@dataclass(frozen=True)
class AppConfig:
    # core
    mode: str
    regions: str
    time_window: int
    min_edge_after_costs: Decimal
    min_edge_after_costs_inplay: Decimal
    min_ev_after_costs: Decimal
    kelly_frac: Decimal
    risk_aversion: Decimal
    start_balance: Decimal
    default_book_cap: Decimal
    book_caps: Dict[str, Decimal]
    fees: Dict[str, Decimal]
    slippage_avg: Dict[str, Decimal]
    slippage_wc: Dict[str, Decimal]
    fx_cost: Decimal
    consensus_method: str
    weights: Dict[str, Decimal]
    telegram_token: Optional[str]
    telegram_chat_id: Optional[str]
    sports: List[str]
    stake_round_step: int
    stake_round_profile: str
    min_ttk_minutes: int
    dedupe_window_sec: int
    provider: str
    prealert_refetch: bool
    prealert_edge_drop_frac: Decimal
    dnb_score_factor: Decimal
    enable_inplay: bool
    bookmakers: Optional[str] = None
    allow_outrights: bool = False
    min_edge_market: Optional[Dict[str, Decimal]] = None
    min_bet_per_book: Optional[Dict[str, int]] = None
    min_step_per_book: Optional[Dict[str, int]] = None
    event_max_stake: Optional[int] = None
    enable_middle_scan: bool = False
    # NEW: trend scoring
    enable_trend_scoring: bool = False
    trend_weight: Decimal = Decimal("0.0")
    trend_ma_short_sec: int = 120
    trend_ma_long_sec: int = 600
    trend_momentum_sec: int = 180
    trend_min_gate: Decimal = Decimal("0.0")
    # NEW: ML layer
    enable_ml: bool = False
    ml_weight: Decimal = Decimal("0.0")
    ml_model_path: Optional[str] = None
    ml_shadow_mode: bool = True
    # NEW: daily calibration
    enable_daily_calibration: bool = True
    calib_run_at_utc: str = "00:05"
    calib_lookback_days: int = 1
    calib_min_edge_bounds: Tuple[Decimal, Decimal] = (Decimal("0.004"), Decimal("0.012"))  # 0.4%..1.2%
    calib_slippage_alpha: Decimal = Decimal("0.5")
    calib_drawdown_brake: bool = True
    # NEW: reminders / echo
    local_echo: bool = True
    reminder_interval_sec: int = 7200
    reminders_enabled: bool = True


def _to_decimal_map(raw: Optional[Dict[str, Any]]) -> Dict[str, Decimal]:
    return _lowercase_key_map_decimal(raw)


def _to_int_map(raw: Optional[Dict[str, Any]]) -> Dict[str, int]:
    return _lowercase_key_map_int(raw)


def load_config(yaml_path: Optional[str] = None) -> AppConfig:
    data: Dict[str, Any] = {}
    if yaml_path:
        if yaml is None:
            raise RuntimeError(
                "Megadtál --config fájlt, de a PyYAML nincs telepítve. "
                "Telepítsd: pip install PyYAML, vagy futtasd config nélkül."
            )
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config fájl nem található: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    # Determine regions from ENV override or YAML (fallback 'global' expands to default set)
    regions_val = _normalize_regions(os.getenv("REGIONS", data.get("regions", "global")))
    book_caps = {str(k).lower(): D(v) for k, v in (data.get("book_caps") or {}).items()}
    fees = _to_decimal_map(data.get("fees"))
    sl_avg = _to_decimal_map(data.get("slippage_avg"))
    sl_wc = _to_decimal_map(data.get("slippage_wc"))
    return AppConfig(
        mode=os.getenv("ARB_MODE", data.get("mode", "MOCK")).upper(),
        regions=regions_val,
        time_window=_env_int("TIME_WINDOW_HOURS", data.get("time_window", 48), lo=1),
        min_edge_after_costs=pct_to_decimal(data.get("min_edge_after_costs", "0.5%")),
        min_edge_after_costs_inplay=pct_to_decimal(data.get("min_edge_after_costs_inplay", "0.8%")),
        min_ev_after_costs=pct_to_decimal(data.get("min_ev_after_costs", "0.3%")),
        kelly_frac=D(data.get("kelly_frac", 0.35)),
        risk_aversion=D(data.get("risk_aversion", 2.0)),
        start_balance=D(data.get("start_balance", 300)),
        default_book_cap=D(data.get("default_book_cap", 150)),
        book_caps=book_caps,
        fees=fees,
        slippage_avg=sl_avg,
        slippage_wc=sl_wc,
        fx_cost=pct_to_decimal(data.get("fx_cost", "0.10%")),
        consensus_method=str(data.get("consensus_method", "median")).lower(),
        weights={k: D(v) for k, v in (data.get("weights") or {"w1": 0.4, "w2": 0.2, "w3": 0.2, "w4": 0.1, "w5": 0.1}).items()},
        telegram_token=os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN") or data.get("telegram_token"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or data.get("telegram_chat_id"),
        sports=data.get("sports", ["AUTO_2WAY"]),
        stake_round_step=_env_int("STAKE_ROUND", data.get("stake_round_step", 1), lo=1),
        stake_round_profile=str(os.getenv("STAKE_ROUND_PROFILE", data.get("stake_round_profile", "flat"))).lower(),
        min_ttk_minutes=_env_int("MIN_TTK_MINUTES", data.get("min_ttk_minutes", 120), lo=0),
        dedupe_window_sec=_env_int("DEDUPE_WINDOW_SEC", data.get("dedupe_window_sec", 600), lo=1),
        provider=str(os.getenv("ODDS_PROVIDER", data.get("provider", "theoddsapi"))).lower(),
        prealert_refetch=bool(int(os.getenv("PREALERT_REFETCH", "1" if data.get("prealert_refetch", True) else "0"))),
        prealert_edge_drop_frac=D(os.getenv("PREALERT_EDGE_DROP_FRAC", str(data.get("prealert_edge_drop_frac", 0.5)))),
        dnb_score_factor=D(os.getenv("DNB_SCORE_FACTOR", str(data.get("dnb_score_factor", 0.5)))),
        enable_inplay=bool(int(os.getenv("ENABLE_INPLAY", "0" if not data.get("enable_inplay") else "1"))),
        bookmakers=os.getenv("BOOKMAKERS", data.get("bookmakers", None)),
        allow_outrights=bool(int(os.getenv("ALLOW_OUTRIGHTS", "1" if data.get("allow_outrights", False) else "0"))),
        min_edge_market=_to_decimal_map(data.get("min_edge_market")),
        min_bet_per_book=_to_int_map(data.get("min_bet_per_book")),
        min_step_per_book=_to_int_map(data.get("min_step_per_book")),
        event_max_stake=int(data.get("event_max_stake", 0)) or None,
        enable_middle_scan=bool(data.get("enable_middle_scan", False)),
        # trend defaults
        enable_trend_scoring=bool(data.get("enable_trend_scoring", False)),
        trend_weight=D(str(data.get("trend_weight", "0.0"))),
        trend_ma_short_sec=int(data.get("trend", {}).get("ma_short_sec", 120) if data.get("trend") else 120),
        trend_ma_long_sec=int(data.get("trend", {}).get("ma_long_sec", 600) if data.get("trend") else 600),
        trend_momentum_sec=int(data.get("trend", {}).get("momentum_horizon_sec", 180) if data.get("trend") else 180),
        trend_min_gate=D(str(data.get("trend", {}).get("min_trend_gate", "0.0") if data.get("trend") else "0.0")),
        # ML defaults
        enable_ml=bool(data.get("enable_ml", False)),
        ml_weight=D(str(data.get("ml_weight", "0.0"))),
        ml_model_path=(str(data.get("ml", {}).get("model_path") or "") or None if data.get("ml") else None),
        ml_shadow_mode=bool(data.get("ml", {}).get("shadow_mode", True) if data.get("ml") else True),
        # calibration defaults
        enable_daily_calibration=bool(data.get("enable_daily_calibration", True)),
        calib_run_at_utc=str(data.get("calibration", {}).get("run_at_utc", "00:05") if data.get("calibration") else "00:05"),
        calib_lookback_days=int(data.get("calibration", {}).get("lookback_days", 1) if data.get("calibration") else 1),
        calib_min_edge_bounds=(
            D(str((data.get("calibration", {}) or {}).get("min_edge_bounds", [0.004, 0.012])[0])),
            D(str((data.get("calibration", {}) or {}).get("min_edge_bounds", [0.004, 0.012])[1])),
        ),
        calib_slippage_alpha=D(str((data.get("calibration", {}) or {}).get("slippage_alpha", 0.5))),
        calib_drawdown_brake=bool((data.get("calibration", {}) or {}).get("drawdown_brake", True)),
        # reminders / echo
        local_echo=bool(data.get("local_echo", True)),
        reminder_interval_sec=int(os.getenv("REMINDER_INTERVAL_SEC", data.get("reminder_interval_sec", 7200))),
        reminders_enabled=bool(data.get("reminders_enabled", True)),
    )


def validate_config(cfg: AppConfig) -> List[str]:
    errs: List[str] = []
    if cfg.stake_round_step < 1:
        errs.append("stake_round_step must be >= 1")
    if cfg.kelly_frac < D(0) or cfg.kelly_frac > D(1):
        errs.append("kelly_frac must be in [0,1]")
    if cfg.time_window <= 0:
        errs.append("time_window must be > 0 (hours)")
    if cfg.min_edge_after_costs < D(0) or cfg.min_edge_after_costs > D("0.10"):
        errs.append("min_edge_after_costs must be in [0,10%]")
    if cfg.min_edge_after_costs_inplay < D(0) or cfg.min_edge_after_costs_inplay > D("0.15"):
        errs.append("min_edge_after_costs_inplay must be in [0,15%]")

    if cfg.mode.upper() == "LIVE":
        if not os.getenv("ODDS_API_KEY", ""):
            errs.append("LIVE mode: missing ODDS_API_KEY")
        if httpx is None:
            errs.append("LIVE mode: httpx not available")
        if not (cfg.bookmakers and str(cfg.bookmakers).strip()):
            errs.append("LIVE mode: BOOKMAKERS kötelező (pl. Bet365,WilliamHill,UnibetUK)")
        if not cfg.prealert_refetch:
            errs.append("LIVE mode: PREALERT_REFETCH kötelező (kapcsold be --prealert vagy env PREALERT_REFETCH=1)")
    # pydantic kötelező LIVE módban
    if cfg.mode.upper() == "LIVE" and not _USING_PYDANTIC:
        errs.append("LIVE mode: pydantic szükséges (pip install pydantic)")

    # ML: ha súlyozás aktív, de nincs sklearn/model, szóljunk
    if cfg.enable_ml and cfg.ml_weight > 0 and (not _SKLEARN_OK or not cfg.ml_model_path):
        errs.append("ML réteg: scikit-learn és érvényes ml_model_path szükséges, vagy állítsd ml_weight=0-ra.")

    if cfg.trend_weight < 0 or cfg.ml_weight < 0:
        errs.append("trend_weight/ml_weight must be >= 0")
    return errs


# ───────────── models ─────────────
class BookOffer(BaseModel):
    book: str
    market: Literal["h2h", "totals", "draw_no_bet", "spreads", "btts"]
    outcome: str  # home/away, over/under, yes/no
    line: Optional[Decimal] = None
    odds: Decimal
    ts: int


class EventOdds(BaseModel):
    sport_key: str
    event_id: str
    commence_time: int
    home: str
    away: str
    offers: Dict[str, Dict[str, List[BookOffer]]] = Field(default_factory=dict)


class ArbOpportunity(BaseModel):
    event_id: str
    sport_key: str
    market: str
    home_team: str
    away_team: str
    offer_a: BookOffer
    offer_b: BookOffer
    inv_sum: Decimal
    edge_before_costs: Decimal
    edge_after_costs: Decimal
    score: Decimal
    stake_a: int
    stake_b: int
    worst_case_profit: Decimal
    cvar_5: Decimal
    latency_ms: int
    offer_count: int
    sport_count: int
    commence_time: int
    # NEW diagnostic fields (optional)
    trend_score: Optional[Decimal] = None
    ml_score: Optional[Decimal] = None


@dataclass(frozen=True)
class StakePlan:
    stakes: Dict[str, int]
    total_stake: int
    method: Literal["prop", "kelly", "robust", "mv"] = "prop"


# ───────────── filters & staleness ─────────────
def valid_price(x: Any, lo: Decimal = ODDS_MIN, hi: Decimal = ODDS_MAX) -> bool:
    try:
        d = D(x)
        if not (lo <= d <= hi):
            raise ValueError(f"Invalid price: {d} not in [{lo}, {hi}]")
        return True
    except Exception:
        return False


def iqr_filter(offers: List[BookOffer]) -> List[BookOffer]:
    xs = [o.odds for o in offers if valid_price(o.odds)]
    if len(xs) < 4:
        return offers
    q = statistics.quantiles(xs, n=4)
    q1, q3 = D(str(q[0])), D(str(q[2]))
    iqr = q3 - q1
    lo = q1 - D("1.5") * iqr
    hi = q3 + D("1.5") * iqr
    return [o for o in offers if lo <= o.odds <= hi]


def _mad(vals: List[Decimal]) -> Decimal:
    med = D(str(statistics.median(vals)))
    devs = [abs(x - med) for x in vals]
    return D(str(statistics.median(devs))) * D("1.4826")


def mad_filter(offers: List[BookOffer], thr: Optional[Decimal] = None) -> List[BookOffer]:
    # biztonságos alapértelmezett küszöb: csak futási időben olvassuk a környezeti változót
    if thr is None:
        thr_str = os.getenv("MAD_Z_THR", "2.5")
        try:
            thr = D(thr_str)
        except Exception:
            thr = D("2.5")
    xs = [o.odds for o in offers if valid_price(o.odds)]
    if len(xs) < 3:
        return offers
    med = D(str(statistics.median(xs)))
    mad = _mad(xs)
    if mad == 0:
        return offers
    out: List[BookOffer] = []
    for o in offers:
        z = abs((o.odds - med) / mad)
        if z <= thr:
            out.append(o)
    return out


def combined_filter(offers: List[BookOffer]) -> List[BookOffer]:
    iqr_first = iqr_filter(offers)
    thr_str = os.getenv("MAD_Z_THR", "2.5")
    try:
        thr_dec = D(thr_str)
    except Exception:
        thr_dec = D("2.5")
    return mad_filter(iqr_first, thr=thr_dec)


@dataclass
class PriceSeries:
    alpha: Decimal = D("0.3")
    ewma: Optional[Decimal] = None
    cusum_pos: Decimal = D("0")
    cusum_neg: Decimal = D("0")
    last_ts: int = 0

    def update(self, price: Decimal, ts: int) -> None:
        self.last_ts = ts
        if self.ewma is None:
            self.ewma = price
            return
        self.ewma = self.alpha * price + (D(1) - self.alpha) * self.ewma
        diff = price - self.ewma
        k = D("0.005")
        self.cusum_pos = max(D(0), self.cusum_pos + diff - k)
        self.cusum_neg = min(D(0), self.cusum_neg + diff + k)

    def signal(self, now_ts: int) -> Decimal:
        age = max(0, now_ts - self.last_ts)
        age_sig = D(min(1.0, age / 600.0))
        mov_sig = clamp(max(self.cusum_pos, abs(self.cusum_neg)) / D("0.02"), D(0), D(1))
        return clamp(D("0.5") * age_sig + D("0.5") * mov_sig, D(0), D(1))


_SERIES: Dict[Tuple[str, str, str, str], PriceSeries] = {}


def purge_series(now_ts: int, ttl: int = 3600) -> None:
    for k in list(_SERIES.keys()):
        if now_ts - _SERIES[k].last_ts > ttl:
            _SERIES.pop(k, None)


def staleness_gate(
    offers: List[BookOffer],
    event_id: str,
    market_key: str,
    now_ts: int,
    thr: Decimal = D("0.75"),
) -> List[BookOffer]:
    kept: List[BookOffer] = []
    for o in offers:
        key = (event_id, market_key, o.outcome, o.book)
        ps = _SERIES.get(key) or PriceSeries()
        ps.update(o.odds, o.ts)
        _SERIES[key] = ps
        if ps.signal(now_ts) <= thr:
            kept.append(o)
    return kept


# ───────────── fair odds / Kelly ─────────────
def _aggregate_price(xs: List[Decimal], method: str = "median") -> Optional[Decimal]:
    xs = [x for x in xs if valid_price(x)]
    if not xs:
        return None
    if method == "trimmed_mean" and len(xs) >= 5:
        k = max(1, len(xs) // 10)
        xs_sorted = sorted(xs)
        trimmed = xs_sorted[k:-k] if len(xs_sorted) - 2 * k >= 1 else xs_sorted
        return D(str(sum(trimmed) / len(trimmed)))
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return xs_sorted[n // 2] if n % 2 == 1 else (xs_sorted[n // 2 - 1] + xs_sorted[n // 2]) / 2


def remove_vig_power(odds: List[Decimal], tol: Decimal = D("1e-9"), max_iter: int = 60) -> List[Decimal]:
    """Find alpha in float-space, return Decimal fair odds. Decimal pow kerülése miatt floatban iterálunk."""
    xs = [float(clamp(D(o), ODDS_MIN, ODDS_MAX)) for o in odds]
    lo, hi = 0.5, 2.5
    alpha = 1.0
    tol_f = float(tol)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        s = sum(math.exp(-mid * math.log(x)) for x in xs)
        if abs(s - 1.0) <= tol_f:
            alpha = mid
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    probs = [math.exp(-alpha * math.log(x)) for x in xs]
    s = sum(probs) or 1.0
    probs = [p / s for p in probs]
    EPSF = 1e-9
    probs = [min(1.0 - EPSF, max(EPSF, p)) for p in probs]
    return [D(1) / D(str(p)) for p in probs]


def fair_probs_from_offers(offers_by_outcome: Dict[str, List[BookOffer]], method: str = "median") -> Dict[str, Decimal]:
    outs = sorted(offers_by_outcome.keys())
    agg: List[Decimal] = []
    for out in outs:
        xs = [o.odds for o in offers_by_outcome[out]]
        val = _aggregate_price(xs, method=method)
        if val is None:
            continue
        agg.append(val)
    if len(agg) < 2:
        return {}
    fair_odds = remove_vig_power(agg)
    probs = [D(1) / fo for fo in fair_odds]
    s = sum(probs)
    probs = [p / s for p in probs]
    return {outs[i]: probs[i] for i in range(len(outs))}


# ───────────── costs & PnL ─────────────
def apply_costs(odds: Dict[str, Decimal], fees: Dict[str, Decimal], slip: Dict[str, Decimal]) -> Dict[str, Decimal]:
    eff: Dict[str, Decimal] = {}
    for k, o in odds.items():
        denom = D(1) + fees.get(k, D(0)) + slip.get(k, D(0))
        eff[k] = clamp(D(o) / max(denom, D("1e-12")), ODDS_MIN, ODDS_MAX)
    return eff


def profit_after_costs_two_way(oa: Decimal, ob: Decimal, sa: int, sb: int, fx: Decimal) -> Decimal:
    total = D(sa + sb)
    win_a = D(sa) * oa * (D(1) - fx) - total
    win_b = D(sb) * ob * (D(1) - fx) - total
    return min(win_a, win_b)


def profit_after_costs_dnb(oa: Decimal, ob: Decimal, sa: int, sb: int, fx: Decimal, refund_cost: Decimal = D(0)) -> Decimal:
    total = D(sa + sb)
    win_a = D(sa) * oa * (D(1) - fx) - total
    win_b = D(sb) * ob * (D(1) - fx) - total
    p_draw = -refund_cost
    return min(win_a, win_b, p_draw)


def inv_sum(oa: Decimal, ob: Decimal) -> Decimal:
    return (D(1) / oa) + (D(1) / ob)


# ───────────── stakes ─────────────
def _equalized_hedge_stakes(oa: Decimal, ob: Decimal, total_cap: Decimal) -> Tuple[int, int]:
    ratio = oa / ob
    s_home = (total_cap / (D(1) + ratio)).to_integral_value(rounding=ROUND_FLOOR)
    s_away = (s_home * ratio).to_integral_value(rounding=ROUND_FLOOR)
    return int(max(s_home, 0)), int(max(s_away, 0))


def _rebalance_with_caps(oa: Decimal, ob: Decimal, cap_home: Decimal, cap_away: Decimal, bank: Decimal) -> Tuple[int, int]:
    total_cap = min(cap_home + cap_away, bank)
    if total_cap <= 0:
        return 0, 0
    ratio = oa / ob
    sH = min(cap_home, (total_cap / (D(1) + ratio)))
    sA = sH * ratio
    if sA > cap_away:
        sA = cap_away
        sH = min(cap_home, sA / ratio)
    sH_i = int(sH.to_integral_value(rounding=ROUND_FLOOR))
    sA_i = int(sA.to_integral_value(rounding=ROUND_FLOOR))
    return max(sH_i, 0), max(sA_i, 0)


def allocate_stakes_proportional(oh: Decimal, oa: Decimal, cap_home: Decimal, cap_away: Decimal, bank: Decimal) -> StakePlan:
    sH, sA = _rebalance_with_caps(oh, oa, cap_home, cap_away, bank)
    return StakePlan(stakes={"home": sH, "away": sA}, total_stake=sH + sA, method="prop")


def allocate_stakes_kelly(
    fair_probs: Dict[str, Decimal],
    odds_map: Dict[str, Decimal],
    caps: Dict[str, Decimal],
    bank: Decimal,
    frac: Decimal,
) -> StakePlan:
    if not {"home", "away"}.issubset(fair_probs.keys()):
        return StakePlan(stakes={"home": 0, "away": 0}, total_stake=0, method="kelly")
    pH, pA = fair_probs["home"], fair_probs["away"]
    oH, oA = odds_map["home"], odds_map["away"]
    bH, bA = oH - D(1), oA - D(1)
    fH = max((bH * pH - (D(1) - pH)) / bH if bH > 0 else D(0), D(0))
    fA = max((bA * pA - (D(1) - pA)) / bA if bA > 0 else D(0), D(0))
    sH = int((bank * fH * frac).to_integral_value(rounding=ROUND_FLOOR))
    sA = int((bank * fA * frac).to_integral_value(rounding=ROUND_FLOOR))
    sH = int(min(D(sH), caps.get("home", bank)))
    sA = int(min(D(sA), caps.get("away", bank)))
    total = sH + sA
    hard_cap = int(min(bank, sum(int(caps.get(k, bank)) for k in ("home", "away"))))
    if total > hard_cap and total > 0:
        scale = Decimal(hard_cap) / Decimal(total)
        sH = int((Decimal(sH) * scale).to_integral_value(rounding=ROUND_FLOOR))
        sA = int((Decimal(sA) * scale).to_integral_value(rounding=ROUND_FLOOR))
    return StakePlan(stakes={"home": sH, "away": sA}, total_stake=sH + sA, method="kelly")


def allocate_stakes_mv(
    fair_probs: Dict[str, Decimal],
    odds_map: Dict[str, Decimal],
    caps: Dict[str, Decimal],
    bank: Decimal,
    risk_aversion: Decimal,
) -> StakePlan:
    oH, oA = odds_map["home"], odds_map["away"]
    total_cap = min(sum(caps.values()), bank)
    shrink = D(1) + max(risk_aversion, D("0.0001"))
    sH, sA = _equalized_hedge_stakes(oH, oA, total_cap / shrink)
    sH = int(min(D(sH), caps.get("home", bank)))
    sA = int(min(D(sA), caps.get("away", bank)))
    return StakePlan(stakes={"home": sH, "away": sA}, total_stake=sH + sA, method="mv")


def allocate_stakes_robust(
    odds_map: Dict[str, Decimal],
    caps: Dict[str, Decimal],
    bank: Decimal,
    fees_map: Dict[str, Decimal],
    fx_cost: Decimal,
    eps_wc: Dict[str, Decimal],
) -> StakePlan:
    # konzervatív effektív odds: fee + worst-case slippage + FX költség
    oH = D(odds_map["home"]) / (D(1) + fees_map.get("home", D(0)) + eps_wc.get("home", D(0)))
    oA = D(odds_map["away"]) / (D(1) + fees_map.get("away", D(0)) + eps_wc.get("away", D(0)))
    oH = clamp(oH * (D(1) - fx_cost), ODDS_MIN, ODDS_MAX)
    oA = clamp(oA * (D(1) - fx_cost), ODDS_MIN, ODDS_MAX)
    sH, sA = _rebalance_with_caps(oH, oA, caps.get("home", bank), caps.get("away", bank), bank)
    # óvatos lefaragás, hogy a rounding után is maradjon puffer
    if sH > 0:
        sH = max(1, sH - 1)
    if sA > 0:
        sA = max(1, sA - 1)
    sH = int(min(D(sH), caps.get("home", bank)))
    sA = int(min(D(sA), caps.get("away", bank)))
    return StakePlan(stakes={"home": sH, "away": sA}, total_stake=sH + sA, method="robust")


# ───────────── HTTP / ConcurrencyGuard ─────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"


class OddsAPIError(Exception):
    pass


class AsyncNullClient:
    async def get(self, *a, **k):
        class Resp:
            status_code = 200
            headers = {}

            def json(self):
                return {}

            def raise_for_status(self):
                return None

        return Resp()

    async def post(self, *a, **k):
        class Resp:
            status_code = 200
            text = '{"ok":true}'

        return Resp()

    async def aclose(self):
        return None


class ConcurrencyGuard:
    """Adaptív guard: hard max az initial, de egy 'target' soft-capre állunk fejlécek alapján.
    A 'limiter()' egy kapu (Gate) context manager-t ad vissza, ami enforce-olja a soft-capet.
    """

    class _Gate:
        def __init__(self, parent: "ConcurrencyGuard"):
            self._p = parent

        async def __aenter__(self):
            while True:
                await self._p._sem.acquire()
                async with self._p._mu:
                    if self._p._inflight < self._p._target:
                        self._p._inflight += 1
                        return self
                self._p._sem.release()
                await asyncio.sleep(0.02)

        async def __aexit__(self, exc_type, exc, tb):
            async with self._p._mu:
                if self._p._inflight > 0:
                    self._p._inflight -= 1
                try:
                    self._p._sem.release()
                except Exception:
                    pass
            return False

    def __init__(self, initial: int = 4):
        self._initial = max(1, initial)
        self._sem = asyncio.Semaphore(self._initial)
        self._target = self._initial
        self._inflight = 0
        self._mu = asyncio.Lock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def limiter(self):
        return ConcurrencyGuard._Gate(self)

    def adapt_from_headers(self, headers: Dict[str, Any]) -> None:
        """A The Odds API 'X-Requests-Remaining' alapján állítjuk a soft-cap targetet."""
        try:
            remain = headers.get("x-requests-remaining") or headers.get("X-Requests-Remaining")
            if remain is None:
                return
            r = int(str(remain))
            if r <= 3:
                new = 1
            elif r <= 10:
                new = 2
            else:
                new = self._initial
            self._target = max(1, min(self._initial, new))
        except Exception:
            pass


_HTTP_CLIENT: Optional[Any] = None
_GUARD = ConcurrencyGuard(_env_int("ODDS_CONCURRENCY", 4, lo=1, hi=32))


async def _get_client() -> Any:
    global _HTTP_CLIENT
    if httpx is None:
        if _HTTP_CLIENT is None:
            _HTTP_CLIENT = AsyncNullClient()
        return _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        try:
            _HTTP_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(10.0))  # type: ignore
        except Exception:
            _HTTP_CLIENT = httpx.AsyncClient(timeout=10.0)  # best effort
    return _HTTP_CLIENT


async def _close_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        try:
            await _HTTP_CLIENT.aclose()
        except Exception:
            pass
        _HTTP_CLIENT = None


async def _respect_retry_after(resp: Any) -> None:
    try:
        ra = resp.headers.get("Retry-After")
        if not ra:
            return
        secs = float(ra) if str(ra).replace(".", "", 1).isdigit() else 1.5
        await asyncio.sleep(min(10.0, max(0.1, secs)))
    except Exception:
        await asyncio.sleep(1.0)


if retry is not None:

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=0.3, max=3.0),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _get_json(client: Any, url: str, params: Dict[str, Any]) -> Any:
        async with _GUARD.limiter():
            resp = await client.get(url, params=params, headers={"Accept": "application/json"}, timeout=10.0)
        if hasattr(resp, "headers"):
            _GUARD.adapt_from_headers(resp.headers)
        if resp.status_code == 429:
            await _respect_retry_after(resp)
            raise OddsAPIError("429")
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, (list, dict)):
            raise OddsAPIError("Unexpected response type from API")
        return data
else:

    async def _get_json(client: Any, url: str, params: Dict[str, Any]) -> Any:
        attempt = 0
        while True:
            try:
                async with _GUARD.limiter():
                    resp = await client.get(
                        url, params=params, headers={"Accept": "application/json"}, timeout=10.0
                    )
                if hasattr(resp, "headers"):
                    _GUARD.adapt_from_headers(resp.headers)
                if resp.status_code == 429:
                    await _respect_retry_after(resp)
                    raise OddsAPIError("429")
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, (list, dict)):
                    raise OddsAPIError("Unexpected response type from API")
                return data
            except Exception:
                if attempt >= 4:
                    raise
                delay = min(3.0, 0.3 * (2**attempt) + random.random() * 0.2)
                await asyncio.sleep(delay)
                attempt += 1


# ───────────── provider layer ─────────────
class ProviderBase:
    name: str = "base"

    async def fetch_sports(self, cfg: AppConfig) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def fetch_odds_for_sport(
        self,
        cfg: AppConfig,
        sport_key: str,
        commence_from_iso: Optional[str],
        commence_to_iso: Optional[str],
        markets: str,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def fetch_inplay_for_sport(self, cfg: AppConfig, sport_key: str, markets: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def fetch_event_odds(
        self,
        cfg: AppConfig,
        sport_key: str,
        event_id: str,
        markets: str,
        bookmakers: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


def _default_mock_event(sport_key: str, now_ts: int, inplay: bool = False) -> Dict[str, Any]:
    dt = datetime.fromtimestamp(now_ts, timezone.utc)
    commence = dt + (timedelta(hours=3) if not inplay else timedelta(hours=-0.5))
    now_iso = commence.isoformat().replace("+00:00", "Z")

    def mk(last_update_shift_sec: int = 0) -> str:
        return (datetime.now(timezone.utc) - timedelta(seconds=last_update_shift_sec)).isoformat().replace(
            "+00:00", "Z"
        )

    return {
        "id": "evt-mock-live" if inplay else "evt-001",
        "commence_time": now_iso,
        "home_team": "Alpha",
        "away_team": "Beta",
        "sport_key": sport_key,
        "bookmakers": [
            {
                "title": "Bet365",
                "key": "bet365",
                "last_update": mk(),
                "markets": [
                    {"key": "h2h", "outcomes": [{"name": "Alpha", "price": 2.10}, {"name": "Beta", "price": 1.80}]},
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 2.05, "point": 2.5},
                            {"name": "Under", "price": 1.85, "point": 2.5},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Alpha", "price": 1.90, "point": -1.5},
                            {"name": "Beta", "price": 2.00, "point": 1.5},
                        ],
                    },
                    {"key": "draw_no_bet", "outcomes": [{"name": "Alpha", "price": 1.95}, {"name": "Beta", "price": 1.95}]},
                    {"key": "btts", "outcomes": [{"name": "Yes", "price": 1.95}, {"name": "No", "price": 1.95}]},
                ],
            },
            {
                "title": "William Hill",
                "key": "williamhill",
                "last_update": mk(30),
                "markets": [
                    {"key": "h2h", "outcomes": [{"name": "Alpha", "price": 1.85}, {"name": "Beta", "price": 2.20}]},
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.85, "point": 2.5},
                            {"name": "Under", "price": 2.05, "point": 2.5},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Alpha", "price": 2.02, "point": -1.5},
                            {"name": "Beta", "price": 1.88, "point": 1.5},
                        ],
                    },
                    {"key": "draw_no_bet", "outcomes": [{"name": "Alpha", "price": 2.05}, {"name": "Beta", "price": 1.85}]},
                    {"key": "btts", "outcomes": [{"name": "Yes", "price": 2.02}, {"name": "No", "price": 1.85}]},
                ],
            },
        ],
    }


class TheOddsAPIProvider(ProviderBase):
    name = "theoddsapi"

    async def fetch_sports(self, cfg: AppConfig) -> List[Dict[str, Any]]:
        # MOCK módban vagy httpx nélkül: fix lista, hogy AUTO_2WAY működjön
        if httpx is None or cfg.mode.upper() == "MOCK":
            return [
                {"key": "basketball_nba"},
                {"key": "soccer_epl"},
                {"key": "tennis_atp"},
                {"key": "mma_mixed_martial_arts"},
            ]
        client = await _get_client()
        key = os.getenv("ODDS_API_KEY", "")
        if not key:
            # Kulcs nélkül csak egy rövid lista, hogy ne dőljön el
            return [{"key": "basketball_nba"}, {"key": "soccer_epl"}]
        data = await _get_json(client, f"{ODDS_API_BASE}/sports", params={"apiKey": key})
        return data if isinstance(data, list) else []

    async def fetch_odds_for_sport(
        self,
        cfg: AppConfig,
        sport_key: str,
        commence_from_iso: Optional[str],
        commence_to_iso: Optional[str],
        markets: str,
    ) -> List[Dict[str, Any]]:
        # MOCK módban mock esemény
        if cfg.mode.upper() == "MOCK":
            return [_default_mock_event(sport_key, _now(), inplay=False)]

        if httpx is None:
            logger.error("[ERROR] LIVE mode requires httpx. Falling back to empty dataset.")
            return []

        key = os.getenv("ODDS_API_KEY", "")
        if cfg.mode.upper() != "MOCK" and not key:
            logger.error("[LIVE] Missing ODDS_API_KEY – skipping fetch for %s", sport_key)
            return []

        client = await _get_client()
        params: Dict[str, Any] = {
            "apiKey": key,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "regions": cfg.regions,
        }
        if commence_from_iso:
            params["commenceTimeFrom"] = commence_from_iso
        if commence_to_iso:
            params["commenceTimeTo"] = commence_to_iso
        if cfg.bookmakers:
            params["bookmakers"] = cfg.bookmakers

        data = await _get_json(client, f"{ODDS_API_BASE}/sports/{sport_key}/odds", params=params)
        if isinstance(data, list):
            return data
        logger.error("[ERROR] Odds API returned non-list payload for sport odds.")
        return []

    async def fetch_inplay_for_sport(self, cfg: AppConfig, sport_key: str, markets: str) -> List[Dict[str, Any]]:
        # MOCK módban mock in-play esemény
        if cfg.mode.upper() == "MOCK":
            return [_default_mock_event(sport_key, _now(), inplay=True)]

        if httpx is None:
            logger.error("[ERROR] LIVE mode requires httpx. Falling back to empty dataset.")
            return []

        key = os.getenv("ODDS_API_KEY", "")
        if cfg.mode.upper() != "MOCK" and not key:
            logger.error("[LIVE] Missing ODDS_API_KEY – skipping in-play fetch for %s", sport_key)
            return []

        client = await _get_client()
        params: Dict[str, Any] = {
            "apiKey": key,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "regions": cfg.regions,
        }
        if cfg.bookmakers:
            params["bookmakers"] = cfg.bookmakers
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds/live"
        try:
            data = await _get_json(client, url, params=params)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    async def fetch_event_odds(
        self,
        cfg: AppConfig,
        sport_key: str,
        event_id: str,
        markets: str,
        bookmakers: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        # MOCK módban ne próbálkozzon refetch-csel
        if cfg.mode.upper() == "MOCK":
            return None
        if httpx is None:
            logger.error("[ERROR] LIVE mode requires httpx. Skipping event refetch.")
            return None

        key = os.getenv("ODDS_API_KEY", "")
        if cfg.mode.upper() != "MOCK" and not key:
            logger.error("[LIVE] Missing ODDS_API_KEY – skipping event refetch for %s", event_id)
            return None

        client = await _get_client()
        params: Dict[str, Any] = {
            "apiKey": key,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
            "regions": cfg.regions,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        elif cfg.bookmakers:
            params["bookmakers"] = cfg.bookmakers

        url = f"{ODDS_API_BASE}/sports/{sport_key}/events/{event_id}/odds"
        try:
            data = await _get_json(client, url, params=params)
            if isinstance(data, list):
                return data[0] if data else None
            if isinstance(data, dict):
                return data
            return None
        except Exception:
            return None



class MockProvider(ProviderBase):
    name = "mock"

    async def fetch_sports(self, cfg: AppConfig) -> List[Dict[str, Any]]:
        return [{"key": "basketball_nba"}, {"key": "soccer_epl"}, {"key": "tennis_atp"}, {"key": "mma_mixed_martial_arts"}]

    async def fetch_odds_for_sport(
        self,
        cfg: AppConfig,
        sport_key: str,
        commence_from_iso: Optional[str],
        commence_to_iso: Optional[str],
        markets: str,
    ) -> List[Dict[str, Any]]:
        return [_default_mock_event(sport_key, _now(), inplay=False)]

    async def fetch_inplay_for_sport(self, cfg: AppConfig, sport_key: str, markets: str) -> List[Dict[str, Any]]:
        return [_default_mock_event(sport_key, _now(), inplay=True)]

async def fetch_event_odds(
    self,
    cfg: AppConfig,
    sport_key: str,
    event_id: str,
    markets: str,
    bookmakers: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # MOCK: no refetch needed
    if cfg.mode.upper() == "MOCK":
        return None
    if httpx is None:
        logger.error("[ERROR] LIVE mode requires httpx. Skipping event refetch.")
        return None
    key = os.getenv("ODDS_API_KEY", "")
    if cfg.mode.upper() == "LIVE" and not key:
        logger.error("[LIVE] Missing ODDS_API_KEY – skipping event refetch for %s", event_id)
        return None
    client = await _get_client()
    params: Dict[str, Any] = {
        "apiKey": key,
        "markets": markets,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        "regions": cfg.regions,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    elif cfg.bookmakers:
        params["bookmakers"] = cfg.bookmakers
    url = f"{ODDS_API_BASE}/sports/{sport_key}/events/{event_id}/odds"
    try:
        data = await _get_json(client, url, params=params)
        if isinstance(data, list):
            return data[0] if data else None
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


# ───────────── helpers ─────────────
def _parse_commence_iso_to_epoch(iso_s: str, default_epoch: int) -> int:
    try:
        dt = datetime.fromisoformat(iso_s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return default_epoch


def _parse_iso_ts(s: str, fallback: int) -> int:
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return fallback


_OUTRIGHT_KEYWORDS = (
    "winner",
    "outright",
    "outrights",
    "special",
    "specials",
    "award",
    "mvp",
    "cy_young",
    "top_scorer",
    "golden_boot",
    "to_win",
    "championship_winner",
    "manager",
    "next_manager",
)


def _is_two_way_sport_key(key: str) -> bool:
    s = (key or "").lower()
    return not any(w in s for w in _OUTRIGHT_KEYWORDS)


def _is_auto_two_way(cfg: AppConfig) -> bool:
    return (len(cfg.sports) == 1 and str(cfg.sports[0]).upper() == "AUTO_2WAY") or (os.getenv("AUTO_TWO_WAY", "0") == "1")


async def _resolve_sports_list(cfg: AppConfig, provider: ProviderBase) -> List[str]:
    if not _is_auto_two_way(cfg):
        return cfg.sports
    all_sports = await provider.fetch_sports(cfg)
    keys = [s.get("key") for s in all_sports if isinstance(s, dict) and isinstance(s.get("key"), str)]
    filtered = keys if cfg.allow_outrights else [k for k in keys if _is_two_way_sport_key(k)]
    if not filtered:
        filtered = keys
    try:
        max_sports = int(os.getenv("MAX_SPORTS", os.getenv("MAX_SPORT", "10")))
    except Exception:
        max_sports = 10
    if max_sports > 0:
        filtered = filtered[: max(1, max_sports)]
    dropped = set(keys) - set(filtered)
    if dropped and not cfg.allow_outrights:
        preview = ", ".join(sorted(dropped)[:10])
        suffix = "..." if len(dropped) > 10 else ""
        logger.info(f"[INFO] Skipped outright/futures sports ({len(dropped)} filtered): {preview}{suffix}")
    return filtered


def _market_is_strict_h2h(market: Dict[str, Any], h_norm: str, a_norm: str) -> bool:
    outs = [o for o in market.get("outcomes", []) if "name" in o and "price" in o]
    if len(outs) != 2:
        return False
    names = {_normalize_name(o.get("name", "")) for o in outs}
    return names == {h_norm, a_norm}


def _market_is_strict_totals(market: Dict[str, Any]) -> Tuple[bool, Optional[Decimal]]:
    outs = [o for o in market.get("outcomes", []) if "name" in o and "price" in o and "point" in o]
    if len(outs) != 2:
        return (False, None)
    names = {_normalize_name(o.get("name", "")) for o in outs}
    if names != {"over", "under"}:
        return (False, None)
    lines = {_normalize_line(o.get("point")) for o in outs}
    if None in lines or len(lines) != 1:
        return (False, None)
    return (True, next(iter(lines)))


def _market_is_strict_spreads(market: Dict[str, Any], h_norm: str, a_norm: str) -> Tuple[bool, Optional[Decimal]]:
    outs = [o for o in market.get("outcomes", []) if "name" in o and "price" in o and "point" in o]
    if len(outs) != 2:
        return (False, None)
    names = {_normalize_name(o.get("name", "")) for o in outs}
    if names != {h_norm, a_norm}:
        return (False, None)
    pts = [_normalize_line(o.get("point")) for o in outs]
    if any(p is None for p in pts):
        return (False, None)
    if abs(pts[0]) != abs(pts[1]):
        return (False, None)
    return (True, abs(pts[0]).quantize(Decimal("0.01")))


def _market_is_strict_dnb(market: Dict[str, Any], h_norm: str, a_norm: str) -> bool:
    outs = [o for o in market.get("outcomes", []) if "name" in o and "price" in o]
    if len(outs) != 2:
        return False
    names = {_normalize_name(o.get("name", "")) for o in outs}
    return names == {h_norm, a_norm}


def _market_is_btts(market: Dict[str, Any]) -> bool:
    outs = [o for o in market.get("outcomes", []) if "name" in o and "price" in o]
    if len(outs) != 2:
        return False
    names = {_normalize_name(o.get("name", "")) for o in outs}
    return names == {"yes", "no"}


def _filter_two_way(eo: EventOdds, market_key: str, now_ts: int) -> Dict[str, List[BookOffer]]:
    offers = dict(eo.offers.get(market_key, {}))
    keys = set(offers.keys())
    if keys == {"over", "under"}:
        offers = {"home": offers.get("over", []), "away": offers.get("under", [])}
    elif keys == {"yes", "no"}:
        offers = {"home": offers.get("yes", []), "away": offers.get("no", [])}
    elif keys != {"home", "away"}:
        return {}
    homes = combined_filter(staleness_gate(offers.get("home", []), eo.event_id, market_key, now_ts))
    aways = combined_filter(staleness_gate(offers.get("away", []), eo.event_id, market_key, now_ts))
    out: Dict[str, List[BookOffer]] = {}
    if homes:
        out["home"] = homes
    if aways:
        out["away"] = aways
    return out


def _build_markets_for_sport(sport_key: str) -> str:
    s = (sport_key or "").lower()
    mk = ["h2h", "totals", "spreads"]
    if s.startswith("soccer_"):
        mk += ["draw_no_bet", "both_teams_to_score"]
    return ",".join(sorted(set(mk)))


# ───────────── collection ─────────────
async def _collect_for_sport(provider: ProviderBase, cfg: AppConfig, sport_key: str, now_ts: int) -> List[EventOdds]:
    from_ts = now_ts + max(0, cfg.min_ttk_minutes) * 60
    to_ts = now_ts + max(0, int(cfg.time_window)) * 3600
    from_iso = datetime.fromtimestamp(from_ts, timezone.utc).isoformat().replace("+00:00", "Z")
    to_iso = datetime.fromtimestamp(to_ts, timezone.utc).isoformat().replace("+00:00", "Z")
    markets = _build_markets_for_sport(sport_key)
    prematch = await provider.fetch_odds_for_sport(cfg, sport_key, from_iso, to_iso, markets=markets)
    inplay = await provider.fetch_inplay_for_sport(cfg, sport_key, markets=markets) if cfg.enable_inplay else []
    raw_all = list(prematch) + list(inplay)
    events: List[EventOdds] = []
    for ev in raw_all:
        def_epoch = now_ts + 3 * 3600
        eo = EventOdds(
            sport_key=ev.get("sport_key", sport_key),
            event_id=ev.get("id", f"{sport_key}-{len(events) + 1:03d}"),
            commence_time=_parse_commence_iso_to_epoch(ev.get("commence_time", ""), def_epoch),
            home=ev.get("home_team", "Home"),
            away=ev.get("away_team", "Away"),
            offers={},
        )
        h_norm = _normalize_name(eo.home)
        a_norm = _normalize_name(eo.away)
        has_any = False
        for bm in ev.get("bookmakers", []):
            book_key = (bm.get("key") or bm.get("title") or "unknown").strip().lower()
            last_ts = _parse_iso_ts(bm.get("last_update", ""), now_ts)
            for m in bm.get("markets", []):
                mkey = (m.get("key") or "").lower()
                if mkey == "h2h":
                    if not _market_is_strict_h2h(m, h_norm, a_norm):
                        continue
                    for o in m.get("outcomes", []):
                        name = _normalize_name(str(o.get("name", "")))
                        outcome = "home" if name == h_norm else "away"
                        try:
                            price = D(o.get("price"))
                        except Exception:
                            continue
                        if not valid_price(price):
                            continue
                        eo.offers.setdefault("h2h", {}).setdefault(outcome, []).append(
                            BookOffer(book=book_key, market="h2h", outcome=outcome, odds=price, ts=last_ts)
                        )
                    has_any = True
                elif mkey == "totals":
                    ok, line = _market_is_strict_totals(m)
                    if not ok or line is None:
                        continue
                    mk = f"totals:{line}"
                    for o in m.get("outcomes", []):
                        name = _normalize_name(str(o.get("name", "")))
                        outcome = name
                        try:
                            price = D(o.get("price"))
                        except Exception:
                            continue
                        if not valid_price(price):
                            continue
                        eo.offers.setdefault(mk, {}).setdefault(outcome, []).append(
                            BookOffer(book=book_key, market="totals", outcome=outcome, line=line, odds=price, ts=last_ts)
                        )
                    has_any = True
                elif mkey == "spreads":
                    ok, line = _market_is_strict_spreads(m, h_norm, a_norm)
                    if not ok or line is None:
                        continue
                    mk = f"spreads:{line}"
                    for o in m.get("outcomes", []):
                        name = _normalize_name(str(o.get("name", "")))
                        outcome = "home" if name == h_norm else "away"
                        try:
                            price = D(o.get("price"))
                        except Exception:
                            continue
                        if not valid_price(price):
                            continue
                        eo.offers.setdefault(mk, {}).setdefault(outcome, []).append(
                            BookOffer(book=book_key, market="spreads", outcome=outcome, line=line, odds=price, ts=last_ts)
                        )
                    has_any = True
                elif mkey == "draw_no_bet":
                    if not _market_is_strict_dnb(m, h_norm, a_norm):
                        continue
                    for o in m.get("outcomes", []):
                        name = _normalize_name(str(o.get("name", "")))
                        outcome = "home" if name == h_norm else "away"
                        try:
                            price = D(o.get("price"))
                        except Exception:
                            continue
                        if not valid_price(price):
                            continue
                        eo.offers.setdefault("draw_no_bet", {}).setdefault(outcome, []).append(
                            BookOffer(book=book_key, market="draw_no_bet", outcome=outcome, odds=price, ts=last_ts)
                        )
                    has_any = True
                elif mkey in ("both_teams_to_score", "btts"):
                    if not _market_is_btts(m):
                        continue
                    for o in m.get("outcomes", []):
                        name = _normalize_name(str(o.get("name", "")))
                        outcome = name  # "yes" vagy "no"
                        try:
                            price = D(o.get("price"))
                        except Exception:
                            continue
                        if not valid_price(price):
                            continue
                        eo.offers.setdefault("btts", {}).setdefault(outcome, []).append(
                            BookOffer(book=book_key, market="btts", outcome=outcome, odds=price, ts=last_ts)
                        )
                    has_any = True
        if not has_any:
            continue
        keep = any(
            ({"home", "away"}.issubset(v.keys()) or {"over", "under"}.issubset(v.keys()) or {"yes", "no"}.issubset(v.keys()))
            for v in eo.offers.values()
        )
        if not keep:
            continue
        events.append(eo)
    return events


async def collect_events_parallel(cfg: AppConfig, provider: ProviderBase) -> Tuple[List[EventOdds], int]:
    now_ts = _now()
    sports_list = await _resolve_sports_list(cfg, provider)
    if not sports_list:
        sports_list = ["basketball_nba"]
    tasks = [_collect_for_sport(provider, cfg, sp, now_ts) for sp in sports_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    events: List[EventOdds] = []
    for sp, res in zip(sports_list, results):
        if isinstance(res, Exception):
            logger.warning(f"[WARN] Sport fetch error ({sp}): {res}")
            continue
        events.extend(res)
    return events, len(sports_list)


# ───────────── rounding ─────────────
def _round_down_to_step(x: int, step: int) -> int:
    if step <= 1:
        return x
    return max(0, (x // step) * step)


def _effective_step(amount: int, base_step: int, profile: str) -> int:
    if profile != "ladder":
        return max(1, base_step)
    if amount < 50:
        m = 1
    elif amount < 200:
        m = 5
    elif amount < 1000:
        m = 10
    else:
        m = 50
    return max(1, base_step * m)


def apply_stake_rounding2(
    stakeH: int,
    stakeA: int,
    stepH_base: int,
    stepA_base: int,
    oa: Decimal,
    ob: Decimal,
    fx: Decimal,
    is_dnb: bool = False,
    profile: str = "flat",
) -> Tuple[int, int]:
    sH = int(stakeH)
    sA = int(stakeA)
    stepH = _effective_step(sH, stepH_base, profile)
    stepA = _effective_step(sA, stepA_base, profile)
    sH = _round_down_to_step(sH, stepH)
    sA = _round_down_to_step(sA, stepA)
    for _ in range(40):
        if sH == 0 and sA == 0:
            break
        worst = profit_after_costs_dnb(oa, ob, sH, sA, fx) if is_dnb else profit_after_costs_two_way(oa, ob, sa=sH, sb=sA, fx=fx)
        if worst >= 0:
            break
        if sH >= sA and sH > 0:
            stepH = _effective_step(sH, stepH_base, profile)
            sH = _round_down_to_step(max(0, sH - stepH), stepH)
        elif sA > 0:
            stepA = _effective_step(sA, stepA_base, profile)
            sA = _round_down_to_step(max(0, sA - stepA), stepA)
        else:
            break
    return sH, sA


# ───────────── scoring core ─────────────
def _recency_factor(age_sec: int) -> Decimal:
    if age_sec <= 60:
        return D("1.0")
    if age_sec <= 5 * 60:
        return D("0.9")
    if age_sec <= 10 * 60:
        return D("0.8")
    if age_sec <= 30 * 60:
        return D("0.5")
    return D("0.2")


def _book_count_factor(n: int) -> Decimal:
    if n <= 2:
        return D("1.0")
    if n == 3:
        return D("0.9")
    return D("0.8")


def _market_group(mk: str) -> str:
    if mk == "h2h":
        return "h2h"
    if mk == "draw_no_bet":
        return "dnb"
    if mk.startswith("totals:"):
        return "totals"
    if mk.startswith("spreads:"):
        return "spreads"
    if mk == "btts":
        return "btts"
    return "other"


def _threshold_for_market(cfg: AppConfig, market_key: str, inplay: bool) -> Decimal:
    base = cfg.min_edge_after_costs_inplay if inplay else cfg.min_edge_after_costs
    group = _market_group(market_key)
    extra = (cfg.min_edge_market or {}).get(group, D(0))
    return max(base, extra or base)


def _compute_score(
    edge_after: Decimal,
    odds_age_sec: int,
    book_count: int,
    is_dnb: bool,
    dnb_factor: Decimal,
    inplay: bool,
    line_density: int,
) -> Decimal:
    rec = _recency_factor(odds_age_sec)
    bc = _book_count_factor(book_count)
    ld = D("1.0") if line_density <= 1 else clamp(D("1.0") + D(str(min(0.2, 0.05 * (line_density - 1)))), D("1.0"), D("1.2"))
    base = edge_after * rec * bc * ld
    if inplay:
        base *= D("0.85")
    return base * (dnb_factor if is_dnb else D(1))


def _book_caps(cfg: AppConfig, a_book: str, b_book: str) -> Dict[str, Decimal]:
    return {
        "home": D(cfg.book_caps.get(a_book.lower(), cfg.default_book_cap)),
        "away": D(cfg.book_caps.get(b_book.lower(), cfg.default_book_cap)),
    }


def _best_pair(homes: List[BookOffer], aways: List[BookOffer]) -> Tuple[Optional[BookOffer], Optional[BookOffer]]:
    best: Tuple[Decimal, Optional[BookOffer], Optional[BookOffer]] = (D("999"), None, None)
    for h in homes:
        for a in aways:
            if h.book == a.book:
                continue
            inv = inv_sum(h.odds, a.odds)
            if inv < best[0]:
                best = (inv, h, a)
    return best[1], best[2]


def _is_inplay_event(cfg: AppConfig, ev: EventOdds, now_ts: int) -> bool:
    return bool(cfg.enable_inplay and ev.commence_time <= now_ts)


def _dynamic_slippage(cfg: AppConfig, book: str, age_sec: int, inplay: bool, worst_case: bool = False) -> Decimal:
    key = (book or "").lower()
    static = (cfg.slippage_wc if worst_case else cfg.slippage_avg).get(key, D(0))
    add = D(0)
    if inplay:
        if age_sec > 300:
            add = D("0.0075")
        elif age_sec > 60:
            add = D("0.0050")
        else:
            add = D("0.0030")
    total = static + add
    MAX_SLIP = D("0.020")  # 2% felső határ
    return clamp(total, D(0), MAX_SLIP)


# ───────────── ledger & dedupe + rotation gzip ─────────────
JSONL = LOG_DIR / "ledger.jsonl"
CSVF = LOG_DIR / "ledger.csv"
DEDUPF = LOG_DIR / "notify_cache.json"
_DEDUPF_TMP = LOG_DIR / "notify_cache.json.tmp"

_CSV_HEADER = [
    "ts",
    "latency_ms",
    "sport_count",
    "offer_count",
    "event_id",
    "sport_key",
    "market",
    "line",
    "book_a",
    "book_b",
    "odds_a",
    "odds_b",
    "inv_sum",
    "edge_before_costs",
    "edge_after_costs",
    "stake_a",
    "stake_b",
    "worst",
    "cvar5",
    "method",
    "commence_time",
]


def _gzip_rotate(path: Path) -> None:
    try:
        import gzip  # noqa
        import shutil

        if not path.exists():
            return
        gz = Path(str(path) + ".gz")
        with open(path, "rb") as f_in, gzip.open(gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"[LEDGER] gzip rotate failed: {e}")


async def append_jsonl(row: Dict[str, Any], max_bytes: Optional[int] = None) -> None:
    try:
        if max_bytes and JSONL.exists() and JSONL.stat().st_size > max_bytes:
            bak = JSONL.with_name(JSONL.name + ".1")
            try:
                if bak.exists():
                    await asyncio.to_thread(bak.unlink)
                await asyncio.to_thread(JSONL.replace, bak)
                await asyncio.to_thread(_gzip_rotate, bak)
            except Exception:
                pass

        def _write():
            with JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        await asyncio.to_thread(_write)
    except Exception as e:
        logger.error(f"[LEDGER] append_jsonl failed: {e}")

async def append_csv(row: Dict[str, Any], max_bytes: Optional[int] = None) -> None:
    try:
        if max_bytes and CSVF.exists() and CSVF.stat().st_size > max_bytes:
            bak = CSVF.with_name(CSVF.name + ".1")
            try:
                if bak.exists():
                    await asyncio.to_thread(bak.unlink)
                await asyncio.to_thread(CSVF.replace, bak)
                await asyncio.to_thread(_gzip_rotate, bak)
            except Exception:
                pass

        def _write():
            exists = CSVF.exists()
            with CSVF.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=_CSV_HEADER)
                if not exists:
                    w.writeheader()
                w.writerow({k: row.get(k) for k in _CSV_HEADER})
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
        await asyncio.to_thread(_write)
    except Exception as e:
        logger.error(f"[LEDGER] append_csv failed: {e}")


def _round3(x: Decimal) -> str:
    try:
        return f"{D(x):.3f}"
    except Exception:
        return str(x)


def _roundN(x: Decimal, digits: int) -> str:
    try:
        d = max(0, min(8, int(digits)))  # védett tartomány
        q = Decimal(1).scaleb(-d)
        return f"{D(x).quantize(q)}"
    except Exception:
        return str(x)


def _dedupe_key(arb: ArbOpportunity) -> str:
    line = ""
    if arb.market.startswith(("totals:", "spreads:")):
        try:
            line = arb.market.split(":", 1)[1]
        except Exception:
            line = ""
    return (
        f"{arb.event_id}|{arb.market}|{line}|"
        f"{arb.offer_a.book}:{_roundN(arb.offer_a.odds, DEDUPE_ODDS_PRECISION)}|"
        f"{arb.offer_b.book}:{_roundN(arb.offer_b.odds, DEDUPE_ODDS_PRECISION)}"
    )


def _load_dedupe() -> Dict[str, int]:
    try:
        return json.loads(DEDUPF.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_dedupe(d: Dict[str, int]) -> None:
    try:
        with _DEDUPF_TMP.open("w", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False))
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        _DEDUPF_TMP.replace(DEDUPF)
    except Exception as e:
        logger.error(f"[DEDUPE] Save failed: {e}")



def should_notify(arb: ArbOpportunity, window_sec: int) -> bool:
    cache = _load_dedupe()
    key = _dedupe_key(arb)
    now = _now()
    cutoff = now - max(1, int(window_sec))
    for k in list(cache.keys()):
        if int(cache.get(k, 0)) < cutoff:
            cache.pop(k, None)
    last = int(cache.get(key, 0))
    if now - last < window_sec:
        _save_dedupe(cache)
        return False
    cache[key] = now
    _save_dedupe(cache)
    return True


# ───────────── reminder cache (pre-match only, odds-NEM-változás esetén) ─────────────
REMINDF = LOG_DIR / "reminder_cache.json"
_REMIND_TMP = LOG_DIR / "reminder_cache.json.tmp"


def _load_cache(path: Path) -> Dict[str, int]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}



def _save_cache(path: Path, tmp: Path, d: Dict[str, int]) -> None:
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False))
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        tmp.replace(path)
    except Exception as e:
        logger.error(f"[REMINDER-CACHE] Save failed: {e}")



def _reminder_key(arb: ArbOpportunity) -> str:
    line = ""
    if arb.market.startswith(("totals:", "spreads:")):
        try:
            line = arb.market.split(":", 1)[1]
        except Exception:
            line = ""
    # odds NINCS a kulcsban -> változás nélkül is ugyanaz marad
    return f"{arb.event_id}|{arb.market}|{line}"


def should_remind(arb: ArbOpportunity, interval_sec: int) -> bool:
    if interval_sec <= 0:
        return False
    now = _now()
    if arb.commence_time <= now:
        return False
    cache = _load_cache(REMINDF)
    key = _reminder_key(arb)
    last = int(cache.get(key, 0))
    if now - last >= interval_sec:
        cache[key] = now
        _save_cache(REMINDF, _REMIND_TMP, cache)
        return True
    return False


# ───────────── Telegram ─────────────
def _uk_time_str(epoch: int) -> str:
    try:
        tz = ZoneInfo("Europe/London") if ZoneInfo else timezone.utc
    except Exception:
        tz = timezone.utc
    dt = datetime.fromtimestamp(epoch, tz)
    return dt.strftime("%d %b %Y, %H:%M %Z")


def _ttk_str(epoch: int, now_ts: Optional[int] = None) -> str:
    now = now_ts or _now()
    delta = epoch - now
    if delta <= 0:
        return "LIVE/started"
    h = delta // 3600
    m = (delta % 3600) // 60
    return f"T-{h}h{m:02d}m" if h > 0 else f"T-{m}m"


def _grade(edge_after: Decimal) -> str:
    if edge_after >= D("0.020"):
        return "HIGH"
    if edge_after >= D("0.010"):
        return "MED"
    return "LOW"


def _fmt_age(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    m = sec // 60
    s = sec % 60
    return f"{m}m{s:02d}s"


def _fmt_pct(x: Decimal, digits: int = 3) -> str:
    q = Decimal(1).scaleb(-int(digits))
    return f"{(x * Decimal(100)).quantize(q)}%"


def _pretty_market(mk: str) -> str:
    if mk.startswith("totals:"):
        return f"totals @ {mk.split(':', 1)[1]}"
    if mk.startswith("spreads:"):
        return f"spread @ {mk.split(':', 1)[1]}"
    if mk == "draw_no_bet":
        return "DNB"
    if mk == "btts":
        return "BTTS Yes/No"
    return mk


from urllib.parse import urlencode

def format_telegram(arb: ArbOpportunity) -> str:
    uk_time = _uk_time_str(arb.commence_time)
    ttk = _ttk_str(arb.commence_time)
    grade = _grade(arb.edge_after_costs)

    if arb.market.startswith("totals:"):
        prices = f"Over({arb.offer_a.book} {arb.offer_a.odds}) / Under({arb.offer_b.book} {arb.offer_b.odds})"
    elif arb.market.startswith("spreads:"):
        prices = f"{arb.home_team}({arb.offer_a.book} {arb.offer_a.odds}) / {arb.away_team}({arb.offer_b.book} {arb.offer_b.odds})"
    elif arb.market == "btts":
        prices = f"Yes({arb.offer_a.book} {arb.offer_a.odds}) / No({arb.offer_b.book} {arb.offer_b.odds})"
    else:
        prices = f"{arb.offer_a.book} {arb.offer_a.odds} / {arb.offer_b.book} {arb.offer_b.odds}"

    now = _now()
    odds_age_sec = max(now - arb.offer_a.ts, now - arb.offer_b.ts)

    extra = []
    if arb.trend_score is not None:
        extra.append(f"TrendScore: {float(arb.trend_score):.3f}")
    if arb.ml_score is not None:
        extra.append(f"ML: {float(arb.ml_score):.3f}")
    xtra = " | " + " ".join(extra) if extra else ""

    lines: List[str] = []
    lines.append(f"ARB — {grade} — {_pretty_market(arb.market)}")
    lines.append(f"{arb.home_team} vs {arb.away_team}")
    lines.append(f"Kickoff: {uk_time} ({ttk})")
    lines.append(f"Prices: {prices}")
    lines.append(
        f"Edge(after): {_fmt_pct(arb.edge_after_costs)} | Score: {float(arb.score):.3f} | Age: {_fmt_age(odds_age_sec)}{xtra}"
    )
    lines.append(f"Books: {arb.offer_a.book} vs {arb.offer_b.book}")

    # Kalkulátor link – biztonságos URL kódolással
    params = {
        "oddsA": f"{arb.offer_a.odds:.2f}",
        "oddsB": f"{arb.offer_b.odds:.2f}",
    }
    calc_url = f"https://proarbitrage.github.io/arb-calc/?{urlencode(params)}"
    lines.append(f"🔗 Calculator: {calc_url}")
    lines.append("ℹ️ Info only. No auto-bets.")
    return "\n".join(lines)



def _split_telegram_text(text: str, max_len: int = 4096) -> List[str]:
    if len(text) <= max_len:
        return [text]
    parts: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for line in text.split("\n"):
        if cur_len + len(line) + 1 > max_len and cur:
            parts.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        parts.append("\n".join(cur))
    return parts


def _http_post_json(url: str, payload: Dict[str, Any]) -> Tuple[int, str]:
    """
    Minimal fallback HTTP POST using urllib when httpx is unavailable.
    Returns (status_code, response_text); on error returns (0, repr(error)).
    """
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:  # type: ignore
            code = getattr(resp, "status", getattr(resp, "code", 200))
            body = resp.read().decode("utf-8", errors="replace")
            return int(code), body
    except Exception as e:
        return 0, repr(e)


async def telegram_send_message(token: str, chat_id: str, text: str) -> Tuple[int, str]:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = _split_telegram_text(text)
    last_code, last_body = 0, ""
    for chunk in chunks:
        payload = {"chat_id": chat_id, "text": chunk}
        if httpx is None:
            last_code, last_body = await asyncio.to_thread(_http_post_json, url, payload)
        else:
            client = await _get_client()
            r = await client.post(url, json=payload)
            last_code, last_body = r.status_code, getattr(r, "text", "")
        if last_code != 200:
            return last_code, last_body
    return last_code, last_body

async def telegram_get_updates(token: str, limit: int = 5) -> Tuple[int, str]:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    if httpx is None:
        return await asyncio.to_thread(_http_post_json, url, {"limit": limit})
    client = await _get_client()
    r = await client.get(url, params={"limit": limit})
    return r.status_code, getattr(r, "text", "")

async def telegram_send_with_retry(token: str, chat_id: str, text: str, attempts: int = 5) -> Tuple[int, str]:
    backoff = 0.5
    last_code, last_body = 0, ""
    for _ in range(max(1, attempts)):
        try:
            code, body = await telegram_send_message(token, chat_id, text)
            if code == 200:
                return code, body
            last_code, last_body = code, body
        except Exception as e:
            last_code, last_body = 0, str(e)
        await asyncio.sleep(min(5.0, backoff))
        backoff = min(5.0, backoff * 2 + random.random() * 0.2)
    return last_code, last_body


def _trend_features_for_offer(event_id: str, market_key: str, offer: BookOffer, now_ts: int) -> Dict[str, Decimal]:
    key = (event_id, market_key, offer.outcome, offer.book)
    ps = _SERIES.get(key)
    if not ps or ps.ewma is None:
        return {
            "ewma_dev": D(0),
            "cusum_mag": D(0),
            "age": clamp(D(max(0, now_ts - offer.ts)) / D(600), D(0), D(1)),
        }
    ewma_dev = abs(offer.odds - ps.ewma) / max(offer.odds, D("1e-9"))
    ewma_dev = clamp(ewma_dev / D("0.02"), D(0), D(1))
    cusum_mag = clamp(max(ps.cusum_pos, abs(ps.cusum_neg)) / D("0.02"), D(0), D(1))
    age = clamp(D(max(0, now_ts - offer.ts)) / D(600), D(0), D(1))
    return {"ewma_dev": ewma_dev, "cusum_mag": cusum_mag, "age": age}


def compute_trend_score(cfg: AppConfig, event_id: str, market_key: str, a: BookOffer, b: BookOffer, now_ts: int) -> Decimal:
    """Egyszerű, robusztus trend-score: stabilitás + momentum jel kombinációja."""
    fa = _trend_features_for_offer(event_id, market_key, a, now_ts)
    fb = _trend_features_for_offer(event_id, market_key, b, now_ts)
    stab = D(1) - (fa["ewma_dev"] + fb["ewma_dev"]) / D(2)
    calm = D(1) - (fa["cusum_mag"] + fb["cusum_mag"]) / D(2)
    fresh = D(1) - (fa["age"] + fb["age"]) / D(2)
    score = clamp((stab * D("0.5") + calm * D("0.3") + fresh * D("0.2")), D(0), D(1))
    return score


# ───────────── ML wrapper (opcionális) ─────────────
class MLWrapper:
    def __init__(self, path: Optional[str], shadow: bool = True, weight: Decimal = D("0")):
        self.path = path
        self.shadow = shadow
        self.weight = weight
        self.model = None
        self.ready = False
        if _SKLEARN_OK and path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)  # Pipeline[StandardScaler+LogReg] javasolt
                self.ready = True
                logger.info(f"[ML] Model loaded: {path}")
            except Exception as e:
                logger.warning(f"[ML] Load failed, fallback to heuristic. {e}")
        else:
            if path:
                logger.warning("[ML] Model path not found or sklearn not available. Using heuristic/shadow.")
            self.ready = False

    @staticmethod
    def _feat_vector(
        inv_sum_f: float,
        age_sec: int,
        ewma_dev: float,
        cusum_mag: float,
        line_density: int,
        market_group: str,
    ) -> List[float]:
        mg_map = {"h2h": 0, "totals": 1, "spreads": 2, "dnb": 3, "btts": 4}
        onehot = [0, 0, 0, 0, 0]
        onehot[mg_map.get(market_group, 0)] = 1
        return [
            inv_sum_f,
            min(1.0, max(0.0, age_sec / 600.0)),
            ewma_dev,
            cusum_mag,
            min(1.0, max(0.0, (line_density - 1) / 10.0)),
            *onehot,
        ]

    def predict01(
        self,
        inv_sum_dec: Decimal,
        age_sec: int,
        ewma_dev: Decimal,
        cusum_mag: Decimal,
        line_density: int,
        market_group: str,
    ) -> float:
        x = [
            self._feat_vector(
                float(inv_sum_dec),
                age_sec,
                float(ewma_dev),
                float(cusum_mag),
                line_density,
                market_group,
            )
        ]
        if _SKLEARN_OK and self.ready and self.model is not None:
            try:
                p = float(self.model.predict_proba(x)[0][1])
                return max(0.0, min(1.0, p))
            except Exception:
                pass
        base = 1.0 - min(0.2, max(0.0, float(inv_sum_dec) - 0.95)) / 0.2
        stabil = 1.0 - float(min(Decimal(1), ewma_dev))
        calm = 1.0 - float(min(Decimal(1), cusum_mag))
        ld = 1.0 - min(1.0, max(0.0, (line_density - 1) / 10.0))
        p = max(0.0, min(1.0, 0.25 * base + 0.25 * stabil + 0.25 * calm + 0.25 * ld))
        if self.shadow:
            try:
                if random.random() < 0.02:  # ~2% mintavétel
                    logger.info(
                        f"[ML-shadow] inv={float(inv_sum_dec):.3f} age={age_sec}s "
                        f"ewma_dev={float(ewma_dev):.3f} cusum={float(cusum_mag):.3f} "
                        f"lines={line_density} mg={market_group} -> p≈{p:.3f}"
                    )
            except Exception:
                pass
        return p


def _compute_trend_and_ml_boosts(
    cfg: AppConfig,
    ev: EventOdds,
    mk: str,
    a: BookOffer,
    b: BookOffer,
    line_density: int,
    inplay_evt: bool,
    inv_raw: Decimal,
) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    trend_s: Optional[Decimal] = None
    ml_s: Optional[Decimal] = None
    now_now = _now()
    if cfg.enable_trend_scoring and cfg.trend_weight > 0:
        trend_s = compute_trend_score(cfg, ev.event_id, mk, a, b, now_now)
    if cfg.enable_ml and cfg.ml_weight > 0:
        fa = _trend_features_for_offer(ev.event_id, mk, a, now_now)
        fb = _trend_features_for_offer(ev.event_id, mk, b, now_now)
        ewma_dev = (fa["ewma_dev"] + fb["ewma_dev"]) / D(2)
        cusum_mag = (fa["cusum_mag"] + fb["cusum_mag"]) / D(2)
        ageA = max(0, now_now - a.ts)
        ageB = max(0, now_now - b.ts)
        oage = max(ageA, ageB)
        mg = _market_group(mk)
        ml: Optional[MLWrapper] = getattr(_compute_trend_and_ml_boosts, "_ml", None)  # type: ignore[attr-defined]
        if ml is None:
            ml = MLWrapper(path=cfg.ml_model_path, shadow=cfg.ml_shadow_mode, weight=cfg.ml_weight)
            setattr(_compute_trend_and_ml_boosts, "_ml", ml)  # type: ignore[attr-defined]
        ml_s = D(str(ml.predict01(inv_raw, oage, ewma_dev, cusum_mag, line_density, mg)))
    return trend_s, ml_s


# ───────────── discovery ─────────────
async def discover_arbs(cfg: AppConfig, provider: ProviderBase, stake_method: str = "robust") -> List[ArbOpportunity]:
    t0 = time.perf_counter()
    now_ts = _now()
    purge_series(now_ts)
    events, sport_count = await collect_events_parallel(cfg, provider)
    offer_count = 0
    for ev in events:
        for _, by_out in ev.offers.items():
            for k in ("home", "away", "over", "under", "yes", "no"):
                offer_count += len(by_out.get(k, []))
    opps: List[ArbOpportunity] = []
    min_ttk = max(0, int(cfg.min_ttk_minutes)) * 60
    max_horizon = max(0, int(cfg.time_window)) * 3600
    for ev in events:
        inplay_evt = bool(cfg.enable_inplay and ev.commence_time <= now_ts)
        if not inplay_evt:
            if (ev.commence_time - now_ts) < min_ttk:
                continue
        if max_horizon and (ev.commence_time - now_ts) > max_horizon:
            continue
        for mk in list(ev.offers.keys()):
            if not (mk == "h2h" or mk.startswith("totals:") or mk == "draw_no_bet" or mk.startswith("spreads:") or mk == "btts"):
                continue
            offers_map = _filter_two_way(eo=ev, market_key=mk, now_ts=now_ts)
            homes = offers_map.get("home", [])
            aways = offers_map.get("away", [])
            if not homes or not aways:
                continue
            line_density = len(set([o.book for o in homes] + [o.book for o in aways]))
            book_count = line_density
            a, b = _best_pair(homes, aways)
            if not a or not b:
                continue
            inv_raw = inv_sum(a.odds, b.odds)
            if inv_raw >= D("1") - EPS:
                continue
            edge_before = D("1") - inv_raw
            now_now = _now()
            ageA = max(0, now_now - a.ts)
            ageB = max(0, now_now - b.ts)
            odds_eff = apply_costs(
                {"home": a.odds, "away": b.odds},
                fees={"home": cfg.fees.get(a.book, D(0)), "away": cfg.fees.get(b.book, D(0))},
                slip={
                    "home": _dynamic_slippage(cfg, a.book, ageA, inplay_evt, worst_case=False),
                    "away": _dynamic_slippage(cfg, b.book, ageB, inplay_evt, worst_case=False),
                },
            )
            odds_eff_for_edge = {
                "home": clamp(odds_eff["home"] * (D(1) - cfg.fx_cost), ODDS_MIN, ODDS_MAX),
                "away": clamp(odds_eff["away"] * (D(1) - cfg.fx_cost), ODDS_MIN, ODDS_MAX),
            }
            s_eff_edge = (D(1) / odds_eff_for_edge["home"]) + (D(1) / odds_eff_for_edge["away"])
            edge_after = D("1") - s_eff_edge
            thr = _threshold_for_market(cfg, mk, inplay_evt)
            if edge_after <= thr - EPS:
                continue
            trend_s, ml_s = _compute_trend_and_ml_boosts(cfg, ev, mk, a, b, line_density, inplay_evt, inv_raw)
            if cfg.enable_trend_scoring and cfg.trend_min_gate > 0 and trend_s is not None:
                if trend_s < cfg.trend_min_gate:
                    continue
            caps = _book_caps(cfg, a.book, b.book)
            bank = cfg.start_balance * 2
            if cfg.event_max_stake:
                bank = min(bank, D(cfg.event_max_stake))
            if stake_method == "kelly":
                fair_probs = fair_probs_from_offers({"home": homes, "away": aways}, method=cfg.consensus_method)
                plan = allocate_stakes_kelly(fair_probs, {"home": D(a.odds), "away": D(b.odds)}, caps=caps, bank=bank, frac=cfg.kelly_frac)
            elif stake_method == "mv":
                fair_probs = fair_probs_from_offers({"home": homes, "away": aways}, method=cfg.consensus_method)
                plan = allocate_stakes_mv(fair_probs, {"home": D(a.odds), "away": D(b.odds)}, caps=caps, bank=bank, risk_aversion=cfg.risk_aversion)
            elif stake_method == "prop":
                plan = allocate_stakes_proportional(D(a.odds), D(b.odds), caps["home"], caps["away"], bank)
            else:
                plan = allocate_stakes_robust(
                    {"home": D(a.odds), "away": D(b.odds)},
                    caps,
                    bank,
                    fees_map={"home": cfg.fees.get(a.book, D(0)), "away": cfg.fees.get(b.book, D(0))},
                    fx_cost=cfg.fx_cost,
                    eps_wc={"home": cfg.slippage_wc.get(a.book, D(0)), "away": cfg.slippage_wc.get(b.book, D(0))},
                )
            stakeH, stakeA = plan.stakes.get("home", 0), plan.stakes.get("away", 0)
            stepH = (cfg.min_step_per_book or {}).get(a.book, cfg.stake_round_step)
            stepA = (cfg.min_step_per_book or {}).get(b.book, cfg.stake_round_step)
            is_dnb = mk == "draw_no_bet"
            stakeH, stakeA = apply_stake_rounding2(
                stakeH, stakeA, stepH, stepA, odds_eff["home"], odds_eff["away"], cfg.fx_cost, is_dnb=is_dnb, profile=cfg.stake_round_profile
            )
            minH = (cfg.min_bet_per_book or {}).get(a.book, 0)
            minA = (cfg.min_bet_per_book or {}).get(b.book, 0)
            if (stakeH and stakeH < minH) or (stakeA and stakeA < minA):
                continue
            if cfg.event_max_stake and (stakeH + stakeA) > cfg.event_max_stake:
                scale = Decimal(cfg.event_max_stake) / Decimal(max(1, stakeH + stakeA))
                stakeH = int((Decimal(stakeH) * scale).to_integral_value(rounding=ROUND_FLOOR))
                stakeA = int((Decimal(stakeA) * scale).to_integral_value(rounding=ROUND_FLOOR))
            worst = (
                profit_after_costs_dnb(odds_eff["home"], odds_eff["away"], stakeH, stakeA, cfg.fx_cost)
                if is_dnb
                else profit_after_costs_two_way(odds_eff["home"], odds_eff["away"], sa=stakeH, sb=stakeA, fx=cfg.fx_cost)
            )
            if worst < 0:
                prop_plan = allocate_stakes_proportional(D(a.odds), D(b.odds), caps["home"], caps["away"], bank)
                sH2, sA2 = prop_plan.stakes["home"], prop_plan.stakes["away"]
                if cfg.event_max_stake and (sH2 + sA2) > cfg.event_max_stake:
                    scale = Decimal(cfg.event_max_stake) / Decimal(max(1, sH2 + sA2))
                    sH2 = int((Decimal(sH2) * scale).to_integral_value(rounding=ROUND_FLOOR))
                    sA2 = int((Decimal(sA2) * scale).to_integral_value(rounding=ROUND_FLOOR))
                sH2, sA2 = apply_stake_rounding2(
                    sH2, sA2, stepH, stepA, odds_eff["home"], odds_eff["away"], cfg.fx_cost, is_dnb=is_dnb, profile=cfg.stake_round_profile
                )
                if (sH2 and sH2 < minH) or (sA2 and sA2 < minA):
                    continue
                worst = (
                    profit_after_costs_dnb(odds_eff["home"], odds_eff["away"], sH2, sA2, cfg.fx_cost)
                    if is_dnb
                    else profit_after_costs_two_way(odds_eff["home"], odds_eff["away"], sa=sH2, sb=sA2, fx=cfg.fx_cost)
                )
                if worst < 0:
                    continue
                stakeH, stakeA = sH2, sA2
            if (stakeH + stakeA) <= 0:
                continue

            # Latency and odds age
            latency_ms = int((time.perf_counter() - t0) * 1000)
            oage = max(_now() - a.ts, _now() - b.ts)

            # Scoring
            base_score = _compute_score(edge_after, oage, book_count, is_dnb, cfg.dnb_score_factor, inplay_evt, line_density)
            fin_score = base_score
            if cfg.enable_trend_scoring and cfg.trend_weight > 0 and trend_s is not None:
                fin_score = max(D(0), fin_score * (D(1) + cfg.trend_weight * (D(2) * trend_s - D(1))))
            if cfg.enable_ml and cfg.ml_weight > 0 and ml_s is not None:
                fin_score = max(D(0), fin_score * (D(1) + cfg.ml_weight * (D(2) * ml_s - D(1))))

            # Append opportunity
            opps.append(ArbOpportunity(
                event_id=ev.event_id,
                sport_key=ev.sport_key,
                market=mk,
                home_team=ev.home,
                away_team=ev.away,
                offer_a=a,
                offer_b=b,
                inv_sum=inv_raw,
                edge_before_costs=edge_before,
                edge_after_costs=edge_after,
                score=fin_score,
                stake_a=stakeH,
                stake_b=stakeA,
                worst_case_profit=worst,
                cvar_5=worst,
                latency_ms=latency_ms,
                offer_count=offer_count,
                sport_count=sport_count,
                commence_time=ev.commence_time,
                trend_score=trend_s,
                ml_score=ml_s,
            ))

            # Ledger row
            line_str = ""
            if mk.startswith(("totals:", "spreads:")):
                try:
                    line_str = mk.split(":", 1)[1]
                except Exception:
                    line_str = ""
            row = {
                "ts": _now(),
                "latency_ms": latency_ms,
                "sport_count": sport_count,
                "offer_count": offer_count,
                "event_id": ev.event_id,
                "sport_key": ev.sport_key,
                "market": mk,
                "line": line_str,
                "book_a": a.book,
                "book_b": b.book,
                "odds_a": str(a.odds),
                "odds_b": str(b.odds),
                "inv_sum": str(inv_raw),
                "edge_before_costs": str(edge_before),
                "edge_after_costs": str(edge_after),
                "stake_a": stakeH,
                "stake_b": stakeA,
                "worst": str(worst),
                "cvar5": str(worst),
                "method": plan.method,
                "commence_time": ev.commence_time,
            }
            try:
                mb_env = os.getenv("LOG_MAX_BYTES")
                max_bytes = int(mb_env) if mb_env else None
            except Exception:
                max_bytes = None
            try:
                await append_jsonl(row, max_bytes=max_bytes)
                await append_csv(row, max_bytes=max_bytes)
            except Exception:
                logger.exception("[LEDGER] append failed")

    # Experimental middle scanning (after processing all markets/events)
    if cfg.enable_middle_scan:
        for ev_m in events:
            total_lines = {k: v for k, v in ev_m.offers.items() if k.startswith("totals:")}
            try:
                lines_sorted = sorted(total_lines.keys(), key=lambda s: D(s.split(":", 1)[1]))
            except Exception:
                lines_sorted = list(total_lines.keys())
            for i in range(len(lines_sorted)):
                for j in range(i + 1, len(lines_sorted)):
                    L1, L2 = lines_sorted[i], lines_sorted[j]
                    overL1 = total_lines[L1].get("over", [])
                    underL2 = total_lines[L2].get("under", [])
                    if not overL1 or not underL2:
                        continue
                    a_mid = max(overL1, key=lambda x: x.odds)
                    b_mid = max(underL2, key=lambda x: x.odds)
                    if a_mid.book == b_mid.book:
                        continue
                    logger.info(
                        f"[EXPERIMENTAL MIDDLE] {ev_m.sport_key} {ev_m.home} vs {ev_m.away}: Over@{L1}({a_mid.book} {a_mid.odds}) / Under@{L2}({b_mid.book} {b_mid.odds})"
                    )
    return opps


# ───────────── prealert refetch ─────────────
async def _prealert_refetch_and_validate(cfg: AppConfig, provider: ProviderBase, opp: ArbOpportunity) -> bool:
    if cfg.mode.upper() != "LIVE":
        return True
    if not cfg.prealert_refetch:
        return True
    if opp.market == "h2h":
        markets = "h2h"
    elif opp.market == "draw_no_bet":
        markets = "draw_no_bet"
    elif opp.market.startswith("totals:"):
        markets = "totals"
    elif opp.market.startswith("spreads:"):
        markets = "spreads"
    elif opp.market == "btts":
        markets = "both_teams_to_score"
    else:
        markets = "h2h"
    books_for_refetch = f"{opp.offer_a.book},{opp.offer_b.book}"
    refreshed = await provider.fetch_event_odds(cfg, opp.sport_key, opp.event_id, markets=markets, bookmakers=books_for_refetch)
    if not refreshed:
        return False
    home_norm = _normalize_name(opp.home_team)
    away_norm = _normalize_name(opp.away_team)

    def _bm_matches(bm: Dict[str, Any], book_name_lc: str) -> bool:
        key_lc = str(bm.get("key", "")).strip().lower()
        title_lc = str(bm.get("title", "")).strip().lower()
        return book_name_lc == key_lc or book_name_lc == title_lc

    def _find(book_name: str, outcome_label: str, want_line: Optional[Decimal]) -> Optional[Decimal]:
        book_name_lc = (book_name or "").lower()
        for bm in refreshed.get("bookmakers", []):
            if not _bm_matches(bm, book_name_lc):
                continue
            for m in bm.get("markets", []):
                mkey = (m.get("key") or "").lower()
                if mkey not in ("h2h", "totals", "draw_no_bet", "spreads", "both_teams_to_score", "btts"):
                    continue
                for o in m.get("outcomes", []):
                    nname = _normalize_name(str(o.get("name", "")))
                    if opp.market.startswith("totals:"):
                        if nname not in ("over", "under"):
                            continue
                        pline = _normalize_line(o.get("point"))
                        if want_line is not None and (pline is None or pline != want_line):
                            continue
                        if (outcome_label == "home" and nname == "over") or (outcome_label == "away" and nname == "under"):
                            try:
                                return D(o.get("price"))
                            except Exception:
                                return None
                    elif opp.market.startswith("spreads:"):
                        if nname not in (home_norm, away_norm):
                            continue
                        pline0 = _normalize_line(o.get("point"))
                        pline = abs(pline0) if pline0 is not None else None
                        if want_line is not None and (pline is None or pline != want_line):
                            continue
                        desired = home_norm if outcome_label == "home" else away_norm
                        if nname == desired:
                            try:
                                return D(o.get("price"))
                            except Exception:
                                return None
                    elif opp.market in ("draw_no_bet", "h2h"):
                        target = home_norm if outcome_label == "home" else away_norm
                        if nname == target:
                            try:
                                return D(o.get("price"))
                            except Exception:
                                return None
                    elif opp.market == "btts":
                        if outcome_label == "home" and nname == "yes":
                            try:
                                return D(o.get("price"))
                            except Exception:
                                return None
                        if outcome_label == "away" and nname == "no":
                            try:
                                return D(o.get("price"))
                            except Exception:
                                return None
        return None

    line = None
    if opp.market.startswith(("totals:", "spreads:")):
        try:
            line = D(opp.market.split(":", 1)[1])
        except Exception:
            line = None
    new_a = _find(opp.offer_a.book, opp.offer_a.outcome, line)
    new_b = _find(opp.offer_b.book, opp.offer_b.outcome, line)
    if new_a is None or new_b is None:
        return False
    inv_raw_new = inv_sum(new_a, new_b)
    if inv_raw_new >= D("1") - EPS:
        return False
    now_now = _now()
    ageA = max(0, now_now - (opp.offer_a.ts or now_now))
    ageB = max(0, now_now - (opp.offer_b.ts or now_now))
    inplay_evt = bool(cfg.enable_inplay and opp.commence_time <= now_now)
    odds_eff_new = apply_costs(
        {"home": new_a, "away": new_b},
        fees={"home": cfg.fees.get(opp.offer_a.book, D(0)), "away": cfg.fees.get(opp.offer_b.book, D(0))},
        slip={
            "home": _dynamic_slippage(cfg, opp.offer_a.book, ageA, inplay_evt, worst_case=False),
            "away": _dynamic_slippage(cfg, opp.offer_b.book, ageB, inplay_evt, worst_case=False),
        },
    )
    odds_eff_new_edge = {
        "home": clamp(odds_eff_new["home"] * (D(1) - cfg.fx_cost), ODDS_MIN, ODDS_MAX),
        "away": clamp(odds_eff_new["away"] * (D(1) - cfg.fx_cost), ODDS_MIN, ODDS_MAX),
    }
    s_eff_new = (D(1) / odds_eff_new_edge["home"]) + (D(1) / odds_eff_new_edge["away"])
    edge_after_new = D("1") - s_eff_new
    if edge_after_new <= D(0):
        return False
    if edge_after_new < opp.edge_after_costs * cfg.prealert_edge_drop_frac:
        return False
    return True


# ───────────── CLI ─────────────
def _build_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("arb-bot-ops+")
    p.add_argument("--healthcheck", action="store_true", help="Gyors önellenőrzés és kilépés (0/2 exit code).")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mode", type=str, default=None, help="MOCK|LIVE")
    p.add_argument("--regions", type=str, default=None)
    p.add_argument("--sports", type=str, default=None, help="Comma-separated list or AUTO_2WAY")
    p.add_argument("--time-window", type=int, default=None, dest="time_window")
    p.add_argument("--min-edge-after-costs", type=str, default=None, dest="min_edge")
    p.add_argument("--min-edge-after-costs-inplay", type=str, default=None, dest="min_edge_inplay")
    p.add_argument("--bookmakers", type=str, default=None)
    p.add_argument("--min-ttk-minutes", type=int, default=None, dest="min_ttk_minutes")
    p.add_argument("--stake-round", type=int, default=None, dest="stake_round_step")
    p.add_argument("--stake-round-profile", type=str, default=None, choices=["flat", "ladder"])
    p.add_argument("--concurrency", type=int, default=None, dest="concurrency")
    p.add_argument("--consensus-method", type=str, default=None, choices=["median", "trimmed_mean"])
    p.add_argument("--provider", type=str, default=None, choices=["theoddsapi", "mock"])
    p.add_argument("--prealert", action="store_true")
    p.add_argument("--no-prealert", dest="no_prealert", action="store_true")
    p.add_argument("--prealert-edge-drop-frac", type=float, default=None)
    p.add_argument("--allow-outrights", action="store_true")
    p.add_argument("--disallow-outrights", action="store_true")
    p.add_argument("--inplay", action="store_true")
    p.add_argument("--stake-method", type=str, default="robust", choices=["robust", "prop", "kelly", "mv"])
    p.add_argument("--notify", action="store_true")
    p.add_argument("--log-max-bytes", type=int, default=None)
    p.add_argument("--tg-test", action="store_true")
    p.add_argument("--tg-debug", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--enable-middle-scan", action="store_true")
    p.add_argument("--interval", type=int, default=int(os.getenv("ARB_INTERVAL", "60")), help="Service loop interval seconds (default 60)")
    p.add_argument("--once", action="store_true", help="Run one cycle and exit")
    # NEW flags
    p.add_argument("--enable-trend-scoring", action="store_true")
    p.add_argument("--trend-weight", type=float, default=None)
    p.add_argument("--trend-min-gate", type=float, default=None)
    p.add_argument("--enable-ml", action="store_true")
    p.add_argument("--ml-weight", type=float, default=None)
    p.add_argument("--ml-model-path", type=str, default=None)
    p.add_argument("--ml-shadow", action="store_true")
    p.add_argument("--enable-daily-calibration", action="store_true")
    # NEW: echo / reminders CLI
    p.add_argument("--local", action="store_true", help="Új (dedupe átment) riasztások írása a konzolra.")
    p.add_argument("--no-local", dest="no_local", action="store_true", help="Ne írjon a konzolra (csak küldés).")
    p.add_argument("--reminder-interval", type=int, default=None, help="Pre-match emlékeztető intervallum mp-ben (alap 7200).")
    p.add_argument("--no-reminders", action="store_true", help="Emlékeztetők kikapcsolása.")
    # folyamat-zár és profilozás kapcsolók
    p.add_argument(
        "--lock-file",
        type=str,
        default=os.getenv("ARB_LOCK_FILE", str(LOG_DIR / ".arb-bot.lock")),
        help="Folyamat-zár fájl (alap: logs/.arb-bot.lock)",
    )
    p.add_argument("--profile", action="store_true", help="cProfile bekapcsolása a teljes futásra")
    p.add_argument("--profile-out", type=str, default=str(LOG_DIR / "profile.stats"), help="Profilozási kimenet (alap: logs/profile.stats)")
    return p.parse_args()


def _apply_cli_overrides(cfg: AppConfig, args: argparse.Namespace) -> AppConfig:
    updated = dict(cfg.__dict__)
    if args.mode:
        updated["mode"] = args.mode.upper()
    if args.regions:
        updated["regions"] = _normalize_regions(args.regions)
    if args.sports:
        updated["sports"] = [x.strip() for x in args.sports.split(",") if x.strip()]
    if args.time_window is not None:
        updated["time_window"] = int(args.time_window)
    if args.min_edge is not None:
        updated["min_edge_after_costs"] = pct_to_decimal(args.min_edge)
    if args.min_edge_inplay is not None:
        updated["min_edge_after_costs_inplay"] = pct_to_decimal(args.min_edge_inplay)
    if args.bookmakers is not None:
        updated["bookmakers"] = args.bookmakers
    if args.min_ttk_minutes is not None:
        updated["min_ttk_minutes"] = int(args.min_ttk_minutes)
    if args.stake_round_step is not None:
        updated["stake_round_step"] = int(args.stake_round_step)
    if args.stake_round_profile is not None:
        updated["stake_round_profile"] = str(args.stake_round_profile).lower()
    if args.consensus_method is not None:
        updated["consensus_method"] = str(args.consensus_method).lower()
    if args.concurrency is not None:
        os.environ["ODDS_CONCURRENCY"] = str(max(1, int(args.concurrency)))
    if args.log_max_bytes is not None:
        os.environ["LOG_MAX_BYTES"] = str(max(0, int(args.log_max_bytes)))
    if args.provider is not None:
        updated["provider"] = str(args.provider).lower()
    if args.prealert:
        updated["prealert_refetch"] = True
    if getattr(args, "no_prealert", False):
        updated["prealert_refetch"] = False
    if args.prealert_edge_drop_frac is not None:
        updated["prealert_edge_drop_frac"] = D(args.prealert_edge_drop_frac)
    if args.allow_outrights:
        updated["allow_outrights"] = True
        os.environ["ALLOW_OUTRIGHTS"] = "1"
    if getattr(args, "disallow_outrights", False):
        updated["allow_outrights"] = False
        os.environ["ALLOW_OUTRIGHTS"] = "0"
    if args.inplay:
        updated["enable_inplay"] = True
        os.environ["ENABLE_INPLAY"] = "1"
    if args.enable_middle_scan:
        updated["enable_middle_scan"] = True
    # NEW
    if args.enable_trend_scoring:
        updated["enable_trend_scoring"] = True
    if args.trend_weight is not None:
        updated["trend_weight"] = D(str(max(0.0, args.trend_weight)))
    if args.trend_min_gate is not None:
        updated["trend_min_gate"] = D(str(max(0.0, min(1.0, args.trend_min_gate))))
    if args.enable_ml:
        updated["enable_ml"] = True
    if args.ml_weight is not None:
        updated["ml_weight"] = D(str(max(0.0, args.ml_weight)))
    if args.ml_model_path is not None:
        updated["ml_model_path"] = args.ml_model_path
    if args.ml_shadow:
        updated["ml_shadow_mode"] = True
    if args.enable_daily_calibration:
        updated["enable_daily_calibration"] = True
    # NEW: echo / reminders
    if args.local:
        updated["local_echo"] = True
    if getattr(args, "no_local", False):
        updated["local_echo"] = False
    if args.reminder_interval is not None:
        updated["reminder_interval_sec"] = max(0, int(args.reminder_interval))
    if getattr(args, "no_reminders", False):
        updated["reminders_enabled"] = False
    return AppConfig(**updated)  # type: ignore


def _start_profiler(enabled: bool):
    if not enabled:
        return None
    try:
        pr = cProfile.Profile()
        pr.enable()
        return pr
    except Exception:
        logger.warning("Profilozás nem indítható (cProfile hiba).")
        return None


def _stop_profiler(pr, out_path: str):
    if not pr:
        return
    try:
        pr.disable()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pr.dump_stats(out_path)
        logger.info(f"[PROFILE] Mentve: {out_path}")
    except Exception as e:
        logger.warning(f"[PROFILE] Mentés sikertelen: {e}")


def _make_provider(name: str) -> ProviderBase:
    return MockProvider() if name == "mock" else TheOddsAPIProvider()


# ───────────── self tests ─────────────
def _self_test_sport_filter():
    assert _is_two_way_sport_key("basketball_nba") is True
    assert _is_two_way_sport_key("soccer_epl") is True
    assert _is_two_way_sport_key("americanfootball_nfl_super_bowl_winner") is False
    assert _is_two_way_sport_key("soccer_uefa_champions_league_winner") is False


def _self_test_apply_costs():
    eff = apply_costs({"home": D("2.00"), "away": D("2.00")}, {"home": D("0.02")}, {"away": D("0.01")})
    assert eff["home"] < D("2.00") and eff["away"] < D("2.00")


def _self_test_format_telegram():
    a = BookOffer(book="a", market="h2h", outcome="home", odds=D("2.1"), ts=_now())
    b = BookOffer(book="b", market="h2h", outcome="away", odds=D("1.9"), ts=_now())
    opp = ArbOpportunity(
        event_id="e1",
        sport_key="basketball_nba",
        market="h2h",
        home_team="Alpha",
        away_team="Beta",
        offer_a=a,
        offer_b=b,
        inv_sum=D("0.99"),
        edge_before_costs=D("0.01"),
        edge_after_costs=D("0.008"),
        score=D("0.006"),
        stake_a=10,
        stake_b=9,
        worst_case_profit=D("0.5"),
        cvar_5=D("0.5"),
        latency_ms=123,
        offer_count=4,
        sport_count=2,
        commence_time=_now() + 3600,
    )
    txt = format_telegram(opp)
    assert "ARB —" in txt and "🔗 Calculator:" in txt and "\n" in txt



def run_self_tests() -> int:
    _self_test_sport_filter()
    _self_test_apply_costs()
    _self_test_format_telegram()
    logger.info("✅ Self-tests passed.")
    return 0


# ───────────── Daily calibration ─────────────
_last_calib_day_key: Optional[str] = None


def _day_key_utc(ts: Optional[int] = None) -> str:
    dt = datetime.fromtimestamp(ts or _now(), timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _should_run_calibration(cfg: AppConfig) -> bool:
    """Egyszer/nap, cfg.calib_run_at_utc (HH:MM) után."""
    global _last_calib_day_key
    if not cfg.enable_daily_calibration:
        return False
    now = datetime.now(timezone.utc)
    try:
        hh, mm = (int(x) for x in (cfg.calib_run_at_utc or "00:05").split(":"))
    except Exception:
        hh, mm = 0, 5
    ts_gate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    dk = _day_key_utc()
    if _last_calib_day_key == dk:
        return False
    if now >= ts_gate:
        return True
    return False


def _read_ledger_rows(days: int = 1) -> List[Dict[str, Any]]:
    """Read last N days of ledger rows from JSONL with basic filtering."""
    rows: List[Dict[str, Any]] = []
    try:
        if JSONL.exists():
            with JSONL.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line.strip()))
                    except Exception:
                        pass
        else:
            return []
        since = _now() - days * 86400
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                if int(r.get("ts", 0)) >= since:
                    out.append(r)
            except Exception:
                pass
        return out
    except Exception:
        return []


def _calibrate(cfg: AppConfig) -> AppConfig:
    """Daily conservative calibration using Q1 of recent edges with optional drawdown brake."""
    try:
        rows = _read_ledger_rows(days=max(1, cfg.calib_lookback_days))
        if not rows:
            return cfg
        edges: List[Decimal] = []
        worst_vals: List[Decimal] = []
        for r in rows:
            try:
                edges.append(D(r.get("edge_after_costs", "0")))
                worst_vals.append(D(r.get("worst", "0")))
            except Exception:
                pass
        if len(edges) < 5:
            return cfg
        q25 = D(str(statistics.quantiles(edges, n=4)[0])) if len(edges) >= 4 else D(str(statistics.median(edges)))
        lo, hi = cfg.calib_min_edge_bounds
        target = clamp(q25, lo, hi)
        new_min_edge = clamp(cfg.min_edge_after_costs * D("0.7") + target * D("0.3"), lo, hi)
        if cfg.calib_drawdown_brake and worst_vals:
            worst_min = min(worst_vals)
            if worst_min < D("-5"):
                new_min_edge = clamp(new_min_edge + D("0.001"), lo, hi)
        alpha = clamp(cfg.calib_slippage_alpha, D(0), D(1))
        sl_avg = dict(cfg.slippage_avg)
        for k, v in sl_avg.items():
            sl_avg[k] = clamp(v * (D("1.0") + D("0.05") * alpha), D(0), D("0.02"))
        updated = dict(cfg.__dict__)
        updated["min_edge_after_costs"] = new_min_edge
        updated["slippage_avg"] = sl_avg
        logger.info(f"[CALIB] min_edge_after_costs -> {float(new_min_edge):.4f}, slippage_avg adj (alpha={float(alpha):.2f})")
        return AppConfig(**updated)  # type: ignore
    except Exception as e:
        logger.warning(f"[CALIB] Skipping calibration due to error: {e}")
        return cfg

async def _service_once(cfg: AppConfig, provider: ProviderBase, stake_method: str) -> None:
    """Run one scan cycle: discover arbs, optional refetch validation, notify & reminders."""
    try:
        opps = await discover_arbs(cfg, provider, stake_method=stake_method)
    except Exception:
        logger.exception("[SERVICE] discover_arbs failed")
        return
    if not opps:
        return
    opps = sorted(opps, key=lambda o: o.score, reverse=True)
    for opp in opps:
        # Optional LIVE prealert refetch validation
        if cfg.prealert_refetch:
            try:
                if not await _prealert_refetch_and_validate(cfg, provider, opp):
                    continue
            except Exception:
                continue
        # Dedupe handling & reminders
        if not should_notify(opp, cfg.dedupe_window_sec):
            if cfg.reminders_enabled and should_remind(opp, cfg.reminder_interval_sec):
                msg = "⏰ Reminder\n" + format_telegram(opp)
                if cfg.telegram_token and cfg.telegram_chat_id:
                    await telegram_send_with_retry(cfg.telegram_token, cfg.telegram_chat_id, msg)
            continue
        # Primary notification
        text = format_telegram(opp)
        if cfg.local_echo:
            logger.info("\n" + text)
        if cfg.telegram_token and cfg.telegram_chat_id:
            await telegram_send_with_retry(cfg.telegram_token, cfg.telegram_chat_id, text)

async def _run(cfg: AppConfig, provider: ProviderBase, interval_sec: int, lock_file: str, stake_method: str) -> None:
    try:
        async with _GUARD:
            with single_instance(lock_file):
                while True:
                    if _should_run_calibration(cfg):
                        cfg2 = _calibrate(cfg)
                        if cfg2 is not None:
                            global _last_calib_day_key
                            _last_calib_day_key = _day_key_utc()
                            cfg = cfg2
                    await _service_once(cfg, provider, stake_method)
                    await asyncio.sleep(max(1, int(interval_sec)))
    except InstanceLockError as e:
        logger.error(str(e))
    finally:
        await _close_client()

def main():
    args = _build_cli()
    if args.healthcheck:
        sys.exit(healthcheck())
    cfg = load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)
    errs = validate_config(cfg)
    if errs:
        for e in errs:
            logger.error(f"[CONFIG] {e}")
        sys.exit(2)
    provider = _make_provider(cfg.provider)
    if args.tg_test and cfg.telegram_token and cfg.telegram_chat_id:
        asyncio.run(telegram_send_with_retry(cfg.telegram_token, cfg.telegram_chat_id, "✅ Telegram OK"))
    if args.once:
        asyncio.run(_service_once(cfg, provider, stake_method=getattr(args, "stake_method", "robust")))
    else:
        asyncio.run(_run(
            cfg, provider,
            interval_sec=getattr(args, "interval", 60),
            lock_file=getattr(args, "lock_file", str(LOG_DIR / ".arb-bot.lock")),
            stake_method=getattr(args, "stake_method", "robust"),
        ))

if __name__ == "__main__":
    main()
