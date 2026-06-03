# -*- coding: utf-8 -*-
"""취업통계 — `output/{연도}_job.csv` (2011~2024) 집계."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from lib.admissions_data import (
    DRILL_ALL_LABEL,
    DRILL_COLUMNS,
    MAJOR_COLUMN,
    normalize_branch,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
JOB_YEAR_START = 2011
JOB_YEAR_END = 2024
LATEST_JOB_YEAR = JOB_YEAR_END

SCHOOL_KEYS = ["학교명", "본분교", "KEDI 학교코드"]
EXTRA_DIM_COLUMNS: list[str] = DRILL_COLUMNS + ["학과코드", "학위구분", "과정구분"]

JOB_SUM_COLUMNS: list[str] = [
    "졸업자_계",
    "졸업자_남",
    "졸업자_여",
    "취업자_합계_계",
    "취업자_합계_남",
    "취업자_합계_여",
    "취업자_교외취업자_계",
    "취업자_교내취업자_계",
    "취업자_해외취업자_계",
    "진학자_계",
    "진학자_남",
    "진학자_여",
    "1차 유지취업자_계",
    "2차 유지취업자_계",
    "취업불가능자_계",
    "제외인정자_계",
    "미상_계",
]

# KPI: kind=sum | rate (가중 비율 = 분자합/분모합×100)
JOB_KPI_DEFINITIONS: list[dict[str, str]] = [
    {"key": "졸업자_계", "label": "졸업자", "unit": "명", "kind": "sum"},
    {"key": "취업자_합계_계", "label": "취업자(합계)", "unit": "명", "kind": "sum"},
    {
        "key": "취업률",
        "label": "취업률",
        "unit": "%",
        "kind": "rate",
        "num": "취업자_합계_계",
        "den": "졸업자_계",
    },
    {"key": "진학자_계", "label": "진학자", "unit": "명", "kind": "sum"},
    {
        "key": "진학률",
        "label": "진학률",
        "unit": "%",
        "kind": "rate",
        "num": "진학자_계",
        "den": "졸업자_계",
    },
    {
        "key": "유지취업률_1차",
        "label": "1차 유지취업률",
        "unit": "%",
        "kind": "rate",
        "num": "1차 유지취업자_계",
        "den": "취업자_합계_계",
    },
    {"key": "1차 유지취업자_계", "label": "1차 유지취업자", "unit": "명", "kind": "sum"},
]


def job_csv_path(year: int) -> Path:
    return OUTPUT_DIR / f"{year}_job.csv"


def list_job_year_paths() -> list[Path]:
    paths: list[Path] = []
    for year in range(JOB_YEAR_START, JOB_YEAR_END + 1):
        p = job_csv_path(year)
        if p.is_file():
            paths.append(p)
    return paths


def year_from_job_path(path: Path) -> int:
    stem = path.stem
    if stem.endswith("_job"):
        return int(stem.replace("_job", ""))
    return int(stem)


def load_job_year_raw(year: int, *, school_name_prefix: str = "") -> pd.DataFrame:
    path = job_csv_path(year)
    if not path.is_file():
        raise FileNotFoundError(f"취업 데이터 없음: {path}")
    want = set(SCHOOL_KEYS + JOB_SUM_COLUMNS + EXTRA_DIM_COLUMNS + ["통계연도", "조사기준일"])
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    cols = [c for c in df.columns if c in want]
    df = df[cols].copy()
    if "통계연도" not in df.columns:
        df["통계연도"] = year
    if school_name_prefix:
        p = str(school_name_prefix).strip()
        df = df[df["학교명"].astype(str).str.startswith(p)].copy()
    return df


def filter_job_school(
    df: pd.DataFrame,
    school_name: str,
    branch: str,
    *,
    kedi_code: str = "",
    major_category: str | None = None,
) -> pd.DataFrame:
    name = str(school_name).strip()
    branch_n = normalize_branch(branch)
    branch_series = df["본분교"].astype(str).map(normalize_branch)

    mask = (df["학교명"].astype(str).str.strip() == name) & (branch_series == branch_n)
    if not mask.any() and kedi_code:
        code = str(kedi_code).strip()
        if code and code.lower() not in ("nan", "none", ""):
            mask = (df["KEDI 학교코드"].astype(str).str.strip() == code) & (
                branch_series == branch_n
            )
    out = df.loc[mask]
    if major_category and str(major_category).strip() not in ("", "전체"):
        if MAJOR_COLUMN in out.columns:
            major = str(major_category).strip()
            out = out[out[MAJOR_COLUMN].astype(str).str.strip() == major]
    return out


def _sum_columns(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in columns:
        if col not in df.columns:
            out[col] = float("nan")
            continue
        out[col] = float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    return out


def summarize_job_kpis(df: pd.DataFrame) -> dict[str, float]:
    sums = _sum_columns(df, JOB_SUM_COLUMNS)
    result: dict[str, float] = {}
    for d in JOB_KPI_DEFINITIONS:
        key = d["key"]
        if d["kind"] == "sum":
            result[key] = sums.get(key, float("nan"))
        elif d["kind"] == "rate":
            num = sums.get(d["num"], float("nan"))
            den = sums.get(d["den"], float("nan"))
            if num != num or den != den or den <= 0:
                result[key] = float("nan")
            else:
                result[key] = num / den * 100
    return result


def load_job_school_options(
    year: int = LATEST_JOB_YEAR,
    *,
    school_name_prefix: str = "",
) -> pd.DataFrame:
    df = load_job_year_raw(year, school_name_prefix=school_name_prefix)
    opts = df[SCHOOL_KEYS].drop_duplicates().sort_values(["학교명", "본분교"])
    opts = opts[opts["학교명"].astype(str).str.strip() != ""]
    return opts.reset_index(drop=True)


def list_job_major_categories(
    school_name: str,
    branch: str,
    year: int,
    *,
    kedi_code: str = "",
    school_name_prefix: str = "",
) -> list[str]:
    df = load_job_year_raw(year, school_name_prefix=school_name_prefix)
    subset = filter_job_school(df, school_name, branch, kedi_code=kedi_code)
    if MAJOR_COLUMN not in subset.columns:
        return []
    majors = (
        subset[MAJOR_COLUMN]
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": ""})
    )
    return sorted({m for m in majors if m})


def school_job_kpis_for_year(
    school_name: str,
    branch: str,
    year: int,
    *,
    kedi_code: str = "",
    major_category: str | None = None,
    school_name_prefix: str = "",
) -> dict[str, float]:
    df = load_job_year_raw(year, school_name_prefix=school_name_prefix)
    subset = filter_job_school(
        df, school_name, branch, kedi_code=kedi_code, major_category=major_category
    )
    return summarize_job_kpis(subset)


def build_job_yearly_series(
    school_name: str,
    branch: str,
    kpi_key: str,
    *,
    kedi_code: str = "",
    major_category: str | None = None,
    school_name_prefix: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in list_job_year_paths():
        year = year_from_job_path(path)
        kpis = school_job_kpis_for_year(
            school_name,
            branch,
            year,
            kedi_code=kedi_code,
            major_category=major_category,
            school_name_prefix=school_name_prefix,
        )
        rows.append({"연도": year, "값": kpis.get(kpi_key, float("nan"))})
    return pd.DataFrame(rows).sort_values("연도")


def job_kpis_by_group(
    df: pd.DataFrame,
    group_column: str,
    *,
    label_column: str | None = None,
) -> pd.DataFrame:
    from lib.admissions_data import apply_hierarchy_filters, list_drill_options

    label_col = label_column or group_column
    if group_column not in df.columns:
        return pd.DataFrame()

    groups = list_drill_options(df, group_column, {})
    rows: list[dict[str, object]] = []
    for group in groups:
        sub = df[df[group_column].astype(str).str.strip() == group]
        kpis = summarize_job_kpis(sub)
        row: dict[str, object] = {label_col: group}
        for d in JOB_KPI_DEFINITIONS:
            unit = d["unit"]
            col = f"{d['label']} ({unit})" if unit else d["label"]
            row[col] = format_job_kpi_value(d["key"], kpis.get(d["key"], float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def format_job_kpi_value(key: str, value: float) -> str:
    if value != value:
        return "—"
    if key in ("취업률", "진학률", "유지취업률_1차"):
        return f"{value:.1f}"
    return f"{value:,.0f}"


def format_job_kpi_raw(key: str, value: float) -> str:
    if value != value:
        return "—"
    defn = next((d for d in JOB_KPI_DEFINITIONS if d["key"] == key), None)
    if defn and defn["unit"] == "%":
        return f"{value:.1f}%"
    return f"{value:,.0f}"


def job_kpi_delta(current: float, previous: float, key: str) -> float | None:
    if current != current or previous != previous:
        return None
    defn = next((d for d in JOB_KPI_DEFINITIONS if d["key"] == key), None)
    if defn and defn["unit"] == "%":
        return current - previous
    if previous == 0:
        return None
    return (current - previous) / abs(previous) * 100
