# -*- coding: utf-8 -*-
"""입시(Admissions) KPI — 연도별 `output/YYYY_data.csv` 집계."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
LATEST_YEAR = 2024
YEAR_START = 2008

# 학교(캠퍼스) 단위 합산 대상
SUM_COLUMNS: list[str] = [
    "입학정원_학부_계",
    "모집인원_학부_계",
    "정원내_모집인원_학부",
    "정원외_모집정원_학부",
    "지원자_전체_계",
    "입학자_전체_계",
    "정원내_입학자_학부_계",
    "정원내_입학자_전체_계",
    "정원외_입학자_전체_계",
]

# 입학자 성별 집계용 컬럼
ENROLLMENT_GENDER_COLUMNS: list[str] = [
    "입학자_전체_계",
    "입학자_전체_남",
    "입학자_전체_여",
    "정원내_입학자_학부_계",
    "정원내_입학자_학부_남",
    "정원내_입학자_학부_여",
    "정원내_입학자_전체_계",
    "정원내_입학자_전체_남",
    "정원내_입학자_전체_여",
]

# 입학·휴학·유예·외국인·졸업 등 학생 현황
STUDENT_STATUS_DEFINITIONS: list[dict[str, str]] = [
    {"key": "입학자_전체_계", "label": "입학자(전체)", "column": "입학자_전체_계"},
    {"key": "휴학생_전체_계", "label": "휴학생(전체)", "column": "휴학생_전체_계"},
    {"key": "유예생_전체_계", "label": "유예생(전체)", "column": "유예생_전체_계"},
    {
        "key": "외국 학생_총계_계",
        "label": "외국인 학생(총계)",
        "column": "외국 학생_총계_계",
    },
    {"key": "졸업자_전체", "label": "졸업자(전체)", "column": "졸업자_전체"},
]

STUDENT_STATUS_COLUMNS: list[str] = [d["column"] for d in STUDENT_STATUS_DEFINITIONS]

# 전임·비전임 교원 성별
FACULTY_GENDER_COLUMNS: list[str] = [
    "전임교원_계",
    "전임교원_남",
    "전임교원_여",
    "비전임교원_계",
    "비전임교원_남",
    "비전임교원_여",
]

FACULTY_SCOPE_ALL = "전체"

FACULTY_GENDER_SCOPES: list[dict[str, str]] = [
    {
        "id": "전임",
        "label": "전임교원",
        "total": "전임교원_계",
        "male": "전임교원_남",
        "female": "전임교원_여",
    },
    {
        "id": "비전임",
        "label": "비전임교원",
        "total": "비전임교원_계",
        "male": "비전임교원_남",
        "female": "비전임교원_여",
    },
]

ENROLLMENT_GENDER_SCOPES: list[dict[str, str]] = [
    {
        "id": "전체",
        "label": "입학자(전체)",
        "total": "입학자_전체_계",
        "male": "입학자_전체_남",
        "female": "입학자_전체_여",
    },
    {
        "id": "정원내_학부",
        "label": "정원내 입학자(학부)",
        "total": "정원내_입학자_학부_계",
        "male": "정원내_입학자_학부_남",
        "female": "정원내_입학자_학부_여",
    },
    {
        "id": "정원내_전체",
        "label": "정원내 입학자(전체)",
        "total": "정원내_입학자_전체_계",
        "male": "정원내_입학자_전체_남",
        "female": "정원내_입학자_전체_여",
    },
]

# KPI 정의: (표시명, 키, 단위, 설명)
KPI_DEFINITIONS: list[dict[str, str]] = [
    {
        "key": "모집인원_학부_계",
        "label": "모집인원(학부)",
        "unit": "명",
        "kind": "sum",
    },
    {
        "key": "지원자_전체_계",
        "label": "지원자(전체)",
        "unit": "명",
        "kind": "sum",
    },
    {
        "key": "입학자_전체_계",
        "label": "입학자(전체)",
        "unit": "명",
        "kind": "sum",
    },
    {
        "key": "정원내_모집인원_학부",
        "label": "정원내 모집인원(학부)",
        "unit": "명",
        "kind": "sum",
    },
    {
        "key": "정원내_입학자_학부_계",
        "label": "정원내 입학자(학부)",
        "unit": "명",
        "kind": "sum",
    },
    {
        "key": "경쟁률",
        "label": "경쟁률",
        "unit": "배",
        "kind": "rate",
    },
    {
        "key": "신입생_충원율",
        "label": "신입생 충원율",
        "unit": "%",
        "kind": "rate",
    },
    {
        "key": "입학률",
        "label": "입학률(지원 대비)",
        "unit": "%",
        "kind": "rate",
    },
]

SCHOOL_KEYS = ["학교코드", "학교명", "본분교"]
MAJOR_COLUMN = "대계열"
DRILL_COLUMNS: list[str] = ["대계열", "중계열", "소계열", "학과명"]
EXTRA_DIM_COLUMNS: list[str] = DRILL_COLUMNS + ["학과코드"]
DRILL_ALL_LABEL = "전체"

# 연도별 본분교 표기 통일 (2024: 본교(제1캠퍼스) ↔ 구형: 본교)
BRANCH_ALIASES: dict[str, str] = {
    "본교(제1캠퍼스)": "본교",
    "본교(제2캠퍼스)": "제2캠퍼스",
    "본교(제3캠퍼스)": "제3캠퍼스",
    "본교(제4캠퍼스)": "제4캠퍼스",
    "분교(제1캠퍼스)": "분교",
    "분교1": "분교",
}


def normalize_branch(value: object) -> str:
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    return BRANCH_ALIASES.get(s, s)


def _coerce_label(value: object) -> str:
    """계열·학과 등 라벨 컬럼 값을 정렬·집합 가능한 문자열로 변환."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, dict, tuple, set)):
        return ""
    s = str(value).strip()
    if s.lower() in ("nan", "none", "<na>", "nat"):
        return ""
    return s


def _unique_non_empty_labels(column_data: pd.Series | pd.DataFrame) -> list[str]:
    """Series(또는 동명 컬럼 DataFrame)에서 비어 있지 않은 고유 라벨 목록."""
    if isinstance(column_data, pd.DataFrame):
        column_data = column_data.iloc[:, 0] if column_data.shape[1] else pd.Series(dtype=object)
    seen: set[str] = set()
    labels: list[str] = []
    for raw in column_data.tolist():
        label = _coerce_label(raw)
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return sorted(labels)


def list_year_data_paths() -> list[Path]:
    paths: list[Path] = []
    for year in range(YEAR_START, LATEST_YEAR + 1):
        p = OUTPUT_DIR / f"{year}_data.csv"
        if p.is_file():
            paths.append(p)
    return paths


def year_from_path(path: Path) -> int:
    m = re.match(r"^(\d{4})_data\.csv$", path.name)
    if not m:
        raise ValueError(path.name)
    return int(m.group(1))


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _sum_school(df: pd.DataFrame) -> dict[str, float]:
    """학과 행 → 학교·캠퍼스 단위 합계 및 파생 KPI."""
    totals: dict[str, float] = {}
    for col in SUM_COLUMNS:
        if col in df.columns:
            totals[col] = float(_to_numeric(df[col]).sum())
        else:
            totals[col] = 0.0

    recruit = totals.get("모집인원_학부_계", 0.0)
    if recruit <= 0:
        recruit = totals.get("정원내_모집인원_학부", 0.0)
    if recruit <= 0:
        recruit = totals.get("입학정원_학부_계", 0.0)

    applicants = totals.get("지원자_전체_계", 0.0)
    enrolled = totals.get("입학자_전체_계", 0.0)
    quota_recruit = totals.get("정원내_모집인원_학부", 0.0)
    if quota_recruit <= 0:
        quota_recruit = recruit
    quota_enrolled = totals.get("정원내_입학자_학부_계", 0.0)

    totals["경쟁률"] = applicants / recruit if recruit > 0 else float("nan")
    totals["신입생_충원율"] = (
        (quota_enrolled / quota_recruit * 100) if quota_recruit > 0 else float("nan")
    )
    totals["입학률"] = (enrolled / applicants * 100) if applicants > 0 else float("nan")
    return totals


def sum_kpis_from_frame(df: pd.DataFrame) -> dict[str, float]:
    """학과 행 집합 → KPI dict."""
    return _sum_school(df)


def enrollment_gender_scope(scope_id: str) -> dict[str, str]:
    for s in ENROLLMENT_GENDER_SCOPES:
        if s["id"] == scope_id:
            return s
    raise KeyError(f"unknown enrollment gender scope: {scope_id}")


def _sum_numeric_columns(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in columns:
        if col not in df.columns:
            out[col] = float("nan")
            continue
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        out[col] = float(s.sum())
    return out


def summarize_enrollment_gender(
    df: pd.DataFrame, scope_id: str = "전체"
) -> dict[str, float]:
    """학과 행 집합 → 입학자 남·여·계·비율."""
    scope = enrollment_gender_scope(scope_id)
    sums = _sum_numeric_columns(df, [scope["total"], scope["male"], scope["female"]])
    total = sums[scope["total"]]
    male = sums[scope["male"]]
    female = sums[scope["female"]]
    if total != total or total <= 0:
        male_pct = female_pct = float("nan")
    else:
        male_pct = male / total * 100
        female_pct = female / total * 100
    return {
        "total": total,
        "male": male,
        "female": female,
        "male_pct": male_pct,
        "female_pct": female_pct,
    }


def enrollment_gender_by_group(
    df: pd.DataFrame,
    group_column: str,
    scope_id: str = "전체",
    *,
    label_column: str | None = None,
) -> pd.DataFrame:
    """행=group_column, 열=남·여·계·비율."""
    label_col = label_column or group_column
    if group_column not in df.columns:
        return pd.DataFrame()

    groups = list_drill_options(df, group_column, {})
    rows: list[dict[str, object]] = []
    for group in groups:
        sub = df[df[group_column].astype(str).str.strip() == group]
        g = summarize_enrollment_gender(sub, scope_id)
        rows.append(
            {
                label_col: group,
                "입학자(계)": g["total"],
                "남": g["male"],
                "여": g["female"],
                "남(%)": g["male_pct"],
                "여(%)": g["female_pct"],
            }
        )
    return pd.DataFrame(rows)


def school_enrollment_gender_for_year(
    school_code: str,
    branch: str,
    year: int,
    school_name: str = "",
    major_category: str | None = None,
    *,
    scope_id: str = "전체",
    school_name_prefix: str = "",
) -> dict[str, float]:
    df = load_year_raw(year, school_name_prefix=school_name_prefix)
    subset = filter_school(
        df, school_code, branch, school_name, major_category=major_category
    )
    return summarize_enrollment_gender(subset, scope_id)


def build_enrollment_gender_yearly_series(
    school_code: str,
    branch: str,
    school_name: str = "",
    major_category: str | None = None,
    *,
    scope_id: str = "전체",
    school_name_prefix: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in list_year_data_paths():
        year = year_from_path(path)
        g = school_enrollment_gender_for_year(
            school_code,
            branch,
            year,
            school_name,
            major_category,
            scope_id=scope_id,
            school_name_prefix=school_name_prefix,
        )
        rows.append(
            {
                "연도": year,
                "입학자(계)": g["total"],
                "남": g["male"],
                "여": g["female"],
                "남(%)": g["male_pct"],
                "여(%)": g["female_pct"],
            }
        )
    return pd.DataFrame(rows).sort_values("연도")


def summarize_student_status(df: pd.DataFrame) -> dict[str, float]:
    """학과 행 집합 → 학생 현황 지표 합계."""
    cols = [d["column"] for d in STUDENT_STATUS_DEFINITIONS]
    sums = _sum_numeric_columns(df, cols)
    return {d["key"]: sums.get(d["column"], float("nan")) for d in STUDENT_STATUS_DEFINITIONS}


def student_status_by_group(
    df: pd.DataFrame,
    group_column: str,
    *,
    label_column: str | None = None,
) -> pd.DataFrame:
    """행=group_column, 열=학생 현황 지표."""
    label_col = label_column or group_column
    if group_column not in df.columns:
        return pd.DataFrame()

    groups = list_drill_options(df, group_column, {})
    rows: list[dict[str, object]] = []
    for group in groups:
        sub = df[df[group_column].astype(str).str.strip() == group]
        status = summarize_student_status(sub)
        row: dict[str, object] = {label_col: group}
        for d in STUDENT_STATUS_DEFINITIONS:
            row[d["label"]] = status.get(d["key"], float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def school_student_status_for_year(
    school_code: str,
    branch: str,
    year: int,
    school_name: str = "",
    major_category: str | None = None,
    *,
    school_name_prefix: str = "",
) -> dict[str, float]:
    df = load_year_raw(year, school_name_prefix=school_name_prefix)
    subset = filter_school(
        df, school_code, branch, school_name, major_category=major_category
    )
    return summarize_student_status(subset)


def build_student_status_yearly_series(
    school_code: str,
    branch: str,
    school_name: str = "",
    major_category: str | None = None,
    *,
    school_name_prefix: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in list_year_data_paths():
        year = year_from_path(path)
        status = school_student_status_for_year(
            school_code,
            branch,
            year,
            school_name,
            major_category,
            school_name_prefix=school_name_prefix,
        )
        row: dict[str, Any] = {"연도": year}
        for d in STUDENT_STATUS_DEFINITIONS:
            row[d["label"]] = status.get(d["key"], float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("연도")


def format_status_value(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:,.0f}"


def faculty_gender_scope(scope_id: str) -> dict[str, str]:
    for s in FACULTY_GENDER_SCOPES:
        if s["id"] == scope_id:
            return s
    raise KeyError(f"unknown faculty gender scope: {scope_id}")


def _faculty_gender_totals(total: float, male: float, female: float) -> dict[str, float]:
    """남·여·계 → 비율 dict."""
    if total != total or total <= 0:
        total = male + female if male == male and female == female else float("nan")
    denom = total if total == total and total > 0 else float("nan")
    if denom != denom:
        male_pct = female_pct = float("nan")
    else:
        male_pct = male / denom * 100 if male == male else float("nan")
        female_pct = female / denom * 100 if female == female else float("nan")
    return {
        "total": total,
        "male": male,
        "female": female,
        "male_pct": male_pct,
        "female_pct": female_pct,
    }


def summarize_faculty_gender_combined(df: pd.DataFrame) -> dict[str, float]:
    """전임+비전임 교원 합산."""
    all_g = summarize_faculty_all(df)
    male = female = total = 0.0
    for s in FACULTY_GENDER_SCOPES:
        g = all_g[s["id"]]
        if g["male"] == g["male"]:
            male += g["male"]
        if g["female"] == g["female"]:
            female += g["female"]
        if g["total"] == g["total"]:
            total += g["total"]
    return _faculty_gender_totals(total, male, female)


def summarize_faculty_gender(df: pd.DataFrame, scope_id: str = "전임") -> dict[str, float]:
    """학과 행 집합 → 교원 남·여·계·비율."""
    if scope_id == FACULTY_SCOPE_ALL:
        return summarize_faculty_gender_combined(df)
    scope = faculty_gender_scope(scope_id)
    sums = _sum_numeric_columns(df, [scope["total"], scope["male"], scope["female"]])
    return _faculty_gender_totals(
        sums[scope["total"]], sums[scope["male"]], sums[scope["female"]]
    )


def summarize_faculty_all(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {s["id"]: summarize_faculty_gender(df, s["id"]) for s in FACULTY_GENDER_SCOPES}


def faculty_gender_by_group(
    df: pd.DataFrame,
    group_column: str,
    scope_id: str = "전임",
    *,
    label_column: str | None = None,
) -> pd.DataFrame:
    label_col = label_column or group_column
    if group_column not in df.columns:
        return pd.DataFrame()

    groups = list_drill_options(df, group_column, {})
    rows: list[dict[str, object]] = []
    for group in groups:
        sub = df[df[group_column].astype(str).str.strip() == group]
        g = summarize_faculty_gender(sub, scope_id)
        rows.append(
            {
                label_col: group,
                "교원(계)": g["total"],
                "남": g["male"],
                "여": g["female"],
                "남(%)": g["male_pct"],
                "여(%)": g["female_pct"],
            }
        )
    return pd.DataFrame(rows)


def school_faculty_gender_for_year(
    school_code: str,
    branch: str,
    year: int,
    school_name: str = "",
    major_category: str | None = None,
    *,
    scope_id: str = "전임",
    school_name_prefix: str = "",
) -> dict[str, float]:
    df = load_year_raw(year, school_name_prefix=school_name_prefix)
    subset = filter_school(
        df, school_code, branch, school_name, major_category=major_category
    )
    return summarize_faculty_gender(subset, scope_id)


def school_faculty_all_for_year(
    school_code: str,
    branch: str,
    year: int,
    school_name: str = "",
    major_category: str | None = None,
    *,
    school_name_prefix: str = "",
) -> dict[str, dict[str, float]]:
    df = load_year_raw(year, school_name_prefix=school_name_prefix)
    subset = filter_school(
        df, school_code, branch, school_name, major_category=major_category
    )
    return summarize_faculty_all(subset)


def faculty_scope_count_label(scope_id: str) -> str:
    if scope_id == FACULTY_SCOPE_ALL:
        return "교원(전체)(계)"
    return f"{faculty_gender_scope(scope_id)['label']}(계)"


def build_faculty_gender_yearly_series(
    school_code: str,
    branch: str,
    school_name: str = "",
    major_category: str | None = None,
    *,
    scope_id: str = "전임",
    school_name_prefix: str = "",
) -> pd.DataFrame:
    count_label = faculty_scope_count_label(scope_id)
    rows: list[dict[str, Any]] = []
    for path in list_year_data_paths():
        year = year_from_path(path)
        g = school_faculty_gender_for_year(
            school_code,
            branch,
            year,
            school_name,
            major_category,
            scope_id=scope_id,
            school_name_prefix=school_name_prefix,
        )
        rows.append(
            {
                "연도": year,
                count_label: g["total"],
                "남": g["male"],
                "여": g["female"],
                "남(%)": g["male_pct"],
                "여(%)": g["female_pct"],
            }
        )
    return pd.DataFrame(rows).sort_values("연도")


def build_faculty_total_yearly_series(
    school_code: str,
    branch: str,
    school_name: str = "",
    major_category: str | None = None,
    *,
    school_name_prefix: str = "",
) -> pd.DataFrame:
    """전임·비전임 교원(계) 연도별 추이."""
    rows: list[dict[str, Any]] = []
    for path in list_year_data_paths():
        year = year_from_path(path)
        all_g = school_faculty_all_for_year(
            school_code,
            branch,
            year,
            school_name,
            major_category,
            school_name_prefix=school_name_prefix,
        )
        row: dict[str, Any] = {"연도": year}
        for s in FACULTY_GENDER_SCOPES:
            g = all_g[s["id"]]
            row[s["label"]] = g["total"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("연도")


def apply_hierarchy_filters(
    df: pd.DataFrame, filters: dict[str, str]
) -> pd.DataFrame:
    out = df
    for col in DRILL_COLUMNS:
        val = str(filters.get(col, "")).strip()
        if val and val != DRILL_ALL_LABEL and col in out.columns:
            out = out[out[col].astype(str).str.strip() == val]
    return out


def list_drill_options(
    df: pd.DataFrame,
    column: str,
    filters: dict[str, str],
) -> list[str]:
    if column not in df.columns:
        return []
    sub = apply_hierarchy_filters(df, filters)
    return _unique_non_empty_labels(sub[column])


def get_school_department_frame(
    school_code: str,
    branch: str,
    school_name: str,
    year: int,
    *,
    school_name_prefix: str = "",
) -> pd.DataFrame:
    df = load_year_raw(year, school_name_prefix=school_name_prefix)
    return filter_school(df, school_code, branch, school_name)


def resolve_drill_group_column(filters: dict[str, str]) -> tuple[str, str] | None:
    """
    현재 필터 기준으로 표시할 하위 그룹 컬럼 반환.
    (group_column, 표시용 라벨 컬럼명) / 학과까지 선택 시 None.
    """
    if filters.get("학과명") and filters["학과명"] != DRILL_ALL_LABEL:
        return None
    if filters.get("소계열") and filters["소계열"] != DRILL_ALL_LABEL:
        return ("학과명", "학과명")
    if filters.get("중계열") and filters["중계열"] != DRILL_ALL_LABEL:
        return ("소계열", "소계열")
    if filters.get("대계열") and filters["대계열"] != DRILL_ALL_LABEL:
        return ("중계열", "중계열")
    return ("대계열", "대계열")


def build_group_kpi_table(
    df: pd.DataFrame,
    group_column: str,
    *,
    label_column: str | None = None,
) -> pd.DataFrame:
    """행=group_column 고유값, 열=KPI."""
    label_col = label_column or group_column
    if group_column not in df.columns:
        return pd.DataFrame()

    col_labels: list[str] = []
    key_by_col: dict[str, str] = {}
    for d in KPI_DEFINITIONS:
        unit = d["unit"]
        col = f"{d['label']} ({unit})" if unit else d["label"]
        col_labels.append(col)
        key_by_col[col] = d["key"]

    groups = list_drill_options(df, group_column, {})
    rows: list[dict[str, str]] = []
    for group in groups:
        sub = df[df[group_column].astype(str).str.strip() == group]
        kpis = _sum_school(sub)
        row: dict[str, str] = {label_col: group}
        for col, key in key_by_col.items():
            row[col] = format_kpi_value(key, kpis.get(key, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def load_year_raw(year: int, *, school_name_prefix: str = "") -> pd.DataFrame:
    path = OUTPUT_DIR / f"{year}_data.csv"
    if not path.is_file():
        raise FileNotFoundError(f"데이터 없음: {path}")
    want = set(
        SCHOOL_KEYS
        + SUM_COLUMNS
        + ENROLLMENT_GENDER_COLUMNS
        + STUDENT_STATUS_COLUMNS
        + FACULTY_GENDER_COLUMNS
        + EXTRA_DIM_COLUMNS
        + ["연도"]
    )
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    cols = [c for c in df.columns if c in want]
    df = df[cols].copy()
    if "연도" not in df.columns:
        df["연도"] = year
    if school_name_prefix:
        df = filter_by_school_prefix(df, school_name_prefix)
    return df


def filter_by_school_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not prefix:
        return df
    p = str(prefix).strip()
    return df[df["학교명"].astype(str).str.startswith(p)].copy()


def load_school_options(
    year: int = LATEST_YEAR,
    *,
    school_name_prefix: str = "",
) -> pd.DataFrame:
    """대학 선택 목록 (학교코드·학교명·본분교)."""
    df = load_year_raw(year, school_name_prefix=school_name_prefix)
    opts = df[SCHOOL_KEYS].drop_duplicates().sort_values(["학교명", "본분교"])
    opts = opts[opts["학교명"].astype(str).str.strip() != ""]
    return opts.reset_index(drop=True)


def list_major_categories(
    school_code: str,
    branch: str,
    year: int,
    school_name: str = "",
    *,
    school_name_prefix: str = "",
) -> list[str]:
    """선택 학교·연도의 대계열 목록(비어 있지 않은 값)."""
    df = load_year_raw(year, school_name_prefix=school_name_prefix)
    subset = filter_school(df, school_code, branch, school_name)
    if MAJOR_COLUMN not in subset.columns:
        return []
    return _unique_non_empty_labels(subset[MAJOR_COLUMN])


def filter_school(
    df: pd.DataFrame,
    school_code: str,
    branch: str,
    school_name: str = "",
    major_category: str | None = None,
) -> pd.DataFrame:
    """학교코드 + 정규화된 본분교로 매칭(연도 간 표기 차이 흡수)."""
    code = str(school_code).strip()
    branch_n = normalize_branch(branch)
    branch_series = df["본분교"].astype(str).map(normalize_branch)

    mask = pd.Series(False, index=df.index)
    if code and code.lower() not in ("nan", "none", ""):
        mask = (df["학교코드"].astype(str).str.strip() == code) & (
            branch_series == branch_n
        )
    if not mask.any() and school_name:
        name = str(school_name).strip()
        mask = (df["학교명"].astype(str).str.strip() == name) & (
            branch_series == branch_n
        )
    out = df.loc[mask]
    if major_category and str(major_category).strip() not in ("", "전체"):
        if MAJOR_COLUMN in out.columns:
            major = str(major_category).strip()
            out = out[out[MAJOR_COLUMN].astype(str).str.strip() == major]
    return out


def school_kpis_for_year(
    school_code: str,
    branch: str,
    year: int,
    school_name: str = "",
    major_category: str | None = None,
    *,
    cached_raw: dict[int, pd.DataFrame] | None = None,
    school_name_prefix: str = "",
) -> dict[str, float]:
    if cached_raw and year in cached_raw:
        df = cached_raw[year]
    else:
        df = load_year_raw(year, school_name_prefix=school_name_prefix)
    subset = filter_school(
        df, school_code, branch, school_name, major_category=major_category
    )
    return _sum_school(subset)


def build_yearly_series(
    school_code: str,
    branch: str,
    kpi_key: str,
    school_name: str = "",
    major_category: str | None = None,
    *,
    school_name_prefix: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in list_year_data_paths():
        year = year_from_path(path)
        kpis = school_kpis_for_year(
            school_code,
            branch,
            year,
            school_name,
            major_category,
            school_name_prefix=school_name_prefix,
        )
        val = kpis.get(kpi_key, float("nan"))
        rows.append({"연도": year, "값": val})
    return pd.DataFrame(rows).sort_values("연도")


def format_kpi_value(key: str, value: float) -> str:
    if value != value:  # NaN
        return "—"
    if key in ("신입생_충원율", "입학률"):
        return f"{value:.1f}"
    if key == "경쟁률":
        return f"{value:.2f}"
    return f"{value:,.0f}"


def kpi_delta(current: float, previous: float, key: str) -> float | None:
    if current != current or previous != previous:
        return None
    if key in ("신입생_충원율", "입학률"):
        return current - previous
    if key == "경쟁률":
        return current - previous
    if previous == 0:
        return None
    return (current - previous) / abs(previous) * 100
