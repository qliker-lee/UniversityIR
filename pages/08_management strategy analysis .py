# -*- coding: utf-8 -*-
"""
총장 관점 대학 경영 전략 분석 Streamlit App
- 입학/재학생/교원/외국인/취업/등록금/장학금/학비감면 데이터를 통합 분석
- 다양한 KPI, 통계 기법, 리스크 스코어, 포트폴리오 매트릭스, 상관/회귀/군집 분석 제공

실행:
    streamlit run president_university_analytics_app.py

필요 패키지:
    pip install streamlit pandas numpy plotly scikit-learn
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Executive View of University Management Strategy Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 14px; }
    .main .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(240, 246, 255, 0.70);
        border: 1px solid rgba(100, 140, 200, 0.25);
        padding: 14px 16px;
        border-radius: 16px;
    }
    .small-note { color:#666; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Default file paths: uploaded files are expected in the same folder or /mnt/data
# -----------------------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parent.parent
PAGES_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_PATH / "output"
FEATURE_GUIDE_MD = PAGES_DIR / "08_management strategy analysis.md"
INTEGRATED_RISK_SCORE_MD = PAGES_DIR / "08_University Integrated Risk Score.md"

DEFAULT_FILES = {
    "university": "output/2024_data.csv",
    "job": "output/2024_job.csv",
    "scholarship": "output/한국장학재단_대학별 장학금 수혜 현황_20250831.csv",
    "tuition": "output/한국장학재단_대학별 평균등록금_20250519.csv",
    "reduction": "output/한국장학재단_대학별 학비감면_20250831.csv",
}


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def read_csv_safely(file_or_path) -> pd.DataFrame:
    """Read CSV using common Korean encodings."""
    if file_or_path is None:
        return pd.DataFrame()

    encodings = ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    last_err = None

    # UploadedFile should be re-read from bytes for each encoding.
    if hasattr(file_or_path, "getvalue"):
        raw = file_or_path.getvalue()
        for enc in encodings:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
            except Exception as e:
                last_err = e
        raise last_err

    path = Path(file_or_path)
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise last_err


def _col_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    """DataFrame에서 컬럼 Series 반환. 없으면 행 수에 맞는 NaN Series."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(default, index=df.index, dtype=float)


def find_existing_default(filename: str) -> Optional[Path]:
    basename = Path(filename).name
    candidates = [
        Path(filename),
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
        BASE_PATH / filename,
        OUTPUT_DIR / basename,
        BASE_PATH / "output" / basename,
        Path("/mnt/data") / basename,
    ]
    for p in candidates:
        if p.exists():
            return p

    # Cloud/Linux 등에서 한글 파일명·경로 차이 대비: output 폴더 패턴 검색
    if OUTPUT_DIR.is_dir():
        glob_map = {
            "장학금": "*장학금*.csv",
            "등록금": "*등록금*.csv",
            "학비감면": "*학비감면*.csv",
        }
        for token, pattern in glob_map.items():
            if token in basename:
                hits = sorted(OUTPUT_DIR.glob(pattern))
                if hits:
                    return hits[-1]
        if basename.endswith("_data.csv"):
            year_match = re.search(r"(20\d{2})_data\.csv$", basename)
            if year_match:
                hits = sorted(OUTPUT_DIR.glob(f"{year_match.group(1)}_data.csv"))
                if hits:
                    return hits[0]
        if basename.endswith("_job.csv"):
            year_match = re.search(r"(20\d{2})_job\.csv$", basename)
            if year_match:
                hits = sorted(OUTPUT_DIR.glob(f"{year_match.group(1)}_job.csv"))
                if hits:
                    return hits[0]
    return None


def resolve_default_data_paths(paths: Dict[str, str]) -> tuple[Dict[str, Optional[Path]], list[str]]:
    """기본 CSV 경로 해석. 누락 파일 목록을 함께 반환."""
    resolved: Dict[str, Optional[Path]] = {}
    missing: list[str] = []
    for key, filename in paths.items():
        p = find_existing_default(filename)
        resolved[key] = p
        if p is None:
            missing.append(filename)
    return resolved, missing


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce",
    )


def safe_div(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return np.where(den.replace(0, np.nan).notna(), num / den.replace(0, np.nan), np.nan)


def pct(x):
    if pd.isna(x):
        return "-"
    return f"{x:.1%}"


def num_fmt(x, suffix=""):
    if pd.isna(x):
        return "-"
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:,.1f}B{suffix}"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:,.1f}M{suffix}"
    return f"{x:,.0f}{suffix}"


def won_fmt(x):
    if pd.isna(x):
        return "-"
    if abs(x) >= 100_000_000:
        return f"{x/100_000_000:,.1f}억"
    if abs(x) >= 10_000:
        return f"{x/10_000:,.0f}만"
    return f"{x:,.0f}원"


def normalize_school_name(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
        .str.replace("대학교", "대", regex=False)
        .str.replace("대학", "대", regex=False)
        .str.replace("(본교)", "", regex=False)
    )


def weighted_mean(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    v = pd.to_numeric(df[value_col], errors="coerce")
    w = pd.to_numeric(df[weight_col], errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(v[mask], weights=w[mask])


def add_percentile(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    rank = s.rank(pct=True)
    if not higher_is_better:
        rank = 1 - rank
    return rank.fillna(0.5)


def _normalize_dimension_columns(df: pd.DataFrame) -> pd.DataFrame:
    """지역/설립 컬럼명을 통합 스키마(시도, 설립)로 맞춘다."""
    if df.empty:
        return df
    out = df.copy()
    if "시도" not in out.columns and "지역별" in out.columns:
        out["시도"] = out["지역별"]
    if "설립" not in out.columns and "설립별" in out.columns:
        out["설립"] = out["설립별"]
    return out


def _hover_columns(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _with_safe_size(df: pd.DataFrame, size_col: str | None) -> pd.DataFrame:
    """Plotly scatter size용 NaN/0 방지."""
    if not size_col or size_col not in df.columns:
        return df
    out = df.copy()
    out[size_col] = pd.to_numeric(out[size_col], errors="coerce").fillna(1).clip(lower=1)
    return out


SELECTED_SCHOOL_COLOR = "#FFC107"
SELECTED_SCHOOL_LINE = "#E65100"
PEER_CHART_COLOR = "#B0BEC5"


def _school_selected(selected_school: str) -> bool:
    return bool(selected_school) and selected_school != "전체"


def highlight_school_on_chart(
    fig: go.Figure,
    df: pd.DataFrame,
    selected_school: str,
    *,
    school_col: str = "학교명",
    x: str | None = None,
    y: str | None = None,
) -> go.Figure:
    """비교군 차트에서 선택 대학을 앰버색으로 강조."""
    if not _school_selected(selected_school) or school_col not in df.columns:
        return fig

    sel = df[df[school_col].astype(str).eq(selected_school)]
    if sel.empty:
        return fig

    row = sel.iloc[0]
    if not fig.data:
        return fig

    trace0 = fig.data[0]
    trace_type = getattr(trace0, "type", None)

    if trace_type == "bar" and x:
        for trace in fig.data:
            if trace.type != "bar":
                continue
            colors = [
                SELECTED_SCHOOL_COLOR if str(v) == selected_school else PEER_CHART_COLOR
                for v in trace.x
            ]
            trace.marker.color = colors
            trace.marker.line = dict(color=SELECTED_SCHOOL_LINE, width=1.5)
        return fig

    if trace_type == "histogram" and x and pd.notna(row.get(x)):
        fig.add_vline(
            x=float(row[x]),
            line_color=SELECTED_SCHOOL_COLOR,
            line_width=3,
            annotation_text=selected_school,
            annotation_position="top",
        )
        return fig

    if trace_type in ("scatter", "box"):
        for trace in fig.data:
            if trace.type == "scatter":
                mode = getattr(trace, "mode", "markers") or "markers"
                if mode == "lines":
                    continue
                if trace.marker is None:
                    trace.marker = {}
                trace.marker.update(opacity=0.4)
            elif trace.type == "box":
                trace.opacity = 0.45

    if x and y and pd.notna(row.get(x)) and pd.notna(row.get(y)):
        fig.add_trace(
            go.Scatter(
                x=[row[x]],
                y=[row[y]],
                mode="markers",
                name=f"★ {selected_school}",
                marker=dict(
                    size=20,
                    color=SELECTED_SCHOOL_COLOR,
                    symbol="circle",
                    line=dict(width=2.5, color=SELECTED_SCHOOL_LINE),
                ),
                hovertemplate=(
                    f"<b>{selected_school}</b><br>"
                    f"{x}=%{{x}}<br>{y}=%{{y}}<extra></extra>"
                ),
            )
        )
    return fig


def show_plot(
    fig: go.Figure,
    df: pd.DataFrame,
    selected_school: str,
    *,
    school_col: str = "학교명",
    x: str | None = None,
    y: str | None = None,
) -> None:
    st.plotly_chart(
        highlight_school_on_chart(fig, df, selected_school, school_col=school_col, x=x, y=y),
        use_container_width=True,
    )


@st.cache_data(show_spinner=False)
def _load_pages_markdown(md_path: str) -> str:
    """pages 폴더 내 마크다운 문서 로드."""
    path = Path(md_path)
    if not path.exists():
        return f"문서를 찾을 수 없습니다: `{path}`"
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_from_paths(paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    resolved, missing = resolve_default_data_paths(paths)
    if missing:
        raise FileNotFoundError(
            "필수 CSV 파일을 찾을 수 없습니다. "
            f"output/ 폴더 또는 Git 저장소에 포함했는지 확인하세요.\n"
            + "\n".join(f"  - {name}" for name in missing)
        )
    data: Dict[str, pd.DataFrame] = {}
    for key, path in resolved.items():
        data[key] = read_csv_safely(path) if path else pd.DataFrame()
    return data


def load_from_uploads(uploaded: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    out = {}
    for key, f in uploaded.items():
        out[key] = read_csv_safely(f) if f else pd.DataFrame()
    return out


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------
def aggregate_university_base(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    if "지역별" in work.columns and "시도" not in work.columns:
        work["시도"] = work["지역별"]
    if "KEDI 학교코드" in work.columns and "학교코드" not in work.columns:
        work["학교코드"] = work["KEDI 학교코드"]
    for c in work.columns:
        if c not in ["학교명", "학제", "대학원구분", "학교상태", "본분교", "시도", "시군구", "설립", "주소", "홈페이지", "학위과정", "대계열", "중계열", "소계열", "학과명"]:
            work[c] = pd.to_numeric(work[c], errors="ignore")

    group_cols = [c for c in ["학교명", "학교코드", "시도", "설립", "학제"] if c in work.columns]
    if "학교명" not in group_cols:
        return pd.DataFrame()
    agg_sum_cols = [
        "학과수_전체", "입학정원_학부_계", "모집인원_학부_계", "지원자_전체_계", "입학자_전체_계",
        "정원내_입학자_학부_계", "재적생_전체_계", "재학생_전체_계", "휴학생_전체_계",
        "유예생_전체_계", "외국 학생_총계_계", "졸업자_전체", "전임교원_계", "비전임교원_계"
    ]
    agg_sum_cols = [c for c in agg_sum_cols if c in work.columns]

    base = work.groupby(group_cols, dropna=False)[agg_sum_cols].sum(min_count=1).reset_index()
    base["학교명_key"] = normalize_school_name(base["학교명"])

    # Derived metrics
    base["지원경쟁률"] = base.get("지원자_전체_계", np.nan) / base.get("모집인원_학부_계", np.nan).replace(0, np.nan)
    base["충원율"] = base.get("입학자_전체_계", np.nan) / base.get("모집인원_학부_계", np.nan).replace(0, np.nan)
    base["재학생비율"] = base.get("재학생_전체_계", np.nan) / base.get("재적생_전체_계", np.nan).replace(0, np.nan)
    base["휴학생비율"] = base.get("휴학생_전체_계", np.nan) / base.get("재적생_전체_계", np.nan).replace(0, np.nan)
    base["외국인학생비율"] = base.get("외국 학생_총계_계", np.nan) / base.get("재적생_전체_계", np.nan).replace(0, np.nan)
    base["전임교원1인당재학생"] = base.get("재학생_전체_계", np.nan) / base.get("전임교원_계", np.nan).replace(0, np.nan)
    return base


def aggregate_job(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    for c in work.columns:
        if c not in ["학교명", "학제", "학교상태", "본분교", "시도", "설립", "과정구분", "대계열", "중계열", "소계열", "학과명", "학위구분", "조사기준일"]:
            work[c] = pd.to_numeric(work[c], errors="ignore")

    group_cols = ["학교명", "시도", "설립", "학제"]
    sum_cols = [
        "졸업자_계", "취업자_합계_계", "진학자_계", "1차 유지취업자_계", "2차 유지취업자_계",
        "3차 유지취업자_계", "4차 유지취업자_계", "외국인유학생_계"
    ]
    sum_cols = [c for c in sum_cols if c in work.columns]
    job = work.groupby(group_cols, dropna=False)[sum_cols].sum(min_count=1).reset_index()

    # Weighted employment rates by graduates.
    rate_cols = [c for c in ["취업률_계", "진학률_계", "1차 유지취업률_계", "2차 유지취업률_계", "3차 유지취업률_계", "4차 유지취업률_계"] if c in work.columns]
    for rc in rate_cols:
        grouped = work.groupby(group_cols, dropna=False)
        try:
            wm = grouped.apply(
                lambda x, r=rc: weighted_mean(x, r, "졸업자_계"),
                include_groups=False,
            ).rename(rc)
        except TypeError:
            wm = grouped.apply(
                lambda x, r=rc: weighted_mean(x, r, "졸업자_계"),
            ).rename(rc)
        job = job.merge(wm.reset_index(), on=group_cols, how="left")

    job["학교명_key"] = normalize_school_name(job["학교명"])
    return job


def aggregate_scholarship(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    numeric = [c for c in work.columns if c not in ["학교명", "설립별", "지역별"]]
    for c in numeric:
        work[c] = to_num(work[c])
    group_cols = ["학교명", "설립별", "지역별"]
    sum_cols = [c for c in work.columns if c not in group_cols and c not in ["조사연도", "기준연도"]]
    sch = work.groupby(group_cols, dropna=False)[sum_cols].sum(min_count=1).reset_index()
    sch["학교명_key"] = normalize_school_name(sch["학교명"])
    sch = sch.rename(columns={"총계(원)": "장학금총액", "교외장학금 소계(원)": "교외장학금", "교내장학금 소계(원)": "교내장학금"})
    return sch


def aggregate_tuition(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    for c in ["입학정원", "평균입학금", "평균등록금(원)"]:
        if c in work:
            work[c] = to_num(work[c])
    group_cols = ["대학명", "학제별", "설립별", "지역별"]
    tuition = work.groupby(group_cols, dropna=False).agg(
        입학정원=("입학정원", "sum"),
        평균입학금=("평균입학금", "mean"),
        평균등록금=("평균등록금(원)", "mean"),
    ).reset_index()
    tuition["학교명_key"] = normalize_school_name(tuition["대학명"])
    return tuition


def aggregate_reduction(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    for c in work.columns:
        if c not in ["학교명", "학종"]:
            work[c] = to_num(work[c])
    if "학종" in work.columns:
        # Prefer total rows when available.
        total = work[work["학종"].astype(str).str.strip().eq("계")].copy()
        if not total.empty:
            work = total

    group_cols = ["학교명"]
    cols = [c for c in work.columns if c not in group_cols and c not in ["기준연도", "학종"]]
    red = work.groupby(group_cols, dropna=False)[cols].sum(min_count=1).reset_index()
    red["학교명_key"] = normalize_school_name(red["학교명"])
    red = red.rename(columns={
        "등록금 수입": "등록금수입",
        "10퍼센트 규정 준수여부 비율": "학비감면10비율",
        "10퍼센트 규정 준수여부 금액": "학비감면10금액",
        "30퍼센트 규정 준수여부 비율": "학비감면30비율",
        "30퍼센트 규정 준수여부 금액": "학비감면30금액",
    })
    return red


def build_integrated(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    base = aggregate_university_base(data.get("university", pd.DataFrame()))
    job = aggregate_job(data.get("job", pd.DataFrame()))
    sch = aggregate_scholarship(data.get("scholarship", pd.DataFrame()))
    tuition = aggregate_tuition(data.get("tuition", pd.DataFrame()))
    red = aggregate_reduction(data.get("reduction", pd.DataFrame()))

    if base.empty:
        return pd.DataFrame(), {
            "base": base,
            "job": job,
            "scholarship": sch,
            "tuition": tuition,
            "reduction": red,
        }

    merged = base.copy()
    if not job.empty:
        job_cols = ["학교명_key"] + [c for c in job.columns if c not in ["학교명", "시도", "설립", "학제", "학교명_key"]]
        merged = merged.merge(job[job_cols], on="학교명_key", how="left", suffixes=("", "_job"))
    if not sch.empty:
        sch_cols = ["학교명_key"] + [c for c in sch.columns if c not in ["학교명", "설립별", "지역별", "학교명_key"]]
        merged = merged.merge(sch[sch_cols], on="학교명_key", how="left")
    if not tuition.empty:
        tui_cols = ["학교명_key"] + [c for c in tuition.columns if c not in ["대학명", "학제별", "설립별", "지역별", "학교명_key"]]
        merged = merged.merge(tuition[tui_cols], on="학교명_key", how="left")
    if not red.empty:
        red_cols = ["학교명_key"] + [c for c in red.columns if c not in ["학교명", "학교명_key"]]
        merged = merged.merge(red[red_cols], on="학교명_key", how="left")

    # Financial/student derived metrics
    enrolled = _col_series(merged, "재적생_전체_계").replace(0, np.nan)
    merged["학생1인당장학금"] = _col_series(merged, "장학금총액") / enrolled
    merged["장학금_등록금대비"] = _col_series(merged, "장학금총액") / _col_series(merged, "등록금수입").replace(0, np.nan)
    merged["등록금_재학생가중부담"] = _col_series(merged, "평균등록금") * _col_series(merged, "재학생_전체_계")

    # Risk/opportunity scores
    merged["입학리스크"] = 1 - add_percentile(merged, "지원경쟁률", True) * 0.45 - add_percentile(merged, "충원율", True) * 0.35 - add_percentile(merged, "재학생비율", True) * 0.20
    merged["재정리스크"] = add_percentile(merged, "평균등록금", False) * 0.25 + add_percentile(merged, "장학금_등록금대비", False) * 0.35 + add_percentile(merged, "재학생_전체_계", True) * 0.40
    merged["성과지수"] = (
        add_percentile(merged, "취업률_계", True) * 0.35 +
        add_percentile(merged, "4차 유지취업률_계", True) * 0.20 +
        add_percentile(merged, "지원경쟁률", True) * 0.20 +
        add_percentile(merged, "외국인학생비율", True) * 0.10 +
        add_percentile(merged, "재학생비율", True) * 0.15
    )
    merged["종합위험도"] = (
        add_percentile(merged, "입학리스크", False) * 0.40 +
        add_percentile(merged, "휴학생비율", False) * 0.20 +
        add_percentile(merged, "취업률_계", True) * 0.25 +
        add_percentile(merged, "장학금_등록금대비", True) * 0.15
    )
    merged["전략분류"] = pd.cut(
        merged["성과지수"],
        bins=[-0.01, 0.35, 0.60, 1.01],
        labels=["구조개선 필요", "선택적 투자", "성장/확산"],
    )

    if "시도" not in merged.columns and not job.empty and "시도" in job.columns:
        region_map = job[["학교명_key", "시도"]].drop_duplicates("학교명_key")
        merged = merged.merge(region_map, on="학교명_key", how="left")

    merged = _normalize_dimension_columns(merged)
    return merged, {"base": base, "job": job, "scholarship": sch, "tuition": tuition, "reduction": red}


# -----------------------------------------------------------------------------
# Visual helpers
# -----------------------------------------------------------------------------
def show_kpis(df: pd.DataFrame, selected_school: Optional[str]):
    if selected_school and selected_school != "전체":
        row = df[df["학교명"].eq(selected_school)].iloc[0]
        kpis = [
            ("재학생", num_fmt(row.get("재학생_전체_계"))),
            ("지원경쟁률", f"{row.get('지원경쟁률', np.nan):.2f}:1" if pd.notna(row.get("지원경쟁률")) else "-"),
            ("충원율", pct(row.get("충원율"))),
            ("취업률", f"{row.get('취업률_계', np.nan):.1f}%" if pd.notna(row.get("취업률_계")) else "-"),
            ("평균등록금", won_fmt(row.get("평균등록금"))),
            ("학생1인당 장학금", won_fmt(row.get("학생1인당장학금"))),
            ("외국인학생비율", pct(row.get("외국인학생비율"))),
            ("종합위험도", f"{row.get('종합위험도', np.nan):.2f}" if pd.notna(row.get("종합위험도")) else "-"),
        ]
    else:
        kpis = [
            ("분석 대학 수", f"{df['학교명'].nunique():,}"),
            ("총 재학생", num_fmt(df["재학생_전체_계"].sum())),
            ("평균 지원경쟁률", f"{df['지원경쟁률'].mean():.2f}:1"),
            ("평균 충원율", pct(df["충원율"].mean())),
            ("평균 취업률", f"{df['취업률_계'].mean():.1f}%"),
            ("평균 등록금", won_fmt(df["평균등록금"].mean())),
            ("총 장학금", won_fmt(df["장학금총액"].sum())),
            ("고위험 대학 수", f"{(df['종합위험도'] < 0.35).sum():,}"),
        ]

    cols = st.columns(4)
    for i, (label, value) in enumerate(kpis):
        cols[i % 4].metric(label, value)


ACTION_ITEM_METRICS: tuple[tuple[str, str], ...] = (
    ("지원경쟁률", "지원경쟁률"),
    ("휴학생비율", "휴학생 비율"),
    ("취업률_계", "취업률"),
    ("외국인학생비율", "외국인 학생 비율"),
)


def _format_action_metric_value(col: str, value: float) -> str:
    if pd.isna(value):
        return "-"
    if col in ("휴학생비율", "외국인학생비율", "충원율", "재학생비율"):
        return pct(value)
    if col == "취업률_계":
        return f"{value:.1f}%"
    return f"{value:.2f}"


def _format_action_metric_delta(col: str, value: float, median: float) -> str | None:
    if pd.isna(value) or pd.isna(median):
        return None
    diff = value - median
    if col in ("휴학생비율", "외국인학생비율", "충원율", "재학생비율"):
        return f"{diff * 100:+.1f}%p (중앙 대비)"
    if col == "취업률_계":
        return f"{diff:+.1f}%p (중앙 대비)"
    return f"{diff:+.2f} (중앙 대비)"


def _action_metric_delta_color(col: str) -> str:
    if col in ("휴학생비율", "입학리스크", "재정리스크", "종합위험도"):
        return "inverse"
    return "normal"


def bar_top(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    n: int = 15,
    ascending: bool = False,
    selected_school: str = "전체",
):
    ranked = df.dropna(subset=[y]).sort_values(y, ascending=ascending)
    plot_df = ranked.head(n)
    if (
        _school_selected(selected_school)
        and x == "학교명"
        and "학교명" in ranked.columns
        and selected_school not in plot_df["학교명"].astype(str).values
    ):
        sel_row = ranked[ranked["학교명"].astype(str).eq(selected_school)]
        if not sel_row.empty:
            plot_df = pd.concat([sel_row, plot_df.head(n - 1)], ignore_index=True)

    fig = px.bar(
        plot_df,
        x=x,
        y=y,
        text=y,
        title=title,
        hover_data=_hover_columns(plot_df, ["학교명", "시도", "설립"]),
    )
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig.update_layout(height=450, xaxis_tickangle=-35)
    show_plot(fig, plot_df, selected_school, x=x, y=y)


def correlation_heatmap(df: pd.DataFrame, cols: list[str]):
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="핵심 지표 상관관계 Heatmap")
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)


def run_regression(df: pd.DataFrame, target: str, features: list[str]):
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn이 설치되어 있지 않아 회귀 분석을 건너뜁니다. `pip install scikit-learn` 후 사용하세요.")
        return
    model_df = df[[target] + features].apply(pd.to_numeric, errors="coerce").dropna()
    if len(model_df) < max(10, len(features) + 3):
        st.info("회귀 분석에 충분한 데이터가 없습니다.")
        return
    X = model_df[features]
    y = model_df[target]
    lr = LinearRegression().fit(X, y)
    r2 = lr.score(X, y)
    coef = pd.DataFrame({"변수": features, "계수": lr.coef_}).sort_values("계수", key=np.abs, ascending=False)

    c1, c2 = st.columns([1, 2])
    c1.metric("회귀모형 설명력 R²", f"{r2:.3f}")
    fig = px.bar(coef, x="변수", y="계수", title=f"{target} 영향요인 회귀계수")
    c2.plotly_chart(fig, use_container_width=True)
    st.dataframe(coef, use_container_width=True, hide_index=True)


def run_clustering(df: pd.DataFrame, features: list[str], k: int):
    if not SKLEARN_AVAILABLE:
        st.warning("scikit-learn이 설치되어 있지 않아 군집 분석을 건너뜁니다. `pip install scikit-learn` 후 사용하세요.")
        return df
    cdf = df.copy()
    model_df = cdf[features].apply(pd.to_numeric, errors="coerce")
    valid = model_df.dropna()
    if len(valid) < k:
        st.info("군집 분석에 충분한 데이터가 없습니다.")
        cdf["Cluster"] = np.nan
        return cdf
    scaler = StandardScaler()
    X = scaler.fit_transform(valid)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    cdf["Cluster"] = np.nan
    cdf.loc[valid.index, "Cluster"] = labels.astype(str)
    return cdf


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.title("🎓 University Management Strategy Analysis Platform")
st.caption("입학 경쟁력 · 재학생 안정성 · 취업 성과 · 재정/등록금 · 장학/학비감면 · 국제화 지표를 통합 분석합니다.")

col1, col2 = st.columns(2)
with col1:  
    with st.expander("📖 기능 설명서 (08_management strategy analysis.md)", expanded=False):
        st.markdown(_load_pages_markdown(str(FEATURE_GUIDE_MD)))

with col2:
    with st.expander("📖 University Integrated Risk Score (08_University Integrated Risk Score.md)", expanded=False):
        st.markdown(_load_pages_markdown(str(INTEGRATED_RISK_SCORE_MD)))

with st.sidebar:
    st.header("📁 데이터 입력")
    mode = st.radio("데이터 로딩 방식", ["기본 경로 사용", "CSV 직접 업로드"], index=0)

    uploaded = {}
    if mode == "CSV 직접 업로드":
        uploaded["university"] = st.file_uploader("output/2024_data.csv", type=["csv"])
        uploaded["job"] = st.file_uploader("2024_job.csv", type=["csv"])
        uploaded["scholarship"] = st.file_uploader("대학별 장학금 수혜 현황", type=["csv"])
        uploaded["tuition"] = st.file_uploader("대학별 평균등록금", type=["csv"])
        uploaded["reduction"] = st.file_uploader("대학별 학비감면", type=["csv"])

    st.header("🎛️ 분석 조건")
    min_students = st.number_input("최소 재학생 수", min_value=0, value=0, step=100)
    # selected_regions = st.multiselect("지역", [])
    # selected_types = st.multiselect("설립", [])


integrated = pd.DataFrame()
marts: Dict[str, pd.DataFrame] = {}
data_load_error: str | None = None

try:
    if mode == "기본 경로 사용":
        raw_data = load_all_from_paths(DEFAULT_FILES)
    else:
        raw_data = load_from_uploads(uploaded)

    integrated, marts = build_integrated(raw_data)
    integrated = _normalize_dimension_columns(integrated)
except Exception as e:
    data_load_error = str(e)

if data_load_error:
    st.error(f"데이터 로딩/통합 중 오류가 발생했습니다: {data_load_error}")
    if mode == "기본 경로 사용":
        st.info(
            "Streamlit Cloud 배포 시 **`University_IR/output/`** 폴더의 CSV 5개가 "
            "저장소에 포함되어 있어야 합니다. "
            "또는 사이드바에서 **「CSV 직접 업로드」** 를 선택해 파일을 올려 주세요."
        )
        with st.expander("필요 파일 목록"):
            for key, rel in DEFAULT_FILES.items():
                p = find_existing_default(rel)
                st.write(f"- **{key}**: `{rel}` → {'✅ ' + str(p) if p else '❌ 없음'}")
    st.stop()
    raise SystemExit(1)

if integrated.empty or "학교명" not in integrated.columns:
    st.warning("분석 가능한 대학 기본 데이터가 없습니다. CSV 파일 위치 또는 업로드 상태를 확인하세요.")
    st.stop()
    raise SystemExit(0)

if "시도" not in integrated.columns:
    st.warning(
        "지역(시도) 정보가 없습니다. `2024_data.csv`의 `시도` 컬럼 또는 "
        "장학/등록금 파일의 `지역별` 컬럼을 확인하세요."
    )
    st.stop()
    raise SystemExit(0)

# Dynamic sidebar filters
with st.sidebar:
    regions = (
        sorted(integrated["시도"].dropna().unique().tolist())
        if "시도" in integrated.columns
        else []
    )
    types = (
        sorted(integrated["설립"].dropna().unique().tolist())
        if "설립" in integrated.columns
        else []
    )
    selected_regions = st.multiselect("지역 선택", regions, default=regions)
    selected_types = st.multiselect("설립 유형 선택", types, default=types)

filtered = integrated.copy()
if "재학생_전체_계" in filtered.columns:
    filtered = filtered[filtered["재학생_전체_계"].fillna(0) >= min_students]
if selected_regions and "시도" in filtered.columns:
    filtered = filtered[filtered["시도"].isin(selected_regions)]
if selected_types and "설립" in filtered.columns:
    filtered = filtered[filtered["설립"].isin(selected_types)]

school_options = ["전체"] + sorted(filtered["학교명"].dropna().unique().tolist())
selected_school = st.sidebar.selectbox("대학 선택", school_options, index=0)

view_df = filtered if selected_school == "전체" else filtered[filtered["학교명"].eq(selected_school)]

# -----------------------------------------------------------------------------
# Main tabs
# -----------------------------------------------------------------------------
tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Executive Summary",
    "입학·재학생 경쟁력",
    "취업·성과 분석",
    "재정·등록금·장학",
    "전략 포트폴리오",
    "통계/AI 분석",
    "데이터 품질/원천"
])

with tab0:
    st.subheader("Executive Summary")
    show_kpis(view_df, selected_school)

    st.divider()
    st.markdown("#### 핵심 전략 진단")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.scatter(
            filtered,
            x="지원경쟁률",
            y="취업률_계",
            size="재학생_전체_계",
            color="전략분류",
            hover_name="학교명",
            hover_data=["시도", "설립", "충원율", "휴학생비율", "평균등록금", "학생1인당장학금"],
            title="입학 경쟁력 × 취업 성과 포지셔닝",
        )
        fig.update_layout(height=520)
        show_plot(fig, filtered, selected_school, x="지원경쟁률", y="취업률_계")

    with c2:
        priority = filtered.copy()
        priority["개선우선점수"] = (
            add_percentile(priority, "입학리스크", True) * 0.35
            + add_percentile(priority, "휴학생비율", True) * 0.25
            + add_percentile(priority, "취업률_계", False) * 0.25
            + add_percentile(priority, "장학금_등록금대비", False) * 0.15
        )
        show_cols = ["학교명", "시도", "설립", "개선우선점수", "지원경쟁률", "충원율", "휴학생비율", "취업률_계"]
        st.dataframe(
            priority.sort_values("개선우선점수", ascending=False)[show_cols].head(20),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Executive View Action Item")
    if selected_school != "전체" and not view_df.empty:
        r = view_df.iloc[0]
        metric_cols = st.columns(len(ACTION_ITEM_METRICS))
        for ui_col, (col_name, label) in zip(metric_cols, ACTION_ITEM_METRICS):
            if col_name not in filtered.columns:
                continue
            val = pd.to_numeric(r.get(col_name), errors="coerce")
            med = pd.to_numeric(filtered[col_name], errors="coerce").median()
            with ui_col:
                st.metric(
                    label,
                    _format_action_metric_value(col_name, val),
                    delta=_format_action_metric_delta(col_name, val, med),
                    delta_color=_action_metric_delta_color(col_name),
                    help=f"비교군({len(filtered):,}개교) 중앙값: {_format_action_metric_value(col_name, med)}",
                )

        actions = []
        if pd.notna(r.get("지원경쟁률")) and r["지원경쟁률"] < filtered["지원경쟁률"].median():
            actions.append("입학 경쟁률이 비교군 중앙값보다 낮습니다. 모집단위별 홍보 ROI와 지원-입학 전환율 개선이 필요합니다.")
        if pd.notna(r.get("휴학생비율")) and r["휴학생비율"] > filtered["휴학생비율"].median():
            actions.append("휴학생 비율이 높습니다. 학사경고/경제곤란/진로불안 요인별 중도이탈 예방 프로그램이 필요합니다.")
        if pd.notna(r.get("취업률_계")) and r["취업률_계"] < filtered["취업률_계"].median():
            actions.append("취업률이 비교군 대비 낮습니다. 학과별 취업성과 하위 그룹을 식별하고 산학협력·현장실습 집중 투자가 필요합니다.")
        if pd.notna(r.get("외국인학생비율")) and r["외국인학생비율"] < filtered["외국인학생비율"].median():
            actions.append("국제화 지표가 낮습니다. 외국인 유학생 유치 국가 다변화와 정착 지원 체계 강화가 필요합니다.")
        if not actions:
            actions.append("주요 지표가 비교군 대비 안정적입니다. 강점 학과 중심의 선택과 집중 전략을 검토할 수 있습니다.")
        for a in actions:
            st.success("• " + a)
    else:
        st.info("왼쪽에서 특정 대학을 선택하면 맞춤형 Action Item을 표시합니다.")

with tab1:
    st.subheader("2. 입학·재학생 경쟁력 분석")
    c1, c2 = st.columns(2)
    with c1:
        bar_top(filtered, "학교명", "지원경쟁률", "지원경쟁률 Top 15", ascending=False, selected_school=selected_school)
    with c2:
        bar_top(filtered, "학교명", "충원율", "충원율 Top 15", ascending=False, selected_school=selected_school)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(filtered, x="설립", y="지원경쟁률", points="all", title="설립 유형별 지원경쟁률 분포")
        show_plot(fig, filtered, selected_school, x="설립", y="지원경쟁률")
    with c4:
        plot_df = _with_safe_size(filtered, "재학생_전체_계")
        fig = px.scatter(plot_df, x="재학생비율", y="휴학생비율", color="시도", size="재학생_전체_계", hover_name="학교명", title="재학생 안정성 진단")
        show_plot(fig, plot_df, selected_school, x="재학생비율", y="휴학생비율")

    st.markdown("#### 학과/계열 상세 분석")
    uni_raw = raw_data.get("university", pd.DataFrame())
    if not uni_raw.empty:
        dept = uni_raw.copy()
        if selected_school != "전체":
            dept = dept[dept["학교명"].eq(selected_school)]
        group_level = st.radio("분석 단위", ["대계열", "중계열", "소계열", "학과명"], horizontal=True)
        dept_agg = dept.groupby(group_level, dropna=False).agg(
            모집인원=("모집인원_학부_계", "sum"),
            지원자=("지원자_전체_계", "sum"),
            입학자=("입학자_전체_계", "sum"),
            재학생=("재학생_전체_계", "sum"),
            휴학생=("휴학생_전체_계", "sum"),
            외국인학생=("외국 학생_총계_계", "sum"),
        ).reset_index()
        dept_agg["경쟁률"] = dept_agg["지원자"] / dept_agg["모집인원"].replace(0, np.nan)
        dept_agg["충원율"] = dept_agg["입학자"] / dept_agg["모집인원"].replace(0, np.nan)
        dept_agg["휴학생비율"] = dept_agg["휴학생"] / (dept_agg["재학생"] + dept_agg["휴학생"]).replace(0, np.nan)
        fig = px.scatter(dept_agg, x="경쟁률", y="충원율", size="재학생", color="휴학생비율", hover_name=group_level, title=f"{group_level} 단위 모집/충원 포트폴리오")
        if _school_selected(selected_school):
            fig.update_traces(marker=dict(color=SELECTED_SCHOOL_COLOR, line=dict(width=1.5, color=SELECTED_SCHOOL_LINE)))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dept_agg.sort_values("경쟁률", ascending=False), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("3. 취업·성과 분석")
    c1, c2 = st.columns(2)
    with c1:
        plot_df = _with_safe_size(filtered, "졸업자_계")
        fig = px.scatter(plot_df, x="취업률_계", y="4차 유지취업률_계", size="졸업자_계", color="설립", hover_name="학교명", title="취업률 × 유지취업률")
        show_plot(fig, plot_df, selected_school, x="취업률_계", y="4차 유지취업률_계")
    with c2:
        fig = px.box(filtered, x="시도", y="취업률_계", color="설립", title="지역/설립별 취업률 분포")
        fig.update_layout(xaxis_tickangle=-45)
        show_plot(fig, filtered, selected_school, x="시도", y="취업률_계")

    st.markdown("#### 계열별 취업성과")
    job_raw = raw_data.get("job", pd.DataFrame())
    if not job_raw.empty:
        job = job_raw.copy()
        if selected_school != "전체":
            job = job[job["학교명"].eq(selected_school)]
        job_group = job.groupby("대계열", dropna=False).agg(
            졸업자=("졸업자_계", "sum"),
            취업자=("취업자_합계_계", "sum"),
            진학자=("진학자_계", "sum"),
        ).reset_index()
        job_group["취업률"] = job_group["취업자"] / job_group["졸업자"].replace(0, np.nan) * 100
        job_group["진학률"] = job_group["진학자"] / job_group["졸업자"].replace(0, np.nan) * 100
        fig = px.bar(job_group.sort_values("취업률", ascending=False), x="대계열", y=["취업률", "진학률"], barmode="group", title="계열별 취업률/진학률")
        if _school_selected(selected_school):
            for trace in fig.data:
                if trace.type == "bar":
                    trace.marker.color = SELECTED_SCHOOL_COLOR
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(job_group.sort_values("취업률", ascending=False), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("4. 재정·등록금·장학 분석")
    c1, c2 = st.columns(2)
    fin_plot_df = _with_safe_size(filtered, "재학생_전체_계")
    with c1:
        fig = px.scatter(fin_plot_df, x="평균등록금", y="학생1인당장학금", size="재학생_전체_계", color="설립", hover_name="학교명", title="등록금 부담 × 학생 1인당 장학금")
        show_plot(fig, fin_plot_df, selected_school, x="평균등록금", y="학생1인당장학금")
    with c2:
        fig = px.scatter(fin_plot_df, x="등록금수입", y="장학금총액", size="재학생_전체_계", color="시도", hover_name="학교명", title="등록금 수입 × 장학금 총액")
        show_plot(fig, fin_plot_df, selected_school, x="등록금수입", y="장학금총액")

    c3, c4 = st.columns(2)
    with c3:
        bar_top(filtered, "학교명", "장학금_등록금대비", "장학금/등록금 수입 비율 Top 15", ascending=False, selected_school=selected_school)
    with c4:
        fig = px.histogram(filtered, x="평균등록금", nbins=30, color="설립", title="평균등록금 분포")
        show_plot(fig, filtered, selected_school, x="평균등록금")

    st.markdown("#### 학비감면 규정 준수 모니터링")
    compliance_cols = ["학교명", "등록금수입", "학비감면10비율", "학비감면10금액", "학비감면30비율", "학비감면30금액", "장학금_등록금대비"]
    st.dataframe(filtered[[c for c in compliance_cols if c in filtered.columns]].sort_values("장학금_등록금대비", ascending=False), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("5. 전략 포트폴리오")
    x_metric = st.selectbox("X축", ["지원경쟁률", "충원율", "재학생비율", "취업률_계", "평균등록금", "학생1인당장학금", "외국인학생비율"], index=0)
    y_metric = st.selectbox("Y축", ["취업률_계", "4차 유지취업률_계", "휴학생비율", "장학금_등록금대비", "성과지수", "종합위험도"], index=0)
    size_metric = st.selectbox("Size", ["재학생_전체_계", "졸업자_계", "장학금총액", "등록금수입"], index=0)

    fig = px.scatter(
        filtered,
        x=x_metric,
        y=y_metric,
        size=size_metric,
        color="전략분류",
        hover_name="학교명",
        hover_data=["시도", "설립", "재학생_전체_계", "지원경쟁률", "취업률_계", "휴학생비율"],
        title=f"{x_metric} × {y_metric} 전략 포트폴리오",
    )
    fig.add_vline(x=filtered[x_metric].median(skipna=True), line_dash="dash")
    fig.add_hline(y=filtered[y_metric].median(skipna=True), line_dash="dash")
    fig.update_layout(height=650)
    show_plot(fig, filtered, selected_school, x=x_metric, y=y_metric)

    st.markdown("#### 전략분류별 요약")
    summary = filtered.groupby("전략분류", dropna=False).agg(
        대학수=("학교명", "nunique"),
        평균재학생=("재학생_전체_계", "mean"),
        평균경쟁률=("지원경쟁률", "mean"),
        평균취업률=("취업률_계", "mean"),
        평균등록금=("평균등록금", "mean"),
        평균장학금비율=("장학금_등록금대비", "mean"),
    ).reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("6. 통계/AI 분석")
    numeric_cols = [
        "지원경쟁률", "충원율", "재학생비율", "휴학생비율", "외국인학생비율", "전임교원1인당재학생",
        "취업률_계", "진학률_계", "1차 유지취업률_계", "4차 유지취업률_계",
        "평균등록금", "학생1인당장학금", "장학금_등록금대비", "성과지수", "종합위험도"
    ]
    numeric_cols = [c for c in numeric_cols if c in filtered.columns]

    correlation_heatmap(filtered, numeric_cols)

    st.markdown("#### 다중회귀 분석")
    target = st.selectbox("목표 지표", [c for c in ["취업률_계", "지원경쟁률", "충원율", "성과지수", "종합위험도"] if c in filtered.columns])
    features = st.multiselect(
        "설명 변수",
        [c for c in numeric_cols if c != target],
        default=[c for c in ["지원경쟁률", "충원율", "휴학생비율", "평균등록금", "학생1인당장학금", "외국인학생비율"] if c in numeric_cols and c != target],
    )
    if st.button("회귀 분석 실행"):
        run_regression(filtered, target, features)

    st.markdown("#### 대학 군집 분석")
    cluster_features = st.multiselect(
        "군집 변수",
        numeric_cols,
        default=[c for c in ["지원경쟁률", "충원율", "취업률_계", "평균등록금", "학생1인당장학금", "외국인학생비율"] if c in numeric_cols],
    )
    k = st.slider("군집 수 K", 2, 8, 4)
    clustered = run_clustering(filtered, cluster_features, k)
    if "Cluster" in clustered.columns and clustered["Cluster"].notna().any():
        fig = px.scatter(clustered, x=cluster_features[0], y=cluster_features[1], color="Cluster", size="재학생_전체_계", hover_name="학교명", title="K-Means 군집 분석 결과")
        show_plot(fig, clustered, selected_school, x=cluster_features[0], y=cluster_features[1])
        st.dataframe(clustered[["학교명", "시도", "설립", "Cluster"] + cluster_features].sort_values("Cluster"), use_container_width=True, hide_index=True)

with tab6:
    st.subheader("7. 데이터 품질/원천 확인")
    st.markdown("#### 원천 데이터 적재 현황")
    load_status = []
    for key, df in raw_data.items():
        load_status.append({"데이터": key, "행수": len(df), "컬럼수": len(df.columns), "컬럼 예시": ", ".join(map(str, df.columns[:8]))})
    st.dataframe(pd.DataFrame(load_status), use_container_width=True, hide_index=True)

    st.markdown("#### 통합 데이터 매칭 현황")
    match_cols = ["학교명", "시도", "설립", "재학생_전체_계", "취업률_계", "평균등록금", "장학금총액", "등록금수입"]
    q = st.text_input("학교명 검색")
    detail = integrated.copy()
    if q:
        detail = detail[detail["학교명"].astype(str).str.contains(q, case=False, na=False)]
    st.dataframe(detail[[c for c in match_cols if c in detail.columns]], use_container_width=True, hide_index=True)

    st.download_button(
        "통합 분석 데이터 CSV 다운로드",
        data=integrated.to_csv(index=False, encoding="utf-8-sig"),
        file_name="president_university_integrated_metrics.csv",
        mime="text/csv",
    )
