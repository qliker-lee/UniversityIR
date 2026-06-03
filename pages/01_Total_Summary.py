# -*- coding: utf-8 -*-
"""신입생 유치·입시 KPI 대시보드 (`output/YYYY_data.csv`)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib import admissions_data as adm

st.set_page_config(page_title="입시 분석", layout="wide")
st.title("📊 대학교 IR(Institutional Research) Analysis")
st.write("분석을 하고자 하는 대학교를 선택후 주요 지표를 분석할 수 있습니다. 주요 지표는 신입생 충원율, 입학률, 경쟁률, 중도탈락률, 취업률, 취업자 수 등이 있습니다.")

OUTPUT_DIR = adm.OUTPUT_DIR
LATEST_YEAR = adm.LATEST_YEAR
PREV_YEAR = LATEST_YEAR - 1


@st.cache_data(show_spinner="데이터 목록 로딩…")
def _school_options() -> pd.DataFrame:
    return adm.load_school_options(LATEST_YEAR)


@st.cache_data(show_spinner="연도별 데이터 로딩…")
def _load_year(year: int) -> pd.DataFrame:
    return adm.load_year_raw(year)


@st.cache_data(show_spinner="연도별 KPI 계산…")
def _yearly_series(
    school_code: str, branch: str, kpi_key: str, school_name: str
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    cached_raw = {
        adm.year_from_path(path): _load_year(adm.year_from_path(path))
        for path in adm.list_year_data_paths()
    }
    for year in sorted(cached_raw):
        kpis = adm.school_kpis_for_year(
            school_code,
            branch,
            year,
            school_name,
            cached_raw=cached_raw,
        )
        rows.append({"연도": year, "값": kpis.get(kpi_key, float("nan"))})
    return pd.DataFrame(rows)


def _school_label(row: pd.Series) -> str:
    return f"{row['학교명']} ({row['본분교']})"


if not OUTPUT_DIR.is_dir() or not (OUTPUT_DIR / f"{LATEST_YEAR}_data.csv").is_file():
    st.error(
        f"`{OUTPUT_DIR}` 에 `{LATEST_YEAR}_data.csv` 가 없습니다. "
        "`01_ist_data_integration.py` 실행 후 다시 열어 주세요."
    )
    st.stop()

schools = _school_options()
if schools.empty:
    st.warning("대학 목록을 불러올 수 없습니다.")
    st.stop()

# ── 1. 대학교 선택 ─────────────────────────────────────────────
st.subheader("1. 대학교 선택")
labels = [_school_label(r) for _, r in schools.iterrows()]
label_to_idx = {lb: i for i, lb in enumerate(labels)}

c1, c2 = st.columns([2, 1])
with c1:
    selected_label = st.selectbox(
        "대학교",
        options=labels,
        index=0,
        help="2024년 데이터 기준 학교·캠퍼스(본분교) 목록",
    )
with c2:
    search = st.text_input("학교명 검색", placeholder="예: 중원대")
    if search.strip():
        filtered = [lb for lb in labels if search.strip() in lb]
        if filtered:
            selected_label = st.selectbox(
                "검색 결과",
                options=filtered,
                key="school_search_pick",
            )
        else:
            st.caption("검색 결과 없음")

row = schools.iloc[label_to_idx[selected_label]]
school_code = str(row["학교코드"]).strip()
branch = str(row["본분교"]).strip()
school_name = str(row["학교명"]).strip()

st.caption(f"학교코드: `{school_code}` · 본분교: `{branch}`")

# ── 2. 최신 KPI + 전년 대비 ───────────────────────────────────
st.subheader(f"2. 주요 KPI ({LATEST_YEAR}년) 및 전년({PREV_YEAR}년) 대비")

year_dfs = {LATEST_YEAR: _load_year(LATEST_YEAR), PREV_YEAR: _load_year(PREV_YEAR)}

kpis_latest = adm.school_kpis_for_year(
    school_code, branch, LATEST_YEAR, school_name, cached_raw=year_dfs
)
kpis_prev = adm.school_kpis_for_year(
    school_code, branch, PREV_YEAR, school_name, cached_raw=year_dfs
)

kpi_rows: list[dict] = []
for defn in adm.KPI_DEFINITIONS:
    key = defn["key"]
    cur = kpis_latest.get(key, float("nan"))
    prev = kpis_prev.get(key, float("nan"))
    delta = adm.kpi_delta(cur, prev, key)
    kpi_rows.append(
        {
            "KPI": defn["label"],
            "kpi_key": key,
            f"{LATEST_YEAR}": adm.format_kpi_value(key, cur),
            f"{PREV_YEAR}": adm.format_kpi_value(key, prev),
            "단위": defn["unit"],
            "전년대비": delta,
        }
    )

kpi_df = pd.DataFrame(kpi_rows)

metric_cols = st.columns(4)
for i, defn in enumerate(adm.KPI_DEFINITIONS[:4]):
    key = defn["key"]
    cur = kpis_latest.get(key, float("nan"))
    prev = kpis_prev.get(key, float("nan"))
    delta = adm.kpi_delta(cur, prev, key)
    val_str = adm.format_kpi_value(key, cur)
    unit = defn["unit"]
    display_val = f"{val_str}{unit}" if unit == "%" else f"{val_str} {unit}".strip()
    with metric_cols[i % 4]:
        if delta is None:
            st.metric(defn["label"], display_val)
        elif key in ("신입생_충원율", "입학률", "경쟁률"):
            st.metric(defn["label"], display_val, f"{delta:+.2f}{unit}")
        else:
            st.metric(defn["label"], display_val, f"{delta:+.1f}%")

metric_cols2 = st.columns(4)
for i, defn in enumerate(adm.KPI_DEFINITIONS[4:]):
    key = defn["key"]
    cur = kpis_latest.get(key, float("nan"))
    prev = kpis_prev.get(key, float("nan"))
    delta = adm.kpi_delta(cur, prev, key)
    val_str = adm.format_kpi_value(key, cur)
    unit = defn["unit"]
    display_val = f"{val_str}{unit}" if unit == "%" else f"{val_str} {unit}".strip()
    with metric_cols2[i % 4]:
        if delta is None:
            st.metric(defn["label"], display_val)
        elif key in ("신입생_충원율", "입학률", "경쟁률"):
            st.metric(defn["label"], display_val, f"{delta:+.2f}{unit}")
        else:
            st.metric(defn["label"], display_val, f"{delta:+.1f}%")

display_table = kpi_df[["KPI", str(LATEST_YEAR), str(PREV_YEAR), "단위"]].copy()
display_table["전년대비"] = kpi_df.apply(
    lambda r: (
        f"{r['전년대비']:+.2f}{r['단위']}"
        if r["kpi_key"] in ("신입생_충원율", "입학률", "경쟁률") and pd.notna(r["전년대비"])
        else (f"{r['전년대비']:+.1f}%" if pd.notna(r["전년대비"]) else "—")
    ),
    axis=1,
)

# ── 3. KPI 선택 → 연도별 추이 ─────────────────────────────────
st.subheader("3. KPI 연도별 추이 분석")

kpi_labels = [d["label"] for d in adm.KPI_DEFINITIONS]
label_to_key = {d["label"]: d["key"] for d in adm.KPI_DEFINITIONS}

selected_kpis = st.multiselect(
    "분석할 KPI 선택",
    options=kpi_labels,
    default=[],
    help="위 표에서 확인한 지표 중 추이를 볼 항목을 고릅니다.",
)

if not selected_kpis:
    st.info("KPI를 하나 이상 선택하면 연도별 추이 차트가 표시됩니다.")
else:
    for label in selected_kpis:
        key = label_to_key[label]
        series = _yearly_series(school_code, branch, key, school_name)
        defn = next(d for d in adm.KPI_DEFINITIONS if d["key"] == key)
        unit = defn["unit"]

        st.markdown(f"**{school_name}** · {label} ({unit})")
        plot_df = series.dropna(subset=["값"]).set_index("연도")[["값"]].rename(
            columns={"값": label}
        )
        st.line_chart(plot_df)
        st.caption(
            f"표시 구간: {int(series['연도'].min())}~{int(series['연도'].max())}년 "
            f"(유효 {len(plot_df)}개 연도)"
        )

        with st.expander(f"{label} — 연도별 수치"):
            show = series.copy()
            show["값"] = show["값"].apply(lambda v: adm.format_kpi_value(key, v))
            st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("신입생 충원율 산술식"):
    st.latex(
        r"""
        \text{신입생 충원율 (\%)} = \left(
        \frac{\text{정원내 입학자(학부)}}{\text{정원내 모집인원(학부)}}
        \right) \times 100
        """
    )
    st.caption(
        "경쟁률 = 지원자(전체) ÷ 모집인원(학부), "
        "입학률 = 입학자(전체) ÷ 지원자(전체) × 100. "
        "학과별 행을 학교·캠퍼스 단위로 합산한 값입니다."
    )
