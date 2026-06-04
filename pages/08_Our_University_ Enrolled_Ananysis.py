# -*- coding: utf-8 -*-
"""선택 대학(중원 접두) 입학·재적·휴학·유예·외국인·졸업 현황 분석."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib import admissions_data as adm

SCHOOL_PREFIX = "중원"
ALL_MAJOR_LABEL = "전체 (대계열 합산)"

STATUS_COLORS: dict[str, str] = {
    "입학자(전체)": "#4C78A8",
    "휴학생(전체)": "#F28E2B",
    "유예생(전체)": "#B279A2",
    "외국인 학생(총계)": "#76B7B2",
    "졸업자(전체)": "#E15759",
}

st.set_page_config(page_title="학생 현황 분석", layout="wide")
st.title("📊 학생 현황 분석")
st.write(
    "**입학자·휴학생·유예생·외국인 학생·졸업자** 현황을 "
    "연도·대계열별로 확인합니다."
)

OUTPUT_DIR = adm.OUTPUT_DIR
LATEST_YEAR = adm.LATEST_YEAR
YEAR_START = adm.YEAR_START
STATUS_LABELS = [d["label"] for d in adm.STUDENT_STATUS_DEFINITIONS]
LABEL_TO_KEY = {d["label"]: d["key"] for d in adm.STUDENT_STATUS_DEFINITIONS}


@st.cache_data(show_spinner="분석 가능 연도 조회…")
def _available_years() -> list[int]:
    return sorted(adm.year_from_path(p) for p in adm.list_year_data_paths())


@st.cache_data(show_spinner="중원대 데이터 목록 로딩…")
def _school_options() -> pd.DataFrame:
    return adm.load_school_options(LATEST_YEAR, school_name_prefix=SCHOOL_PREFIX)


@st.cache_data(show_spinner="대계열 목록 로딩…")
def _major_list(school_code: str, branch: str, school_name: str, year: int) -> list[str]:
    return adm.list_major_categories(
        school_code, branch, year, school_name, school_name_prefix=SCHOOL_PREFIX
    )


@st.cache_data(show_spinner="학생 현황 집계…")
def _status_summary(
    school_code: str,
    branch: str,
    school_name: str,
    year: int,
    major_category: str | None,
) -> dict[str, float]:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return adm.school_student_status_for_year(
        school_code,
        branch,
        year,
        school_name,
        major,
        school_name_prefix=SCHOOL_PREFIX,
    )


@st.cache_data(show_spinner="대계열별 학생 현황…")
def _status_by_major(
    school_code: str,
    branch: str,
    school_name: str,
    year: int,
    majors: tuple[str, ...],
) -> pd.DataFrame:
    df = adm.get_school_department_frame(
        school_code, branch, school_name, year, school_name_prefix=SCHOOL_PREFIX
    )
    rows: list[dict[str, object]] = []
    total = adm.summarize_student_status(df)
    row_total: dict[str, object] = {"대계열": "전체"}
    for d in adm.STUDENT_STATUS_DEFINITIONS:
        row_total[d["label"]] = total.get(d["key"], float("nan"))
    rows.append(row_total)
    for major in majors:
        sub = df[df[adm.MAJOR_COLUMN].astype(str).str.strip() == major]
        status = adm.summarize_student_status(sub)
        row: dict[str, object] = {"대계열": major}
        for d in adm.STUDENT_STATUS_DEFINITIONS:
            row[d["label"]] = status.get(d["key"], float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="연도별 학생 현황 추이…")
def _status_yearly_series(
    school_code: str,
    branch: str,
    school_name: str,
    major_category: str | None,
) -> pd.DataFrame:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return adm.build_student_status_yearly_series(
        school_code,
        branch,
        school_name,
        major,
        school_name_prefix=SCHOOL_PREFIX,
    )


def _school_label(row: pd.Series) -> str:
    return f"{row['학교명']} ({row['본분교']})"


def _status_color_scale(labels: list[str]) -> alt.Scale:
    return alt.Scale(
        domain=labels,
        range=[STATUS_COLORS.get(lb, "#888888") for lb in labels],
    )


def _plot_status_bar(summary: dict[str, float], year: int, *, height: int = 360) -> None:
    long = pd.DataFrame(
        {
            "지표": STATUS_LABELS,
            "인원": [summary.get(LABEL_TO_KEY[lb], float("nan")) for lb in STATUS_LABELS],
        }
    )
    long = long[long["인원"] == long["인원"]]
    if long.empty:
        st.info(f"{year}년 표시할 현황 데이터가 없습니다.")
        return
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("인원:Q", title="인원(명)"),
            y=alt.Y("지표:N", sort=STATUS_LABELS, title=""),
            color=alt.Color(
                "지표:N",
                scale=_status_color_scale(STATUS_LABELS),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("지표:N", title="지표"),
                alt.Tooltip("인원:Q", title="인원", format=",.0f"),
            ],
        )
        .properties(height=height, title=f"{year}년 학생 현황")
    )
    st.altair_chart(chart, width="stretch")


def _plot_status_area(
    series: pd.DataFrame,
    selected_labels: list[str],
    *,
    height: int = 360,
) -> None:
    if not selected_labels:
        st.info("추이를 볼 지표를 하나 이상 선택하세요.")
        return
    plot = series.set_index("연도")[selected_labels].copy()
    long = plot.reset_index().melt(id_vars="연도", var_name="지표", value_name="인원")
    long = long.dropna(subset=["인원"])
    if long.empty:
        st.info("선택 지표에 유효한 연도별 데이터가 없습니다.")
        return
    chart = (
        alt.Chart(long)
        .mark_area(opacity=0.55, line={"strokeWidth": 2}, point=True, interpolate="monotone")
        .encode(
            x=alt.X("연도:O", title="연도"),
            y=alt.Y("인원:Q", title="인원(명)", stack=None),
            color=alt.Color(
                "지표:N",
                scale=_status_color_scale(selected_labels),
                legend=alt.Legend(title="지표"),
            ),
            tooltip=[
                alt.Tooltip("연도:O", title="연도"),
                alt.Tooltip("지표:N", title="지표"),
                alt.Tooltip("인원:Q", title="인원", format=",.0f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def _display_status_table(df: pd.DataFrame) -> None:
    show = df.copy()
    for col in STATUS_LABELS:
        if col in show.columns:
            show[col] = show[col].apply(adm.format_status_value)
    st.dataframe(show, use_container_width=True, hide_index=True)


def _render_status_metrics(
    current: dict[str, float],
    prev: dict[str, float] | None,
    year: int,
    prev_year: int | None,
) -> None:
    cols = st.columns(len(adm.STUDENT_STATUS_DEFINITIONS))
    for i, d in enumerate(adm.STUDENT_STATUS_DEFINITIONS):
        val = current.get(d["key"], float("nan"))
        with cols[i]:
            delta = None
            if prev is not None and prev_year is not None:
                pval = prev.get(d["key"], float("nan"))
                if val == val and pval == pval:
                    delta = val - pval
            st.metric(
                d["label"],
                adm.format_status_value(val),
                delta=f"{delta:+,.0f}" if delta is not None else None,
                help=f"{d['column']} 합산",
            )
    cap = f"{year}년 · 선택 범위 합산"
    if prev_year is not None:
        cap += f" · 전년({prev_year}년) 대비 증감 표시"
    st.caption(cap)


def _render_drilldown(
    school_code: str,
    branch: str,
    school_name: str,
    year: int,
) -> None:
    st.subheader("계층별 학생 현황")
    st.caption(
        f"{year}년 · 대계열 → 중계열 → 소계열 → 학과 순으로 드릴다운합니다."
    )

    df = adm.get_school_department_frame(
        school_code, branch, school_name, year, school_name_prefix=SCHOOL_PREFIX
    )
    if df.empty:
        st.warning("선택한 캠퍼스에 해당 연도 데이터가 없습니다.")
        return

    filters: dict[str, str] = {c: adm.DRILL_ALL_LABEL for c in adm.DRILL_COLUMNS}
    cols = st.columns(len(adm.DRILL_COLUMNS))
    for i, col_name in enumerate(adm.DRILL_COLUMNS):
        opts = adm.list_drill_options(df, col_name, filters)
        options = [adm.DRILL_ALL_LABEL, *opts] if opts else [adm.DRILL_ALL_LABEL]
        with cols[i]:
            filters[col_name] = st.selectbox(
                col_name, options, key=f"status_drill_{col_name}"
            )

    sub = adm.apply_hierarchy_filters(df, filters)
    resolved = adm.resolve_drill_group_column(filters)

    if resolved is None:
        status = adm.summarize_student_status(sub)
        st.markdown("**선택 학과**")
        _render_status_metrics(status, None, year, None)
        _plot_status_bar(status, year)
        return

    group_col, label = resolved
    table = adm.student_status_by_group(sub, group_col, label_column=label)
    if table.empty:
        st.info("하위 그룹 데이터가 없습니다.")
        return

    st.markdown(f"**{label}별**")
    _display_status_table(table)

    chart_src = table.copy()
    for lb in STATUS_LABELS:
        if lb in chart_src.columns:
            chart_src[lb] = pd.to_numeric(chart_src[lb], errors="coerce")
    long = chart_src.melt(id_vars=[label], value_vars=STATUS_LABELS, var_name="지표", value_name="인원")
    long = long.dropna(subset=["인원"])
    if not long.empty:
        chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("인원:Q", title="인원(명)"),
                y=alt.Y(f"{label}:N", title=label),
                color=alt.Color("지표:N", scale=_status_color_scale(STATUS_LABELS)),
                yOffset="지표:N",
                tooltip=[label, "지표", alt.Tooltip("인원:Q", format=",.0f")],
            )
            .properties(height=max(280, 40 * table.shape[0]))
        )
        st.altair_chart(chart, width="stretch")


# ── 데이터 존재 확인 ─────────────────────────────────────────────
if not OUTPUT_DIR.is_dir() or not (OUTPUT_DIR / f"{LATEST_YEAR}_data.csv").is_file():
    st.error(
        f"`{OUTPUT_DIR}` 에 `{LATEST_YEAR}_data.csv` 가 없습니다. "
        "`01_ist_data_integration.py` 실행 후 다시 열어 주세요."
    )
    st.stop()

schools = _school_options()
if schools.empty:
    st.warning(f"학교명이 「{SCHOOL_PREFIX}」으로 시작하는 데이터가 없습니다.")
    st.stop()

# ── 선택 UI ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("대학·캠퍼스")
    labels = [_school_label(r) for _, r in schools.iterrows()]
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    selected_label = st.selectbox(
        "기관",
        options=labels,
        index=0,
        help=f"학교명이 「{SCHOOL_PREFIX}」으로 시작하는 목록만 표시됩니다.",
    )
    row = schools.iloc[label_to_idx[selected_label]]
    school_code = str(row["학교코드"]).strip()
    branch = str(row["본분교"]).strip()
    school_name = str(row["학교명"]).strip()
    st.caption(f"학교코드: `{school_code}` · 본분교: `{branch}`")

with col2:
    st.subheader("연도")
    available_years = _available_years()
    if not available_years:
        st.error("`output` 폴더에 연도별 CSV가 없습니다.")
        st.stop()
    default_year_idx = (
        available_years.index(LATEST_YEAR)
        if LATEST_YEAR in available_years
        else len(available_years) - 1
    )
    selected_year = st.selectbox(
        "분석 연도",
        options=available_years,
        index=default_year_idx,
        format_func=lambda y: f"{y}년",
    )
    prev_year = selected_year - 1 if selected_year > YEAR_START else None

with col3:
    st.subheader("대계열")
    majors = _major_list(school_code, branch, school_name, selected_year)
    major_options = [ALL_MAJOR_LABEL, *majors]
    selected_major = st.selectbox(
        "대계열",
        options=major_options,
        index=0,
        help="「전체」는 캠퍼스 전체 학과 합산입니다.",
    )

major_label = selected_major if selected_major != ALL_MAJOR_LABEL else "전체"

# ── 1. 요약 지표 · 막대 차트 ─────────────────────────────────────
st.divider()
st.subheader(f"학생 현황 요약 ({selected_year}년 · {major_label})")

current = _status_summary(
    school_code, branch, school_name, selected_year, selected_major
)
prev: dict[str, float] | None = None
if prev_year is not None:
    prev = _status_summary(
        school_code, branch, school_name, prev_year, selected_major
    )

_render_status_metrics(current, prev, selected_year, prev_year)
_plot_status_bar(current, selected_year)

# ── 2. 전년 대비 표 ───────────────────────────────────────────────
if prev is not None and prev_year is not None:
    st.divider()
    st.subheader(f"전년({prev_year}년) 대비")
    rows = []
    for d in adm.STUDENT_STATUS_DEFINITIONS:
        cur = current.get(d["key"], float("nan"))
        prv = prev.get(d["key"], float("nan"))
        diff = cur - prv if cur == cur and prv == prv else float("nan")
        pct = (diff / prv * 100) if diff == diff and prv not in (0, float("nan")) else float("nan")
        rows.append(
            {
                "지표": d["label"],
                str(prev_year): adm.format_status_value(prv),
                str(selected_year): adm.format_status_value(cur),
                "증감(명)": adm.format_status_value(diff) if diff == diff else "—",
                "증감률(%)": f"{pct:+.1f}%" if pct == pct else "—",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── 3. 대계열별 표 ─────────────────────────────────────────────────
st.divider()
st.subheader(f"대계열별 학생 현황 ({selected_year}년)")
if not majors:
    st.info("대계열 정보가 없어 표를 만들 수 없습니다.")
else:
    major_table = _status_by_major(
        school_code, branch, school_name, selected_year, tuple(majors)
    )
    _display_status_table(major_table)

# ── 4. 계층 드릴다운 ─────────────────────────────────────────────
st.divider()
_render_drilldown(school_code, branch, school_name, selected_year)

# ── 5. 연도별 추이 ─────────────────────────────────────────────────
st.divider()
st.subheader(f"연도별 학생 현황 추이 ({major_label})")
st.caption(f"{YEAR_START}~{LATEST_YEAR}년 · 학과 행 합산")

series = _status_yearly_series(school_code, branch, school_name, selected_major)
if series.empty:
    st.info("연도별 데이터가 없습니다.")
else:
    default_metrics = STATUS_LABELS[:3]
    selected_metrics = st.multiselect(
        "추이 지표",
        options=STATUS_LABELS,
        default=default_metrics,
    )
    _plot_status_area(series, selected_metrics)

    with st.expander("연도별 수치 표"):
        show = series.copy()
        for lb in STATUS_LABELS:
            if lb in show.columns:
                show[lb] = show[lb].apply(adm.format_status_value)
        st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("집계 방식"):
    st.markdown(
        """
        | 지표 | 원본 컬럼 |
        |------|-----------|
        | 입학자(전체) | `입학자_전체_계` |
        | 휴학생(전체) | `휴학생_전체_계` |
        | 유예생(전체) | `유예생_전체_계` |
        | 외국인 학생(총계) | `외국 학생_총계_계` |
        | 졸업자(전체) | `졸업자_전체` |
        """
    )
    st.caption(
        f"데이터: `output/YYYY_data.csv` · 학교명「{SCHOOL_PREFIX}」시작 기관만 표시."
    )
