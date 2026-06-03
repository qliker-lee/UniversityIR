# -*- coding: utf-8 -*-
"""선택 대학(중원 접두) 취업통계 분석 (2011~2024 job CSV)."""

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
from lib import job_data as job

SCHOOL_PREFIX = "중원"
ALL_MAJOR_LABEL = "전체 (대계열 합산)"

KPI_COLORS: dict[str, str] = {
    "졸업자": "#4C78A8",
    "취업자(합계)": "#59A14F",
    "취업률": "#E15759",
    "진학자": "#B279A2",
    "진학률": "#76B7B2",
    "1차 유지취업률": "#F28E2B",
    "1차 유지취업자": "#EDC948",
}

st.set_page_config(page_title="취업 통계 분석", layout="wide")
st.title("💼 취업 통계 분석")
st.write(
    f"**{job.JOB_YEAR_START}~{job.JOB_YEAR_END}년** `output/YYYY_job.csv` 데이터로 "
    "졸업자·취업자·취업률·진학·유지취업 현황을 분석합니다. "
    "학교명이 **「중원」**으로 시작하는 기관만 표시됩니다."
)

OUTPUT_DIR = job.OUTPUT_DIR
KPI_LABELS = [d["label"] for d in job.JOB_KPI_DEFINITIONS]
LABEL_TO_KEY = {d["label"]: d["key"] for d in job.JOB_KPI_DEFINITIONS}


@st.cache_data(show_spinner="취업통계 연도 조회…")
def _available_years(year_start: int, year_end: int) -> list[int]:
    """year_start·year_end를 인자로 받아 캐시 키에 반영 (연도 범위 변경 시 갱신)."""
    return sorted(job.year_from_job_path(p) for p in job.list_job_year_paths())


@st.cache_data(show_spinner="기관별 데이터 보유 연도…")
def _years_with_school_data(
    school_name: str,
    branch: str,
    kedi_code: str,
    year_start: int,
    year_end: int,
) -> list[int]:
    years: list[int] = []
    for year in range(year_start, year_end + 1):
        if not job.job_csv_path(year).is_file():
            continue
        kpis = job.school_job_kpis_for_year(
            school_name,
            branch,
            year,
            kedi_code=kedi_code,
            school_name_prefix=SCHOOL_PREFIX,
        )
        grads = kpis.get("졸업자_계", float("nan"))
        if grads == grads and grads > 0:
            years.append(year)
    return years


@st.cache_data(show_spinner="기관 목록 로딩…")
def _school_options(year_end: int) -> pd.DataFrame:
    return job.load_job_school_options(year_end, school_name_prefix=SCHOOL_PREFIX)


@st.cache_data(show_spinner="대계열 목록…")
def _major_list(
    school_name: str, branch: str, kedi_code: str, year: int, year_end: int
) -> list[str]:
    return job.list_job_major_categories(
        school_name, branch, year, kedi_code=kedi_code, school_name_prefix=SCHOOL_PREFIX
    )


@st.cache_data(show_spinner="취업 KPI 집계…")
def _job_kpis(
    school_name: str,
    branch: str,
    kedi_code: str,
    year: int,
    major_category: str | None,
) -> dict[str, float]:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return job.school_job_kpis_for_year(
        school_name,
        branch,
        year,
        kedi_code=kedi_code,
        major_category=major,
        school_name_prefix=SCHOOL_PREFIX,
    )


@st.cache_data(show_spinner="대계열별 취업 KPI…")
def _kpis_by_major(
    school_name: str,
    branch: str,
    kedi_code: str,
    year: int,
    majors: tuple[str, ...],
) -> pd.DataFrame:
    df = job.load_job_year_raw(year, school_name_prefix=SCHOOL_PREFIX)
    subset = job.filter_job_school(df, school_name, branch, kedi_code=kedi_code)

    def _row(label: str, sub: pd.DataFrame) -> dict[str, object]:
        kpis = job.summarize_job_kpis(sub)
        row: dict[str, object] = {"대계열": label}
        for d in job.JOB_KPI_DEFINITIONS:
            unit = d["unit"]
            col = f"{d['label']} ({unit})" if unit else d["label"]
            row[col] = job.format_job_kpi_value(d["key"], kpis.get(d["key"], float("nan")))
        return row

    rows = [_row("전체", subset)]
    for major in majors:
        sub = subset[subset[adm.MAJOR_COLUMN].astype(str).str.strip() == major]
        rows.append(_row(major, sub))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="연도별 KPI 추이…")
def _yearly_series(
    school_name: str,
    branch: str,
    kedi_code: str,
    kpi_key: str,
    major_category: str | None,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return job.build_job_yearly_series(
        school_name,
        branch,
        kpi_key,
        kedi_code=kedi_code,
        major_category=major,
        school_name_prefix=SCHOOL_PREFIX,
    )


def _school_label(row: pd.Series) -> str:
    return f"{row['학교명']} ({row['본분교']})"


def _kpi_color_scale(labels: list[str]) -> alt.Scale:
    return alt.Scale(
        domain=labels,
        range=[KPI_COLORS.get(lb, "#888888") for lb in labels],
    )


def _plot_kpi_bar(kpis: dict[str, float], year: int) -> None:
    count_kpis = [d for d in job.JOB_KPI_DEFINITIONS if d["kind"] == "sum"]
    long = pd.DataFrame(
        {
            "지표": [d["label"] for d in count_kpis],
            "값": [kpis.get(d["key"], float("nan")) for d in count_kpis],
        }
    )
    long = long[long["값"] == long["값"]]
    if long.empty:
        st.info(f"{year}년 표시할 인원 지표가 없습니다.")
        return
    labels = long["지표"].tolist()
    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("값:Q", title="인원(명)"),
            y=alt.Y("지표:N", sort=labels, title=""),
            color=alt.Color("지표:N", scale=_kpi_color_scale(labels), legend=None),
            tooltip=[
                alt.Tooltip("지표:N", title="지표"),
                alt.Tooltip("값:Q", title="인원", format=",.0f"),
            ],
        )
    )
    text = bars.mark_text(dy=-6, fontSize=12).encode(
        text=alt.Text("값:Q", format=",.0f")
    )
    chart = (bars + text).properties(height=280, title=f"{year}년 주요 인원 지표")
    st.altair_chart(chart, use_container_width=True)


def _plot_rate_cards(kpis: dict[str, float], year: int) -> None:
    rate_defs = [d for d in job.JOB_KPI_DEFINITIONS if d["kind"] == "rate"]
    cols = st.columns(len(rate_defs))
    for i, d in enumerate(rate_defs):
        val = kpis.get(d["key"], float("nan"))
        with cols[i]:
            st.metric(d["label"], job.format_job_kpi_raw(d["key"], val))


def _plot_yearly_area(series: pd.DataFrame, kpi_label: str) -> None:
    plot = series.sort_values("연도").copy()
    valid = plot[plot["값"].notna()]
    if valid.empty:
        st.info("유효한 연도별 데이터가 없습니다.")
        return
    defn = next(d for d in job.JOB_KPI_DEFINITIONS if d["label"] == kpi_label)
    is_pct = defn["unit"] == "%"
    color = KPI_COLORS.get(kpi_label, "#4C78A8")
    chart = (
        alt.Chart(plot)
        .mark_area(
            opacity=0.6,
            line={"strokeWidth": 2, "color": color},
            interpolate="monotone",
        )
        .encode(
            x=alt.X("연도:O", title="연도", sort=None),
            y=alt.Y("값:Q", title=f"{kpi_label} ({defn['unit']})"),
            tooltip=[
                alt.Tooltip("연도:O", title="연도"),
                alt.Tooltip(
                    "값:Q",
                    title=kpi_label,
                    format=",.1f" if is_pct else ",.0f",
                ),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
    first_y = int(valid["연도"].min())
    last_y = int(valid["연도"].max())
    if first_y > plot["연도"].min():
        st.caption(
            f"※ {int(plot['연도'].min())}~{first_y - 1}년은 선택 기관에 통계 행이 없거나 "
            f"졸업자가 0명입니다. 표시 구간: **{first_y}~{last_y}년**"
        )


def _render_drilldown(
    school_name: str,
    branch: str,
    kedi_code: str,
    year: int,
) -> None:
    st.subheader("계층별 취업 현황")
    st.caption(f"{year}년 · 대계열 → 중계열 → 소계열 → 학과")

    df = job.load_job_year_raw(year, school_name_prefix=SCHOOL_PREFIX)
    subset = job.filter_job_school(df, school_name, branch, kedi_code=kedi_code)
    if subset.empty:
        st.warning("해당 연도·기관 데이터가 없습니다.")
        return

    filters: dict[str, str] = {c: adm.DRILL_ALL_LABEL for c in adm.DRILL_COLUMNS}
    cols = st.columns(len(adm.DRILL_COLUMNS))
    for i, col_name in enumerate(adm.DRILL_COLUMNS):
        opts = adm.list_drill_options(subset, col_name, filters)
        options = [adm.DRILL_ALL_LABEL, *opts] if opts else [adm.DRILL_ALL_LABEL]
        with cols[i]:
            filters[col_name] = st.selectbox(col_name, options, key=f"job_drill_{col_name}")

    sub = adm.apply_hierarchy_filters(subset, filters)
    resolved = adm.resolve_drill_group_column(filters)

    if resolved is None:
        kpis = job.summarize_job_kpis(sub)
        st.markdown("**선택 학과**")
        c1, c2, c3 = st.columns(3)
        c1.metric("졸업자", job.format_job_kpi_raw("졸업자_계", kpis.get("졸업자_계", float("nan"))))
        c2.metric("취업자(합계)", job.format_job_kpi_raw("취업자_합계_계", kpis.get("취업자_합계_계", float("nan"))))
        c3.metric("취업률", job.format_job_kpi_raw("취업률", kpis.get("취업률", float("nan"))))
        return

    group_col, label = resolved
    table = job.job_kpis_by_group(sub, group_col, label_column=label)
    if table.empty:
        st.info("하위 그룹 데이터가 없습니다.")
        return
    st.dataframe(table, use_container_width=True, hide_index=True)


# ── 데이터 확인 ──────────────────────────────────────────────────
if not OUTPUT_DIR.is_dir() or not job.job_csv_path(job.LATEST_JOB_YEAR).is_file():
    st.error(
        f"`{OUTPUT_DIR}` 에 `{job.LATEST_JOB_YEAR}_job.csv` 가 없습니다. "
        "`11_1st_data_integration.py` 실행 후 다시 열어 주세요."
    )
    st.stop()

_y0, _y1 = job.JOB_YEAR_START, job.JOB_YEAR_END
years_avail = _available_years(_y0, _y1)
if not years_avail:
    st.error(f"{_y0}~{_y1}년 job CSV가 없습니다.")
    st.stop()

schools = _school_options(_y1)
if schools.empty:
    st.warning(f"학교명이 「{SCHOOL_PREFIX}」으로 시작하는 취업 데이터가 없습니다.")
    st.stop()

# ── 선택 UI ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("대학·캠퍼스")
    labels = [_school_label(r) for _, r in schools.iterrows()]
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    selected_label = st.selectbox("기관", options=labels, index=0)
    row = schools.iloc[label_to_idx[selected_label]]
    school_name = str(row["학교명"]).strip()
    branch = str(row["본분교"]).strip()
    kedi_code = str(row.get("KEDI 학교코드", "")).strip()
    st.caption(f"KEDI 코드: `{kedi_code}` · 본분교: `{branch}`")

years_with_data = _years_with_school_data(
    school_name, branch, kedi_code, _y0, _y1
)

with col2:
    st.subheader("졸업(통계) 연도")
    default_idx = (
        years_avail.index(job.LATEST_JOB_YEAR)
        if job.LATEST_JOB_YEAR in years_avail
        else len(years_avail) - 1
    )
    selected_year = st.selectbox(
        "분석 연도",
        options=years_avail,
        index=default_idx,
        format_func=lambda y: f"{y}년",
        help=f"파일 보유 연도 {_y0}~{_y1}. 기관 데이터: "
        + (
            f"{years_with_data[0]}~{years_with_data[-1]}년"
            if years_with_data
            else "없음"
        ),
    )
    prev_year = selected_year - 1 if selected_year > _y0 else None
    if years_with_data:
        st.caption(
            f"선택 기관 통계 행: **{years_with_data[0]}~{years_with_data[-1]}년** "
            f"({len(years_with_data)}개 연도)"
        )
    elif selected_year:
        st.caption("선택 기관·연도에 졸업자 데이터가 없을 수 있습니다.")

with col3:
    st.subheader("대계열")
    majors = _major_list(school_name, branch, kedi_code, selected_year, _y1)
    selected_major = st.selectbox(
        "대계열",
        options=[ALL_MAJOR_LABEL, *majors],
        index=0,
    )

major_label = selected_major if selected_major != ALL_MAJOR_LABEL else "전체"

# ── 1. 요약 ───────────────────────────────────────────────────────
st.divider()
st.subheader(f"취업 현황 요약 ({selected_year}년 · {major_label})")

current = _job_kpis(school_name, branch, kedi_code, selected_year, selected_major)
prev: dict[str, float] | None = None
if prev_year is not None:
    prev = _job_kpis(school_name, branch, kedi_code, prev_year, selected_major)

metric_cols = st.columns(len(job.JOB_KPI_DEFINITIONS))
for i, d in enumerate(job.JOB_KPI_DEFINITIONS):
    val = current.get(d["key"], float("nan"))
    with metric_cols[i]:
        delta = None
        if prev is not None:
            pval = prev.get(d["key"], float("nan"))
            if val == val and pval == pval:
                diff = val - pval
                delta = (
                    f"{diff:+.1f}%p"
                    if d["unit"] == "%"
                    else f"{diff:+,.0f}"
                )
        st.metric(
            d["label"],
            job.format_job_kpi_raw(d["key"], val),
            delta=delta,
        )

if prev_year is not None and prev is not None:
    st.caption(f"전년({prev_year}년) 대비 증감 · 취업률·진학률·유지취업률은 %p, 인원은 %")

_plot_kpi_bar(current, selected_year)
_plot_rate_cards(current, selected_year)

# ── 2. 전년 대비 표 ───────────────────────────────────────────────
if prev is not None and prev_year is not None:
    st.divider()
    st.subheader(f"전년({prev_year}년) 대비")
    rows = []
    for d in job.JOB_KPI_DEFINITIONS:
        cur = current.get(d["key"], float("nan"))
        prv = prev.get(d["key"], float("nan"))
        if cur == cur and prv == prv:
            diff = cur - prv
            if d["unit"] == "%":
                chg = f"{diff:+.1f}%p"
            else:
                chg = f"{diff:+,.0f}"
        else:
            chg = "—"
        rows.append(
            {
                "지표": d["label"],
                str(prev_year): job.format_job_kpi_raw(d["key"], prv),
                str(selected_year): job.format_job_kpi_raw(d["key"], cur),
                "증감": chg,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── 3. 대계열별 ───────────────────────────────────────────────────
st.divider()
st.subheader(f"대계열별 취업 현황 ({selected_year}년)")
if not majors:
    st.info("대계열 정보가 없습니다.")
else:
    st.dataframe(
        _kpis_by_major(school_name, branch, kedi_code, selected_year, tuple(majors)),
        use_container_width=True,
        hide_index=True,
    )

# ── 4. 드릴다운 ───────────────────────────────────────────────────
st.divider()
_render_drilldown(school_name, branch, kedi_code, selected_year)

# ── 5. 연도별 추이 ─────────────────────────────────────────────────
st.divider()
st.subheader(f"연도별 추이 ({major_label})")
st.caption(f"{job.JOB_YEAR_START}~{job.JOB_YEAR_END}년 · 학과 행 합산 후 비율은 가중 평균")

default_trend = [KPI_LABELS[0], KPI_LABELS[2], KPI_LABELS[4]]
selected_kpis = st.multiselect(
    "추이 KPI",
    options=KPI_LABELS,
    default=[lb for lb in default_trend if lb in KPI_LABELS],
)

if not selected_kpis:
    st.info("KPI를 하나 이상 선택하세요.")
else:
    for label in selected_kpis:
        key = LABEL_TO_KEY[label]
        series = _yearly_series(
            school_name, branch, kedi_code, key, selected_major, _y0, _y1
        )
        st.markdown(f"**{label}**")
        _plot_yearly_area(series, label)
        with st.expander(f"{label} — 연도별 수치"):
            show = series.copy()
            show["값"] = show["값"].apply(
                lambda v, k=key: job.format_job_kpi_raw(k, v)
            )
            st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("집계 방식"):
    st.markdown(
        f"""
        - 데이터: `output/{{연도}}_job.csv` ({job.JOB_YEAR_START}~{job.JOB_YEAR_END})
        - **인원** 지표: 학과 행 합산
        - **취업률·진학률·유지취업률**: 분자·분모를 각각 합산한 뒤 비율 계산 (가중 평균)
        - 예: 취업률 = Σ취업자(합계) ÷ Σ졸업자 × 100
        """
    )
