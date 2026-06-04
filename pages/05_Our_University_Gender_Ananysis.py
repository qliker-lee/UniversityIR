# -*- coding: utf-8 -*-
"""선택 대학(중원 접두) 입학자 남·여 비율 분석."""

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
SCOPE_OPTIONS = {s["label"]: s["id"] for s in adm.ENROLLMENT_GENDER_SCOPES}

COLOR_MALE = "#87CEEB"  # skyblue
COLOR_FEMALE = "#FFBF00"  # amber
CHART_BAR = "막대"
CHART_PIE = "파이"

st.set_page_config(page_title="입학자 성별 비율", layout="wide")
st.title("👥 입학자 남·여 비율 분석")
st.write(
    "학교명이 **「중원」**으로 시작하는 기관을 선택한 뒤, "
    "**입학자** 기준 남·여 인원과 비율을 확인합니다. "
    "대계열별·연도별 추이·계층 드릴다운도 지원합니다."
)

OUTPUT_DIR = adm.OUTPUT_DIR
LATEST_YEAR = adm.LATEST_YEAR
YEAR_START = adm.YEAR_START


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


@st.cache_data(show_spinner="입학자 성별 집계…")
def _gender_summary(
    school_code: str,
    branch: str,
    school_name: str,
    year: int,
    scope_id: str,
    major_category: str | None,
) -> dict[str, float]:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return adm.school_enrollment_gender_for_year(
        school_code,
        branch,
        year,
        school_name,
        major,
        scope_id=scope_id,
        school_name_prefix=SCHOOL_PREFIX,
    )


@st.cache_data(show_spinner="대계열별 성별 집계…")
def _gender_by_major(
    school_code: str,
    branch: str,
    school_name: str,
    year: int,
    scope_id: str,
    majors: tuple[str, ...],
) -> pd.DataFrame:
    df = adm.get_school_department_frame(
        school_code, branch, school_name, year, school_name_prefix=SCHOOL_PREFIX
    )
    rows: list[dict[str, object]] = []
    total_g = adm.summarize_enrollment_gender(df, scope_id)
    rows.append(
        {
            "대계열": "전체",
            "입학자(계)": total_g["total"],
            "남": total_g["male"],
            "여": total_g["female"],
            "남(%)": total_g["male_pct"],
            "여(%)": total_g["female_pct"],
        }
    )
    for major in majors:
        sub = df[df[adm.MAJOR_COLUMN].astype(str).str.strip() == major]
        g = adm.summarize_enrollment_gender(sub, scope_id)
        rows.append(
            {
                "대계열": major,
                "입학자(계)": g["total"],
                "남": g["male"],
                "여": g["female"],
                "남(%)": g["male_pct"],
                "여(%)": g["female_pct"],
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="연도별 성별 추이…")
def _gender_yearly_series(
    school_code: str,
    branch: str,
    school_name: str,
    scope_id: str,
    major_category: str | None,
) -> pd.DataFrame:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return adm.build_enrollment_gender_yearly_series(
        school_code,
        branch,
        school_name,
        major,
        scope_id=scope_id,
        school_name_prefix=SCHOOL_PREFIX,
    )


def _school_label(row: pd.Series) -> str:
    return f"{row['학교명']} ({row['본분교']})"


def _fmt_count(v: float) -> str:
    if v != v:
        return "—"
    return f"{v:,.0f}"


def _fmt_pct(v: float) -> str:
    if v != v:
        return "—"
    return f"{v:.1f}%"


def _gender_color_scale(categories: list[str]) -> alt.Scale:
    color_map = {
        "남": COLOR_MALE,
        "여": COLOR_FEMALE,
        "남(%)": COLOR_MALE,
        "여(%)": COLOR_FEMALE,
    }
    return alt.Scale(
        domain=categories,
        range=[color_map.get(c, "#888888") for c in categories],
    )


def _plot_gender_bar_simple(
    labels: list[str],
    values: list[float],
    y_title: str,
    *,
    height: int = 320,
) -> None:
    long = pd.DataFrame({"성별": labels, "값": values})
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("성별:N", sort=labels, title=""),
            y=alt.Y("값:Q", title=y_title),
            color=alt.Color(
                "성별:N",
                scale=_gender_color_scale(labels),
                legend=alt.Legend(title="성별"),
            ),
            tooltip=[
                alt.Tooltip("성별:N", title="성별"),
                alt.Tooltip("값:Q", title=y_title, format=",.0f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def _plot_gender_pie_simple(
    labels: list[str],
    values: list[float],
    *,
    height: int = 320,
    value_title: str = "인원",
) -> None:
    total = sum(v for v in values if v == v)
    if total <= 0:
        st.info("표시할 데이터가 없습니다.")
        return
    long = pd.DataFrame({"성별": labels, "값": values})
    long["비율"] = long["값"] / total * 100
    long["label_text"] = long.apply(
        lambda r: f"{r['성별']}\n{r['비율']:.1f}%", axis=1
    )

    base = alt.Chart(long).encode(
        theta=alt.Theta("값:Q", stack=True),
        color=alt.Color(
            "성별:N",
            scale=_gender_color_scale(labels),
            legend=alt.Legend(title="성별"),
        ),
        order=alt.Order("성별:N", sort="ascending"),
        tooltip=[
            alt.Tooltip("성별:N", title="성별"),
            alt.Tooltip("값:Q", title=value_title, format=",.0f"),
            alt.Tooltip("비율:Q", title="비율(%)", format=".1f"),
        ],
    )
    pie = base.mark_arc(outerRadius=120)
    text = base.mark_text(radius=75, size=13).encode(text="label_text:N")
    chart = (pie + text).properties(height=height)
    st.altair_chart(chart, width="stretch")


def _plot_gender_bar_grouped(
    df: pd.DataFrame,
    category_col: str,
    *,
    y_title: str = "인원",
    height: int = 400,
) -> None:
    value_cols = [c for c in ("남", "여") if c in df.columns]
    if not value_cols:
        return
    long = df.reset_index()
    if category_col not in long.columns and long.columns[0] != category_col:
        long = long.rename(columns={long.columns[0]: category_col})
    long = long.melt(
        id_vars=[category_col],
        value_vars=value_cols,
        var_name="성별",
        value_name="값",
    )
    chart = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X(f"{category_col}:N", title=category_col),
            y=alt.Y("값:Q", title=y_title),
            xOffset="성별:N",
            color=alt.Color(
                "성별:N",
                scale=_gender_color_scale(value_cols),
                legend=alt.Legend(title="성별"),
            ),
            tooltip=[
                alt.Tooltip(f"{category_col}:N", title=category_col),
                alt.Tooltip("성별:N", title="성별"),
                alt.Tooltip("값:Q", title=y_title, format=",.0f"),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def _plot_gender_pie_grouped(
    df: pd.DataFrame,
    category_col: str,
    *,
    value_title: str = "인원",
    height: int = 400,
) -> None:
    value_cols = [c for c in ("남", "여") if c in df.columns]
    if not value_cols:
        return
    long = df.reset_index()
    if category_col not in long.columns and long.columns[0] != category_col:
        long = long.rename(columns={long.columns[0]: category_col})
    long = long.melt(
        id_vars=[category_col],
        value_vars=value_cols,
        var_name="성별",
        value_name="값",
    )
    long["값"] = pd.to_numeric(long["값"], errors="coerce").fillna(0)
    group_totals = long.groupby(category_col)["값"].transform("sum")
    long = long[group_totals > 0].copy()
    if long.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    long["비율"] = long["값"] / group_totals * 100
    long["label_text"] = long.apply(
        lambda r: f"{r['성별']}\n{r['비율']:.0f}%", axis=1
    )

    n_groups = long[category_col].nunique()
    ncol = min(4, max(1, n_groups))
    row_h = max(160, min(220, height // max(1, (n_groups + ncol - 1) // ncol)))

    base = alt.Chart(long).encode(
        theta=alt.Theta("값:Q", stack=True),
        color=alt.Color(
            "성별:N",
            scale=_gender_color_scale(value_cols),
            legend=alt.Legend(title="성별"),
        ),
        order=alt.Order("성별:N", sort="ascending"),
        tooltip=[
            alt.Tooltip(f"{category_col}:N", title=category_col),
            alt.Tooltip("성별:N", title="성별"),
            alt.Tooltip("값:Q", title=value_title, format=",.0f"),
            alt.Tooltip("비율:Q", title="비율(%)", format=".1f"),
        ],
    )
    pie = base.mark_arc(outerRadius=70)
    text = base.mark_text(radius=42, size=11).encode(text="label_text:N")
    chart = (
        (pie + text)
        .properties(width=170, height=row_h)
        .facet(
            column=alt.Column(f"{category_col}:N", title=category_col),
            columns=ncol,
        )
    )
    st.altair_chart(chart, width="stretch")


def _plot_gender_simple(
    chart_type: str,
    labels: list[str],
    values: list[float],
    y_title: str,
    *,
    height: int = 320,
) -> None:
    if chart_type == CHART_PIE:
        _plot_gender_pie_simple(labels, values, height=height, value_title=y_title)
    else:
        _plot_gender_bar_simple(labels, values, y_title, height=height)


def _plot_gender_grouped(
    chart_type: str,
    df: pd.DataFrame,
    category_col: str,
    *,
    y_title: str = "인원",
    height: int = 400,
) -> None:
    if chart_type == CHART_PIE:
        _plot_gender_pie_grouped(df, category_col, value_title=y_title, height=height)
    else:
        _plot_gender_bar_grouped(df, category_col, y_title=y_title, height=height)


def _plot_gender_area(
    df: pd.DataFrame,
    *,
    y_title: str,
    height: int = 320,
    stacked: bool = False,
) -> None:
    value_cols = [c for c in df.columns if c in ("남", "여", "남(%)", "여(%)")]
    if not value_cols:
        return
    long = df.reset_index().melt(
        id_vars=[df.index.name or "index"],
        value_vars=value_cols,
        var_name="성별",
        value_name="값",
    )
    year_col = long.columns[0]
    long = long.rename(columns={year_col: "연도"})
    fmt = ",.1f" if y_title.endswith("%") else ",.0f"
    y_stack: str | None = "zero" if stacked else None
    chart = (
        alt.Chart(long)
        .mark_area(opacity=0.65, line={"strokeWidth": 2}, point=True, interpolate="monotone")
        .encode(
            x=alt.X("연도:O", title="연도"),
            y=alt.Y("값:Q", title=y_title, stack=y_stack),
            color=alt.Color(
                "성별:N",
                scale=_gender_color_scale(value_cols),
                legend=alt.Legend(title="성별"),
            ),
            tooltip=[
                alt.Tooltip("연도:O", title="연도"),
                alt.Tooltip("성별:N", title="성별"),
                alt.Tooltip("값:Q", title=y_title, format=fmt),
            ],
        )
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def _display_gender_table(df: pd.DataFrame, group_col: str) -> None:
    show = df.copy()
    for col in ("입학자(계)", "남", "여"):
        if col in show.columns:
            show[col] = show[col].apply(_fmt_count)
    for col in ("남(%)", "여(%)"):
        if col in show.columns:
            show[col] = show[col].apply(_fmt_pct)
    st.dataframe(show, use_container_width=True, hide_index=True)


def _render_summary_metrics(g: dict[str, float], scope_label: str, year: int) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("입학자(계)", _fmt_count(g["total"]))
    c2.metric("남", _fmt_count(g["male"]))
    c3.metric("여", _fmt_count(g["female"]))
    c4.metric("남 비율", _fmt_pct(g["male_pct"]))
    c5.metric("여 비율", _fmt_pct(g["female_pct"]))
    st.caption(f"{year}년 · {scope_label} · 선택 범위 합산")


def _render_pie_column(g: dict[str, float], year: int) -> None:
    """단일 연도 남·여 파이 + 캡션."""
    if g["total"] != g["total"] or g["total"] <= 0:
        st.info(f"{year}년 입학자 데이터가 없습니다.")
        return
    _plot_gender_pie_simple(
        ["남", "여"], [g["male"], g["female"]], height=360, value_title="인원"
    )
    st.caption(
        f"남 {_fmt_count(g['male'])}명 ({_fmt_pct(g['male_pct'])}) · "
        f"여 {_fmt_count(g['female'])}명 ({_fmt_pct(g['female_pct'])})"
    )


def _render_gender_charts(
    g: dict[str, float],
    chart_type: str,
    year: int,
    *,
    prev_g: dict[str, float] | None = None,
    prev_year: int | None = None,
) -> None:
    if chart_type == CHART_PIE:
        has_current = g["total"] == g["total"] and g["total"] > 0
        has_prev = (
            prev_g is not None
            and prev_year is not None
            and prev_g["total"] == prev_g["total"]
            and prev_g["total"] > 0
        )
        if not has_current and not has_prev:
            st.info("해당 조건에 입학자 데이터가 없습니다.")
            return

        st.markdown("**남·여 구성 (파이)**")
        col_cur, col_prev = st.columns(2)
        with col_cur:
            st.markdown(f"**{year}년**")
            if has_current:
                _render_pie_column(g, year)
            else:
                st.info(f"{year}년 입학자 데이터가 없습니다.")
        with col_prev:
            if prev_year is not None:
                st.markdown(f"**{prev_year}년 (전년)**")
                if has_prev and prev_g is not None:
                    _render_pie_column(prev_g, prev_year)
                else:
                    st.info(f"{prev_year}년 입학자 데이터가 없습니다.")
            else:
                st.markdown("**전년**")
                st.info("비교 가능한 전년 데이터가 없습니다.")
        return

    if g["total"] != g["total"] or g["total"] <= 0:
        st.info("해당 조건에 입학자 데이터가 없습니다.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**남·여 인원**")
        _plot_gender_bar_simple(
            ["남", "여"], [g["male"], g["female"]], "인원", height=320
        )
    with col_b:
        st.markdown("**남·여 비율**")
        _plot_gender_bar_simple(
            ["남", "여"], [g["male_pct"], g["female_pct"]], "비율(%)", height=320
        )


def _render_drilldown(
    school_code: str,
    branch: str,
    school_name: str,
    year: int,
    scope_id: str,
    scope_label: str,
    chart_type: str,
) -> None:
    st.subheader("계층별 입학자 성별")
    st.caption(
        f"{year}년 · {scope_label} · "
        "대계열 → 중계열 → 소계열 → 학과 순으로 드릴다운합니다."
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
            filters[col_name] = st.selectbox(col_name, options, key=f"drill_{col_name}")

    sub = adm.apply_hierarchy_filters(df, filters)
    resolved = adm.resolve_drill_group_column(filters)

    if resolved is None:
        g = adm.summarize_enrollment_gender(sub, scope_id)
        st.markdown("**선택 학과**")
        _render_summary_metrics(g, scope_label, year)
        _render_gender_charts(
            g,
            chart_type,
            year,
            prev_g=adm.school_enrollment_gender_for_year(
                school_code,
                branch,
                year - 1,
                school_name,
                None,
                scope_id=scope_id,
                school_name_prefix=SCHOOL_PREFIX,
            )
            if year > YEAR_START
            else None,
            prev_year=year - 1 if year > YEAR_START else None,
        )
        return

    group_col, label = resolved
    table = adm.enrollment_gender_by_group(sub, group_col, scope_id, label_column=label)
    if table.empty:
        st.info("하위 그룹 데이터가 없습니다.")
        return

    st.markdown(f"**{label}별**")
    _display_gender_table(table, label)

    chart_src = table.copy()
    for c in ("남", "여"):
        chart_src[c] = pd.to_numeric(chart_src[c], errors="coerce")
    _plot_gender_grouped(
        chart_type,
        chart_src.set_index(label)[["남", "여"]],
        label,
        height=400,
    )


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
col1, col2, col3, col4 = st.columns(4)

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
    st.subheader("입학자 구분")
    scope_label = st.selectbox(
        "집계 대상",
        options=list(SCOPE_OPTIONS.keys()),
        index=0,
        help="입학자(전체) 또는 정원내 입학자 기준을 선택합니다.",
    )
    scope_id = SCOPE_OPTIONS[scope_label]

with col4:
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

chart_type = st.radio(
    "차트 유형",
    options=[CHART_PIE, CHART_BAR],
    horizontal=True,
    index=0,
    help="막대: 인원·비율을 나란히 비교. 파이: 남·여 구성 비율을 원형으로 표시.",
)

# ── 1. 요약 · 차트 ───────────────────────────────────────────────
st.divider()
st.subheader(f"입학자 성별 요약 ({selected_year}년 · {major_label})")

current = _gender_summary(
    school_code, branch, school_name, selected_year, scope_id, selected_major
)
_render_summary_metrics(current, scope_label, selected_year)

prev: dict[str, float] | None = None
if prev_year is not None:
    prev = _gender_summary(
        school_code, branch, school_name, prev_year, scope_id, selected_major
    )
    d_total = (
        current["total"] - prev["total"]
        if current["total"] == current["total"] and prev["total"] == prev["total"]
        else None
    )
    d_male_pct = (
        current["male_pct"] - prev["male_pct"]
        if current["male_pct"] == current["male_pct"]
        and prev["male_pct"] == prev["male_pct"]
        else None
    )
    if d_total is not None:
        st.caption(f"전년({prev_year}년) 대비: 입학자 {d_total:+,.0f}명")
    else:
        st.caption(f"전년({prev_year}년) 대비: —")
    if d_male_pct is not None:
        st.caption(f"남 비율 변화: {d_male_pct:+.1f}%p")

_render_gender_charts(
    current,
    chart_type,
    selected_year,
    prev_g=prev,
    prev_year=prev_year,
)

# ── 2. 대계열별 표 · 차트 ─────────────────────────────────────────
st.divider()
st.subheader(f"대계열별 입학자 성별 ({selected_year}년)")
if not majors:
    st.info("대계열 정보가 없어 표를 만들 수 없습니다.")
else:
    major_table = _gender_by_major(
        school_code, branch, school_name, selected_year, scope_id, tuple(majors)
    )
    _display_gender_table(major_table, "대계열")

    chart_src = major_table[major_table["대계열"] != "전체"].copy()
    for c in ("남", "여"):
        chart_src[c] = pd.to_numeric(chart_src[c], errors="coerce")
    title = "**대계열별 남·여 구성 (파이)**" if chart_type == CHART_PIE else "**대계열별 남·여 인원**"
    st.markdown(title)
    _plot_gender_grouped(
        chart_type,
        chart_src.set_index("대계열")[["남", "여"]],
        "대계열",
        height=400,
    )

# ── 3. 계층 드릴다운 ─────────────────────────────────────────────
st.divider()
_render_drilldown(
    school_code, branch, school_name, selected_year, scope_id, scope_label, chart_type
)

# ── 4. 연도별 추이 ───────────────────────────────────────────────
st.divider()
st.subheader(f"연도별 입학자 성별 추이 ({major_label})")
st.caption(f"{YEAR_START}~{LATEST_YEAR}년 · {scope_label}")

series = _gender_yearly_series(
    school_code, branch, school_name, scope_id, selected_major
)
valid = series.dropna(subset=["입학자(계)"])
if valid.empty:
    st.info("연도별 입학자 데이터가 없습니다.")
else:
    plot_count = valid.set_index("연도")[["남", "여"]]
    st.markdown("**남·여 인원 추이**")
    _plot_gender_area(plot_count, y_title="인원", height=320, stacked=True)

    plot_pct = valid.set_index("연도")[["남(%)", "여(%)"]]
    st.markdown("**남·여 비율 추이**")
    _plot_gender_area(plot_pct, y_title="비율(%)", height=320, stacked=True)

    with st.expander("연도별 수치 표"):
        show = valid.copy()
        show["입학자(계)"] = show["입학자(계)"].apply(_fmt_count)
        show["남"] = show["남"].apply(_fmt_count)
        show["여"] = show["여"].apply(_fmt_count)
        show["남(%)"] = show["남(%)"].apply(_fmt_pct)
        show["여(%)"] = show["여(%)"].apply(_fmt_pct)
        st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("집계 방식"):
    st.markdown(
        """
        - **입학자(전체)**: `입학자_전체_남` / `입학자_전체_여` 합산
        - **정원내 입학자(학부)**: `정원내_입학자_학부_남` / `정원내_입학자_학부_여` 합산
        - **정원내 입학자(전체)**: `정원내_입학자_전체_남` / `정원내_입학자_전체_여` 합산
        - **남(%)** = 남 ÷ 계 × 100 (학과 행을 선택 범위 내에서 합산한 뒤 비율 계산)
        """
    )
    st.caption(
        f"데이터: `output/YYYY_data.csv` · 학교명「{SCHOOL_PREFIX}」시작 기관만 표시."
    )
