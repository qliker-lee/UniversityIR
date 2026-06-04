# -*- coding: utf-8 -*-
"""선택 대학(중원 접두) 전임·비전임 교원 남·여 현황 분석."""

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
SCOPE_OPTIONS = {s["label"]: s["id"] for s in adm.FACULTY_GENDER_SCOPES}
TREND_SCOPE_OPTIONS = {
    "전체 (전임+비전임)": adm.FACULTY_SCOPE_ALL,
    **SCOPE_OPTIONS,
}

COLOR_MALE = "#87CEEB"
COLOR_FEMALE = "#FFBF00"
CHART_BAR = "막대"
CHART_PIE = "파이"
SCOPE_COLOR_MAP = {
    s["label"]: color
    for s, color in zip(adm.FACULTY_GENDER_SCOPES, ["#4C78A8", "#F28E2B"])
}


def _scope_color_scale(labels: list[str]) -> alt.Scale:
    return alt.Scale(
        domain=labels,
        range=[SCOPE_COLOR_MAP.get(lb, "#888888") for lb in labels],
    )

st.set_page_config(page_title="교원 현황 분석", layout="wide")
st.title("👨‍🏫 교원 현황 분석")
st.write(
    "학교명이 **「중원」**으로 시작하는 기관을 선택한 뒤, "
    "**전임교원·비전임교원**의 인원과 남·여 구성을 분석합니다."
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


@st.cache_data(show_spinner="교원 현황 집계…")
def _faculty_all(
    school_code: str,
    branch: str,
    school_name: str,
    year: int,
    major_category: str | None,
) -> dict[str, dict[str, float]]:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return adm.school_faculty_all_for_year(
        school_code,
        branch,
        year,
        school_name,
        major,
        school_name_prefix=SCHOOL_PREFIX,
    )


@st.cache_data(show_spinner="대계열별 교원 현황…")
def _faculty_by_major(
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
    total = adm.summarize_faculty_all(df)
    row_total: dict[str, object] = {"대계열": "전체"}
    for s in adm.FACULTY_GENDER_SCOPES:
        g = total[s["id"]]
        row_total[f"{s['label']}(계)"] = g["total"]
        row_total[f"{s['label']}_남"] = g["male"]
        row_total[f"{s['label']}_여"] = g["female"]
    rows.append(row_total)
    for major in majors:
        sub = df[df[adm.MAJOR_COLUMN].astype(str).str.strip() == major]
        status = adm.summarize_faculty_all(sub)
        row: dict[str, object] = {"대계열": major}
        for s in adm.FACULTY_GENDER_SCOPES:
            g = status[s["id"]]
            row[f"{s['label']}(계)"] = g["total"]
            row[f"{s['label']}_남"] = g["male"]
            row[f"{s['label']}_여"] = g["female"]
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="연도별 교원 추이…")
def _faculty_gender_yearly(
    school_code: str,
    branch: str,
    school_name: str,
    scope_id: str,
    major_category: str | None,
) -> pd.DataFrame:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return adm.build_faculty_gender_yearly_series(
        school_code,
        branch,
        school_name,
        major,
        scope_id=scope_id,
        school_name_prefix=SCHOOL_PREFIX,
    )


@st.cache_data(show_spinner="연도별 교원(계) 추이…")
def _faculty_total_yearly(
    school_code: str,
    branch: str,
    school_name: str,
    major_category: str | None,
) -> pd.DataFrame:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return adm.build_faculty_total_yearly_series(
        school_code,
        branch,
        school_name,
        major,
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
    color_map = {"남": COLOR_MALE, "여": COLOR_FEMALE, "남(%)": COLOR_MALE, "여(%)": COLOR_FEMALE}
    return alt.Scale(
        domain=categories,
        range=[color_map.get(c, "#888888") for c in categories],
    )


def _plot_pie(g: dict[str, float], *, height: int = 300, chart_width: int = 320) -> None:
    male = float(g.get("male", 0) or 0)
    female = float(g.get("female", 0) or 0)
    if male + female <= 0:
        st.info("표시할 데이터가 없습니다.")
        return
    total = male + female
    long = pd.DataFrame(
        {
            "성별": ["남", "여"],
            "값": [male, female],
            "비율": [male / total * 100, female / total * 100],
        }
    )
    long["label_text"] = long.apply(
        lambda r: f"{r['성별']}\n{int(r['값']):,}명\n{r['비율']:.1f}%", axis=1
    )
    base = alt.Chart(long).encode(
        theta=alt.Theta("값:Q", stack=True),
        color=alt.Color(
            "성별:N",
            scale=_gender_color_scale(["남", "여"]),
            legend=alt.Legend(title="성별"),
        ),
        order=alt.Order("성별:N", sort="ascending"),
        tooltip=[
            alt.Tooltip("성별:N", title="성별"),
            alt.Tooltip("값:Q", format=",.0f", title="인원"),
            alt.Tooltip("비율:Q", format=".1f", title="비율(%)"),
        ],
    )
    pie = base.mark_arc(outerRadius=95)
    text = base.mark_text(radius=58, size=12, fill="white").encode(text="label_text:N")
    chart = (pie + text).properties(height=height, width=chart_width)
    st.altair_chart(chart, use_container_width=False)


def _plot_scope_compare_bar(all_g: dict[str, dict[str, float]], year: int) -> None:
    rows = []
    for s in adm.FACULTY_GENDER_SCOPES:
        g = all_g[s["id"]]
        rows.append({"구분": s["label"], "인원": g["total"] if g["total"] == g["total"] else 0})
    long = pd.DataFrame(rows)
    long["인원"] = pd.to_numeric(long["인원"], errors="coerce").fillna(0)
    if long["인원"].sum() <= 0:
        st.info("교원 데이터가 없습니다.")
        return
    labels = long["구분"].tolist()
    total = float(long["인원"].sum())
    long["비율"] = long["인원"] / total * 100
    long["label_text"] = long.apply(
        lambda r: f"{r['구분']}\n{int(r['인원']):,}명\n{r['비율']:.1f}%", axis=1
    )

    bar_base = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("구분:N", title="", sort=labels),
            y=alt.Y("인원:Q", title="인원(명)"),
            color=alt.Color("구분:N", scale=_scope_color_scale(labels), legend=None),
            tooltip=[
                alt.Tooltip("구분:N", title="구분"),
                alt.Tooltip("인원:Q", title="인원", format=",.0f"),
                alt.Tooltip("비율:Q", title="비율(%)", format=".1f"),
            ],
        )
    )
    bar_text = bar_base.mark_text(dy=-8, fontSize=13).encode(
        text=alt.Text("인원:Q", format=",.0f")
    )
    bar_chart = (bar_base + bar_text).properties(
        height=320,
        title=f"{year}년 전임·비전임 교원(계) 막대차트",
    )

    pie_base = alt.Chart(long).encode(
        theta=alt.Theta("인원:Q", stack=True),
        color=alt.Color(
            "구분:N",
            scale=_scope_color_scale(labels),
            legend=alt.Legend(title="구분"),
        ),
        order=alt.Order("구분:N", sort="ascending"),
        tooltip=[
            alt.Tooltip("구분:N", title="구분"),
            alt.Tooltip("인원:Q", title="인원", format=",.0f"),
            alt.Tooltip("비율:Q", title="비율(%)", format=".1f"),
        ],
    )
    pie_chart = (
        pie_base.mark_arc(outerRadius=105)
        + pie_base.mark_text(radius=62, size=12, fill="white").encode(
            text="label_text:N"
        )
    ).properties(
        height=320,
        width=320,
        title=f"{year}년 전임·비전임 교원(계) 비율",
    )

    col1, col2 = st.columns([1.3, 1], gap="medium")
    with col1:
        st.altair_chart(bar_chart, width="stretch")
    with col2:
        st.altair_chart(pie_chart, width="content")


def _plot_gender_area(series: pd.DataFrame, value_cols: list[str], y_title: str) -> None:
    plot = series.set_index("연도")[value_cols].copy()
    long = plot.reset_index().melt(id_vars="연도", var_name="지표", value_name="값")
    long = long.dropna(subset=["값"])
    if long.empty:
        st.info("유효한 연도별 데이터가 없습니다.")
        return
    chart = (
        alt.Chart(long)
        .mark_area(opacity=0.55, line={"strokeWidth": 2}, point=True, interpolate="monotone")
        .encode(
            x=alt.X("연도:O", title="연도"),
            y=alt.Y("값:Q", title=y_title, stack=None),
            color=alt.Color(
                "지표:N",
                scale=_gender_color_scale(value_cols),
                legend=alt.Legend(title="지표"),
            ),
            tooltip=[
                alt.Tooltip("연도:O"),
                alt.Tooltip("지표:N"),
                alt.Tooltip("값:Q", format=",.1f" if "%" in y_title else ",.0f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")


def _render_scope_block(
    label: str,
    g: dict[str, float],
    prev: dict[str, float] | None,
    year: int,
    prev_year: int | None,
) -> None:
    st.markdown(f"**{label}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("교원(계)", _fmt_count(g["total"]))
    c2.metric("남", _fmt_count(g["male"]))
    c3.metric("여", _fmt_count(g["female"]))
    c4.metric("남 비율", _fmt_pct(g["male_pct"]))
    # c5.metric("여 비율", _fmt_pct(g["female_pct"]))
    if prev is not None and prev_year is not None:
        d = g["total"] - prev["total"] if g["total"] == g["total"] and prev["total"] == prev["total"] else None
        if d is not None:
            st.caption(f"전년({prev_year}년) 대비 {label} {d:+,.0f}명")

    col_cur, col_prev = st.columns(2)
    with col_cur:
        st.caption(f"{year}년 남·여 구성")
        _plot_pie(g)
    with col_prev:
        if prev_year is not None and prev is not None:
            st.caption(f"{prev_year}년 남·여 구성")
            _plot_pie(prev)
        else:
            st.caption("전년")
            st.info("비교 가능한 전년 데이터가 없습니다.")


def _render_drilldown(
    school_code: str,
    branch: str,
    school_name: str,
    year: int,
    scope_id: str,
    scope_label: str,
) -> None:
    st.subheader("계층별 교원 성별")
    st.caption(f"{year}년 · {scope_label}")

    df = adm.get_school_department_frame(
        school_code, branch, school_name, year, school_name_prefix=SCHOOL_PREFIX
    )
    if df.empty:
        st.warning("데이터가 없습니다.")
        return

    filters: dict[str, str] = {c: adm.DRILL_ALL_LABEL for c in adm.DRILL_COLUMNS}
    cols = st.columns(len(adm.DRILL_COLUMNS))
    for i, col_name in enumerate(adm.DRILL_COLUMNS):
        opts = adm.list_drill_options(df, col_name, filters)
        options = [adm.DRILL_ALL_LABEL, *opts] if opts else [adm.DRILL_ALL_LABEL]
        with cols[i]:
            filters[col_name] = st.selectbox(col_name, options, key=f"prof_drill_{col_name}")

    sub = adm.apply_hierarchy_filters(df, filters)
    resolved = adm.resolve_drill_group_column(filters)

    if resolved is None:
        g = adm.summarize_faculty_gender(sub, scope_id)
        _render_scope_block(scope_label, g, None, year, None)
        return

    group_col, label = resolved
    table = adm.faculty_gender_by_group(sub, group_col, scope_id, label_column=label)
    if table.empty:
        st.info("하위 그룹 데이터가 없습니다.")
        return
    show = table.copy()
    for col in ("교원(계)", "남", "여"):
        show[col] = show[col].apply(_fmt_count)
    for col in ("남(%)", "여(%)"):
        show[col] = show[col].apply(_fmt_pct)
    st.dataframe(show, use_container_width=True, hide_index=True)


# ── 데이터 확인 ──────────────────────────────────────────────────
if not OUTPUT_DIR.is_dir() or not (OUTPUT_DIR / f"{LATEST_YEAR}_data.csv").is_file():
    st.error(f"`{OUTPUT_DIR}` 에 `{LATEST_YEAR}_data.csv` 가 없습니다.")
    st.stop()

schools = _school_options()
if schools.empty:
    st.warning(f"학교명이 「{SCHOOL_PREFIX}」으로 시작하는 데이터가 없습니다.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("대학·캠퍼스")
    labels = [_school_label(r) for _, r in schools.iterrows()]
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    selected_label = st.selectbox("기관", options=labels, index=0)
    row = schools.iloc[label_to_idx[selected_label]]
    school_code = str(row["학교코드"]).strip()
    branch = str(row["본분교"]).strip()
    school_name = str(row["학교명"]).strip()

with col2:
    st.subheader("연도")
    available_years = _available_years()
    if not available_years:
        st.stop()
    default_idx = (
        available_years.index(LATEST_YEAR)
        if LATEST_YEAR in available_years
        else len(available_years) - 1
    )
    selected_year = st.selectbox(
        "분석 연도", options=available_years, index=default_idx, format_func=lambda y: f"{y}년"
    )
    prev_year = selected_year - 1 if selected_year > YEAR_START else None

with col3:
    st.subheader("대계열")
    majors = _major_list(school_code, branch, school_name, selected_year)
    selected_major = st.selectbox(
        "대계열", options=[ALL_MAJOR_LABEL, *majors], index=0
    )

major_label = selected_major if selected_major != ALL_MAJOR_LABEL else "전체"
drill_scope_label = st.selectbox(
    "드릴다운·성별 상세 구분",
    options=list(SCOPE_OPTIONS.keys()),
    index=0,
)
drill_scope_id = SCOPE_OPTIONS[drill_scope_label]

# ── 1. 전임·비전임 요약 ───────────────────────────────────────────
st.divider()
st.subheader(f"교원 현황 요약 ({selected_year}년 · {major_label})")

current_all = _faculty_all(
    school_code, branch, school_name, selected_year, selected_major
)
prev_all: dict[str, dict[str, float]] | None = None
if prev_year is not None:
    prev_all = _faculty_all(
        school_code, branch, school_name, prev_year, selected_major
    )

_plot_scope_compare_bar(current_all, selected_year)

scope_cols = st.columns(len(adm.FACULTY_GENDER_SCOPES), gap="medium")
for col, s in zip(scope_cols, adm.FACULTY_GENDER_SCOPES):
    prev_g = prev_all[s["id"]] if prev_all else None
    with col:
        _render_scope_block(
            s["label"],
            current_all[s["id"]],
            prev_g,
            selected_year,
            prev_year,
        )

# ── 2. 대계열별 표 ─────────────────────────────────────────────────
st.divider()
st.subheader(f"대계열별 교원 현황 ({selected_year}년)")
if not majors:
    st.info("대계열 정보가 없습니다.")
else:
    major_table = _faculty_by_major(
        school_code, branch, school_name, selected_year, tuple(majors)
    )
    show = major_table.copy()
    for col in show.columns:
        if col != "대계열":
            show[col] = pd.to_numeric(major_table[col], errors="coerce").apply(_fmt_count)
    st.dataframe(show, use_container_width=True, hide_index=True)

# ── 3. 드릴다운 ───────────────────────────────────────────────────
st.divider()
_render_drilldown(
    school_code, branch, school_name, selected_year, drill_scope_id, drill_scope_label
)

# ── 4. 연도별 추이 ─────────────────────────────────────────────────
st.divider()
st.subheader(f"연도별 교원 추이 ({major_label})")
st.caption(f"{YEAR_START}~{LATEST_YEAR}년")

total_series = _faculty_total_yearly(school_code, branch, school_name, selected_major)
if not total_series.empty:
    st.markdown("**전임·비전임 교원(계) 추이**")
    chart = (
        alt.Chart(total_series.melt("연도", var_name="구분", value_name="인원"))
        .mark_area(opacity=0.55, line={"strokeWidth": 2}, point=True)
        .encode(
            x="연도:O",
            y=alt.Y("인원:Q", title="인원(명)", stack=None),
            color=alt.Color(
                "구분:N",
                scale=_scope_color_scale(
                    [s["label"] for s in adm.FACULTY_GENDER_SCOPES]
                ),
            ),
            tooltip=[
                alt.Tooltip("연도:O", title="연도"),
                alt.Tooltip("구분:N", title="구분"),
                alt.Tooltip("인원:Q", title="인원", format=",.0f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)

trend_scope_label = st.selectbox(
    "성별 추이 구분",
    options=list(TREND_SCOPE_OPTIONS.keys()),
    index=0,
    help="「전체」는 전임·비전임 교원을 합산합니다.",
)
trend_scope_id = TREND_SCOPE_OPTIONS[trend_scope_label]
gender_series = _faculty_gender_yearly(
    school_code, branch, school_name, trend_scope_id, selected_major
)
if gender_series.empty:
    st.info("연도별 데이터가 없습니다.")
else:
    st.markdown(f"**{trend_scope_label} 남·여 인원 추이**")
    _plot_gender_area(gender_series, ["남", "여"], "인원(명)")
    st.markdown(f"**{trend_scope_label} 남·여 비율 추이**")
    _plot_gender_area(gender_series, ["남(%)", "여(%)"], "비율(%)")

    with st.expander("연도별 수치 표"):
        show = gender_series.copy()
        for col in show.columns:
            if col != "연도":
                show[col] = show[col].apply(
                    lambda v, c=col: _fmt_pct(v) if "%" in c else _fmt_count(v)
                )
        st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("집계 방식"):
    st.markdown(
        """
        | 구분 | 컬럼 |
        |------|------|
        | 전체 | 전임·비전임 교원 합산 |
        | 전임교원(계·남·여) | `전임교원_계`, `전임교원_남`, `전임교원_여` |
        | 비전임교원(계·남·여) | `비전임교원_계`, `비전임교원_남`, `비전임교원_여` |
        """
    )
    st.caption(f"데이터: `output/YYYY_data.csv` · 「{SCHOOL_PREFIX}」시작 기관만 표시.")
