# -*- coding: utf-8 -*-
"""중원대학교(학교명 '중원' 시작) 입시·대계열별 KPI 분석."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib import admissions_data as adm

SCHOOL_PREFIX = "중원"
ALL_MAJOR_LABEL = "전체 (대계열 합산)"
DRILL_ALL = adm.DRILL_ALL_LABEL

st.set_page_config(page_title="중원대학교 IR", layout="wide")
st.title("🎓 중원 대학교  IR(Institutional Research) Analysis")
st.write(
    "대학·캠퍼스를 선택한 뒤 **대계열**별 입시 KPI와 연도별 추이를 확인할 수 있습니다."
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


@st.cache_data(show_spinner="연도별 데이터 로딩…")
def _load_year(year: int) -> pd.DataFrame:
    return adm.load_year_raw(year, school_name_prefix=SCHOOL_PREFIX)


@st.cache_data(show_spinner="대계열 목록 로딩…")
def _major_list(school_code: str, branch: str, school_name: str, year: int) -> list[str]:
    return adm.list_major_categories(
        school_code, branch, year, school_name, school_name_prefix=SCHOOL_PREFIX
    )


@st.cache_data(show_spinner="연도별 KPI 계산…")
def _yearly_series(
    school_code: str,
    branch: str,
    kpi_key: str,
    school_name: str,
    major_category: str | None,
) -> pd.DataFrame:
    major = None if major_category == ALL_MAJOR_LABEL else major_category
    return adm.build_yearly_series(
        school_code,
        branch,
        kpi_key,
        school_name,
        major,
        school_name_prefix=SCHOOL_PREFIX,
    )


def _school_label(row: pd.Series) -> str:
    return f"{row['학교명']} ({row['본분교']})"


def _format_delta(key: str, delta: float | None, unit: str) -> str | None:
    if delta is None:
        return None
    if key in ("신입생_충원율", "입학률", "경쟁률"):
        return f"{delta:+.2f}{unit}"
    return f"{delta:+.1f}%"


def _kpi_column_label(defn: dict[str, str]) -> str:
    unit = defn["unit"]
    if unit in ("%", "배"):
        return f"{defn['label']} ({unit})"
    return f"{defn['label']} ({unit})" if unit else defn["label"]


@st.cache_data(show_spinner="대계열×KPI 표 계산…")
def _build_major_kpi_matrix(
    school_code: str,
    branch: str,
    school_name: str,
    majors: tuple[str, ...],
    year: int,
) -> pd.DataFrame:
    """행=대계열, 열=KPI 현황 표."""
    col_labels = [_kpi_column_label(d) for d in adm.KPI_DEFINITIONS]
    key_by_col = {col: d["key"] for col, d in zip(col_labels, adm.KPI_DEFINITIONS)}

    def _row_for_major(major: str | None, label: str) -> dict[str, str]:
        kpis = adm.school_kpis_for_year(
            school_code,
            branch,
            year,
            school_name,
            major,
            school_name_prefix=SCHOOL_PREFIX,
        )
        row: dict[str, str] = {"대계열": label}
        for col, key in key_by_col.items():
            row[col] = adm.format_kpi_value(key, kpis.get(key, float("nan")))
        return row

    rows = [_row_for_major(None, "전체")]
    for major in majors:
        rows.append(_row_for_major(major, major))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="학과 단위 데이터 로딩…")
def _department_frame(
    school_code: str, branch: str, school_name: str, year: int
) -> pd.DataFrame:
    return adm.get_school_department_frame(
        school_code, branch, school_name, year, school_name_prefix=SCHOOL_PREFIX
    )


def _kpi_row_dict(kpis: dict[str, float], label_col: str, label: str) -> dict[str, str]:
    row: dict[str, str] = {label_col: label}
    for defn in adm.KPI_DEFINITIONS:
        col = _kpi_column_label(defn)
        row[col] = adm.format_kpi_value(defn["key"], kpis.get(defn["key"], float("nan")))
    return row


def _render_drilldown_section(
    school_code: str,
    branch: str,
    school_name: str,
    majors: list[str],
    year: int,
) -> None:
    """대계열 → 중계열 → 소계열 → 학과명 드릴다운."""
    st.markdown("### 계열·학과 드릴다운")
    st.caption(
        "대계열 → 중계열 → 소계열 → 학과명 순으로 선택하면 "
        "하위 단위 KPI 표가 갱신됩니다."
    )

    frame = _department_frame(school_code, branch, school_name, year)
    if frame.empty:
        st.warning("선택한 기관의 학과 데이터가 없습니다.")
        return

    filters: dict[str, str] = {}
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        dae_opts = [DRILL_ALL, *majors] if majors else [DRILL_ALL]
        sel_dae = st.selectbox("대계열", options=dae_opts, key="drill_dae")
        if sel_dae != DRILL_ALL:
            filters["대계열"] = sel_dae

    with c2:
        jung_opts = (
            [DRILL_ALL, *adm.list_drill_options(frame, "중계열", filters)]
            if filters.get("대계열")
            else [DRILL_ALL]
        )
        sel_jung = st.selectbox(
            "중계열",
            options=jung_opts,
            key="drill_jung",
            disabled=not filters.get("대계열"),
        )
        if sel_jung != DRILL_ALL:
            filters["중계열"] = sel_jung

    with c3:
        so_opts = (
            [DRILL_ALL, *adm.list_drill_options(frame, "소계열", filters)]
            if filters.get("중계열")
            else [DRILL_ALL]
        )
        sel_so = st.selectbox(
            "소계열",
            options=so_opts,
            key="drill_so",
            disabled=not filters.get("중계열"),
        )
        if sel_so != DRILL_ALL:
            filters["소계열"] = sel_so

    with c4:
        dept_opts = (
            [DRILL_ALL, *adm.list_drill_options(frame, "학과명", filters)]
            if filters.get("소계열")
            else [DRILL_ALL]
        )
        sel_dept = st.selectbox(
            "학과명",
            options=dept_opts,
            key="drill_dept",
            disabled=not filters.get("소계열"),
        )
        if sel_dept != DRILL_ALL:
            filters["학과명"] = sel_dept

    path_parts = [filters[c] for c in adm.DRILL_COLUMNS if c in filters]
    if path_parts:
        st.info("선택 경로: " + " › ".join(path_parts))
    else:
        st.info("드릴다운 미선택 — 아래는 대계열 단위 요약입니다.")

    subset = adm.apply_hierarchy_filters(frame, filters)

    if filters.get("학과명"):
        st.markdown(f"**학과 상세 KPI** — {filters['학과명']} ({year}년)")
        if subset.empty:
            st.warning("해당 학과 데이터가 없습니다.")
            return
        kpis = adm.sum_kpis_from_frame(subset)
        detail = _kpi_row_dict(kpis, "학과명", filters["학과명"])
        st.dataframe(
            pd.DataFrame([detail]),
            use_container_width=True,
            hide_index=True,
        )
        return

    group_info = adm.resolve_drill_group_column(filters)
    if group_info is None:
        return
    group_col, label_col = group_info

    level_names = {
        "대계열": "대계열별",
        "중계열": "중계열별",
        "소계열": "소계열별",
        "학과명": "학과별",
    }
    st.markdown(f"**{level_names.get(group_col, group_col)} KPI** ({year}년)")

    drill_table = adm.build_group_kpi_table(subset, group_col, label_column=label_col)
    if drill_table.empty:
        st.warning("표시할 하위 항목이 없습니다.")
    else:
        st.dataframe(drill_table, use_container_width=True, hide_index=True)


def _render_kpi_section(
    *,
    school_code: str,
    branch: str,
    school_name: str,
    major_category: str | None,
    analysis_year: int,
    prev_year: int | None,
    section_title: str,
) -> None:
    st.subheader(section_title)
    major = None if major_category == ALL_MAJOR_LABEL else major_category

    year_dfs: dict[int, pd.DataFrame] = {
        analysis_year: _load_year(analysis_year),
    }
    if prev_year is not None:
        year_dfs[prev_year] = _load_year(prev_year)

    kpis_cur = adm.school_kpis_for_year(
        school_code,
        branch,
        analysis_year,
        school_name,
        major,
        cached_raw=year_dfs,
        school_name_prefix=SCHOOL_PREFIX,
    )
    kpis_prev: dict[str, float] = {}
    if prev_year is not None:
        kpis_prev = adm.school_kpis_for_year(
            school_code,
            branch,
            prev_year,
            school_name,
            major,
            cached_raw=year_dfs,
            school_name_prefix=SCHOOL_PREFIX,
        )

    kpi_rows: list[dict] = []
    for defn in adm.KPI_DEFINITIONS:
        key = defn["key"]
        cur = kpis_cur.get(key, float("nan"))
        prev = kpis_prev.get(key, float("nan")) if prev_year else float("nan")
        delta = adm.kpi_delta(cur, prev, key) if prev_year else None
        row = {
            "KPI": defn["label"],
            "kpi_key": key,
            str(analysis_year): adm.format_kpi_value(key, cur),
            "단위": defn["unit"],
            "전년대비": delta,
        }
        if prev_year is not None:
            row[str(prev_year)] = adm.format_kpi_value(key, prev)
        kpi_rows.append(row)
    kpi_df = pd.DataFrame(kpi_rows)

    cols_top = st.columns(4)
    cols_bot = st.columns(4)
    for i, defn in enumerate(adm.KPI_DEFINITIONS):
        key = defn["key"]
        cur = kpis_cur.get(key, float("nan"))
        prev = kpis_prev.get(key, float("nan")) if prev_year else float("nan")
        delta = adm.kpi_delta(cur, prev, key) if prev_year else None
        val_str = adm.format_kpi_value(key, cur)
        unit = defn["unit"]
        display_val = f"{val_str}{unit}" if unit == "%" else f"{val_str} {unit}".strip()
        target = cols_top[i] if i < 4 else cols_bot[i - 4]
        with target:
            st.metric(
                defn["label"],
                display_val,
                _format_delta(key, delta, unit),
            )

    table_cols = ["KPI", str(analysis_year), "단위"]
    if prev_year is not None:
        table_cols.insert(2, str(prev_year))
    display_table = kpi_df[table_cols].copy()
    display_table["전년대비"] = kpi_df.apply(
        lambda r: (
            f"{r['전년대비']:+.2f}{r['단위']}"
            if r["kpi_key"] in ("신입생_충원율", "입학률", "경쟁률")
            and pd.notna(r["전년대비"])
            else (f"{r['전년대비']:+.1f}%" if pd.notna(r["전년대비"]) else "—")
        ),
        axis=1,
    )
    st.dataframe(display_table, use_container_width=True, hide_index=True)


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

col1, col2, col3 = st.columns(3)
with col1:
    # ── 1. 대학교 선택 ─────────────────────────────────────────────
    st.subheader("대학·캠퍼스 선택")
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
    # ── 2. 연도 선택 ─────────────────────────────────────────────
    st.subheader("연도 선택")
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
        help=f"{YEAR_START}~{LATEST_YEAR}년 중 선택. KPI·드릴다운 표는 이 연도 기준입니다.",
    )
    prev_year = selected_year - 1 if selected_year > YEAR_START else None
    if prev_year is not None:
        st.caption(f"전년 대비: {prev_year}년 ↔ {selected_year}년")
    else:
        st.caption(f"{selected_year}년은 비교 가능한 이전 연도 데이터가 없습니다.")

with col3:
    # ── 3. 대계열 선택 ─────────────────────────────────────────────
    st.subheader("대계열 선택")
    majors = _major_list(school_code, branch, school_name, selected_year)
    major_options = [ALL_MAJOR_LABEL, *majors]

    selected_major = st.selectbox(
        "대계열",
        options=major_options,
        index=0,
        help=f"{selected_year}년 데이터 기준. 「전체」는 선택 캠퍼스의 모든 학과를 합산합니다.",
    )

    if majors:
        st.caption(f"{selected_year}년 등록 대계열 {len(majors)}개: {', '.join(majors)}")
    else:
        st.caption(f"{selected_year}년 데이터에 대계열 정보가 없습니다.")

# ── 4. KPI 현황 (선택 연도·전년) ───────────────────────────────
st.divider()
major_label = selected_major if selected_major != ALL_MAJOR_LABEL else "전체"
_yoy_title = (
    f"주요 KPI ({selected_year}년 · {major_label}) 및 전년({prev_year}년) 대비"
    if prev_year
    else f"4. 주요 KPI ({selected_year}년 · {major_label})"
)
_render_kpi_section(
    school_code=school_code,
    branch=branch,
    school_name=school_name,
    major_category=selected_major,
    analysis_year=selected_year,
    prev_year=prev_year,
    section_title=_yoy_title,
)

# ── 5. 대계열별 × KPI 현황 (행=대계열, 열=KPI) ─────────────────
st.divider()
st.subheader(f"대계열별 KPI 현황 ({selected_year}년)")
st.caption("행은 대계열, 열은 주요 KPI입니다. 맨 위 「전체」는 캠퍼스 전체 합산입니다.")

if not majors:
    st.info("대계열 정보가 없어 표를 만들 수 없습니다.")
else:
    major_kpi_table = _build_major_kpi_matrix(
        school_code, branch, school_name, tuple(majors), selected_year
    )
    st.dataframe(major_kpi_table, use_container_width=True, hide_index=True)

    all_kpi_cols = [
        _kpi_column_label(d)
        for d in adm.KPI_DEFINITIONS
        if _kpi_column_label(d) in major_kpi_table.columns
    ]
    count_cols = [
        _kpi_column_label(d)
        for d in adm.KPI_DEFINITIONS
        if d["unit"] == "명" and _kpi_column_label(d) in major_kpi_table.columns
    ]

    if all_kpi_cols:
        chart_src = major_kpi_table[major_kpi_table["대계열"] != "전체"].copy()
        chart_df = chart_src.set_index("대계열")
        for col in all_kpi_cols:
            chart_df[col] = (
                chart_df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .replace("—", None)
            )
            chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")

        if len(count_cols) >= 4:
            st.markdown("**대계열별 모집·지원·입학 (명) — 비교**")
            st.bar_chart(chart_df[count_cols[:4]], stack=False, height=400)

        tab1, tab2, tab3 = st.tabs(["1~3번째 KPI", "4~6번째 KPI", "7~9번째 KPI"])
        with tab1:  
            individual_cols = all_kpi_cols[:3]
            if individual_cols:
                st.markdown("**대계열별 KPI 개별 (1~3번째)**")
                kpi_chart_cols = st.columns(len(individual_cols))
                for i, col in enumerate(individual_cols):
                    with kpi_chart_cols[i]:
                        st.caption(col)
                        st.bar_chart(chart_df[[col]])

        with tab2:
            individual_cols = all_kpi_cols[3:6]
            if individual_cols:
                st.markdown("**대계열별 KPI 개별 (4~6번째)**")
                kpi_chart_cols = st.columns(len(individual_cols))
                for i, col in enumerate(individual_cols):
                    with kpi_chart_cols[i]:
                        st.caption(col)
                        st.bar_chart(chart_df[[col]])
        with tab3:
            individual_cols = all_kpi_cols[6:9]
            if individual_cols:
                st.markdown("**대계열별 KPI 개별 (7~9번째)**")
                kpi_chart_cols = st.columns(len(individual_cols))
                for i, col in enumerate(individual_cols):
                    with kpi_chart_cols[i]:
                        st.caption(col)
                        st.bar_chart(chart_df[[col]])

    _render_drilldown_section(
        school_code, branch, school_name, majors, selected_year
    )

# ── 6. KPI 연도별 추이 (2008~2024) ─────────────────────────────
st.divider()
st.subheader(f"KPI 연도별 추이 ({major_label})")
st.caption("연도별 추이는 선택 연도와 무관하게 전체 기간(2008~2024)을 표시합니다.")

kpi_labels = [d["label"] for d in adm.KPI_DEFINITIONS]
label_to_key = {d["label"]: d["key"] for d in adm.KPI_DEFINITIONS}

selected_kpis = st.multiselect(
    "분석할 KPI",
    options=kpi_labels,
    default=[kpi_labels[0], kpi_labels[5]] if len(kpi_labels) > 5 else kpi_labels[:2],
)

if not selected_kpis:
    st.info("KPI를 하나 이상 선택하세요.")
else:
    for label in selected_kpis:
        key = label_to_key[label]
        series = _yearly_series(school_code, branch, key, school_name, selected_major)
        defn = next(d for d in adm.KPI_DEFINITIONS if d["key"] == key)
        unit = defn["unit"]

        st.markdown(f"**{school_name}** · {major_label} · {label} ({unit})")
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
            show["값"] = show["값"].apply(lambda v, k=key: adm.format_kpi_value(k, v))
            st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("지표 산출 방식"):
    st.latex(
        r"""
        \text{신입생 충원율 (\%)} = \left(
        \frac{\text{정원내 입학자(학부)}}{\text{정원내 모집인원(학부)}}
        \right) \times 100
        """
    )
    st.caption(
        f"데이터: `output/YYYY_data.csv` 중 학교명「{SCHOOL_PREFIX}」시작 행만 사용. "
        "대계열을 선택하면 해당 계열 학과만 합산합니다."
    )
