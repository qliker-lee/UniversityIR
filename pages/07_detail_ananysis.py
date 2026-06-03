import altair as alt
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="University IR Intelligence", layout="wide")

BASE_PATH = Path(__file__).resolve().parent.parent
BILLION_WON = 1_000_000_000

COUNT_COLS = ("학과수", "입학정원", "입학자_학부")
AMOUNT_COLS = ("평균등록금(원)", "교외장학금", "교내장학금", "장학금_총계")
DERIVED_METRIC_COLS = ("인당장학금", "입학률", "실질부담금")
FILTER_COLS = ("학교명", "학제별", "설립별", "지역별")
QUOTA_BIN_COL = "입학정원_구간"
QUOTA_BIN_COUNT = 10
METRIC_CHART_COLS = COUNT_COLS + AMOUNT_COLS + DERIVED_METRIC_COLS
METRIC_CHART_LABELS = {
    "학과수": "학과수",
    "입학정원": "입학정원",
    "입학자_학부": "입학자(학부)",
    "평균등록금(원)": "평균등록금(원)",
    "교외장학금": "교외장학금",
    "교내장학금": "교내장학금",
    "장학금_총계": "장학금 총계",
    "인당장학금": "인당장학금(원)",
    "입학률": "입학률(%)",
    "실질부담금": "실질부담금(등록금-장학금)",
}
MAX_SCHOOLS_ON_CHART = 50


def _read_csv(path: Path) -> pd.DataFrame:
    """UTF-8(BOM) 우선, 실패 시 cp949."""
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


# --- 1. 데이터 클리닝 핵심 함수 (에러 방지용) ---
def _clean_to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def fix_count_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """건수·정원 등 정수 컬럼 변환."""
    for col in columns:
        if col in df.columns:
            df[col] = _clean_to_numeric(df[col]).astype("int64")
    return df


def convert_amounts_to_billion_won(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """금액(원)을 억원 단위로 변환."""
    for col in columns:
        if col in df.columns:
            df[col] = (_clean_to_numeric(df[col]) / BILLION_WON).round(2)
    return df


def _apply_multiselect_filters(
    df: pd.DataFrame, selections: dict[str, list]
) -> pd.DataFrame:
    """선택된 값이 있을 때만 해당 컬럼으로 필터 (미선택 = 전체)."""
    filtered = df
    for col, selected in selections.items():
        if selected and col in filtered.columns:
            filtered = filtered[filtered[col].astype(str).isin(selected)]
    return filtered


def _assign_enrollment_quota_bins(
    df: pd.DataFrame,
    column: str = "입학정원",
    n_bins: int = QUOTA_BIN_COUNT,
) -> tuple[pd.DataFrame, list[str]]:
    """display_df의 입학정원을 분위 기준 n_bins 구간으로 나눈 라벨 컬럼을 추가한다."""
    out = df.copy()
    values = pd.to_numeric(out[column], errors="coerce")
    try:
        _, bin_edges = pd.qcut(values, q=n_bins, duplicates="drop", retbins=True)
    except ValueError:
        _, bin_edges = pd.cut(values, bins=n_bins, retbins=True)

    labels: list[str] = []
    for i in range(len(bin_edges) - 1):
        lo = int(bin_edges[i])
        hi = int(bin_edges[i + 1])
        labels.append(f"{lo:,}~{hi:,}")

    out[QUOTA_BIN_COL] = pd.cut(
        values,
        bins=bin_edges,
        include_lowest=True,
        labels=labels,
    ).astype(str)
    return out, labels


def _drop_pivot_margins(pivot: pd.DataFrame) -> pd.DataFrame:
    """피벗 테이블의 합계 행·열 제거."""
    chart_df = pivot.copy()
    chart_df = chart_df.loc[chart_df.index.astype(str) != "합계"]
    if isinstance(chart_df.columns, pd.MultiIndex):
        keep_cols = [
            c
            for c in chart_df.columns
            if not any(str(v) == "합계" for v in (c if isinstance(c, tuple) else (c,)))
        ]
        chart_df = chart_df[keep_cols]
    else:
        total_cols = [c for c in chart_df.columns if str(c) == "합계"]
        chart_df = chart_df.drop(columns=total_cols, errors="ignore")
    return chart_df


def _pivot_counts_to_long(
    pivot: pd.DataFrame,
    index_name: str = "지역별",
    series_name: str = "학제별",
) -> pd.DataFrame:
    """피벗 → long (melt/stack 없이 직접 변환)."""
    chart_df = _drop_pivot_margins(pivot)
    if chart_df.empty or chart_df.shape[1] == 0:
        return pd.DataFrame()

    records: list[dict] = []
    multi_cols = isinstance(chart_df.columns, pd.MultiIndex)

    for region in chart_df.index:
        if str(region) == "합계":
            continue
        for col in chart_df.columns:
            if multi_cols:
                if any(str(v) == "합계" for v in col):
                    continue
                degree, estab = col[0], col[1]
                records.append(
                    {
                        index_name: region,
                        "학제별": degree,
                        "설립별": estab,
                        "학교수": int(chart_df.loc[region, col]),
                    }
                )
            else:
                if str(col) == "합계":
                    continue
                records.append(
                    {
                        index_name: region,
                        series_name: col,
                        "학교수": int(chart_df.loc[region, col]),
                    }
                )
    return pd.DataFrame(records)


def _pivot_to_chart_long(
    pivot: pd.DataFrame, index_name: str = "지역별", series_name: str = "학제별"
) -> pd.DataFrame:
    """피벗(합계 행·열 제외) → 막대차트용 long 형식."""
    long_df = _pivot_counts_to_long(pivot, index_name, series_name)
    if long_df.empty:
        return pd.DataFrame(columns=[index_name, series_name, "학교수"])
    return long_df


def _pivot_multi_column_to_chart_long(
    pivot: pd.DataFrame, index_name: str = "지역별"
) -> pd.DataFrame:
    """학제별×설립별 등 MultiIndex 열 피벗 → long 형식."""
    long_df = _pivot_counts_to_long(pivot, index_name)
    if long_df.empty:
        return pd.DataFrame(
            columns=[index_name, "학제별", "설립별", "학교수", "범례"]
        )
    if "학제별" in long_df.columns and "설립별" in long_df.columns:
        long_df["범례"] = (
            long_df["학제별"].astype(str) + " · " + long_df["설립별"].astype(str)
        )
    return long_df


def _plot_region_degree_bar(pivot_schools: pd.DataFrame) -> None:
    long_df = _pivot_to_chart_long(pivot_schools)
    if long_df.empty:
        st.info("차트에 표시할 데이터가 없습니다.")
        return

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("지역별:N", title="지역", sort="-y"),
            y=alt.Y("학교수:Q", title="학교 수"),
            color=alt.Color("학제별:N", title="학제"),
            xOffset=alt.XOffset("학제별:N"),
            tooltip=[
                alt.Tooltip("지역별:N", title="지역"),
                alt.Tooltip("학제별:N", title="학제"),
                alt.Tooltip("학교수:Q", title="학교 수", format=",.0f"),
            ],
        )
        .properties(height=380, title="지역별 × 학제별 학교 수")
    )
    st.altair_chart(chart, width="stretch")


def _plot_region_establishment_bar(pivot: pd.DataFrame) -> None:
    long_df = _pivot_to_chart_long(pivot, series_name="설립별")
    if long_df.empty:
        st.info("차트에 표시할 데이터가 없습니다.")
        return

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("지역별:N", title="지역", sort="-y"),
            y=alt.Y("학교수:Q", title="학교 수"),
            color=alt.Color("설립별:N", title="설립"),
            xOffset=alt.XOffset("설립별:N"),
            tooltip=[
                alt.Tooltip("지역별:N", title="지역"),
                alt.Tooltip("설립별:N", title="설립"),
                alt.Tooltip("학교수:Q", title="학교 수", format=",.0f"),
            ],
        )
        .properties(height=380, title="지역별 × 설립별 학교 수")
    )
    st.altair_chart(chart, width="stretch")


def _plot_correlation_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    """두 지표 간의 상관관계 산점도"""
    chart = (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                f"{x_col}:Q",
                title=METRIC_CHART_LABELS.get(x_col, x_col),
            ),
            y=alt.Y(
                f"{y_col}:Q",
                title=METRIC_CHART_LABELS.get(y_col, y_col),
            ),
            color="설립별:N",
            shape="학제별:N",
            tooltip=["학교명", x_col, y_col, "지역별"],
        )
        .properties(
            height=450,
            title=f"{METRIC_CHART_LABELS.get(x_col, x_col)} vs {METRIC_CHART_LABELS.get(y_col, y_col)} 상관관계",
        )
    )
    return chart.interactive()


def _plot_metric_histogram(df: pd.DataFrame, col: str):
    """지표의 분포 확인 (히스토그램)"""
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{col}:Q",
                bin=alt.Bin(maxbins=30),
                title=METRIC_CHART_LABELS.get(col, col),
            ),
            y=alt.Y("count()", title="대학 수"),
            color=alt.value("#636EFA"),
        )
        .properties(height=300, title=f"{METRIC_CHART_LABELS.get(col, col)} 분포")
    )
    return chart


def _plot_group_box_plot(df: pd.DataFrame, group_col: str, val_col: str):
    """그룹별 편차 확인 (박스플롯)"""
    chart = (
        alt.Chart(df)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X(f"{group_col}:N", title=group_col),
            y=alt.Y(
                f"{val_col}:Q",
                title=METRIC_CHART_LABELS.get(val_col, val_col),
            ),
            color=f"{group_col}:N",
        )
        .properties(
            height=400,
            title=f"{group_col}별 {METRIC_CHART_LABELS.get(val_col, val_col)} 편차",
        )
    )
    return chart


def display_metrics(df: pd.DataFrame):
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("총 대학 수", len(df))
    with col2:
        st.metric("대학교 수", len(df[df["학제별"] == "대학교"]))
    with col3:
        st.metric("전문대학 수", len(df[df["학제별"] == "전문대학"]))
    with col4:
        st.metric("국공립 대학 수", len(df[(df["설립별"] == "국립") & (df["학제별"] == "대학교")]))
    with col5:
        st.metric("사립 대학 수", len(df[(df["설립별"] == "사립") & (df["학제별"] == "대학교")]))
    with col6:
        st.metric("국공립 전문대학 수", len(df[(df["설립별"] == "국립") & (df["학제별"] == "전문대학")]))
    with col7:
        st.metric("사립 전문대학 수", len(df[(df["설립별"] == "사립") & (df["학제별"] == "전문대학")]))

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        pivot_schools = pd.pivot_table(
            df.dropna(subset=["지역별", "학제별"]),
            index="지역별",
            columns="학제별",
            values="학교명",
            aggfunc="count",
            fill_value=0,
            margins=True,
            margins_name="합계",
        ).astype(int)

        tab1, tab2 = st.tabs(["지역별 × 학제별 학교수 (차트)", "지역별 × 학제별 학교수 (데이터)"])
        with tab1:
            _plot_region_degree_bar(pivot_schools)
        with tab2:
            st.dataframe(pivot_schools, width="stretch")

    with col2:
        pivot_by_establishment = pd.pivot_table(
            df.dropna(subset=["지역별",  "설립별"]),
            index="지역별",
            columns=["설립별"],
            values="학교명",
            aggfunc="count",
            fill_value=0,
            margins=True,
            margins_name="합계",
        ).astype(int)

        tab1, tab2 = st.tabs(["지역별 × 설립별 학교수 (차트)", "지역별 × 설립별 학교수 (데이터)"])
        with tab1:
            _plot_region_establishment_bar(pivot_by_establishment)
        with tab2:
            st.dataframe(pivot_by_establishment, width="stretch")


def _build_group_amount_statistics(
    df: pd.DataFrame,
    group_cols: str | list[str],
    sort_order: list[str] | None = None,
) -> pd.DataFrame:
    """그룹(복수 디멘전 포함)별 금액 지표 평균 및 학교 수."""
    amount_cols = [c for c in AMOUNT_COLS if c in df.columns]
    cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    if not amount_cols or any(c not in df.columns for c in cols):
        return pd.DataFrame()

    work = df.dropna(subset=cols).copy()
    for col in amount_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    mean_df = work.groupby(cols, as_index=False)[amount_cols].mean().round(0)
    if "학교명" in work.columns:
        counts = (
            work.groupby(cols)["학교명"]
            .nunique()
            .reset_index(name="학교수")
        )
        out = mean_df.merge(counts, on=cols, how="left")
        col_order = [*cols, "학교수", *amount_cols]
    else:
        out = mean_df
        col_order = [*cols, *amount_cols]

    out = out[col_order]
    if sort_order and cols:
        group_key = cols[0]
        order_map = {value: idx for idx, value in enumerate(sort_order)}
        out = (
            out.assign(_sort=out[group_key].map(order_map))
            .sort_values("_sort", na_position="last")
            .drop(columns="_sort")
        )
    else:
        out = out.sort_values(cols)
    return out.reset_index(drop=True)


def _statistics_column_config() -> dict:
    amount_format = dict(format="localized", step=1)
    return {
        "학교수": st.column_config.NumberColumn("학교 수", format="localized", step=1),
        "평균등록금(원)": st.column_config.NumberColumn("평균등록금(원)", **amount_format),
        "교외장학금": st.column_config.NumberColumn("교외장학금", **amount_format),
        "교내장학금": st.column_config.NumberColumn("교내장학금", **amount_format),
        "장학금_총계": st.column_config.NumberColumn("장학금 총계", **amount_format),
    }


def display_statistics(
    df: pd.DataFrame, quota_bin_options: list[str] | None = None
):
    st.subheader("통계 분석")
    st.caption(
        "지역별×학제별, 설립별×학제별, 입학정원 구간×학제별, 입학정원 구간별 "
        "**평균등록금**, **교외·교내 장학금**, **장학금 총계**의 학교 단위 평균입니다."
    )

    column_config = _statistics_column_config()

    tab_region_degree, tab_establishment_degree, tab_quota_degree, tab_quota = st.tabs(
        [
            "지역별 × 학제별",
            "설립별 × 학제별",
            "입학정원 구간 × 학제별",
            "입학정원 구간",
        ]
    )

    group_title_map = {
        "지역별": "지역",
        "학제별": "학제",
        "설립별": "설립",
        QUOTA_BIN_COL: "입학정원 구간",
    }

    with tab_region_degree:
        stats_df = _build_group_amount_statistics(df, ["지역별", "학제별"])
        if stats_df.empty:
            st.info("지역별 × 학제별 기준 통계를 표시할 데이터가 없습니다.")
        else:
            st.dataframe(
                stats_df,
                column_config={
                    "지역별": st.column_config.TextColumn(group_title_map["지역별"]),
                    "학제별": st.column_config.TextColumn(group_title_map["학제별"]),
                    **column_config,
                },
                hide_index=True,
                width="stretch",
            )

    with tab_establishment_degree:
        stats_df = _build_group_amount_statistics(df, ["설립별", "학제별"])
        if stats_df.empty:
            st.info("설립별 × 학제별 기준 통계를 표시할 데이터가 없습니다.")
        else:
            st.dataframe(
                stats_df,
                column_config={
                    "설립별": st.column_config.TextColumn(group_title_map["설립별"]),
                    "학제별": st.column_config.TextColumn(group_title_map["학제별"]),
                    **column_config,
                },
                hide_index=True,
                width="stretch",
            )

    with tab_quota_degree:
        stats_df = _build_group_amount_statistics(
            df, [QUOTA_BIN_COL, "학제별"], sort_order=quota_bin_options
        )
        if stats_df.empty:
            st.info("입학정원 구간 × 학제별 기준 통계를 표시할 데이터가 없습니다.")
        else:
            st.dataframe(
                stats_df,
                column_config={
                    QUOTA_BIN_COL: st.column_config.TextColumn(
                        group_title_map[QUOTA_BIN_COL]
                    ),
                    "학제별": st.column_config.TextColumn(group_title_map["학제별"]),
                    **column_config,
                },
                hide_index=True,
                width="stretch",
            )

    with tab_quota:
        stats_df = _build_group_amount_statistics(
            df, QUOTA_BIN_COL, sort_order=quota_bin_options
        )
        if stats_df.empty:
            st.info("입학정원 구간 기준 통계를 표시할 데이터가 없습니다.")
        else:
            st.dataframe(
                stats_df,
                column_config={
                    QUOTA_BIN_COL: st.column_config.TextColumn(
                        group_title_map[QUOTA_BIN_COL]
                    ),
                    **column_config,
                },
                hide_index=True,
                width="stretch",
            )


def _prepare_school_chart_df(
    df: pd.DataFrame, metrics: list[str]
) -> tuple[pd.DataFrame, str | None]:
    """학교별 차트용 데이터프레임 (정렬·상위 N개교 제한)."""
    chart_df = df[["학교명", *metrics]].copy()
    for col in metrics:
        chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce").fillna(0)

    chart_df = chart_df.sort_values(metrics[0], ascending=False)
    caption = None
    if len(chart_df) > MAX_SCHOOLS_ON_CHART:
        chart_df = chart_df.head(MAX_SCHOOLS_ON_CHART)
        caption = (
            f"학교 수가 많아 {METRIC_CHART_LABELS.get(metrics[0], metrics[0])} "
            f"기준 상위 {MAX_SCHOOLS_ON_CHART}개교만 표시합니다."
        )
    return chart_df, caption


def _value_tooltip_format(col: str) -> str:
    if col == "입학률":
        return ",.2f"
    if col in COUNT_COLS:
        return ",.0f"
    if col in DERIVED_METRIC_COLS:
        return ",.0f"
    if col in AMOUNT_COLS:
        return ",.2f"
    return ",.0f"


def _plot_school_combo_chart(df: pd.DataFrame, metric_left: str, metric_right: str) -> None:
    """학교별 2지표 콤보차트 (좌 Y: 막대, 우 Y: 선, 독립 스케일)."""
    chart_df, caption = _prepare_school_chart_df(df, [metric_left, metric_right])
    if caption:
        st.caption(caption)

    label_left = METRIC_CHART_LABELS.get(metric_left, metric_left)
    label_right = METRIC_CHART_LABELS.get(metric_right, metric_right)
    school_order = chart_df["학교명"].tolist()
    fmt_left = _value_tooltip_format(metric_left)
    fmt_right = _value_tooltip_format(metric_right)

    base = alt.Chart(chart_df).encode(
        x=alt.X(
            "학교명:N",
            title="학교",
            sort=school_order,
            axis=alt.Axis(labelAngle=-45, labelLimit=120),
        ),
    )

    bars = base.mark_bar(color="#4C78A8", opacity=0.85).encode(
        y=alt.Y(
            f"{metric_left}:Q",
            title=label_left,
            axis=alt.Axis(titleColor="#4C78A8", format=fmt_left),
        ),
        tooltip=[
            alt.Tooltip("학교명:N", title="학교"),
            alt.Tooltip(f"{metric_left}:Q", title=label_left, format=fmt_left),
        ],
    )

    line = base.mark_line(
        color="#F58518",
        point=alt.OverlayMarkDef(filled=True, size=60),
    ).encode(
        y=alt.Y(
            f"{metric_right}:Q",
            title=label_right,
            axis=alt.Axis(orient="right", titleColor="#F58518", format=fmt_right),
        ),
        tooltip=[
            alt.Tooltip("학교명:N", title="학교"),
            alt.Tooltip(f"{metric_right}:Q", title=label_right, format=fmt_right),
        ],
    )

    chart = (
        alt.layer(bars, line)
        .resolve_scale(y="independent")
        .properties(
            height=480,
            width=max(600, 18 * len(school_order)),
            title=f"학교별 지표 비교 (좌: {label_left}, 우: {label_right})",
        )
    )
    st.altair_chart(chart, width="stretch")


def _plot_school_metrics_grouped_bar(df: pd.DataFrame, metrics: list[str]) -> None:
    """학교별 선택 지표 막대차트 (가로 그룹 막대)."""
    chart_df, caption = _prepare_school_chart_df(df, metrics)
    if caption:
        st.caption(caption)

    school_order = chart_df["학교명"].tolist()
    long_df = chart_df.melt(id_vars="학교명", var_name="지표", value_name="값")
    long_df["지표"] = long_df["지표"].map(
        lambda c: METRIC_CHART_LABELS.get(c, c)
    )

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            y=alt.Y("학교명:N", title="학교", sort=school_order),
            x=alt.X("값:Q", title="값"),
            color=alt.Color("지표:N", title="지표"),
            yOffset=alt.YOffset("지표:N"),
            tooltip=[
                alt.Tooltip("학교명:N", title="학교"),
                alt.Tooltip("지표:N", title="지표"),
                alt.Tooltip("값:Q", title="값", format=",.0f"),
            ],
        )
        .properties(
            height=max(400, 22 * len(school_order)),
            title="학교별 지표 비교",
        )
    )
    st.altair_chart(chart, width="stretch")


def _plot_school_metrics_bar(df: pd.DataFrame, metrics: list[str]) -> None:
    """학교별 지표 차트 (2개 선택 시 듀얼 Y축 콤보, 그 외 그룹 막대)."""
    if len(metrics) == 2:
        _plot_school_combo_chart(df, metrics[0], metrics[1])
    else:
        _plot_school_metrics_grouped_bar(df, metrics)


def detail_analysis(df: pd.DataFrame):
    st.subheader("상세 분석")

    available_metrics = [c for c in METRIC_CHART_COLS if c in df.columns]
    selected_metrics = st.multiselect(
        "측정값 선택",
        options=available_metrics,
        default=available_metrics[:3] if len(available_metrics) >= 3 else available_metrics,
        format_func=lambda c: METRIC_CHART_LABELS.get(c, c),
    )

    if not selected_metrics:
        st.info("차트에 표시할 측정값을 하나 이상 선택해 주세요.")
        return

    if len(selected_metrics) == 2:
        st.caption(
            "2개 지표 선택: 첫 번째는 좌측 Y축(막대), 두 번째는 우측 Y축(선) 콤보 차트로 표시됩니다."
        )
    elif len(selected_metrics) > 2:
        st.caption("3개 이상 선택 시 가로 그룹 막대 차트로 표시됩니다. 콤보 차트는 지표 2개만 선택하세요.")

    _plot_school_metrics_bar(df, selected_metrics)


@st.cache_data
def load_and_merge_data():
    output_dir = BASE_PATH / "output"

    # [입시 데이터 로드]
    df_adm = _read_csv(output_dir / "2024_data.csv")

    # 대학 마스터 정보 (학제/설립/지역)
    # 2024_data.csv 실제 컬럼: 학제, 설립, 시도
    df_master = (
        df_adm[["학교명", "학제", "설립", "시도"]]
        .dropna(subset=["학교명"])
        .drop_duplicates("학교명")
        .rename(columns={"학제": "학제별", "설립": "설립별", "시도": "지역별"})
    )

    # 학교별 합계 집계
    df_adm_sum = (
        df_adm.groupby("학교명")
        .agg(
            {
                "학과수_전체": "sum",
                "입학정원_학부_계": "sum",
                "정원내_입학자_학부_계": "sum",
            }
        )
        .reset_index()
        .rename(
            columns={
                "학과수_전체": "학과수",
                "입학정원_학부_계": "입학정원",
                "정원내_입학자_학부_계": "입학자_학부",
            }
        )
    )

    # [등록금 데이터 로드]
    df_tuition = _read_csv(output_dir / "한국장학재단_대학별 평균등록금_20250519.csv")
    df_tuition.columns = [c.replace('"', '').strip() for c in df_tuition.columns]
    df_tuition = df_tuition.rename(columns={"대학명": "학교명"})
    # 등록금 파일에는 학제/설립/지역/입학정원 컬럼도 있어 충돌(suffix) 방지를 위해
    # 본 대시보드에서 필요한 컬럼만 유지한다.
    keep_tuition_cols = [c for c in ["학교명", "평균등록금(원)"] if c in df_tuition.columns]
    df_tuition = df_tuition[keep_tuition_cols].copy()

    # [장학금 수혜 현황 데이터 로드]
    df_scholarship = _read_csv(
        output_dir / "한국장학재단_대학별 장학금 수혜 현황_20250831.csv"
    )
    df_scholarship.columns = [c.replace('"', '').strip() for c in df_scholarship.columns]
    df_scholarship = df_scholarship.rename(
        columns={
            "교외장학금 소계(원)": "교외장학금",
            "교내장학금 소계(원)": "교내장학금",
            "총계(원)": "장학금_총계",
        }
    )

    # [데이터 병합] 학교명 기준
    merged = pd.merge(df_master, df_adm_sum, on="학교명", how="inner")
    merged = pd.merge(merged, df_tuition, on="학교명", how="left")
    merged = pd.merge(
        merged,
        df_scholarship[["학교명", "교외장학금", "교내장학금", "장학금_총계"]],
        on="학교명",
        how="left",
    )

    merged = fix_count_columns(merged, COUNT_COLS)
    return merged


# --- 2. 메인 화면 구현 ---
st.title("🎓 대학 IR 통합 데이터 대시보드")
st.write("대학 입시, 등록금, 장학금 수혜 현황을 통합 데이터로 분석합니다. (입학정원 > 0 이상인 현황입니다.)")

with st.sidebar:
    if st.button("캐시 새로고침", help="코드 수정 후 차트 오류 시 클릭"):
        st.cache_data.clear()
        st.rerun()

df = load_and_merge_data()

if not df.empty:
    display_df = df[df["입학정원"] > 0].copy()
    enrolled = pd.to_numeric(display_df["입학자_학부"], errors="coerce")
    quota = pd.to_numeric(display_df["입학정원"], errors="coerce")
    scholarship_total = pd.to_numeric(display_df["장학금_총계"], errors="coerce")
    tuition_avg = pd.to_numeric(display_df["평균등록금(원)"], errors="coerce")

    display_df["인당장학금"] = scholarship_total.div(enrolled.where(enrolled > 0))
    display_df["입학률"] = (enrolled.div(quota.where(quota > 0)) * 100).round(2)
    display_df["실질부담금"] = (tuition_avg - display_df["인당장학금"]).round(0)

    st.success(
        f"데이터 로드 성공: 총 {len(display_df):,}개 대학 "
        "(입시·등록금·장학금 수혜 현황 매칭)."
    )

    preview_cols = [
        "학교명",
        "학제별",
        "설립별",
        "지역별",
        *COUNT_COLS,
        *AMOUNT_COLS,
        "인당장학금",
    ]
    count_format = dict(format="localized", step=1)
    amount_format = dict(format="localized", step=1)

    display_df, quota_bin_options = _assign_enrollment_quota_bins(display_df)

    display_metrics(display_df)
    display_statistics(display_df, quota_bin_options=quota_bin_options)

    st.divider()
    filter_ui_cols = st.columns(len(FILTER_COLS) + 1)
    filter_selections: dict[str, list] = {}
    for ui_col, filter_col in zip(filter_ui_cols, FILTER_COLS):
        options = sorted(display_df[filter_col].dropna().astype(str).unique())
        with ui_col:
            filter_selections[filter_col] = st.multiselect(
                filter_col,
                options=options,
                default=[],
                placeholder="전체",
            )
    with filter_ui_cols[-1]:
        filter_selections[QUOTA_BIN_COL] = st.multiselect(
            "입학정원 구간",
            options=quota_bin_options,
            default=[],
            placeholder="전체",
            help=f"입학정원을 분위 기준 {QUOTA_BIN_COUNT}개 구간으로 나눈 값입니다.",
        )

    filtered_df = _apply_multiselect_filters(display_df, filter_selections)
    st.caption(f"표시: {len(filtered_df):,}개 / 전체 {len(display_df):,}개")

    st.dataframe(
        filtered_df[[c for c in preview_cols if c in filtered_df.columns]],
        column_config={
            "학교명": st.column_config.TextColumn("학교명", width="medium"),
            "학제별": st.column_config.TextColumn("학제"),
            "설립별": st.column_config.TextColumn("설립"),
            "지역별": st.column_config.TextColumn("지역"),
            "학과수": st.column_config.NumberColumn("학과수", **count_format),
            "입학정원": st.column_config.NumberColumn("입학정원", **count_format),
            "입학자_학부": st.column_config.NumberColumn("입학자 수", **count_format),
            "평균등록금(원)": st.column_config.NumberColumn("평균등록금(원)", **amount_format),
            "교외장학금": st.column_config.NumberColumn("교외장학금", **amount_format),
            "교내장학금": st.column_config.NumberColumn("교내장학금", **amount_format),
            "장학금_총계": st.column_config.NumberColumn("장학금_총계", **amount_format),
            "인당장학금": st.column_config.NumberColumn(
                "인당장학금", help="장학금_총계 ÷ 입학자_학부", **amount_format
            ),
        },
        hide_index=True,
        width="stretch",
    )

    # 요청 기능: 상관분석(산점도) / 분포(히스토그램) / 편차(박스플롯)
    if filtered_df.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
    else:
        tabs_main = st.tabs(["📊 통합 대시보드", "🔍 상세 상관분석", "📈 분포 및 편차"])

        with tabs_main[0]:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("총 분석 대학", f"{len(filtered_df)}개")
            col2.metric("평균 입학률", f"{filtered_df['입학률'].mean():.1f}%")
            col3.metric(
                "평균 등록금",
                f"{filtered_df['평균등록금(원)'].mean()/10000:.0f}만원",
            )
            col4.metric(
                "평균 인당 장학금",
                f"{filtered_df['인당장학금'].mean()/10000:.0f}만원",
            )

            st.divider()
            st.dataframe(
                filtered_df[
                    [
                        "학교명",
                        "지역별",
                        "설립별",
                        "입학률",
                        "평균등록금(원)",
                        "인당장학금",
                        "실질부담금",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        with tabs_main[1]:
            st.subheader("지표 간 상관관계 분석")
            sc1, sc2 = st.columns([1, 3])
            with sc1:
                x_axis = st.selectbox(
                    "X축 지표",
                    ["입학정원", "평균등록금(원)", "학과수"],
                    index=1,
                )
                y_axis = st.selectbox(
                    "Y축 지표",
                    ["인당장학금", "입학률", "실질부담금", "장학금_총계"],
                    index=0,
                )
                st.info("산점도를 통해 두 지표 간의 경향성을 파악할 수 있습니다.")
                st.write("상관관계 산점도 (Scatter Plot):")

                st.write("예: '등록금이 비싼 대학이 장학금을 더 많이 주는가?' 혹은 '입학정원이 큰 대학이 입학률이 높은가?'와 같은 가설을 시각적으로 검증할 수 있습니다.")

                st.write("색상과 모양을 통해 설립별(국공립/사립), 학제별(대학/전문대) 차이를 동시에 볼 수 있습니다.")

            with sc2:
                scatter_chart = _plot_correlation_scatter(
                    filtered_df, x_axis, y_axis
                )
                st.altair_chart(scatter_chart, width="stretch")

        with tabs_main[2]:
            st.subheader("지표 분포 및 그룹별 비교")
            dist_col, box_col = st.columns(2)

            selected_target = st.selectbox(
                "분석 대상 지표 선택",
                ["평균등록금(원)", "인당장학금", "입학률", "실질부담금"],
            )

            with dist_col:
                st.altair_chart(
                    _plot_metric_histogram(filtered_df, selected_target),
                    width="stretch",
                )

                st.write("히스토그램 (Histogram):")
                st.write("우리나라 대학들의 등록금이 특정 구간에 얼마나 몰려 있는지, 장학금 혜택이 상향 평준화되어 있는지 등의 데이터 분포 밀도를 확인할 수 있습니다.")


            with box_col:
                group_target = st.radio(
                    "비교 그룹", ["지역별", "설립별", "학제별"], horizontal=True
                )
                st.altair_chart(
                    _plot_group_box_plot(filtered_df, group_target, selected_target),
                    width="stretch",
                )

                st.write("박스플롯 (Box Plot):")
                st.write("단순 평균만 보는 것이 아니라, 지역별/설립별로 지표의 편차(최대, 최소, 중앙값, 이상치)를 확인할 수 있습니다.")
                st.write("예를 들어, 수도권 대학과 비수도권 대학 간의 등록금 격차나 편차 범위를 한눈에 비교하기에 가장 좋은 그래프입니다.")
    if not filtered_df.empty:
        detail_analysis(filtered_df)
else:
    st.error("데이터 파일을 읽을 수 없습니다. 파일명과 인코딩(cp949)을 확인해주세요.")