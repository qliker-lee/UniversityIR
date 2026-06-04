# -*- coding: utf-8 -*-
"""
DataSense IR Module - University Enrollment Risk Matrix
Qliker, 2026.05.29 Version 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

PAGES_DIR = Path(__file__).resolve().parent
FEATURE_GUIDE_MD = PAGES_DIR / "03_risk diagnosis system.md"

# --- 1. 페이지 초기 설정 ---
st.set_page_config(
    page_title="대학 입학수요위험도 진단 시스템",
    page_icon="⚠️",
    layout="wide"
)

# --- 2. 가상 마스터 데이터 생성 (국내 주요 권역별 지표 모사) ---
@st.cache_data
def load_enrollment_risk_data():
    regions = ['강원권', '충북권', '충남/대전권', '전북권', '전남/광주권', '경북/대구권', '경남/부산권', '수도권']
    
    # 지역별 인구 및 학령 유출 원천 데이터 구조화
    data = {
        '권역': regions,
        '출생아수_변동률': [-4.2, -2.1, -1.5, -5.3, -4.8, -5.0, -3.9, 0.5],       # 전년 대비 %
        '고교졸업자_수': [12000, 14500, 21000, 15000, 17500, 23000, 28000, 85000], 
        '청년_순이동자수': [-1200, -850, 400, -1900, -2100, -2500, -1800, 9800],   # 수도권 유입 대응
        '청년유출률': [12.5, 9.2, 4.1, 15.4, 16.8, 14.2, 11.0, -2.5],             # %
        
        # 대학 자체 변수 (분석 대상 가상 대학 스펙)
        '대학_자체_충원율': [88.5, 94.2, 97.0, 82.1, 85.4, 89.0, 91.2, 100.0],
        '입시_경쟁률': [4.2, 6.5, 7.8, 3.1, 3.8, 4.5, 5.2, 11.5],
        '지역내_대학수': [12, 10, 18, 11, 14, 19, 22, 45],
        '등록금_재정의존도': [72.5, 68.0, 65.2, 78.4, 76.0, 71.2, 69.5, 48.0]        # 전체 재정 중 등록금 %
    }
    
    df = pd.DataFrame(data)
    
    # ------------------------------------------------------------------
    # 3. 입학수요위험도 알고리즘 연산 가동 (4대 인자 표준화 후 가중치 결합)
    # ------------------------------------------------------------------
    
    # 인자 1: 지역 학령인구 감소 위험도 (출생아 감소 및 청년유출이 높을수록 위험)
    df['인자_학령인구위험'] = (df['출생아수_변동률'] * -5) + (df['청년유출률'] * 3)
    
    # 인자 2: 대학 자체 경쟁력 위험도 (충원율과 경쟁률이 낮을수록 위험 -> 역산)
    df['인자_자체경쟁력위험'] = (100 - df['대학_자체_충원율']) * 2.5 + (15 - df['입시_경쟁률']) * 3
    
    # 인자 3: 지역 내 경쟁 강도 (지역 내 대학 수가 많고 고교 졸업자가 적을수록 위험)
    df['인자_경쟁강도위험'] = (df['지역내_대학수'] / df['고교졸업자_수'] * 100000)
    
    # 인자 4: 대학 규모 의존성 위험 (등록금 의존도가 높을수록 위험)
    df['인자_규모의존위험'] = df['등록금_재정의존도'] * 1.2

    # 각 인자별 0~25점 정규화 연산 (Max 100점 만점 구조 설계)
    for col in ['인자_학령인구위험', '인자_자체경쟁력위험', '인자_경쟁강도위험', '인자_규모의존위험']:
        min_v = df[col].min()
        max_v = df[col].max()
        df[col+'_점수'] = ((df[col] - min_v) / (max_v - min_v) * 25).round(1)
        
    # 최종 입학수요위험도 도출
    df['입학수요위험도'] = (
        df['인자_학령인구위험_점수'] + 
        df['인자_자체경쟁력위험_점수'] + 
        df['인자_경쟁강도위험_점수'] + 
        df['인자_규모의존위험_점수']
    ).round(1)
    
    return df


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

# --- 4. UI 및 시각화 파트 ---
def main():
    st.title("⚠️ 지역 인구 기반 대학 입학수요위험도 진단 시스템")
    st.caption("지역별 학령인구 변동성 및 대학 재정/경쟁력 지표를 결합한 통합 시점 리스크 시뮬레이터 (가상의 데이터임)")
    st.caption("⚠️ 향후 KOSIS(국가통계 연계) 데이터 기반으로 생성된 실제 데이터로 변경 예정")
    st.write("---")
    
    with st.expander("🔍 데이터 소스 및 기능 설명서 (risk diagnosis system.md)", expanded=False):
        st.markdown(_load_pages_markdown(str(FEATURE_GUIDE_MD)))

        
    df = load_enrollment_risk_data()
    
    # ------------------------------------------------------------------
    # TOP 레벨: 위험 징후 알림 및 KPI 스코어 요약
    # ------------------------------------------------------------------
    highest_risk_row = df.loc[df['입학수요위험도'].idxmax()]
    lowest_risk_row = df.loc[df['입학수요위험도'].idxmin()]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="📈 전국 최고 위험 권역", 
            value=f"{highest_risk_row['권역']}", 
            delta=f"위험도: {highest_risk_row['입학수요위험도']}점", 
            delta_color="inverse"
        )
    with col2:
        st.metric(
            label="📉 전국 최저 위험 권역", 
            value=f"{lowest_risk_row['권역']}", 
            delta=f"위험도: {lowest_risk_row['입학수요위험도']}점",
            delta_color="normal"
        )
    with col3:
        st.metric(
            label="📋 분석 대상 권역 수", 
            value=f"{len(df)} 개 권역",
            delta="인구 통계 매핑 완료"
        )
        
    st.write("")
    
    # 레이아웃 분할
    left_chart, right_chart = st.columns([2, 2])
    
    with left_chart:
        st.markdown("#### 📊 권역별 입학수요위험도 정량 비교")
        # 위험도 정렬 후 바 차트 바인딩
        df_sorted = df.sort_values(by='입학수요위험도', ascending=True)
        
        # 위험도 점수에 따른 고대비 색상 바 조절
        fig_bar = px.bar(
            df_sorted,
            x='입학수요위험도',
            y='권역',
            orientation='h',
            text='입학수요위험도',
            color='입학수요위험도',
            color_continuous_scale='Reds',
            labels={'입학수요위험도': '통합 위험도 점수 (0 ~ 100)'},
            height=500
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(plot_bgcolor='#fdfdfd')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with right_chart:
        st.markdown("#### 🎯 핵심 리스크 인자 구성 비율 심층 분석")
        selected_region = st.selectbox("구조 분석 대상 권역 선택", options=df['권역'].unique())
        region_detail = df[df['권역'] == selected_region].iloc[0]
        
        # 방사형(Radar) 차트를 활용한 4대 리스크 세부 요인 시각화
        categories = ['학령인구감소', '자체경쟁력부족', '지역경쟁강도', '규모의존성(재정)']
        scores = [
            region_detail['인자_학령인구위험_점수'],
            region_detail['인자_자체경쟁력위험_점수'],
            region_detail['인자_경쟁강도위험_점수'],
            region_detail['인자_규모의존위험_점수']
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=scores + [scores[0]], # 순환 구조 마감
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(214, 39, 40, 0.2)',
            line=dict(color='#d62728', width=2),
            name=selected_region
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 25])
            ),
            showlegend=False,
            height=450
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ------------------------------------------------------------------
    # 하단 탭: 원천 데이터 뷰어 및 리스크 제언 컨텍스트
    # ------------------------------------------------------------------
    st.write("---")
    st.markdown("### 🔍 리스크 진단 상세 데이터 시트")
    
    # 보기 편하게 가독성 높은 컬럼명 변환 후 매핑
    display_df = df.copy()
    display_df.columns = [
        '권역', '출생아수 변동률(%)', '고교졸업자 수(명)', '청년 순이동자(명)', '청년유출률(%)',
        '대학 자체 충원율(%)', '입시 경쟁률(:1)', '지역내 대학 수', '등록금 재정의존도(%)',
        '학령인구점수', '자체경쟁력점수', '경쟁강도점수', '규모의존점수',
        '종합 위험 지수', '학령인구위험_점수', '자체경쟁력위험_점수', '경쟁강도위험_점수', '규모의존위험_점수'
    ]
    
    st.dataframe(
        display_df[[
            '권역', '종합 위험 지수', '출생아수 변동률(%)', '고교졸업자 수(명)', 
            '청년 순이동자(명)', '청년유출률(%)', '대학 자체 충원율(%)', '등록금 재정의존도(%)'
        ]].sort_values(by='종합 위험 지수', ascending=False), 
        use_container_width=True
    )

if __name__ == "__main__":
    main()