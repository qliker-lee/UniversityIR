# -*- coding: utf-8 -*-
"""
JWU Institutional Research (IR) Strategic Dashboard
Designed for University Decision Support Systems
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

PAGES_DIR = Path(__file__).resolve().parent
FEATURE_GUIDE_MD = PAGES_DIR / "09_Strategic IR Dashboard.md"

# --- 1. 기본 페이지 설정 (다크 모드 고대비 대응용 패스텔 & 헤드룸 스타일) ---
st.set_page_config(
    page_title="중원대학교(JWU) IR 전략 의사결정 시스템",
    page_icon="🏛️",
    layout="wide"
)

# --- 스타일 사용자 지정 (Streamlit 기본 테마 보정) ---
st.markdown("""
    <style>
    .reportview-container { background: #fdfdfd; }
    h1, h2, h3 { color: #0f2d59 !important; font-family: 'Malgun Gothic', sans-serif; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; }
    </style>
""", unsafe_allow_html=True)


# --- 2. 가상 IR 마스터 데이터 생성 (중원대 실제 공시 지표 추세 반영) ---
@st.cache_data
def get_jwu_ir_mock_data():
    # 2025~2026 공시 및 기관 현황 기반 데이터 셋 프레임워크
    departments = [
        '항공운항학과', '의료보건계열', '드론봇군사학과', '스포츠지도학과', 
        '컴퓨터공학과', '경영학과', '바이오메디컬학과', '융합디자인학과'
    ]
    
    data = {
        '학과명': departments,
        '계열': ['공학/항공', '의료보건', '인문사회', '예체능', '공학/항공', '인문사회', '자연과학', '예체능'],
        '입학정원': [40, 120, 35, 80, 60, 50, 45, 40],
        '실입학자': [40, 118, 35, 76, 54, 42, 41, 35],
        '입시경쟁률': [12.4, 10.8, 9.6, 8.5, 7.2, 5.4, 6.8, 6.1],
        '평균등록금_천원': [7816, 7801, 6372, 7436, 7816, 6372, 7801, 7436],
        '인당장학금_천원': [4100, 3950, 3849, 3600, 3750, 3800, 3900, 3550],
        '산학연구과제수': [8, 15, 6, 4, 12, 3, 11, 2]
    }
    df = pd.DataFrame(data)
    # 계산 유도 파생 지표 수립
    df['신입생충원율'] = (df['실입학자'] / df['입학정원'] * 100).round(1)
    df['실질등록금_천원'] = df['평균등록금_천원'] - df['인당장학금_천원']
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


# --- 3. 컨트롤러 메인 로직 가동 ---
def main():
    st.title("🏛️ 중원대학교(JWU) 전략적 IR 대시보드")
    st.caption("대학 정량 지표 데이터 모델링 기반 총장단 및 기획처 의사결정 지원 시스템")
    st.write("---")
    
    with st.expander("🔍 데이터 소스 및 기능 설명서 (09_Strategic IR Dashboard.md)", expanded=False):
        st.markdown(_load_pages_markdown(str(FEATURE_GUIDE_MD)))
    df = get_jwu_ir_mock_data()
    
    # [사이드바 필터 그룹 컨텍스트]
    st.header("🧭 IR 분석 영역 필터")
    selected_fields = st.multiselect(
        "분석 대상 학과 계열",
        options=df['계열'].unique(),
        default=df['계열'].unique()
    )
    
    filtered_df = df[df['계열'].isin(selected_fields)]

    # --- TOP 레벨 상단 마스터 경영 KPI 스코어보드 ---
    st.markdown("### 📊 대학 핵심 운영 현황 (University Top KPIs)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_capacity = filtered_df['입학정원'].sum()
    total_enrolled = filtered_df['실입학자'].sum()
    avg_fulfillment = (total_enrolled / total_capacity * 100) if total_capacity > 0 else 0
    avg_competition = filtered_df['입시경쟁률'].mean()
    net_tuition_avg = filtered_df['실질등록금_천원'].mean() * 1000
    total_research = filtered_df['산학연구과제수'].sum()
    
    kpi1.metric("평균 신입생 충원율", f"{avg_fulfillment:.1f} %", delta=f"{avg_fulfillment-95.0:.1f} % vs 목표")
    kpi2.metric("평균 입시 경쟁률", f"{avg_competition:.2f} : 1")
    kpi3.metric("학생 실질 연간등록금 부담액", f"{net_tuition_avg/10000:.0f} 만원")
    kpi4.metric("활성 산학연구 과제 수", f"{total_research} 건", delta="▲ 3건 상반기 대비")
    
    st.write("")

    # --- 탭 구조 설계 (파트 분리) ---
    tab1, tab2, tab3 = st.tabs(["🚀 모집 경쟁력 집중 진단", "💰 학과 계열별 재정 & 장학금 포트폴리오", "🧪 특성화 연구 산학 매트릭스"])

    # --- Tab 1: 모집 경쟁력 진단 ---
    with tab1:
        st.subheader("학과별 충원율 및 경쟁률 크로스 분석")
        st.write("충원율이 낮으면서 경쟁률마저 저하되는 '위험 학과(At Risk)' 도출을 위한 매트릭스")
        
        # 경쟁률 대 충원율 분석 시각화
        fig1 = px.scatter(
            filtered_df,
            x='입시경쟁률',
            y='신입생충원율',
            size='입학정원',
            color='계열',
            hover_name='학과명',
            text='학과명',
            labels={'입시경쟁률': '입시 경쟁률 (X : 1)', '신입생충원율': '신입생 충원율 (%)'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        # 텍스트 위치 조절 및 가이드 가로/세로 기준선(경영 방어선) 추가
        fig1.update_traces(textposition='top center')
        fig1.add_hline(y=95.0, line_dash="dash", line_color="black", annotation_text="충원율 관리 방어선(95%)")
        fig1.add_vline(x=df['입시경쟁률'].mean(), line_dash="dash", line_color="gray", annotation_text="평균 경쟁률")
        
        fig1.update_layout(height=500, plot_bgcolor='#fcfcfc')
        st.plotly_chart(fig1, use_container_width=True)

    # --- Tab 2: 재정 및 장학 수혜 구조 분석 ---
    with tab2:
        st.subheader("계열별 명목 등록금 vs 장학금 지급률 비교 분석")
        st.write("학생 1인당 실제로 부담하는 실질 등록금(Net Tuition)의 구조적 밸런스를 점검합니다.")
        
        # 가공 바 플롯 구조화
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=filtered_df['학과명'],
            y=filtered_df['평균등록금_천원'],
            name='명목 평균 등록금',
            marker_color='#1f77b4'
        ))
        fig2.add_trace(go.Bar(
            x=filtered_df['학과명'],
            y=filtered_df['인당장학금_천원'],
            name='1인당 수혜 장학금',
            marker_color='#aec7e8'
        ))
        fig2.add_trace(go.Scatter(
            x=filtered_df['학과명'],
            y=filtered_df['실질등록금_천원'],
            name='실질 등록금 부담액',
            line=dict(color='#ff7f0e', width=3, dash='dot')
        ))
        
        fig2.update_layout(
            barmode='group',
            height=500,
            xaxis_title="학과명",
            yaxis_title="금액 (단위: 천원)",
            legend_orientation="h",
            legend_y=1.1
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 3: 산학 협력 및 연구소 성과 성과 분석 ---
    with tab3:
        st.subheader("중원대 특성화 연구 인프라 및 과제 현황 수주 스펙")
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            # 계열별 산학협력 연구과제 분포 트리맵(Treemap) 구성
            fig3 = px.treemap(
                filtered_df,
                path=['계열', '학과명'],
                values='산학연구과제수',
                color='산학연구과제수',
                color_continuous_scale='Blues',
                title="학과 계열별 산학협력 활성화 비중 포트폴리오"
            )
            st.plotly_chart(fig3, use_container_width=True)
            
        with col_right:
            st.markdown("""
            **💡 IR 연구원 코멘트 및 전략 제언**
            * **보건·의료 및 항공 특성화 강화**: '의료보건계열'과 '항공운항학과'가 전체 모집 충원율 상승을 견인 중이며 외부 산학협력 수주액 및 연구 과제 참여 빈도 역시 상위 지표를 마크하고 있음.
            * **리스크 헤징**: 일부 예체능 및 인문사회 계열 학과의 충원율 방어선(95%) 이탈 징후가 포착되므로 해당 학과들을 **'ESG경영센터'** 등의 교내 부설연구소 연구 프로젝트와 융합 연계하여 취업 연계형 산학 커리큘럼으로의 개편(학과 구조 고도화)이 시급함.
            """)
            
            st.markdown("##### 📌 원천 분석 데이터 뷰어")
            st.dataframe(filtered_df[['학과명', '계열', '신입생충원율', '입시경쟁률', '실질등록금_천원', '산학연구과제수']], height=200)

if __name__ == "__main__":
    main()