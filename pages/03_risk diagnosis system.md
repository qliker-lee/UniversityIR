# 대학 입학수요위험도 진단 시스템 (University Enrollment Risk Matrix)
**사용자 및 개발자용 공식 프로그램 설명서**

> **작성자:** Qliker  
> **최종 수정일:** 2026.06.01  
> **시스템 버전:** v1.0 (지역 인구 지표 및 대학 재정 건전성 통합 모듈)

---

## 1. 시스템 개요 (Overview)

**대학 입학수요위험도 진단 시스템**은 학령인구 감소와 청년 인구의 수도권 유출 등 급격한 환경 변화 직격탄을 맞고 있는 대학교 기획처 및 IR(기관연구) 센터를 위한 정량적 위기 진단 솔루션입니다. 

본 시스템은 대학 외부의 **지역 정주 환경(인구 구조)** 지표와 대학 내부의 **운영 경쟁력 및 재정 구조** 지표를 결합하는 표준 알고리즘을 통해, 각 권역별/대학별 위기 징후를 0~100점의 점수로 환산하고 이를 입포테인먼트 대시보드로 시각화합니다.

---

## 2. 입학수요위험도 데이터 모델링 구조 (Data Modeling)

시스템의 정합성과 다차원 분석을 위해 입력 데이터를 다음 4대 지표로 표준화(Min-Max Scaling)하여 복합 연산을 수행합니다. 각 인자는 **최대 25점**을 가집니다.

1. **지역 학령인구 감소 지표 (Demographic Weight)**
   * **주요 요소:** 출생아 수 변동률, 고교 졸업자 수 추세, 지역 청년 유출률 및 인구 순이동
   * **설명:** 대학이 위치한 배후 지역의 학령 자원 고갈 속도와 타 지역(수도권 등)으로의 인구 이탈 심각성을 측정합니다.
2. **대학 자체 경쟁력 지표 (Institutional Power)**
   * **주요 요소:** 최근 신입생 충원율, 수시/정시 입시 경쟁률 역산 데이터
   * **설명:** 외부 요인과 별개로 대학 자체의 브랜드 및 학과 매력도를 평가하여 학생 모집 능력을 진단합니다.
3. **지역 내 경쟁 강도 지표 (Regional Rivalry)**
   * **주요 요소:** 동일 권역 내 타 대학교 수, 권역 총 정원 대비 고교 졸업자 수 비율
   * **설명:** 제한된 학령 자원을 두고 지역 내 대학들이 벌이는 생존 경쟁의 밀도를 측정합니다.
4. **대학 규모 의존성 지표 (Scale Dependency)**
   * **주요 요소:** 대학 총 재정 대비 등록금 수입 비중, 전체 편제 정원 규모
   * **설명:** 대학의 등록금 의존도 구조를 분석합니다. 정원 규모가 크고 등록금 의존도가 높을수록 미충원 발생 시 대학 재정이 받는 타격이 기하급수적으로 증가하는 리스크를 반영합니다.

---

## 3. 프로그램 디렉토리 구조 (Directory Structure)

본 패키지는 독립형 Streamlit 애플리케이션으로 가동되며 다음과 같은 아키텍처 구조를 가집니다.

```
Enrollment_Risk_System/
│
├── src/
│   └── app.py                      # Streamlit 대시보드 메인 실행 스크립트
│
├── requirements.txt                # 시스템 구동을 위한 외부 라이브러리 명세
└── README.md                       # 프로그램 설명서 (본 파일)
```

---

## 4. 핵심 환경 설정 및 소스 코드 (Implementation)

### 4.1 의존성 패키지 설정 (`requirements.txt`)
시스템 가동을 위해 아래의 라이브러리 설치가 필요합니다.
```text
streamlit
pandas
numpy
plotly
```

### 4.2 메타 소스 코드 (`src/app.py`)
아래 코드는 4대 리스크 인자의 정규화 연산 엔진과 레이더(Radar) 차트, 수평 바(Bar) 차트를 활용한 대시보드 전체 구현체입니다.

```python
# -*- coding: utf-8 -*-
"""
DataSense IR Module - University Enrollment Risk Matrix
Qliker, 2026.06.01 Version 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# --- 4. UI 및 시각화 파트 ---
def main():
    st.title("⚠️ 지역 인구 기반 대학 입학수요위험도 진단 시스템")
    st.caption("지역별 학령인구 변동성 및 대학 재정/경쟁력 지표를 결합한 통합 시점 리스크 시뮬레이터")
    st.write("---")
    
    df = load_enrollment_risk_data()
    
    # TOP 레벨: 위험 징후 알림 및 KPI 스코어 요약
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
    left_chart, right_chart = st.columns([3, 2])
    
    with left_chart:
        st.markdown("### 📊 권역별 입학수요위험도 정량 비교")
        df_sorted = df.sort_values(by='입학수요위험도', ascending=True)
        
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
        st.markdown("### 🎯 핵심 리스크 인자 구성 비율 심층 분석")
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
            r=scores + [scores[0]], 
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

    st.write("---")
    st.markdown("### 🔍 리스크 진단 상세 데이터 시트")
    
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
```

---

## 5. 분석 효과 및 운영 제언 (Strategic Value)

1. **위기 진단의 객관화:** * 단편적인 학생 충원율 분석에서 벗어나, 지역 거주 청년 유출과 고교 졸업자 공급 추세를 복합 스캔하여 향후 **5~10년 뒤의 장기적 생존 한계선**을 선제적으로 예측합니다.
2. **맞춤형 구조 개편 전략 수립:** * **방사형 레이더 차트**를 통해 대학 위기의 근본적 취약점을 파악할 수 있습니다. 예를 들어, `학령인구감소` 점수가 높은 대학은 타 권역 학생 유치를 위한 기숙사 인프라 확대 및 온라인 학위 과정 개설을, `규모의존성`이 높은 대학은 산학협력단 중심의 연구 과제 수주를 통한 재정 다각화 전략을 매핑해야 합니다.
