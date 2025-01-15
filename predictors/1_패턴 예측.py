import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import time
import plotly.express as px

# st.title("차트 패턴 유사도 검색")

st.set_page_config(
    page_title="차트 패턴 유사도 검색",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded")

col = st.columns((2, 1), gap='medium')

with col[1]:
    st.subheader('오늘의 지표 📩')
    st.markdown('***')
    dt = yf.download('BTC-USD',
                    start='2020-01-14',
                    end='2025-01-14',
                    progress=False)
    
    dates = list((dt.index))
    today = str(dates[-1])[:10]
    yester = str(dates[-2])[:10]
    tclose = dt[dt.index == today].Close.values[0][0]
    yclose = dt[dt.index == yester].Close.values[0][0]
    change = tclose - yclose
    st.metric('today', round(tclose, 3), round(change, 5))

    # 그래프
    st.markdown("#### BTC-USD")
    st.write("가격 변동 그래프")

    dt = dt.reset_index()
    fig = px.line(dt, x='Date', y=dt['Close'].values.flatten())
    fig.update_traces(line=dict(color='#00afad')) 

    st.plotly_chart(fig)

with col[0]:
    # --- (1) CSV 훈련 데이터 불러오기 ---
    try:
        train_data = pd.read_csv(
            'btc_1h_data.csv',   # CSV 경로를 본인 환경에 맞게 수정
            parse_dates=['Open time'],
            index_col='Open time'
        )
        #st.write(f"**CSV 로드 성공**: {train_data.shape} rows")
        #st.dataframe(train_data.head(3))
    except FileNotFoundError:
        st.error("btc_1h_data.csv 파일이 없습니다. 경로를 확인해 주세요.")
        st.stop()

    # 훈련 데이터 범위 (예: 2018-01-01 ~ 2024-12-31)
    TRAIN_START = pd.Timestamp("2018-01-01")
    TRAIN_END   = pd.Timestamp("2024-12-31")

    train = train_data.loc[TRAIN_START:TRAIN_END].copy()

    if 'Close' not in train.columns:
        st.error("'Close' 컬럼이 없습니다. CSV 컬럼명을 확인해주세요.")
        st.stop()

    train_close = train['Close']

    # --- (2) 사용자에게 기준 구간(Base) 날짜 입력 ---
    base_start_date = st.date_input(
        "기준 구간 시작 날짜", 
        value=pd.to_datetime("2024-12-24")
    )
    base_end_date = st.date_input(
        "기준 구간 종료 날짜", 
        value=pd.to_datetime("2024-12-27")
    )

    # --- (3) 버튼 클릭 -> 분석 ---
    if st.button("그래프 분석 & 시각화"):
        with st.spinner("현재 분석/예측 중입니다. 잠시만 기다려주세요..."):
            time.sleep(1)
            # 3-1) st.date_input()은 datetime.date이므로 pd.Timestamp로 변환
            base_start_ts = pd.Timestamp(base_start_date)
            base_end_ts   = pd.Timestamp(base_end_date)
            today_ts      = pd.Timestamp.today().normalize()  # 오늘 날짜 (자정 기준)

            # yfinance는 미래(오늘 이후) 데이터가 없으므로, 미래 날짜를 잘라낸다
            adj_start = min(base_start_ts, today_ts)
            adj_end   = min(base_end_ts,   today_ts)

            # end 날짜에 +1일 -> 해당 날짜 종가까지 포함
            adj_end_for_download = adj_end + pd.Timedelta(days=1)

            if adj_end_for_download <= adj_start:
                st.warning(f"입력 기간이 유효하지 않습니다. (start={base_start_ts}, end={base_end_ts})")
                st.stop()

            st.info(f"yfinance에서 데이터를 다운받는 기간: {adj_start} ~ {adj_end} (end={adj_end_for_download})")

            # 3-2) yfinance 다운로드 (Timestamp 그대로 전달)
            with st.spinner("yfinance로 데이터를 가져오는 중..."):
                try:
                    yfin_data = yf.download(
                        tickers="BTC-USD",
                        interval="1h",
                        start=adj_start,
                        end=adj_end_for_download
                    )
                except Exception as e:
                    st.error(f"yfinance 데이터 수집 오류: {e}")
                    st.stop()

            if yfin_data.empty:
                st.warning("yfinance에서 해당 구간의 데이터를 가져오지 못했습니다. (미래 날짜이거나 거래 데이터 없음)")
                st.stop()

            st.write(f"**yfinance Base 구간 로드 성공**: {yfin_data.shape} rows")
            st.dataframe(yfin_data)

            # --- (4) Base 구간: 'Close' 컬럼만 추출 => Series ---
            base = yfin_data['Close'].iloc[:,0]

            # window_size = base 길이
            window_size = len(base)
            next_date = 30

            if window_size == 0:
                st.warning("[주의] 기준 구간(Base)에 데이터가 없습니다. (window_size=0)")
                st.stop()

            # --- (5) 코사인 유사도 함수 ---
            def cosine_similarity(x, y):
                x = np.array(x).flatten()
                y = np.array(y).flatten()
                if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
                    return 0.0
                return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

            # (6) Base 정규화
            # base는 Series이므로 base.max(), base.min()은 스칼라(float)
            if base.max() != base.min():
                base_norm = (base - base.min()) / (base.max() - base.min())
            else:
                base_norm = np.zeros(len(base))

            # (7) 훈련 데이터에서 window_size만큼 슬라이딩하며 유사도 계산
            moving_cnt = len(train_close) - window_size
            if moving_cnt <= 0:
                st.warning("훈련 데이터가 너무 적거나 윈도우가 너무 큽니다. (pattern 탐색 불가)")
                st.stop()

            sim_list = []
            for i in range(moving_cnt):
                target = train_close.iloc[i : i + window_size]
                if (len(target) == window_size) and (target.max() != target.min()):
                    target_norm = (target - target.min()) / (target.max() - target.min())
                    sim = cosine_similarity(base_norm, target_norm)
                    sim_list.append(sim)
                else:
                    sim_list.append(0.0)

            sim_series = pd.Series(sim_list)

            if sim_series.empty:
                st.write("유사도 계산 결과가 없습니다.")
                st.stop()

            # 디버깅: TOP 10 similarity
            top10 = sim_series.sort_values(ascending=False).head(10)
            st.write("TOP 10 similarity:")
            st.dataframe(top10.rename("Similarity"))

            # (8) 유사도 0.98 이상 구간들의 미래 패턴 평균
            threshold = 0.98
            high_indices = sim_series[sim_series >= threshold].index
            patterns_ext = []

            for idx in high_indices:
                future_segment = train_close.iloc[idx + window_size : idx + window_size + next_date]
                if (len(future_segment) == next_date) and (future_segment.max() != future_segment.min()):
                    f_norm = (future_segment - future_segment.min()) / (future_segment.max() - future_segment.min())
                    patterns_ext.append(f_norm.values)

            if len(patterns_ext) > 0:
                mean_pattern_extended = np.mean(patterns_ext, axis=0)
                st.write(f"유사도 {threshold} 이상 패턴 수:", len(patterns_ext))
            else:
                mean_pattern_extended = None
                st.write(f"유사도 {threshold} 이상인 패턴이 없습니다.")

            # (9) 유사도 1위 구간 idx
            best_idx = sim_series.idxmax()

            # (10) 시각화
            fig, ax = plt.subplots(figsize=(12, 6))

            # Base (Normalized)
            ax.plot(
                range(len(base_norm)), 
                base_norm, 
                label=f"[Base] {adj_start} ~ {adj_end}", 
                color='black'
            )

            # Best Match
            if best_idx is not None and best_idx >= 0:
                pred_start = best_idx + window_size
                pred_end   = pred_start + next_date
                best_prediction = train_close.iloc[pred_start:pred_end]

                if (len(best_prediction) == next_date) and (best_prediction.max() != best_prediction.min()):
                    best_prediction_norm = (best_prediction - best_prediction.min()) / (best_prediction.max() - best_prediction.min())
                else:
                    best_prediction_norm = np.zeros(len(best_prediction))

                ax.plot(
                    range(len(base_norm), len(base_norm) + len(best_prediction_norm)),
                    best_prediction_norm,
                    label='[Prediction] Best Match', 
                    color='red'
                )

            # Mean pattern
            if mean_pattern_extended is not None:
                ax.plot(
                    range(len(base_norm), len(base_norm) + len(mean_pattern_extended)),
                    mean_pattern_extended,
                    label=f"[Prediction] Mean (sim >= {threshold})", 
                    color='blue'
                )

            # 기준 구간 & 예측 영역 표시
            ax.axvline(x=len(base_norm)-1, color='gray', linestyle='--')
            ax.axvspan(
                len(base_norm)-1,
                len(base_norm) + next_date - 1,
                facecolor='yellow', alpha=0.2, 
                label='Prediction Area'
            )

            ax.set_title("BTC 1H Pattern Matching (Close as Series to avoid ambiguous truth value)")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Normalized Price")
            ax.legend()
            plt.tight_layout()

            st.pyplot(fig)
    st.success("분석 완료!")


st.sidebar.image("logo.png", width = 300)
st.sidebar.header("How to use program")

with st.sidebar:
    model_provider = st.selectbox(
        "예측 방식을 선택해주세요. ",
        ["cosine similarity", "LLM"],
        key="model_provider"
    )

    if model_provider == "cosine similarity":
        st.markdown(
        """

        **코사인 유사도**를 이용하여 유사 패턴을 찾아내 예측하는 방식입니다.
        1. 보고자 하는 기간을 선택해주세요. 🔑
        2. 실행버튼 클릭  📝
        3. 기다리세용 🚀
        """
    )
        
    if model_provider == "LLM":
        st.markdown(
        """
        
        **LLM**를 이용하여 유사 패턴을 찾아내 예측하는 방식입니다.
        1. 보고자 하는 기간을 선택해주세요. 🔑
        2. 실행버튼 클릭  📝
        3. 기다리세용 🚀
        """
    )