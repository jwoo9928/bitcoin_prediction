import streamlit as st
import yfinance as yf
import plotly.express as px
from bs4 import BeautifulSoup
import requests
import numpy as np
import datetime
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta, date
import predictors.similarity as sr
import matplotlib.pyplot as plt
import predictors.gpt_predict as gpts
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from dotenv import load_dotenv
import re
import os as OS
import gdown
import json

load_dotenv()

st.set_page_config(
    page_title="BTC predicting",
    page_icon="💰",
    layout="wide",
)

shared_link = OS.getenv('CSV_LINK')
file_id = shared_link.split('/d/')[1].split('/view')[0]
download_url = f"https://drive.google.com/uc?id={file_id}"
output_file = "btc_1h_data.csv"

# 파일이 이미 존재하는지 확인
if not OS.path.exists(output_file):
    print("CSV 파일이 로컬에 존재하지 않습니다. 다운로드를 시작합니다.")
    gdown.download(download_url, output_file, quiet=False)
else:
    print("CSV 파일이 이미 존재합니다. 로컬에서 불러옵니다.")

# CSV 파일 로드
try:
    train_data = pd.read_csv(
        output_file,   # CSV 경로
        parse_dates=['Open time'],
        index_col='Open time'
    )
    print(f"**CSV 로드 성공**: {train_data.shape[0]} rows")
except FileNotFoundError:
    print("btc_1h_data.csv 파일이 없습니다. 경로를 확인해 주세요.")
    st.stop()
except Exception as e:
    print(f"CSV 파일을 로드하는 중 에러가 발생했습니다: {e}")
    st.stop()

# 훈련 데이터 범위 (예: 2018-01-01 ~ 2024-12-31)
TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END   = pd.Timestamp("2024-12-31")

train = train_data.loc[TRAIN_START:TRAIN_END].copy()

if 'Close' not in train.columns:
    st.error("'Close' 컬럼이 없습니다. CSV 컬럼명을 확인해주세요.")
    st.stop()

train_close = train['Close']

st.session_state.page = "home"


# 사이드바
st.sidebar.image("assets/logo.png", width = 300)
st.sidebar.header("How to use program")

with st.sidebar:
    model_provider = st.selectbox(
        "예측 방식을 선택해주세요. ",
        ["cosine similarity", "LLM-GPT","LLM-GEMINI"],
        key="model_provider"
    )

    if model_provider == "cosine similarity":
        st.markdown(
        """

        **코사인 유사도**를 이용하여 유사 패턴을 찾아내 예측하는 방식입니다.
        1. 보고자 하는 기간을 선택해주세요. 🔑
        2. 그래프분석및시각화 클릭  📝
        3. 기다려주세요 🚀
        """
        )
            
        
    if model_provider == "LLM-GPT":
        st.markdown(
        """
        
        **GPT**를 이용하여 예측하는 방식입니다.
        1. 보고자 하는 기간을 선택해주세요. 🔑
        2. 그래프분석및시각화 클릭  📝
        3. 기다려주세요 🚀
        """
        )

    if model_provider == "LLM-GEMINI(시간 소요)":
        st.markdown(
        """
        
        **GEMINI**를 이용하여 예측하는 방식입니다.
        1. 보고자 하는 기간을 선택해주세요. 🔑
        2. 실행버튼 클릭  📝
        3. 기다리세용 🚀
        """
        )
       
    
    base_start_date = st.date_input(
        "기준 구간 시작 날짜", 
        value=pd.to_datetime("2024-12-24")
    )
    base_end_date = st.date_input(
        "기준 구간 종료 날짜", 
        value=pd.to_datetime("2024-12-27")
    )

    if st.button("그래프 분석 & 시각화"):
        st.session_state.page = model_provider
        
        
    st.markdown('\n')
    # 관련 기사 확인하기
    url = 'https://m.stock.naver.com/crypto/news/UPBIT/BTC'
    st.link_button("관련 기사 확인하기", url)


if st.session_state.page == 'home':
    print(st.session_state.page)
    img = Image.open("assets/logo2.png")
    new_size = (100, 100)
    img = img.resize(new_size)
    st.image(img)
        


    st.markdown('<h1>\
        <span style="color: #2482C5;">비트코인 가격 예측 프로그램 </span>\
        <span style="color: #0067ac;">B-redict💱</span>\
    </h1>\
    ', unsafe_allow_html=True)

    st.write('2025.01.15 - 2025.01.16.')
    st.markdown("***")
    st.subheader("What's B-redict?")
    st.markdown('과거 데이터를 토대로 비트코인의 가격을 예측하는 프로그램입니다.\n\
                첫째로 지정한 기간의 패턴을 분석하여 과거 여러 시점 중 코사인 유사도가 높게 나오는 시점의\
                패턴을 토대로 미래 데이터를 예측합니다. 둘째로 LLM과 프롬프트를 활용하여 미래 데이터를 예측합니다.\n')


    st.markdown('\n\n\n')
    # why

    st.subheader('Why did we make it?')

    with st.chat_message("user"):
        st.markdown("""
    **비트코인의 가격은 매우 변동성이 큽니다.**\n저희는 과거 데이터를 기반으로 향후 가격 변동 패턴을 분석하고\
    예측하는 프로그램을 만들었습니다. 이를 통해 가격 변동의 트렌드를 이해하고, \
    시장 흐름에 대한 직관적 이해를 얻고자 합니다. 또한 경험이 부족한 투자자에게도 예측 정보를 제공하여,\
    보다 효율적이고 객관적인 투자 결정을 할 수 있게 돕고자 합니다.\
    비트코인의 가격은 종종 투자자들의 심리적 요인에 의해 영향을 받습니다. 과거 데이터에서 비트코인의\
    패턴을 분석하여 향후 시장 움직임을 예측할 수 있습니다.
    **예측 모델을 바탕으로 다양한 투자 전략을 개발하고 수립하는 것**을 최종 목표로 두고 있습니다.\
                    
            """)

    st.markdown('\n')
    st.markdown('***')

    # USD/KRW 환율 데이터
    st.subheader("환전 고시 환율 💱")

    dt = yf.download('KRW=X',
                        start='2020-01-14',
                        end='2025-01-14',
                        progress=False)
        
    dates = list(dt.index)
    today = str(dates[-1])[:10]
    yester = str(dates[-2])[:10]
    tclose = dt[dt.index == today].Close.values[0][0]
    yclose = dt[dt.index == yester].Close.values[0][0]
    change = tclose - yclose
    st.metric(label='today', value=round(tclose, 3), delta=round(change, 5))


    st.markdown('***')
    # 그래프
    st.markdown('<h3 style="color: #123f6d;">환전 고시 환율 그래프\n</h3>', unsafe_allow_html=True)
    st.write("USD/KRW 환율 데이터")

    dt = dt.reset_index()
    fig = px.line(dt, x='Date', y=dt['Close'].values.flatten())
    fig.update_traces(line=dict(color='#00afad')) 

    st.plotly_chart(fig)



    st.markdown('***')
    # 주요 암호화폐 가격 차트

    st.markdown('<h3 style="color: #123f6d;">주요 암호화폐 가격 차트\n</h3>', unsafe_allow_html=True)

    cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD']

    crypto_data = {}
    for ticker in cryptos:
        crypto_data[ticker] = yf.download(ticker, start='2020-01-01', end='2025-01-01')

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    
    with col1:
        st.subheader("Bitcoin (BTC)") 
        fig1 = px.line(crypto_data['BTC-USD'].reset_index(), x='Date', y=crypto_data['BTC-USD']['Close'].values.flatten(), title='Bitcoin (BTC)')
        st.plotly_chart(fig1)

    with col2:
        st.subheader("Ethereum (ETH)")
        fig2 = px.line(crypto_data['ETH-USD'].reset_index(), x='Date', y=crypto_data['ETH-USD']['Close'].values.flatten(), title='Ethereum (ETH)')
        st.plotly_chart(fig2)

    with col3:
        st.subheader("Binance Coin (BNB)")
        fig3 = px.line(crypto_data['BNB-USD'].reset_index(), x='Date', y=crypto_data['BNB-USD']['Close'].values.flatten(), title='Binance Coin (BNB)')
        st.plotly_chart(fig3)

    with col4: 
        st.subheader("Cardano (ADA)")
        fig4 = px.line(crypto_data['ADA-USD'].reset_index(), x='Date', y=crypto_data['ADA-USD']['Close'].values.flatten(), title='Cardano (ADA)')
        st.plotly_chart(fig4)

    with col5:
        st.subheader("Solana (SOL)")
        fig5 = px.line(crypto_data['SOL-USD'].reset_index(), x='Date', y=crypto_data['SOL-USD']['Close'].values.flatten(), title='Solana (SOL)')
        st.plotly_chart(fig5)

    with col6:
        st.subheader("XRP (XRP)")
        fig6 = px.line(crypto_data['XRP-USD'].reset_index(), x='Date', y=crypto_data['XRP-USD']['Close'].values.flatten(), title='XRP (XRP)')
        st.plotly_chart(fig6)

    st.markdown("- 위 차트는 2020년부터 2024년까지의 주요 암호화폐 등락을 나타냅니다.")

elif st.session_state.page == 'LLM-GPT':
# 3-1) st.date_input()은 datetime.date이므로 pd.Timestamp로 변환
    base_start_ts = pd.Timestamp(base_start_date)
    base_end_ts   = pd.Timestamp(base_end_date)
    today_ts      = pd.Timestamp.today().normalize()  # 오늘 날짜 (자정 기준)
    next_date = 24

    # yfinance는 미래(오늘 이후) 데이터가 없으므로, 미래 날짜를 잘라낸다
    adj_start = min(base_start_ts, today_ts)
    adj_end   = min(base_end_ts,   today_ts)
    
    adj_end_for_download = adj_end + pd.Timedelta(days=1)
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

    base = yfin_data['Close'].iloc[:,0]
    base_norm = sr.nomarize_base(base)

    weekly_data = gpts.get_weekly_data(train_data, base_end_ts)

    predictions = gpts.predict_bitcoin_prices(weekly_data, base.tail(24), base_end_ts, 'assets/prompt.txt')
    predicted_prices = predictions['Close'].values

    # 예측값을 정규화
    predicted_prices_norm = (predicted_prices - min(predicted_prices)) / (max(predicted_prices) - min(predicted_prices))

    sim_series = sr.get_sim_series(base, base_norm, train_close)

    best_idx = sim_series.idxmax()

    # (10) 시각화
    fig, ax = plt.subplots(figsize=(12, 6))

    # 실제 이후 30시간의 실제 가격 데이터 가져오기
    future_start_date = base_end_date

    actual_future = train.loc[future_start_date:].iloc[1 : 1 + next_date]['Close'] if not train.loc[future_start_date:].empty else pd.Series(dtype=float)
    if actual_future.max() != actual_future.min():
        actual_future_norm = (actual_future - actual_future.min()) / (actual_future.max() - actual_future.min())
    else:
        actual_future_norm = np.zeros(len(actual_future))
    # x축에서의 시작 위치 설정 (기준 구간의 끝)
    start_x = len(base_norm)  # 24

    # 실제 미래 데이터를 진한 빨간색 선으로 플롯 (24~54)
    ax.plot(range(start_x, start_x + len(actual_future_norm)),
             actual_future_norm.values, label='[Actual Future] 24 Hours', color='darkred')

    # Base (Normalized)
    ax.plot(
        range(len(base_norm)), 
        base_norm, 
        label=f"[Base] {adj_start} ~ {adj_end}", 
        color='black'
    )


    # 예측된 값 시각화
    ax.plot(range(len(base_norm), len(base_norm) + len(predicted_prices_norm)),
            predicted_prices_norm, label='[Prediction] GPT', color='green', linewidth=2)
    
    next_date = 24
    
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


elif st.session_state.page == 'LLM-GEMINI':
    gemini_key =OS.getenv('GOOGLE_API_KEY')

    model = genai.GenerativeModel("gemini-1.5-flash")
    df = train
    predicted_df = pd.DataFrame()
    script_dir = OS.path.dirname(OS.path.abspath(__file__)) # 현재 스크립트의 경로
    file_path = OS.path.join(script_dir, 'assets', 'prompt2.txt') # 경로 합치기

    f = open(file_path, 'r')
    prompt = f.read()
    genai.configure(api_key=gemini_key)
# 3-1) st.date_input()은 datetime.date이므로 pd.Timestamp로 변환
    base_start_ts = pd.Timestamp(base_start_date)
    base_end_ts   = pd.Timestamp(base_end_date)
    today_ts      = pd.Timestamp.today().normalize()  # 오늘 날짜 (자정 기준)
    next_date = 30

    # yfinance는 미래(오늘 이후) 데이터가 없으므로, 미래 날짜를 잘라낸다
    adj_start = min(base_start_ts, today_ts)
    end_date  = min(base_end_ts,   today_ts)
    
    adj_end_for_download = end_date + pd.Timedelta(days=1)
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

    
    print(df.info)
    df['Open time'] = pd.to_datetime(df.index)
    df['Close time'] = pd.to_datetime(df['Close time'])


    # 선택된 날짜 범위로 데이터 필터링
    # start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    start_datetime = end_datetime - timedelta(days=3 * 30)
    data = df[(df['Open time'] >= start_datetime) & (df['Close time'] <= end_datetime)]

    data = data.to_dict()
    full_prompt = f"{prompt}\n\n**데이터:**\n{data}\n\n**요청사항:**\n1.  제공된 데이터의 마지막 시간부터 30시간까지의 값을 작성해주세요.\n2.  30시간치 데이터를 생략 없이 json형식으로 꼭 반환하세요."
     
    # 모델에 프롬프트를 전달하고 응답을 받습니다.
    response: GenerateContentResponse = model.generate_content(full_prompt)

    print(response.text)
    try:
        json_start = response.text.find('{')
        json_end = response.text.rfind('}')
        json_str = response.text[json_start:json_end+1]

        # 숫자 키를 문자열 키로 변환
        json_str = re.sub(r'{\s*(\d+):', r'{"\1":', json_str)
        json_str = re.sub(r',\s*(\d+):', r',"\1":', json_str)
        
        predicted_data = json.loads(json_str)
        # 데이터프레임으로 변환
        predicted_df = pd.DataFrame(predicted_data)
        predicted_df['Open time'] = pd.to_datetime(predicted_df['Open time'])
        st.line_chart(predicted_df.set_index('Open time')['Close'])

    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print("Gemini response content:", response.text)
    except Exception as e:
        print(f"기타 오류: {e}")
        print("Gemini response content:", response.text)
    

    

else:
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
    base_norm = sr.nomarize_base(base)
    window_size = len(base)
    next_date = 24

    sim_series = sr.get_sim_series(base, base_norm, train_close)

    if sim_series.empty:
        st.write("유사도 계산 결과가 없습니다.")
        st.stop()

    # 디버깅: TOP 10 similarity
    top10 = sim_series.sort_values(ascending=False).head(10)
    st.write("TOP 10 similarity:")
    st.dataframe(top10.rename("Similarity"))

    #(8) 유사도 0.98 이상 구간들의 미래 패턴 평균
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

    # 실제 이후 24시간의 실제 가격 데이터 가져오기
    future_start_date = base_end_date

    actual_future = train.loc[future_start_date:].iloc[1 : 1 + next_date]['Close'] if not train.loc[future_start_date:].empty else pd.Series(dtype=float)
    if actual_future.max() != actual_future.min():
        actual_future_norm = (actual_future - actual_future.min()) / (actual_future.max() - actual_future.min())
    else:
        actual_future_norm = np.zeros(len(actual_future))
    # x축에서의 시작 위치 설정 (기준 구간의 끝)
    start_x = len(base_norm)  # 24

    # 실제 미래 데이터를 진한 빨간색 선으로 플롯 (24~54)
    ax.plot(range(start_x, start_x + len(actual_future_norm)),
             actual_future_norm.values, label='[Actual Future] 24 Hours', color='darkred')



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
