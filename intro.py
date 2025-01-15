import streamlit as st
import yfinance as yf
import plotly.express as px
from bs4 import BeautifulSoup
import requests
import numpy as np
import datetime
import pandas as pd
from PIL import Image

st.set_page_config(
    page_title="BTC predicting",
    page_icon="💰",
    layout="wide",
)

img = Image.open("logo2.png")
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
            지정한 기간의 패턴을 분석하여 과거 여러 시점 중 코사인 유사도가 높게 나오는 시점의\
            패턴을 토대로 미래 데이터를 예측합니다.\n')


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






# 사이드바
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
        
    st.markdown('\n')
    # 관련 기사 확인하기
    url = 'https://m.stock.naver.com/crypto/news/UPBIT/BTC'
    st.link_button("관련 기사 확인하기", url)

