import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st

def cosine_similarity(x, y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return 0.0
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def nomarize_base(base):
    # (6) Base 정규화
    # base는 Series이므로 base.max(), base.min()은 스칼라(float)
    if base.max() != base.min():
        base_norm = (base - base.min()) / (base.max() - base.min())
    else:
        base_norm = np.zeros(len(base))
    return base_norm

def get_sim_series(base,base_norm, train_close):
    # window_size = base 길이
    window_size = len(base)

    if window_size == 0:
        st.warning("[주의] 기준 구간(Base)에 데이터가 없습니다. (window_size=0)")
        st.stop()

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
    return sim_series