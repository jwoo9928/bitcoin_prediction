from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import List

from io import StringIO
import re


def get_weekly_data(data, base_end_date):
    # 데이터 필터링 (11/30 23시까지)
    cutoff_date = pd.Timestamp(base_end_date)
    filtered_data = data[data.index <= cutoff_date].copy()

    # 1) 인덱스에서 주 정보 추출 (년-주 형태)
    filtered_data['Week'] = filtered_data.index.map(lambda x: f'{x.year}-W{x.isocalendar()[1]}')

    # 2) 주 단위로 데이터 집계
    weekly_data = filtered_data.groupby('Week').agg({
        'Open': 'first',          # 주 시작 시가
        'Close': 'last',         # 주 종료 종가
        'High': 'max',           # 주 최고가
        'Low': 'min',           # 주 최저가
        'Volume': 'sum'         # 주 총 거래량
    }).reset_index()

    # 3) 주간 변화량 계산 (Close - Open)
    weekly_data['Change'] = weekly_data['Close'] - weekly_data['Open']
    return weekly_data

def load_prompt_template(file_path: str) -> str:
    """프롬프트 템플릿을 파일에서 로드하는 함수"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_future_timestamps(start_time: datetime, periods: int) -> List[datetime]:
    """미래 시간대 생성 함수"""
    return [start_time + timedelta(hours=i) for i in range(periods)]

def create_gpt_prompt(weekly_df: pd.DataFrame, recent_hourly_data: pd.DataFrame, standard: str, prompt_template: str) -> str:
    """GPT 프롬프트 생성 함수"""
    print(weekly_df[["Week", "Volume"]])
    return prompt_template.format(
        standard=standard,
        weekly_data=weekly_df.to_string(),
        hourly_data=recent_hourly_data.to_string(),
    )

def parse_gpt_response(response: str) -> pd.Series:
    """GPT 응답을 파싱하여 시계열 데이터로 변환하는 함수"""
    pattern = re.compile(r"^(Open time,Close|[\d]{4}-[\d]{2}-[\d]{2} [\d]{2}:[\d]{2}:[\d]{2}, [\d]+\.\d{2})$")
    response = "\n".join([line for line in response.splitlines() if pattern.match(line)])
    print(response)
    # CSV 형식으로 읽어들여서 pandas DataFrame 생성
    df = pd.read_csv(StringIO(response), parse_dates=['Open time'])
    return df


def predict_bitcoin_prices(weekly_data: pd.DataFrame, hourly_data: pd.DataFrame,standard:str, prompt_path: str) -> pd.Series:
    """메인 예측 함수"""
    # 시간별 데이터 필터링
    filtered_hourly_data = hourly_data
    
    # 프롬프트 템플릿 로드
    prompt_template = load_prompt_template(prompt_path)
    
    # GPT 모델 설정
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
    )
    
    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_template("{input}")
    # GPT에 질의
    chain = prompt | model
    response = chain.invoke({
        "input": create_gpt_prompt(weekly_data, filtered_hourly_data,standard, prompt_template)
    })
    
    # 결과 파싱 및 반환
    return parse_gpt_response(response.content)