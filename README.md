<div align="center">
<h2>비트코인 가격 예측 프로그램 B-redict💱</h2>
  
![Mobile App Screen Mockup, Mosaic](https://github.com/user-attachments/assets/455190a2-0287-4175-87de-13e60f7fa83b)

</div>

## ✅ Overview

- **프로젝트 이름**   비트코인 가격 예측 프로그램 B-redict💱
- **프로젝트 기간**   2025.01.15 - 2025.01.16
- **배포 주소**  [B-redict](https://jwoo9928-bitcoin-prediction-app-wzhrj9.streamlit.app/)
- **멤버**   우리FISA 4기 AI 엔지니어링 김정수 박재우 박혁준 허정원


## 🏁 목차

- **📖 Description**

- **🧩 Why?**

- **🔧 사용기술**

- **💣 트러블슈팅**

  

## 📖 Description


![20250116_102550-front](https://github.com/user-attachments/assets/6c991edf-e552-4d45-b37f-c57fa33d74d3)|![20250116_114624-front](https://github.com/user-attachments/assets/64a6fdfd-5f9b-4cd7-ad9e-c6c944cb7303)
---|---|
초기화면|주요 암호화폐 가격차트

![20250116_113616-front](https://github.com/user-attachments/assets/98922579-96c5-4ba3-ab7d-d87f2cb94efa)|![20250116_114046-front](https://github.com/user-attachments/assets/71d1ef19-aed0-401c-b03b-4b0db7556385)
---|---|
데이터 불러오기|결과 시각화


**과거 데이터를 토대로 비트코인의 가격을 예측하는 프로그램**입니다. 예측 방식은 두 가지를 사용합니다. 

  

**📍 similarity**

과거 데이터의 패턴과 **코사인 유사도가 높은 패턴을 찾아** 예측합니다.

지정한 기간의 패턴을 분석하여 과거 여러 시점 중 코사인 유사도가 높게 나오는 시점의 패턴을 토대로 미래 데이터를 예측합니다. **유사도 0.98 이상인 구간들의 미래 패턴 평균과 유사도가 가장 높은 패턴**을 시각화합니다.

(파란선 - 유사도 0.98 이상 패턴의 평균값 / 빨간선 - 유사도가 가장 높은 패턴 1개의 패턴)

**정규화 시행**

BTC-USD의 패턴을 기준으로 예측하기 때문에 단위 구간 내의 최저점을 0, 최고점을 1로 정규화하였습니다. 

  

**📍 LLM predict**

**LLM 프롬프트를 활용**하는 방식입니다.

지정한 기간의 데이터를 과거 데이터에서 필터링 한 후, 토큰을 최소화하기 위해 주 단위로 데이터를 집계합니다. 프롬프트 템플릿을 활용해 프롬프트를 생성하여 LLM에 질의를 시행합니다. LLM의 응답을 파싱하여 시계열 데이터로 변환합니다. 


## 🧩 Why?

**비트코인의 가격은 매우 변동성이 큽니다.**

저희는 과거 데이터를 기반으로 향후 가격 변동 패턴을 분석하고 예측하는 프로그램을 만들었습니다. 이를 통해 가격 변동의 트렌드를 이해하고, 시장 흐름에 대한 직관적 이해를 얻고자 합니다. 또한 경험이 부족한 투자자에게도 예측 정보를 제공하여, 보다 효율적이고 객관적인 투자 결정을 할 수 있게 돕고자 합니다. 비트코인의 가격은 종종 투자자들의 심리적 요인에 의해 영향을 받습니다. 과거 데이터에서 비트코인의 패턴을 분석하여 향후 시장 움직임을 예측할 수 있습니다.

 **예측 모델을 바탕으로 다양한 투자 전략을 개발하고 수립하는 것**을 최종 목표로 하고 있습니다.



## 🔧 사용기술

### Environment

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

### Development

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)  ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

### Domain & Communication

![Bitcoin](https://img.shields.io/badge/Bitcoin-000?style=for-the-badge&logo=bitcoin&logoColor=white)  ![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white) ![Google Gemini](https://img.shields.io/badge/google%20gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)	
![Slack](https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
 



## 💣 트러블슈팅

### 이슈 1. 코사인 유사도 검증

**문제** 코사인 유사도를 측정하였을 때, 1위가 항상 동일한 시점의 데이터가 return되는 오류 발생

**원인** 전체 과거 데이터와 일부인 특정 시점의 데이터 사이의 코사인 유사도를 측정하기 때문에 동일한 시점의 데이터가 출력됩니다.

**해결** 코사인 유사도 1위를 채택하는 것이 아닌 2위의 패턴을 사용합니다.


### 이슈 2. LLM 토큰 크기 조정

**문제 & 원인** 2018년 데이터부터 불러와야 하는 데이터 용량 문제

**해결** 토큰의 크기를 일주일 단위로 설정하였습니다.


## 데이터 출처

![kaggle bitcoin hour](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024)
