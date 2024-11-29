# streamlit run gyro.py

import streamlit as st
import psycopg2
import pandas.io.sql as psql
from sqlalchemy import create_engine
import datetime
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 데이터베이스 설정
HOSTNAME = '203.253.202.70'
PORT = 5432
USERNAME = 'kkilab'
PASSWORD = 'kkilab'
DATABASE = 'gyro'


# 접속 엔진 생성
con_str_fmt = "postgresql://{0}:{1}@{2}:{3}/{4}"
con_str = con_str_fmt.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)
engine = create_engine(con_str)

# 데이터베이스에서 선박 데이터 가져오기
def get_ship_names(start_date, end_date):
    query = f"""
        SELECT device_id, COUNT(*) as ea
        FROM raw_imu
        WHERE device_id IS NOT NULL AND device_id != ''
        AND created_at >= '{start_date}' AND created_at <= '{end_date}'
        GROUP BY device_id
        ORDER BY ea DESC;
    """
    with psycopg2.connect(host=HOSTNAME, dbname=DATABASE, user=USERNAME, password=PASSWORD) as conn:
        return psql.read_sql(query, conn)

# 데이터베이스에서 특정 device_id의 데이터를 가져오기
def get_gyro_data(start_date, end_date, device_id):
    if device_id == '전체 선박':
        query_raw = ''
    else:
        query_raw = f"device_id = '{device_id}' AND"

    query1 = f"""
        SELECT orientation_x, orientation_y, orientation_z, device_id, created_at
        FROM raw_imu
        WHERE {query_raw} created_at >= '{start_date}' AND created_at <= '{end_date}'
        ORDER BY created_at DESC;
    """
    query2 = f"""
        SELECT *
        FROM imu_period
        WHERE {query_raw} datetime >= '{start_date}' AND datetime <= '{end_date}'
        ORDER BY datetime DESC;
    """
    with psycopg2.connect(host=HOSTNAME, dbname=DATABASE, user=USERNAME, password=PASSWORD) as conn:
        raw_imu_data = psql.read_sql(query1, conn)
        imu_period_data = psql.read_sql(query2, conn)
    raw_imu_data.set_index(raw_imu_data['device_id'], inplace=True)
    raw_imu_data.drop(columns=['device_id'], inplace=True)
    
    imu_period_data.set_index(imu_period_data['device_id'], inplace=True)
    imu_period_data.drop(columns=['device_id','datetime2'], inplace=True)

    return raw_imu_data, imu_period_data

# 세션 상태 초기화
if 'ship_name_data' not in st.session_state:
    st.session_state.update({
        "ship_name_data": None,
        "start_date": None,
        "end_date": None,
        "selected_device": None,
    })

def paging(data, option):
    items_per_page = st.sidebar.number_input("페이지당 항목 수", min_value=5, max_value=50, value=10, step=5)
    total_items = len(data)
    total_pages = (total_items // items_per_page) + (1 if total_items % items_per_page > 0 else 0)

    # 페이지 선택
    current_page = st.sidebar.slider("페이지 선택", 1, total_pages, 1)
    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    current_data = data.iloc[start_idx:end_idx]

    st.write(f"현재 페이지: {current_page}/{total_pages}")
    st.table(current_data)

    if option == 'imu_period':
        # 특정 컬럼 제외
        excluded_columns = ['datetime']  # 제외할 컬럼 리스트
        data_col = [col for col in current_data.columns if col not in excluded_columns]

        # 여러 컬럼 선택 및 그래프 생성
        selected_columns = st.multiselect("그래프로 그릴 컬럼을 선택하세요:", data_col)
        if selected_columns:
            plt.figure(figsize=(12, 6))
            for column in selected_columns:
                plt.plot(current_data['datetime'], current_data[column], marker='o', label=column)

            plt.title(f"{selected_device} - 선택된 컬럼", fontsize=16)
            plt.xlabel("날짜", fontsize=12)
            plt.ylabel("값", fontsize=12)
            plt.xticks(rotation=45, fontsize=10)  # x축 레이블 회전
            plt.legend(title="컬럼", fontsize=10)
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.info("그래프로 표시할 컬럼을 선택하세요.")

# 레이아웃 설정
st.set_page_config(layout="wide")  # 와이드 레이아웃 활성화

# 사이드바: 날짜 및 조회 설정
with st.sidebar:
    st.title("Gyro 데이터 조회")
    start_date = st.date_input("시작일", value=datetime.date.today() - datetime.timedelta(days=14))
    end_date = st.date_input("종료일", value=datetime.date.today())

    if st.button("조회"):
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.ship_name_data = get_ship_names(start_date, end_date)

    
# 중앙과 오른쪽 영역 나누기
col1, col2 = st.columns([99, 1])  # 비율 설정 (중앙 9, 오른쪽 1)

with col1:
    # 중앙 영역: 데이터 테이블 출력
    if st.session_state.ship_name_data is not None:
        ship_name_data = st.session_state.ship_name_data

        if ship_name_data.empty:
            st.warning("조회 결과가 없습니다.")
        else:
            total_users = ship_name_data['ea'].sum()
            total_row = pd.DataFrame([{'device_id': '전체 선박', 'ea': total_users}])
            ship_name_data = pd.concat([total_row, ship_name_data], ignore_index=True)
            ship_name_data.rename(columns={'ea': '개수'}, inplace=True)

            st.subheader("조회 결과:")
            st.table(ship_name_data)

            device_options = ["Device ID를 선택하세요"] + list(ship_name_data['device_id'])
            selected_device = st.selectbox("Device ID 선택", device_options)
            st.session_state.selected_device = selected_device

            if selected_device == "Device ID를 선택하세요":
                st.info("Device ID를 선택하세요.")
            else:
                st.session_state.raw_imu_data, st.session_state.imu_period = get_gyro_data(
                    st.session_state.start_date, st.session_state.end_date, selected_device
                )
                option = st.radio("옵션을 선택하세요", ["raw_imu", "imu_period"])

                data = st.session_state.raw_imu_data.drop_duplicates().sort_values(by='created_at') if option == "raw_imu" else st.session_state.imu_period.drop_duplicates().sort_values(by='datetime')
            
                if not data.empty:
                    paging(data,option)
                else:
                    st.warning("데이터가 없습니다.")
    else:
        st.info("사이드바에서 날짜를 선택하고 '조회'를 클릭하세요.")

with col2:
    # 오른쪽 영역: 빈 공간 유지
    st.empty()
