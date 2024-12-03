# streamlit run gyro.py

import streamlit as st
import psycopg2
import pandas.io.sql as psql
from sqlalchemy import create_engine
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import find_peaks
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
        SELECT device_id, COUNT(DISTINCT CONCAT(orientation_x, orientation_y, orientation_z, orientation_w, device_id, created_at)) as ea
                FROM raw_imu
                WHERE created_at2 >= '{start_date}' AND created_at2 <= '{end_date}'
                GROUP BY device_id
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
        SELECT orientation_x, orientation_y, orientation_z, orientation_w, device_id, created_at
        FROM raw_imu
        WHERE {query_raw} created_at2 >= '{start_date}' AND created_at2 <= '{end_date}'
        ORDER BY created_at DESC;
    """
    
    with psycopg2.connect(host=HOSTNAME, dbname=DATABASE, user=USERNAME, password=PASSWORD) as conn:
        raw_imu_data = psql.read_sql(query1, conn)
        
    raw_imu_data.set_index(raw_imu_data['device_id'], inplace=True)
    raw_imu_data.drop(columns=['device_id'], inplace=True)
    

    return raw_imu_data
def quaternion_to_euler(x, y, z, w):
    # Roll (X axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (Z axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)


    return roll, pitch, yaw


def sensor_moving_peak(sensor, window_size=1, min_distance=10, feature='roll'):
    sensor_period = pd.DataFrame()
    
    
    # 이동평균 평활화
    sensor.loc[:, feature] = sensor[feature].rolling(window=window_size).mean()
    
    sensor = sensor.dropna(subset=[feature])  # 결측값 제거
    # peak 값 구하기
    peaks, _ = find_peaks(sensor[feature], distance=min_distance)
    
    
    #주기, 속력 구하기
    period_data = []  # 데이터를 담을 리스트

    for i in range(1, peaks.shape[0]):
        start_idx = peaks[i - 1]
        end_idx = peaks[i]
        max_deg = sensor.iloc[start_idx:end_idx][feature].max()
        min_deg = sensor.iloc[start_idx:end_idx][feature].min()
        deg_diff = max_deg - min_deg
        period = end_idx - start_idx
        spd = deg_diff / period
    
        device_id = sensor.index[0]
        datetime = sensor.created_at.iloc[start_idx]

        # period_data 리스트에 딕셔너리 추가
        period_data.append({
            'device_id': device_id,
            'datetime': datetime,
            f'max_deg_{feature}': max_deg,
            f'min_deg_{feature}': min_deg,
            f'deg_diff_{feature}': deg_diff,
            f'{feature}_spd': spd,
            'period': period
        })
    
    # 리스트를 데이터프레임으로 변환
    sensor_period = pd.DataFrame(period_data)

    # 하위 30%, 상위 30% 제거
    lower_quantile = sensor_period[f'deg_diff_{feature}'].quantile(0.2)
    upper_quantile = sensor_period[f'deg_diff_{feature}'].quantile(0.8)
    sensor_period = sensor_period[
        (sensor_period[f'deg_diff_{feature}'] > lower_quantile) & 
        (sensor_period[f'deg_diff_{feature}'] < upper_quantile)
    ]

    
    return sensor_period, peaks



def paging(data):
    
    if data.shape[0] == 0:
        st.warning("데이터가 없습니다.")
        return
    
    elif data.shape[0] < 10:
        st.sidebar.warning("데이터가 너무 작아 페이지당 항목 수를 설정할 수 없습니다.")
        items_per_page = data.shape[0]  # 모든 데이터를 한 페이지로 표시
    
    default_items_per_page = max(1, data.shape[0] // 100)
    items_per_page = st.sidebar.slider("페이지당 항목 수", min_value=1, max_value=data.shape[0], value=min(default_items_per_page,data.shape[0]), step=10)
    
    total_items = len(data)
    
    if total_items == 0:
        st.warning("데이터가 없습니다.")  # 데이터가 없을 경우 경고 메시지 표시
        return

    total_pages = (total_items // items_per_page) + (1 if total_items % items_per_page > 0 else 0)

    # 페이지 선택 슬라이더
    if total_pages > 1:
        # current_page = st.sidebar.slider("페이지 선택", 1, total_pages, 1)
        page_options = list(range(1, total_pages + 1))
        current_page = st.sidebar.selectbox("페이지 선택",  page_options,index=0)
        
    else:
        current_page = 1  # 총 페이지가 1개 이하인 경우 기본값 설정

    start_idx = max(0, (current_page - 1) * items_per_page)
    end_idx = min(total_items, start_idx + items_per_page)
    
    current_data = data.iloc[start_idx:end_idx]

    st.write(f"현재 페이지: {current_page}/{total_pages}")
    st.dataframe(current_data)

    current_data.loc[:, 'roll'], current_data.loc[:, 'pitch'], current_data.loc[:, 'yaw'] = zip(*current_data.apply(
        lambda row: np.degrees(quaternion_to_euler(
            row['orientation_x'], row['orientation_y'], row['orientation_z'], row['orientation_w']
        )), axis=1))

    # 여러 컬럼 선택 및 그래프 생성
    selected_columns = st.multiselect("그래프로 그릴 컬럼을 선택하세요:", ['roll', 'pitch', 'yaw'])

    if selected_columns:
        # 데이터 크기 계산
        data_size = current_data.shape[0]

        if data_size // 10 > 1:  # 슬라이더를 생성할 수 있는 조건
            current_window_size = st.slider(
                label="빈도 선택",
                min_value=1,
                max_value=data_size // 10,
                value=data_size // 10 // 2,
                step=1
            )

            current_distance = st.slider(
                label="피크 간 최소 거리(distance)",
                min_value=1,
                max_value=100,
                value=20,
                step=1,
                help="피크 간 최소 거리를 설정하세요"
            )

            # 데이터 보기 옵션 추가
            view_option = st.radio(
                "보기 옵션을 선택하세요",
                options=["그래프 보기", "데이터프레임 보기"],
                help="그래프 또는 sensor_period 데이터프레임 중에서 선택하세요."
            )

            for feature in selected_columns:
                # 각 feature마다 피크 계산 및 데이터프레임 생성
                sensor_period, peaks = sensor_moving_peak(
                    current_data,
                    window_size=current_window_size,
                    min_distance=current_distance,
                    feature=feature
                )

                if view_option == "그래프 보기":
                    if sensor_period.shape[0] > 0:
                        # 그래프 생성
                        fig, ax = plt.subplots(figsize=(15, 8))

                        # 선형 그래프 그리기
                        ax.plot(
                            current_data['created_at'],
                            current_data[feature],
                            label=f"{feature} (line)",
                            zorder=1
                        )

                        # 피크 데이터 그리기
                        if len(peaks) > 0:  # peaks가 있는 경우만 처리
                            peaks_data = current_data.iloc[peaks]
                            ax.scatter(
                                peaks_data['created_at'],
                                peaks_data[feature],
                                color='orange',
                                s=50,
                                label=f"{feature} (peaks)",
                                zorder=2
                            )

                        # x축 범위 설정
                        ax.set_xlim(current_data['created_at'].min(), current_data['created_at'].max())
                        ax.set_xlabel("날짜")
                        ax.set_ylabel("Value")
                        ax.set_title(f"Sensor Data: {feature}")
                        ax.legend()
                        plt.xticks(rotation=45, fontsize=10)  # x축 레이블 회전
                        ax.grid(True)

                        # Streamlit에 그래프 표시
                        st.pyplot(fig)
                    else:
                        st.warning(f"{feature} 데이터가 없습니다.")
                
                elif view_option == "데이터프레임 보기":
                    if not sensor_period.empty:
                        st.subheader(f"Sensor Period Data: {feature}")
                        st.table(sensor_period)
                    else:
                        st.warning(f"{feature}에 대한 데이터가 없습니다.")
        else:
            st.warning("데이터 크기가 너무 작아 슬라이더를 생성할 수 없습니다.")
    else:
        st.info("그래프로 표시할 컬럼을 선택하세요.")






# 세션 상태 초기화
if 'ship_name_data' not in st.session_state:
    st.session_state.update({
        "ship_name_data": None,
        "start_date": None,
        "end_date": None,
        "selected_device": None,
    })
# 레이아웃 설정
st.set_page_config(layout="wide")  # 와이드 레이아웃 활성화

# 사이드바: 날짜 및 조회 설정
with st.sidebar:
    st.title("Gyro 데이터 조회")
    start_date = st.date_input("시작일", value=datetime.date.today() - datetime.timedelta(days=14))
    end_date = st.date_input("종료일", value=datetime.date.today())

    if st.button("조회"):
        try: 
            with st.spinner("데이터를 불러오는 중입니다... 잠시만 기다려주세요."):
                st.session_state.start_date = start_date
                st.session_state.end_date = end_date
                st.session_state.ship_name_data = get_ship_names(start_date, end_date)
            st.success("데이터를 성공적으로 불러왔습니다!")
        except Exception as e:
            st.error("오류 발생")
        
    
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

            device_opt = ["Device ID를 선택하세요"] + list(ship_name_data['device_id'])
            device_options = [i for i in device_opt if i != "전체 선박"]

            selected_device = st.selectbox("Device ID 선택", device_options)
            st.session_state.selected_device = selected_device

            if selected_device == "Device ID를 선택하세요":
                st.info("Device ID를 선택하세요.")
            else:
                st.session_state.raw_imu_data = get_gyro_data(
                    st.session_state.start_date, st.session_state.end_date, selected_device
                )
                
                data = st.session_state.raw_imu_data.drop_duplicates().sort_values(by='created_at') 
            
                if not data.empty:
                    paging(data)
                else:
                    st.warning("데이터가 없습니다.")
    else:
        st.info("사이드바에서 날짜를 선택하고 '조회'를 클릭하세요.")

with col2:
    # 오른쪽 영역: 빈 공간 유지
    st.empty()
    
    
    
    
    
    
    
    
    
    
    
