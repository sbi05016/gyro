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
import matplotlib.dates as mdates


plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 데이터베이스 설정
HOSTNAME = '203.253.202.70'
PORT = 5432
USERNAME = 'kkilab'
PASSWORD = 'kkilab'
DATABASE = 'safetyship'


# 접속 엔진 생성
con_str_fmt = "postgresql://{0}:{1}@{2}:{3}/{4}"
con_str = con_str_fmt.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)
engine = create_engine(con_str)

# 데이터베이스에서 선박 데이터 가져오기
def get_ship_names(start_date, end_date):
    query = f"""
                            SELECT 
                        u.uid,
                        u.mmsi,
                        r.device_id,
                        COUNT(sensor_value_1) as ea
                    FROM 
                        PUBLIC."Sensor" r
                    JOIN 
                        PUBLIC."User" u
                    ON 
                        r.device_id = u.device_name 
                    WHERE 
                        r.created_at >= '{start_date}' AND r.created_at <= '{end_date} 23:59:59'
                    GROUP BY 
                        u.uid, u.mmsi, r.device_id;
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
        SELECT sensor_name,sensor_value_1,device_id,created_at
        FROM PUBLIC."Sensor"
        WHERE {query_raw} created_at >= '{start_date}' AND created_at <= '{end_date} 23:59:59'
        ORDER BY created_at DESC;
    """
    
    with psycopg2.connect(host=HOSTNAME, dbname=DATABASE, user=USERNAME, password=PASSWORD) as conn:
        raw_imu_data = psql.read_sql(query1, conn)
        
    raw_imu_data.set_index(raw_imu_data['device_id'], inplace=True)
    raw_imu_data.drop(columns=['device_id'], inplace=True)
    

    return raw_imu_data



def sensor_moving_peak(sensor, feature, window_size=1, min_distance=10):
    sensor['smoothed_value'] = sensor[feature].rolling(window=window_size).mean()
    sensor = sensor.dropna(subset=['smoothed_value'])  # 결측값 제거

    # peak 값 구하기
    peaks, _ = find_peaks(sensor['smoothed_value'], distance=min_distance)

    # 주기, 속력 구하기
    period_data = []
 
    sensor_period = pd.DataFrame(columns=['start_idx','end_idx','max_deg','min_deg','deg_diff','period','spd','datetime'])
    for i in range(1,peaks.shape[0]):
        start_idx =  peaks[i-1]
        end_idx =  peaks[i]
        max_deg = sensor.iloc[start_idx:end_idx][feature].max()
        min_deg = sensor.iloc[start_idx:end_idx][feature].min()
        deg_diff = sensor.iloc[start_idx:end_idx][feature].max()-sensor.iloc[start_idx:end_idx][feature].min()
        period = peaks[i] - peaks[i-1]
        feature_spd = deg_diff / period

        
        datetime = sensor.created_at.iloc[peaks[i-1]]
        new_df = pd.DataFrame(columns=['start_idx','end_idx','max_deg','min_deg','deg_diff','period','spd','datetime'],
                                                         data=[[start_idx,end_idx,int(max_deg),int(min_deg),deg_diff,period,feature_spd,datetime]])
       
        sensor_period = pd.concat([sensor_period,new_df])


    # 하위 30%, 상위 30% 제거. 
    lower_quantile = sensor_period.deg_diff.quantile(0.2)
    upper_quantile = sensor_period.deg_diff.quantile(0.8)
    sensor_period =  sensor_period[(sensor_period.deg_diff > lower_quantile) & (sensor_period.deg_diff < upper_quantile)]

    sensor_period = sensor_period[['max_deg','min_deg','spd','datetime']]

    
    return sensor_period, peaks



def paging(data):
    
    if data.shape[0] == 0:
        st.warning("데이터가 없습니다.")
        return
    

    
    elif 0<data.shape[0] < 10:
        st.sidebar.warning("데이터가 너무 작아 페이지당 항목 수를 설정할 수 없습니다.")
        items_per_page = data.shape[0]  # 모든 데이터를 한 페이지로 표시
    
    else:
        if data.shape[0]//10>0:
            default_items_per_page = max(10, data.shape[0] // 10)
        else:
            default_items_per_page = max(1, data.shape[0] // 10)
        items_per_page = st.sidebar.slider("페이지당 항목 수", min_value=1, max_value=data.shape[0], value=default_items_per_page, step=10)
    
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

    # 여러 컬럼 선택 및 그래프 생성
    selected_columns = st.multiselect("그래프로 그릴 컬럼을 선택하세요:", ['Roll', 'Pitch'])
    
    
    if selected_columns:
        for feature in selected_columns:

            current_data2 = current_data[current_data['sensor_name']==feature]
            current_data2 = current_data2.rename(columns={"sensor_value_1": feature})
            current_data2 = current_data2.drop(columns=['sensor_name'])
            
            
            
            # Roll 또는 Pitch 필터링 추가
            # filter_column = st.selectbox("필터링할 컬럼 선택:", ["Roll", "Pitch"], key="filter_column")
            # print(current_data2)
            
            # min_value = st.number_input(f"{filter_column} 최소값", value=current_data[filter_column].min())
            # max_value = st.number_input(f"{filter_column} 최대값", value=current_data[filter_column].max())

            # filtered_data = data[(data[filter_column] >= min_value) & (data[filter_column] <= max_value)]
            # if filtered_data.empty:
            #     st.warning("선택한 범위에 해당하는 데이터가 없습니다.")
            #     return
            
            
            
            # 데이터 크기 계산
            data_size = current_data2.shape[0]
       
            if data_size> 1:  # 슬라이더를 생성할 수 있는 조건
                current_window_size = st.slider(
                    label="빈도 선택",
                    min_value=1,
                    max_value=max(20, data_size // 2),
                    value=min(5, data_size // 2),
                    key=f"window_size_{feature}",  # 고유 key 추가
                    step=1
                )

                current_distance = st.slider(
                    label="피크 간 최소 거리(distance)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    step=1,
                    key=f"distance_{feature}",  # 고유 key 추가
                    help="피크 간 최소 거리를 설정하세요"
                )

                # 데이터 보기 옵션 추가
                view_option = st.radio(
                    "보기 옵션을 선택하세요",
                    options=["그래프 보기", "데이터프레임 보기"],
                    help="그래프 또는 sensor_period 데이터프레임 중에서 선택하세요.",
                    key=f"view_option_{feature}",  # 고유 key 추가
                )

            
                # 각 feature마다 피크 계산 및 데이터프레임 생성
                
            
                sensor_period, peaks = sensor_moving_peak(
                    current_data2,feature,
                    window_size=current_window_size,
                    min_distance=current_distance,
                    
                )
                # 슬라이더로 시간 범위 조절
                min_date = current_data2['created_at'].min().to_pydatetime()  # 최소 날짜
                max_date = current_data2['created_at'].max().to_pydatetime()  # 최대 날짜

                x_axis_range = st.slider(
                    "X축 날짜 범위를 선택하세요",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date),  # 기본값은 전체 범위
                    format="MM/DD HH:mm",
                    step=datetime.timedelta(minutes=1),
                    key=f"x_axis_range_{feature}"
                )

                # 필터링된 데이터
                
                x_start, x_end = x_axis_range
                
                if x_start >= x_end:
                    st.warning("범위가 올바르지 않습니다")
                    return
                
                current_data_filtered = current_data2[
                    (current_data2['created_at'] >= x_start) &
                    (current_data2['created_at'] <= x_end)
                    ]
                
                
                if current_data_filtered.empty:
                    st.warning("선택된 X축 범위 내 데이터가 없습니다. 범위를 변경해주세요.")
                    return
                
                if view_option == "그래프 보기":
                    if sensor_period.shape[0] > 0:
                        
                        
                        
                        # X축 간격 조절 방식 선택
                        x_axis_mode = st.radio(
                            "X축 간격 조절 방식",
                            options=["자동", "수동"],
                            index=0,
                            key=f"x_axis_mode_{feature}"
                        )
                        
                        # 수동 모드 간격 선택 (콤보박스)
                        manual_interval = None
                        if x_axis_mode == "수동":
                            manual_interval = st.selectbox(
                                "수동 간격 선택",
                                options=["1분", "5분", "10분", "30분", "1시간"],
                                index=4,
                                key=f"manual_interval_{feature}"
                            )
                            
                        fig, ax = plt.subplots(figsize=(20, 10))    
                        ax.plot(
                                current_data_filtered['created_at'],
                                current_data_filtered[feature],
                                label=f"{feature} 원본 데이터",
                                linestyle='--',
                                alpha=0.7,
                            )
                        
                        
                        # 평활화 그래프 그리기
                        ax.plot(
                            current_data_filtered['created_at'],
                            current_data_filtered['smoothed_value'],
                            label=f"{feature} 평활화 데이터",
                            zorder=1
                        )

                        # 피크 데이터 그리기
                        
                        if len(peaks) > 0:
                            

                            filtered_peaks, _ = find_peaks(
                            current_data_filtered['smoothed_value'],  # 평활화된 값 기준으로 피크 탐지
                            distance=current_distance  # 슬라이더에서 설정한 최소 거리 사용
                        )

                            

                            if len(filtered_peaks) > 0:
                                # 유효한 피크 인덱스만 사용
                                peak_times = current_data_filtered.iloc[filtered_peaks]['created_at']
                                peak_values = current_data_filtered.iloc[filtered_peaks]['smoothed_value']

                                # 피크 데이터 표시
                                ax.scatter(
                                    peak_times,
                                    peak_values,
                                    color='orange',
                                    s=50,
                                    label=f"{feature} (peaks)",
                                    zorder=2
                                )
                            else:
                                st.warning("선택된 범위 내 유효한 피크 데이터가 없습니다.")
                            
                        else:
                            st.warning("피크 데이터가 없습니다.")
                            
                        # X축 간격 설정
                        if x_axis_mode == "자동":
                            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # 자동 간격 설정
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))  # 날짜 형식 지정
                        elif x_axis_mode == "수동":
                            # 수동 간격 처리
                            interval_map = {
                                "1분": mdates.MinuteLocator(interval=1),
                                "5분": mdates.MinuteLocator(interval=5),
                                "10분": mdates.MinuteLocator(interval=10),
                                "30분": mdates.MinuteLocator(interval=30),
                                "1시간": mdates.HourLocator(interval=1),
                            }
                            ax.xaxis.set_major_locator(interval_map[manual_interval])
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 시간 형식 표시 (시:분)

                        # X축 레이블 회전 및 폰트 크기
                        plt.xticks(rotation=45, fontsize=10)

                        # 그래프 꾸미기
                        ax.set_xlabel("시간")
                        ax.set_ylabel("값")
                        ax.set_title(f"Sensor Data: {feature}")
                        ax.legend()
                        ax.grid(True)

                        # Streamlit에 그래프 표시
                        st.pyplot(fig)
                            
                    else:
                        st.warning(f"{feature} 데이터가 없습니다.")
                
                elif view_option == "데이터프레임 보기":
                    if not sensor_period.empty:
                        st.subheader(f"Sensor Period Data: {feature}")
                        st.table(sensor_period)
                        st.subheader(f"{feature} Data")
                        st.tabel(current_data_filtered)
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
    start_date = st.date_input("시작일", value=datetime.date.today() - datetime.timedelta(days=1))
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
            total_row = pd.DataFrame([{'device_id': '전체 선박', 'uid':'---','mmsi':0,'ea': int(total_users)}])
            ship_name_data = pd.concat([total_row, ship_name_data], ignore_index=True)
            ship_name_data.rename(columns={'uid':'사용자명','ea': '개수'}, inplace=True)

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
    
    
    
    
    
    
    
    
    
    
    
