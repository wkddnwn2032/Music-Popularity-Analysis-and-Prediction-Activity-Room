
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ==========================================
# 1. 기본 설정 및 디자인
# ==========================================

st.title("🔎 음악 및 장르 검색")
st.write("""
원하는 장르, 가수, 음악 제목을 검색하고
검색한 음악의 인기도와 음악적 특징을 전체 데이터와 비교해 보세요.
""")

GREEN = "#1DB954"

st.markdown("""
<style>
    .stApp {
        background-color: #FFFFFF;
        color: #222222;
    }

    h1, h2, h3 {
        color: #191919 !important;
    }

    [data-testid="stMetric"] {
        background-color: #F1F8F3;
        border: 1px solid #DCE8DF;
        padding: 15px;
        border-radius: 10px;
    }

    [data-testid="stMetricValue"] {
        color: #15803D !important;
    }

    .stButton > button {
        background-color: #1DB954;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #15803D;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. 데이터 불러오기
# ==========================================

DATA_URL = "https://raw.githubusercontent.com/wkddnwn2032/spotify-music-data1/refs/heads/main/spotify-tracks-dataset.csv"


@st.cache_data
def load_data():

    data = pd.read_csv(DATA_URL)

    # 불필요한 인덱스 열 제거
    data = data.loc[
        :,
        ~data.columns.astype(str).str.startswith("Unnamed")
    ]

    data = data.drop_duplicates()

    # 숫자형 변수 변환
    numeric_columns = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature"
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

    # 문자열 결측값 정리
    text_columns = [
        "track_name",
        "artists",
        "track_genre"
    ]

    for col in text_columns:
        if col in data.columns:
            data[col] = data[col].fillna("정보 없음").astype(str)

    data = data.dropna(subset=["popularity"])

    return data


try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(error)
    st.stop()


required_columns = [
    "popularity",
    "track_name",
    "artists",
    "track_genre"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "데이터에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# ==========================================
# 3. 검색 조건
# ==========================================

st.divider()
st.header("🎵 원하는 음악 검색하기")

st.write(
    "검색하고 싶은 조건을 입력하거나 선택하세요. "
    "조건을 여러 개 사용하면 모든 조건을 만족하는 음악만 검색됩니다."
)

all_genres = sorted(
    df["track_genre"].dropna().unique().tolist()
)

all_artists = sorted(
    df["artists"].dropna().unique().tolist()
)

col1, col2 = st.columns(2)

with col1:

    selected_genres = st.multiselect(
        "음악 장르 선택",
        options=all_genres,
        placeholder="장르를 선택하세요. 여러 개 선택 가능"
    )

    artist_keyword = st.text_input(
        "가수 이름 검색",
        placeholder="예: Taylor Swift"
    )

with col2:

    track_keyword = st.text_input(
        "음악 제목 검색",
        placeholder="예: Love"
    )

    min_popularity = st.slider(
        "최소 인기도",
        min_value=0,
        max_value=100,
        value=0
    )

    max_popularity = st.slider(
        "최대 인기도",
        min_value=0,
        max_value=100,
        value=100
    )


# ==========================================
# 4. 가수 선택 기능
# ==========================================

st.subheader("가수 선택")

if artist_keyword.strip():

    matching_artists = [
        artist for artist in all_artists
        if artist_keyword.strip().lower() in artist.lower()
    ]

else:
    matching_artists = all_artists

artist_options = ["전체"] + matching_artists[:500]

if len(matching_artists) > 500:
    st.caption(
        f"검색된 가수가 {len(matching_artists):,}명입니다. "
        "선택 목록에는 처음 500명만 표시합니다. "
        "가수 이름을 더 구체적으로 입력해 보세요."
    )

selected_artist = st.selectbox(
    "검색된 가수 중 정확한 이름 선택",
    options=artist_options,
    index=0
)


# ==========================================
# 5. 검색 결과 만들기
# ==========================================

filtered_df = df.copy()

# 장르 조건
if selected_genres:

    filtered_df = filtered_df[
        filtered_df["track_genre"].isin(selected_genres)
    ]

# 가수 검색어 조건
if artist_keyword.strip():

    filtered_df = filtered_df[
        filtered_df["artists"].str.contains(
            artist_keyword.strip(),
            case=False,
            na=False,
            regex=False
        )
    ]

# 특정 가수 선택 조건
if selected_artist != "전체":

    filtered_df = filtered_df[
        filtered_df["artists"] == selected_artist
    ]

# 음악 제목 조건
if track_keyword.strip():

    filtered_df = filtered_df[
        filtered_df["track_name"].str.contains(
            track_keyword.strip(),
            case=False,
            na=False,
            regex=False
        )
    ]

# 인기도 조건
filtered_df = filtered_df[
    filtered_df["popularity"].between(
        min_popularity,
        max_popularity
    )
]

filtered_df = filtered_df.copy()


# ==========================================
# 6. 검색 결과 요약
# ==========================================

st.divider()
st.header("📋 검색 결과")

total_count = len(df)
result_count = len(filtered_df)

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 음악 수",
    f"{total_count:,}곡"
)

col2.metric(
    "검색 결과 음악 수",
    f"{result_count:,}곡"
)

col3.metric(
    "검색 결과 비율",
    f"{result_count / total_count * 100:.2f}%"
    if total_count > 0 else "0%"
)


# ==========================================
# 7. 검색된 음악 목록
# ==========================================

if result_count > 0:

    st.subheader("검색된 음악 목록")

    display_columns = [
        col for col in [
            "track_name",
            "artists",
            "track_genre",
            "popularity",
            "danceability",
            "energy",
            "tempo"
        ]
        if col in filtered_df.columns
    ]

    result_table = (
        filtered_df[display_columns]
        .sort_values(
            "popularity",
            ascending=False
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        result_table.head(500),
        use_container_width=True,
        hide_index=True
    )

    if result_count > 500:
        st.caption(
            "화면에는 인기도가 높은 상위 500곡만 표시합니다. "
            "CSV 파일을 다운로드하면 전체 검색 결과를 확인할 수 있습니다."
        )

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 검색 결과 CSV 다운로드",
        data=csv_data,
        file_name="spotify_search_results.csv",
        mime="text/csv"
    )

else:

    st.warning(
        "검색 조건에 맞는 음악이 없습니다. "
        "검색 조건을 변경해 주세요."
    )

    st.stop()


# ==========================================
# 8. 전체 데이터 및 검색 결과 비교
# ==========================================

st.divider()
st.header("📊 전체 데이터와 검색 결과 비교")

st.write("""
검색된 음악의 평균 인기도가 전체 데이터의 평균보다 높은지,
낮은지 비교할 수 있습니다.
""")

overall_mean = df["popularity"].mean()
search_mean = filtered_df["popularity"].mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 평균 인기도",
    f"{overall_mean:.2f}점"
)

col2.metric(
    "검색 결과 평균 인기도",
    f"{search_mean:.2f}점"
)

col3.metric(
    "평균 인기도 차이",
    f"{search_mean - overall_mean:+.2f}점"
)

comparison_data = pd.DataFrame({
    "구분": [
        "전체 데이터",
        "검색 결과"
    ],
    "평균 인기도": [
        overall_mean,
        search_mean
    ],
    "음악 수": [
        len(df),
        len(filtered_df)
    ]
})

fig_comparison = px.bar(
    comparison_data,
    x="구분",
    y="평균 인기도",
    color="구분",
    text="평균 인기도",
    color_discrete_sequence=[
        "#A7A7A7",
        GREEN
    ],
    title="전체 데이터와 검색 결과의 평균 인기도 비교"
)

fig_comparison.update_traces(
    texttemplate="%{text:.2f}점",
    textposition="outside"
)

fig_comparison.update_layout(
    template="plotly_white",
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(color="#222222"),
    showlegend=False,
    yaxis=dict(
        title="평균 인기도",
        range=[0, 110],
        gridcolor="#EAEAEA"
    ),
    xaxis_title=""
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)


# ==========================================
# 9. 선택 장르와 검색 결과 비교
# ==========================================

st.divider()
st.header("🎼 장르별 평균 인기도 비교")

st.write("""
선택한 장르에 속한 전체 음악과 검색 결과를 비교합니다.
가수나 음악 제목으로 검색했다면 해당 조건에 맞는 음악의 평균 인기도도 함께 확인할 수 있습니다.
""")

if selected_genres:

    genre_df = df[
        df["track_genre"].isin(selected_genres)
    ].copy()

    genre_mean = genre_df["popularity"].mean()

    genre_comparison = pd.DataFrame({
        "구분": [
            "전체 데이터",
            "선택한 장르",
            "검색 결과"
        ],
        "평균 인기도": [
            overall_mean,
            genre_mean,
            search_mean
        ],
        "음악 수": [
            len(df),
            len(genre_df),
            len(filtered_df)
        ]
    })

    fig_genre = px.bar(
        genre_comparison,
        x="구분",
        y="평균 인기도",
        color="구분",
        text="평균 인기도",
        color_discrete_sequence=[
            "#A7A7A7",
            "#15803D",
            GREEN
        ],
        title="전체 데이터 · 선택 장르 · 검색 결과 비교"
    )

    fig_genre.update_traces(
        texttemplate="%{text:.2f}점",
        textposition="outside"
    )

    fig_genre.update_layout(
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#222222"),
        showlegend=False,
        yaxis=dict(
            title="평균 인기도",
            range=[0, 110],
            gridcolor="#EAEAEA"
        ),
        xaxis_title=""
    )

    st.plotly_chart(
        fig_genre,
        use_container_width=True
    )

    genre_summary = pd.DataFrame({
        "비교 대상": [
            "전체 데이터",
            "선택한 장르",
            "검색 결과"
        ],
        "음악 수": [
            len(df),
            len(genre_df),
            len(filtered_df)
        ],
        "평균 인기도": [
            round(overall_mean, 2),
            round(genre_mean, 2),
            round(search_mean, 2)
        ]
    })

    st.dataframe(
        genre_summary,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "장르를 선택하면 해당 장르 전체의 평균 인기도와 "
        "검색 결과를 비교할 수 있습니다."
    )


# ==========================================
# 10. 음악적 특징 비교
# ==========================================

st.divider()
st.header("🎹 음악적 특징 비교")

st.write("""
검색 결과에 포함된 음악들은 전체 음악과 비교했을 때
어떤 음악적 특징을 가지고 있을까요?

각 음악적 특징의 평균값을 계산하여 비교합니다.
""")

feature_columns = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

available_features = [
    col for col in feature_columns
    if col in df.columns
]

feature_labels = {
    "danceability": "댄서빌리티",
    "energy": "에너지",
    "loudness": "음량",
    "speechiness": "말소리 포함 정도",
    "acousticness": "어쿠스틱 정도",
    "instrumentalness": "연주곡 정도",
    "liveness": "라이브 정도",
    "valence": "음악의 긍정적인 분위기",
    "tempo": "템포"
}

if available_features:

    feature_means = []

    for feature in available_features:

        feature_means.append({
            "음악적 특징": feature_labels.get(
                feature, feature
            ),
            "전체 데이터": df[feature].mean(),
            "검색 결과": filtered_df[feature].mean()
        })

    feature_df = pd.DataFrame(feature_means)

    feature_long = feature_df.melt(
        id_vars="음악적 특징",
        value_vars=[
            "전체 데이터",
            "검색 결과"
        ],
        var_name="비교 대상",
        value_name="평균값"
    )

    fig_features = px.bar(
        feature_long,
        x="음악적 특징",
        y="평균값",
        color="비교 대상",
        barmode="group",
        color_discrete_map={
            "전체 데이터": "#A7A7A7",
            "검색 결과": GREEN
        },
        title="전체 데이터와 검색 결과의 음악적 특징 비교",
        hover_data={
            "평균값": ":.3f"
        }
    )

    fig_features.update_layout(
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#222222"),
        xaxis_title="음악적 특징",
        yaxis_title="평균값",
        xaxis_tickangle=-30,
        legend_title=""
    )

    fig_features.update_xaxes(
        showgrid=False
    )

    fig_features.update_yaxes(
        gridcolor="#EAEAEA"
    )

    st.plotly_chart(
        fig_features,
        use_container_width=True
    )

    st.dataframe(
        feature_df.round(3),
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 11. 검색 결과 인기도 분포
# ==========================================

st.divider()
st.header("📈 검색 결과의 인기도 분포")

st.write(
    "검색한 음악들의 인기도가 어느 구간에 많이 분포하는지 확인합니다."
)

fig_hist = px.histogram(
    filtered_df,
    x="popularity",
    nbins=20,
    color_discrete_sequence=[GREEN],
    title="검색 결과 인기도 분포"
)

fig_hist.update_layout(
    template="plotly_white",
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(color="#222222"),
    xaxis_title="인기도",
    yaxis_title="음악 수",
    bargap=0.08
)

fig_hist.update_xaxes(
    gridcolor="#EAEAEA"
)

fig_hist.update_yaxes(
    gridcolor="#EAEAEA"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)


# ==========================================
# 12. 마무리
# ==========================================

st.divider()

st.subheader("💡 검색 결과 분석 안내")

st.write("""
이 페이지에서는 원하는 음악을 장르, 가수, 제목 및 인기도 조건으로 검색할 수 있습니다.

또한 검색 결과를 전체 데이터와 비교하여 평균 인기도와 음악적 특징의 차이를 확인할 수 있습니다.

다만 평균값의 차이가 특정 음악적 특징 때문에 인기도가 높아졌다는 인과관계를 의미하지는 않습니다.
검색 결과의 음악 수와 장르 구성도 함께 고려해야 합니다.
""")

st.caption(
    "자료: Spotify Tracks Dataset"
)
