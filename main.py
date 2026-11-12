
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 기본 설정
# ==========================================

st.set_page_config(
    page_title="음악 인기도 분석 및 예측 실습실",
    page_icon="🎧",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/wkddnwn2032/spotify-music-data1/refs/heads/main/spotify-tracks-dataset.csv"

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

    p, label, span, li {
        color: #333333;
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

    .description {
        background-color: #F1F8F3;
        border-left: 5px solid #1DB954;
        color: #222222;
        padding: 15px;
        border-radius: 7px;
        margin-bottom: 12px;
    }

    .description b {
        color: #15803D;
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
# 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)

    data = data.loc[
        :,
        ~data.columns.astype(str).str.startswith("Unnamed")
    ]

    data = data.drop_duplicates()

    numeric_columns = [
        "duration_ms", "popularity", "danceability",
        "energy", "key", "loudness", "mode",
        "speechiness", "acousticness",
        "instrumentalness", "liveness",
        "valence", "tempo", "time_signature"
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            )

    if "explicit" in data.columns:
        data["explicit"] = (
            data["explicit"]
            .astype(str)
            .str.lower()
            .map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0
            })
        )

    data = data.dropna(subset=["popularity"])

    return data


try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.write(error)
    st.info(
        "GitHub 저장소가 Public인지, CSV 주소가 정확한지 확인해 주세요."
    )
    st.stop()


if "popularity" not in df.columns:
    st.error("popularity 열을 찾을 수 없습니다.")
    st.stop()


# ==========================================
# 제목 및 데이터 요약
# ==========================================

st.title("🎧 음악 인기도 분석 및 예측 실습실")

st.write("""
음악에는 춤추기 좋은 정도, 에너지, 분위기, 템포 등
다양한 음악적 특징이 존재합니다.

이러한 음악적 특징은 음악의 인기도와 어떤 관계가 있을까요?

Spotify 음악 데이터를 활용하여 음악의 특징과 인기도 사이의
관계를 분석하고 인기도를 예측해 봅니다.
""")

st.divider()

st.header("📊 데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 음악 수", f"{len(df):,}곡")
col2.metric("평균 인기도", f"{df['popularity'].mean():.2f}점")
col3.metric("최고 인기도", f"{df['popularity'].max():.0f}점")

genre_count = (
    df["track_genre"].nunique()
    if "track_genre" in df.columns else 0
)

col4.metric("음악 장르 수", f"{genre_count:,}개")


# ==========================================
# 1. 주요 변수 살펴보기
# ==========================================

st.divider()
st.header("1. 주요 변수 살펴보기")

st.write("데이터에 포함된 주요 변수와 각 변수의 의미입니다.")

variable_info = {
    "track_name": ["음악 제목", "문자형"],
    "artists": ["가수 또는 아티스트 이름", "문자형"],
    "track_genre": ["음악 장르", "문자형"],
    "popularity": ["Spotify에서 측정한 음악 인기도 (0~100)", "정수형"],
    "duration_ms": ["음악 재생 시간 (밀리초)", "정수형"],
    "danceability": ["춤추기에 적합한 정도 (0~1)", "실수형"],
    "energy": ["음악의 에너지와 강렬함 (0~1)", "실수형"],
    "key": ["음악의 조성", "정수형"],
    "loudness": ["음악의 평균 음량 (dB)", "실수형"],
    "mode": ["장조 또는 단조 여부", "정수형"],
    "speechiness": ["말소리가 포함된 정도", "실수형"],
    "acousticness": ["어쿠스틱 음악일 가능성 (0~1)", "실수형"],
    "instrumentalness": ["연주곡일 가능성 (0~1)", "실수형"],
    "liveness": ["라이브 공연 녹음일 가능성 (0~1)", "실수형"],
    "valence": ["음악의 밝고 긍정적인 분위기 (0~1)", "실수형"],
    "tempo": ["음악의 빠르기 (BPM)", "실수형"],
    "time_signature": ["음악의 박자", "정수형"],
    "explicit": ["욕설이나 성인용 표현 포함 여부", "논리형"]
}

variable_rows = []

for col, info in variable_info.items():
    if col in df.columns:
        variable_rows.append({
            "변수 이름": col,
            "의미": info[0],
            "데이터 유형": info[1]
        })

st.dataframe(
    pd.DataFrame(variable_rows),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 2. 인기도 TOP 10
# ==========================================

st.divider()
st.header("2. 인기도가 가장 높은 음악 TOP 10")

if "track_name" in df.columns:

    top10 = df.sort_values(
        "popularity",
        ascending=False
    ).head(10).copy()

    show_columns = [
        col for col in [
            "track_name",
            "artists",
            "track_genre",
            "popularity"
        ]
        if col in top10.columns
    ]

    top10 = top10[show_columns].reset_index(drop=True)
    top10.index = top10.index + 1
    top10.index.name = "순위"

    st.dataframe(
        top10,
        use_container_width=True
    )

else:
    st.warning("음악 제목을 표시할 수 없습니다.")


# ==========================================
# 3. 장르별 인기도 비교 (트리맵)
# ==========================================

st.divider()
st.header("3. 장르별 인기도 비교")

st.write("""
장르별 평균 인기도를 계산하여 인기도가 높은 상위 20개 장르를 비교합니다.
사각형의 넓이가 클수록 평균 인기도가 높습니다.
""")

if "track_genre" in df.columns:

    genre_popularity = (
        df.dropna(subset=["track_genre", "popularity"])
        .groupby("track_genre", as_index=False)
        .agg(
            **{
                "평균 인기도": ("popularity", "mean"),
                "곡 수": ("popularity", "count")
            }
        )
        .sort_values(
            "평균 인기도",
            ascending=False
        )
        .head(20)
        .reset_index(drop=True)
    )

    if not genre_popularity.empty:

        # 장르별 서로 다른 색상
        genre_colors = (
            px.colors.qualitative.Alphabet
            + px.colors.qualitative.Dark24
        )[:len(genre_popularity)]

        # 평균 인기도를 숫자형으로 명확하게 변환
        genre_values = genre_popularity[
            "평균 인기도"
        ].astype(float).tolist()

        genre_names = genre_popularity[
            "track_genre"
        ].astype(str).tolist()

        genre_counts = genre_popularity[
            "곡 수"
        ].astype(int).tolist()

        # Graph Objects를 이용해 면적을 직접 지정
        fig_treemap = go.Figure(
            go.Treemap(
                labels=genre_names,
                parents=[""] * len(genre_names),

                # 사각형의 넓이를 평균 인기도에 비례시킴
                values=genre_values,

                # 각 장르별 서로 다른 색상
                marker=dict(
                    colors=genre_colors,
                    line=dict(
                        color="#FFFFFF",
                        width=3
                    )
                ),

                # 사각형 안에 장르명과 인기도 표시
                texttemplate=(
                    "%{label}<br>%{value:.1f}점"
                ),

                textfont=dict(
                    size=14,
                    color="#FFFFFF"
                ),

                # 마우스를 올렸을 때 표시할 정보
                customdata=genre_counts,

                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "평균 인기도: %{value:.2f}점<br>"
                    "곡 수: %{customdata:,}곡"
                    "<extra></extra>"
                ),

                # 큰 값부터 정렬
                sort=True,

                # 사각형 배치 방식
                tiling=dict(
                    packing="squarify",
                    pad=3
                )
            )
        )

        fig_treemap.update_layout(
            template="plotly_white",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#222222"),
            title=dict(
                text="평균 인기도 TOP 20 장르",
                font=dict(
                    color="#191919",
                    size=20
                )
            ),
            margin=dict(
                t=55,
                l=10,
                r=10,
                b=10
            ),
            height=550
        )

        st.plotly_chart(
            fig_treemap,
            use_container_width=True
        )

        st.caption(
            "사각형의 넓이는 평균 인기도에 비례합니다. "
            "각 장르는 서로 다른 색상으로 구분했으며, "
            "마우스를 올리면 평균 인기도와 곡 수를 확인할 수 있습니다."
        )

    else:
        st.info("장르별 인기도를 계산할 데이터가 없습니다.")

else:
    st.warning(
        "track_genre 열을 찾을 수 없어 장르별 그래프를 표시할 수 없습니다."
    )


# ==========================================
# 4. 인기도 분포 그래프
# ==========================================

st.divider()
st.header("4. 인기도에 따른 음악 수의 분포")

st.write(
    "인기도를 구간별로 나누어 각 구간에 해당하는 음악이 "
    "얼마나 많은지 살펴봅니다."
)

bins = [-1, 0, 20, 40, 60, 80, 100]

labels = [
    "0점",
    "1~20점",
    "21~40점",
    "41~60점",
    "61~80점",
    "81~100점"
]

df["인기도 구간"] = pd.cut(
    df["popularity"],
    bins=bins,
    labels=labels
)

popularity_count = (
    df["인기도 구간"]
    .value_counts(sort=False)
    .reindex(labels, fill_value=0)
    .reset_index()
)

popularity_count.columns = [
    "인기도 구간",
    "음악 수"
]

fig = px.bar(
    popularity_count,
    x="인기도 구간",
    y="음악 수",
    text="음악 수",
    color="음악 수",
    color_continuous_scale=[
        "#14532D",
        "#15803D",
        "#1DB954",
        "#86EFAC"
    ],
    title="인기도 구간별 음악 수"
)

fig.update_traces(
    texttemplate="%{text:,}곡",
    textposition="outside"
)

fig.update_layout(
    template="plotly_white",
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(color="#222222"),
    title_font=dict(color="#191919", size=20),
    coloraxis_showscale=False,
    xaxis_title="인기도",
    yaxis_title="음악 수"
)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="#EAEAEA")

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 5. 그래프 해석
# ==========================================

st.subheader("💡 그래프를 통해 알 수 있는 내용")

total = popularity_count["음악 수"].sum()

if total > 0:

    highest_group = popularity_count.loc[
        popularity_count["음악 수"].idxmax()
    ]

    highest_ratio = (
        highest_group["음악 수"] / total * 100
    )

    low_count = popularity_count.loc[
        popularity_count["인기도 구간"].isin(
            ["0점", "1~20점", "21~40점"]
        ),
        "음악 수"
    ].sum()

    low_ratio = low_count / total * 100

    st.markdown(f"""
    <div class="description">
    <b>① 음악이 가장 많이 분포한 구간</b><br>
    인기도 {highest_group['인기도 구간']} 구간에
    <b>{highest_group['음악 수']:,}곡</b>이 존재하며,
    전체 음악 중 약 <b>{highest_ratio:.1f}%</b>를 차지합니다.
    </div>

    <div class="description">
    <b>② 인기도가 낮은 음악의 비율</b><br>
    인기도 0~40점에 해당하는 음악은
    <b>{low_count:,}곡</b>으로,
    전체의 약 <b>{low_ratio:.1f}%</b>입니다.
    </div>

    <div class="description">
    <b>③ 분포의 특징</b><br>
    인기도가 높은 음악과 낮은 음악이 고르게 분포하는지,
    특정 구간에 집중되어 있는지를 확인할 수 있습니다.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 6. 음악적 특징과 인기도의 관계
# ==========================================

st.divider()
st.header("6. 음악적 특징과 인기도의 관계")

st.write(
    "상관계수를 이용하여 음악의 여러 특징과 인기도 사이의 관계를 분석합니다."
)

feature_columns = [
    "duration_ms",
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

if available_features:

    corr_data = df[
        available_features + ["popularity"]
    ].corr(numeric_only=True)

    popularity_corr = (
        corr_data["popularity"]
        .drop("popularity")
        .sort_values()
    )

    corr_df = popularity_corr.reset_index()
    corr_df.columns = [
        "음악적 특징",
        "상관계수"
    ]

    fig_corr = px.bar(
        corr_df,
        x="상관계수",
        y="음악적 특징",
        orientation="h",
        color="상관계수",
        color_continuous_scale=[
            "#15803D",
            "#1DB954",
            "#86EFAC"
        ],
        title="음악적 특징과 인기도의 상관관계"
    )

    fig_corr.update_layout(
        template="plotly_white",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#222222"),
        title_font=dict(color="#191919", size=20),
        coloraxis_showscale=False,
        xaxis_title="상관계수 (-1 ~ +1)",
        yaxis_title="음악적 특징"
    )

    fig_corr.update_xaxes(
        gridcolor="#EAEAEA",
        zeroline=True,
        zerolinecolor="#888888"
    )

    fig_corr.update_yaxes(showgrid=False)

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    st.caption(
        "상관계수가 -1에 가까울수록 음의 관계, "
        "+1에 가까울수록 양의 관계입니다. "
        "0에 가까우면 선형적인 관계가 약하다는 뜻입니다."
    )

    strongest = popularity_corr.abs().idxmax()
    strongest_value = popularity_corr[strongest]

    positive_features = popularity_corr[
        popularity_corr > 0
    ]

    negative_features = popularity_corr[
        popularity_corr < 0
    ]

    st.subheader("📌 상관관계 분석 결과")

    st.markdown(f"""
    <div class="description">
    <b>① 가장 뚜렷한 관계</b><br>
    분석한 특징 중 <b>{strongest}</b>가 인기도와
    가장 큰 절댓값의 상관계수를 보였습니다.
    상관계수는 <b>{strongest_value:.3f}</b>입니다.
    </div>
    """, unsafe_allow_html=True)

    if len(positive_features) > 0:
        st.write(
            "**양의 상관관계를 보이는 특징:** "
            + ", ".join(positive_features.index.tolist())
        )

    if len(negative_features) > 0:
        st.write(
            "**음의 상관관계를 보이는 특징:** "
            + ", ".join(negative_features.index.tolist())
        )

    if abs(strongest_value) < 0.1:
        st.info(
            "분석한 음악적 특징과 인기도 사이의 선형적 상관관계는 "
            "전반적으로 약한 편입니다. 인기도는 한 가지 음악적 "
            "특징만으로 설명하기 어려울 수 있습니다."
        )


# ==========================================
# 7. 머신러닝으로 음악 인기도 예측
# ==========================================

st.divider()
st.header("7. 머신러닝으로 음악 인기도 예측하기")

st.write("""
음악적 특징을 입력 변수로 사용하여 인기도를 예측하는 모델을 만듭니다.

- **선형 회귀:** 음악적 특징과 인기도 사이의 선형적인 관계를 학습합니다.
- **랜덤 포레스트:** 여러 의사결정나무의 예측을 결합하여 복잡한 관계를 학습합니다.
""")

model_candidates = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "explicit",
    "key",
    "mode",
    "time_signature"
]

model_features = [
    col for col in model_candidates
    if col in df.columns
]

if model_features:

    default_features = [
        col for col in [
            "danceability",
            "energy",
            "loudness",
            "acousticness",
            "valence",
            "tempo"
        ]
        if col in model_features
    ]

    selected_features = st.multiselect(
        "예측에 사용할 음악적 특징을 선택하세요.",
        options=model_features,
        default=default_features
    )

    st.caption(
        "같은 음악이 학습 데이터와 평가 데이터에 동시에 "
        "들어가는 문제를 줄이기 위해 track_id 기준으로 나눕니다."
    )

    if st.button(
        "🎯 모델 학습 및 예측 결과 확인",
        type="primary"
    ):

        if not selected_features:
            st.warning("최소 한 개 이상의 특징을 선택해 주세요.")

        else:

            model_df = df.copy()

            for col in selected_features:
                model_df[col] = pd.to_numeric(
                    model_df[col],
                    errors="coerce"
                )

            model_df = model_df.dropna(
                subset=["popularity"]
            )

            model_df = model_df.loc[
                model_df[selected_features].notna().any(axis=1)
            ]

            if len(model_df) > 30000:
                model_df = model_df.sample(
                    n=30000,
                    random_state=42
                )

            if len(model_df) < 10:
                st.error("모델을 학습하기 위한 데이터가 부족합니다.")
                st.stop()

            X = model_df[selected_features]
            y = model_df["popularity"]

            # track_id 기준으로 데이터 분리
            if "track_id" in model_df.columns:

                groups = model_df["track_id"].astype("string")

                missing_group_values = pd.Series(
                    [
                        "missing_" + str(idx)
                        for idx in model_df.index
                    ],
                    index=model_df.index,
                    dtype="string"
                )

                groups = groups.fillna(
                    missing_group_values
                ).to_numpy()

                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=0.2,
                    random_state=42
                )

                try:
                    train_idx, test_idx = next(
                        splitter.split(X, y, groups)
                    )

                except ValueError:
                    st.warning(
                        "track_id 기준 분할이 어려워 "
                        "일반 무작위 분할을 사용합니다."
                    )

                    train_idx, test_idx = train_test_split(
                        range(len(X)),
                        test_size=0.2,
                        random_state=42
                    )

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42
                )

            # 선형 회귀 모델
            linear_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression())
            ])

            # 랜덤 포레스트 모델
            forest_model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(
                    n_estimators=50,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ))
            ])

            with st.spinner(
                "모델을 학습하고 있습니다. 잠시 기다려 주세요."
            ):

                linear_model.fit(X_train, y_train)
                forest_model.fit(X_train, y_train)

                linear_pred = linear_model.predict(X_test)
                forest_pred = forest_model.predict(X_test)

            def evaluate_model(name, actual, predicted):

                return {
                    "모델": name,
                    "MAE": mean_absolute_error(
                        actual, predicted
                    ),
                    "RMSE": mean_squared_error(
                        actual, predicted
                    ) ** 0.5,
                    "R²": r2_score(
                        actual, predicted
                    )
                }

            results = pd.DataFrame([
                evaluate_model(
                    "선형 회귀",
                    y_test,
                    linear_pred
                ),
                evaluate_model(
                    "랜덤 포레스트",
                    y_test,
                    forest_pred
                )
            ])

            st.subheader("모델 성능 비교")

            st.dataframe(
                results.style.format({
                    "MAE": "{:.2f}",
                    "RMSE": "{:.2f}",
                    "R²": "{:.3f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            best_index = results["MAE"].idxmin()
            best_model_name = results.loc[
                best_index, "모델"
            ]

            if best_model_name == "선형 회귀":
                best_model = linear_model
                best_pred = linear_pred
            else:
                best_model = forest_model
                best_pred = forest_pred

            st.success(
                f"MAE 기준으로 더 나은 모델은 "
                f"{best_model_name}입니다."
            )

            st.caption(
                "MAE와 RMSE는 낮을수록 좋고, R²는 일반적으로 "
                "1에 가까울수록 설명력이 높습니다."
            )

            # 실제값과 예측값 비교
            compare_df = pd.DataFrame({
                "실제 인기도": y_test,
                "예측 인기도": best_pred
            })

            fig_compare = px.scatter(
                compare_df,
                x="실제 인기도",
                y="예측 인기도",
                opacity=0.5,
                title=f"{best_model_name}: 실제 인기도와 예측 인기도 비교"
            )

            fig_compare.add_trace(
                go.Scatter(
                    x=[0, 100],
                    y=[0, 100],
                    mode="lines",
                    name="이상적인 예측",
                    line=dict(
                        color=GREEN,
                        dash="dash"
                    )
                )
            )

            fig_compare.update_layout(
                template="plotly_white",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#222222"),
                title_font=dict(color="#191919", size=20),
                xaxis_title="실제 인기도",
                yaxis_title="예측 인기도"
            )

            fig_compare.update_xaxes(
                gridcolor="#EAEAEA"
            )

            fig_compare.update_yaxes(
                gridcolor="#EAEAEA"
            )

            st.plotly_chart(
                fig_compare,
                use_container_width=True
            )

            # 변수 중요도
            st.subheader("랜덤 포레스트의 변수 중요도")

            importance = forest_model.named_steps[
                "model"
            ].feature_importances_

            importance_df = pd.DataFrame({
                "음악적 특징": selected_features,
                "중요도": importance
            }).sort_values(
                "중요도",
                ascending=True
            )

            fig_importance = px.bar(
                importance_df,
                x="중요도",
                y="음악적 특징",
                orientation="h",
                color="중요도",
                color_continuous_scale=[
                    "#14532D",
                    "#1DB954",
                    "#86EFAC"
                ],
                title="인기도 예측에 사용된 변수 중요도"
            )

            fig_importance.update_layout(
                template="plotly_white",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#222222"),
                title_font=dict(color="#191919", size=20),
                coloraxis_showscale=False
            )

            fig_importance.update_xaxes(
                gridcolor="#EAEAEA"
            )

            fig_importance.update_yaxes(
                showgrid=False
            )

            st.plotly_chart(
                fig_importance,
                use_container_width=True
            )

            st.caption(
                "변수 중요도는 모델이 예측할 때 각 변수를 얼마나 "
                "활용했는지를 나타냅니다. 인과관계를 의미하지는 않습니다."
            )

            # 새로운 음악 인기도 예측
            st.divider()
            st.subheader("🎵 새로운 음악의 인기도 예측")

            st.write(
                "음악적 특징을 입력하면 학습된 모델이 예상 인기도를 계산합니다."
            )

            input_values = {}
            input_columns = st.columns(2)

            for i, feature in enumerate(selected_features):

                with input_columns[i % 2]:

                    valid_values = model_df[
                        feature
                    ].dropna()

                    if len(valid_values) == 0:
                        default_value = 0.0
                        min_value = 0.0
                        max_value = 1.0

                    else:
                        default_value = float(
                            valid_values.median()
                        )

                        min_value = float(
                            valid_values.min()
                        )

                        max_value = float(
                            valid_values.max()
                        )

                    if min_value == max_value:
                        max_value = min_value + 1.0

                    input_values[feature] = st.number_input(
                        feature,
                        min_value=min_value,
                        max_value=max_value,
                        value=max(
                            min_value,
                            min(max_value, default_value)
                        ),
                        key=f"input_{feature}"
                    )

            if st.button("새로운 음악의 인기도 예측하기"):

                new_music = pd.DataFrame(
                    [input_values],
                    columns=selected_features
                )

                prediction = best_model.predict(
                    new_music
                )[0]

                prediction = max(
                    0,
                    min(100, prediction)
                )

                st.metric(
                    "예상 인기도",
                    f"{prediction:.1f}점"
                )

                st.caption(
                    "예측값은 0~100점 범위로 표시했습니다. "
                    "실제 Spotify 인기도와 다를 수 있습니다."
                )


# ==========================================
# 8. 최종 해석
# ==========================================

st.divider()
st.header("8. 분석 결과 종합 해석")

st.write("""
음악의 인기도는 춤추기 좋은 정도, 에너지, 음량, 템포와 같은
다양한 음악적 특징과 관련될 수 있습니다.

하지만 상관관계 분석에서 나타난 수치는 두 변수 사이의
선형적인 관계를 보여줄 뿐, 특정 특징이 인기도를 직접 높이거나
낮춘다는 사실을 증명하지는 않습니다.

또한 음악의 인기도는 음악 자체의 특징뿐 아니라 가수의 인지도,
홍보, 발매 시기, 유행, 청취자의 취향 등 여러 요인의 영향을 받을 수 있습니다.

따라서 음악적 특징만으로 인기도를 완벽하게 예측하기는 어렵지만,
데이터 분석과 머신러닝을 활용하면 음악의 특징과 인기도 사이의
관계를 탐색하고 예측 가능성을 확인할 수 있습니다.
""")

st.caption(
    "자료: Spotify Tracks Dataset | "
    "분석 결과는 해당 데이터셋에 포함된 음악을 기준으로 합니다."
)
