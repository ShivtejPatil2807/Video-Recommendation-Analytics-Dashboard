from pathlib import Path
import sys

# Ensure project root (C:\Users\Shivtej\video-recommender) is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.express as px
import streamlit as st

from src.analytics import (
    analyze_user_category_preferences,
    calculate_global_kpis,
    get_daily_engagement_trends,
    get_trending_videos,
    load_data,
)
from src.data_generator import generate_datasets
from src.recommender import VideoRecommender
import plotly.express as px
import streamlit as st

from src.analytics import (
    analyze_user_category_preferences,
    calculate_global_kpis,
    get_daily_engagement_trends,
    get_trending_videos,
    load_data,
)
from src.data_generator import generate_datasets
from src.recommender import VideoRecommender

st.set_page_config(page_title="Video RecSys & Analytics", layout="wide")


@st.cache_data
def init_data():
    generate_datasets()
    return load_data()


videos_df, interactions_df = init_data()
recommender = VideoRecommender(videos_df, interactions_df)

st.title("Video Analytics & Recommendation Engine")
tab1, tab2, tab3 = st.tabs(
    ["Phase 1: Data Analytics", "Phase 2: Recommendation Engine", "Phase 3: JSON API Preview"]
)

with tab1:
    kpis = calculate_global_kpis(interactions_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Views", f"{kpis['total_views']:,}")
    c2.metric("Total Likes", f"{kpis['total_likes']:,}")
    c3.metric("Total Comments", f"{kpis['total_comments']:,}")
    c4.metric("Total Watch Hours", f"{kpis['total_watch_hours']:,.1f} hrs")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Daily Engagement Trends")
        trends_df = get_daily_engagement_trends(interactions_df)
        fig_trends = px.line(
            trends_df, x="date", y=["views", "likes", "comments"], markers=True
        )
        st.plotly_chart(fig_trends, use_container_width=True)

    with col_right:
        st.subheader("Category Preferences (Watch Time Share)")
        cat_pref = analyze_user_category_preferences(videos_df, interactions_df)
        fig_cat = px.bar(
            cat_pref,
            x="category",
            y="preference_share_pct",
            color="category",
            labels={"preference_share_pct": "Watch Share (%)"},
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Top Trending Videos")
    trending_df = get_trending_videos(videos_df, interactions_df, top_n=10)
    st.dataframe(
        trending_df[
            ["video_id", "title", "category", "tags", "views", "likes", "comments", "trending_score"]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("Interactive Hybrid Recommender")
    r_col1, r_col2, r_col3 = st.columns(3)
    user_list = ["None (Use Custom Preferences)"] + sorted(
        interactions_df["user_id"].unique().tolist()
    )
    selected_user = r_col1.selectbox("Select User ID", user_list)
    pref_cats = r_col2.multiselect(
        "Override / Add Preferred Categories", videos_df["category"].unique()
    )
    all_tags = sorted(
        {tag for tags in videos_df["tags"].dropna() for tag in tags.split(",")}
    )
    pref_tags = r_col3.multiselect("Override / Add Preferred Tags", all_tags)

    uid = None if selected_user.startswith("None") else selected_user
    recs = recommender.recommend(
        user_id=uid,
        preferred_categories=pref_cats or None,
        preferred_tags=pref_tags or None,
        top_k=10,
    )
    st.dataframe(recs, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Simulated JSON Output (`POST /recommendations`)")
    st.json(
        {
            "user_id": uid,
            "preferred_categories": pref_cats,
            "preferred_tags": pref_tags,
            "count": len(recs),
            "recommendations": recs.to_dict(orient="records"),
        }
    )