import numpy as np
import pandas as pd


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    videos = pd.read_csv(f"{data_dir}/videos.csv", parse_dates=["upload_date"])
    interactions = pd.read_csv(
        f"{data_dir}/interactions.csv", parse_dates=["timestamp"]
    )
    return videos, interactions


def calculate_global_kpis(interactions: pd.DataFrame) -> dict:
    return {
        "total_views": int(interactions["viewed"].sum()),
        "total_likes": int(interactions["liked"].sum()),
        "total_comments": int(interactions["commented"].sum()),
        "total_watch_hours": round(float(interactions["watch_time_sec"].sum()) / 3600, 2),
    }


def get_trending_videos(
    videos: pd.DataFrame, interactions: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    ref_time = interactions["timestamp"].max()
    agg = (
        interactions.groupby("video_id", as_index=False)
        .agg(
            views=("viewed", "sum"),
            likes=("liked", "sum"),
            comments=("commented", "sum"),
            watch_time_sec=("watch_time_sec", "sum"),
            latest_interaction=("timestamp", "max"),
        )
        .merge(videos, on="video_id", how="left")
    )

    days_old = (
        (ref_time - agg["latest_interaction"]).dt.total_seconds() / 86400
    ).clip(lower=0)
    raw_engagement = (
        agg["views"] * 1.0
        + agg["likes"] * 3.0
        + agg["comments"] * 5.0
        + (agg["watch_time_sec"] / 3600) * 2.0
    )
    agg["trending_score"] = (raw_engagement / np.power(days_old + 2.0, 1.5)).round(2)
    return agg.sort_values("trending_score", ascending=False).head(top_n)


def analyze_user_category_preferences(
    videos: pd.DataFrame, interactions: pd.DataFrame, user_id: str | None = None
) -> pd.DataFrame:
    merged = interactions.merge(
        videos[["video_id", "category", "duration_sec"]], on="video_id", how="left"
    )
    if user_id:
        merged = merged[merged["user_id"] == user_id]

    pref = (
        merged.groupby("category", as_index=False)
        .agg(
            views=("viewed", "sum"),
            likes=("liked", "sum"),
            watch_time_sec=("watch_time_sec", "sum"),
        )
    )
    total_watch = pref["watch_time_sec"].sum()
    pref["preference_share_pct"] = (
        (pref["watch_time_sec"] / total_watch * 100).round(2)
        if total_watch > 0
        else 0.0
    )
    return pref.sort_values("preference_share_pct", ascending=False)


def get_daily_engagement_trends(interactions: pd.DataFrame) -> pd.DataFrame:
    return (
        interactions.assign(date=interactions["timestamp"].dt.date)
        .groupby("date", as_index=False)
        .agg(
            views=("viewed", "sum"),
            likes=("liked", "sum"),
            comments=("commented", "sum"),
            watch_hours=("watch_time_sec", lambda s: round(s.sum() / 3600, 2)),
        )
    )