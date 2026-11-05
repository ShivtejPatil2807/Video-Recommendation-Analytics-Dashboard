import numpy as np
import pandas as pd


class VideoRecommender:
    def __init__(self, videos: pd.DataFrame, interactions: pd.DataFrame):
        self.videos = videos.copy()
        self.interactions = interactions.copy()
        self.video_stats = self._build_video_stats()

    def _build_video_stats(self) -> pd.DataFrame:
        stats = (
            self.interactions.groupby("video_id", as_index=False)
            .agg(
                views=("viewed", "sum"),
                likes=("liked", "sum"),
                comments=("commented", "sum"),
                total_watch_sec=("watch_time_sec", "sum"),
            )
            .merge(self.videos, on="video_id", how="right")
        )
        stats[["views", "likes", "comments", "total_watch_sec"]] = stats[
            ["views", "likes", "comments", "total_watch_sec"]
        ].fillna(0)

        # Normalize engagement score to [0, 1] using Min-Max scaling on log-transformed engagement
        raw_eng = np.log1p(
            stats["views"] * 1.0
            + stats["likes"] * 4.0
            + stats["comments"] * 6.0
            + (stats["total_watch_sec"] / 60.0) * 0.5
        )
        min_e, max_e = raw_eng.min(), raw_eng.max()
        stats["engagement_score"] = (
            (raw_eng - min_e) / (max_e - min_e) if max_e > min_e else 0.0
        )
        stats["tag_set"] = stats["tags"].apply(
            lambda t: set(str(t).split(",")) if pd.notna(t) else set()
        )
        return stats

    def extract_user_profile(self, user_id: str) -> tuple[dict[str, float], set[str], set[str]]:
        user_hist = self.interactions[self.interactions["user_id"] == user_id].merge(
            self.videos[["video_id", "category", "tags"]], on="video_id", how="left"
        )
        if user_hist.empty:
            return {}, set(), set()

        # Weight category affinity by watch time + likes
        user_hist["weight"] = user_hist["watch_time_sec"] * (1 + 0.5 * user_hist["liked"])
        cat_weights = user_hist.groupby("category")["weight"].sum()
        cat_affinity = (cat_weights / cat_weights.sum()).to_dict()

        # Extract top tags from liked or heavily watched videos
        liked_tags: set[str] = set()
        for tags_str in user_hist["tags"].dropna():
            liked_tags.update(tags_str.split(","))

        watched_ids = set(user_hist["video_id"].unique())
        return cat_affinity, liked_tags, watched_ids

    def recommend(
        self,
        user_id: str | None = None,
        preferred_categories: list[str] | None = None,
        preferred_tags: list[str] | None = None,
        top_k: int = 10,
        exclude_watched: bool = True,
    ) -> pd.DataFrame:
        candidates = self.video_stats.copy()
        cat_affinity: dict[str, float] = {}
        target_tags: set[str] = set()
        watched_ids: set[str] = set()

        if user_id:
            cat_affinity, target_tags, watched_ids = self.extract_user_profile(user_id)

        if preferred_categories:
            uniform_weight = 1.0 / len(preferred_categories)
            for c in preferred_categories:
                cat_affinity[c] = max(cat_affinity.get(c, 0.0), uniform_weight)

        if preferred_tags:
            target_tags.update([t.strip().lower() for t in preferred_tags])

        if exclude_watched and watched_ids:
            candidates = candidates[~candidates["video_id"].isin(watched_ids)].copy()

        # 1. Category Match Score
        candidates["category_score"] = candidates["category"].map(cat_affinity).fillna(0.0)

        # 2. Tag Jaccard Similarity Score
        if target_tags:
            candidates["tag_score"] = candidates["tag_set"].apply(
                lambda s: len(s.intersection(target_tags)) / len(s.union(target_tags))
                if s
                else 0.0
            )
        else:
            candidates["tag_score"] = 0.0

        # 3. Combine into Relevance Score & Final Hybrid Rank Score
        has_preferences = bool(cat_affinity or target_tags)
        if has_preferences:
            candidates["relevance_score"] = (
                0.6 * candidates["category_score"] + 0.4 * candidates["tag_score"]
            )
            candidates["final_score"] = (
                0.65 * candidates["relevance_score"]
                + 0.35 * candidates["engagement_score"]
            )
        else:
            # Cold-start fallback: rank purely by engagement
            candidates["relevance_score"] = 0.0
            candidates["final_score"] = candidates["engagement_score"]

        cols = [
            "video_id",
            "title",
            "category",
            "tags",
            "views",
            "likes",
            "comments",
            "relevance_score",
            "engagement_score",
            "final_score",
        ]
        return (
            candidates[cols]
            .round({"relevance_score": 3, "engagement_score": 3, "final_score": 3})
            .sort_values("final_score", ascending=False)
            .head(top_k)
        )