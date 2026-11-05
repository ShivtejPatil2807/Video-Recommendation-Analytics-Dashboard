from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.analytics import calculate_global_kpis, get_trending_videos, load_data
from src.recommender import VideoRecommender

app = FastAPI(title="Video Recommendation Engine API", version="1.0.0")

videos_df, interactions_df = load_data()
engine = VideoRecommender(videos_df, interactions_df)


class RecommendationRequest(BaseModel):
    user_id: str | None = Field(default=None, examples=["user_05"])
    preferred_categories: list[str] | None = Field(
        default=None, examples=[["Tech", "Education"]]
    )
    preferred_tags: list[str] | None = Field(
        default=None, examples=[["python", "fastapi", "ai"]]
    )
    top_k: int = Field(default=5, ge=1, le=50)
    exclude_watched: bool = True


@app.get("/analytics/summary")
def get_analytics_summary():
    return {
        "kpis": calculate_global_kpis(interactions_df),
        "trending_videos": get_trending_videos(videos_df, interactions_df, top_n=5)[
            ["video_id", "title", "category", "trending_score", "views", "likes"]
        ].to_dict(orient="records"),
    }


@app.post("/recommendations")
def get_recommendations(payload: RecommendationRequest):
    if (
        not payload.user_id
        and not payload.preferred_categories
        and not payload.preferred_tags
    ):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: user_id, preferred_categories, or preferred_tags.",
        )

    recs_df = engine.recommend(
        user_id=payload.user_id,
        preferred_categories=payload.preferred_categories,
        preferred_tags=payload.preferred_tags,
        top_k=payload.top_k,
        exclude_watched=payload.exclude_watched,
    )
    return {
        "user_id": payload.user_id,
        "count": len(recs_df),
        "recommendations": recs_df.to_dict(orient="records"),
    }