from pathlib import Path
import numpy as np
import pandas as pd


def generate_datasets(
    output_dir: str = "data", n_videos: int = 100, n_interactions: int = 3000
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    categories = {
        "Tech": ["python", "ai", "coding", "fastapi", "cloud", "linux"],
        "Gaming": ["fps", "rpg", "esports", "walkthrough", "speedrun", "indie"],
        "Education": ["math", "physics", "history", "engineering", "tutorial", "exam"],
        "Music": ["lofi", "rock", "synth", "acoustic", "live", "remix"],
        "Fitness": ["hiit", "yoga", "strength", "cardio", "nutrition", "mobility"],
    }

    # 1. Generate Videos Dataset
    video_rows = []
    for vid_idx in range(1, n_videos + 1):
        cat = rng.choice(list(categories.keys()))
        tags = rng.choice(categories[cat], size=3, replace=False)
        video_rows.append(
            {
                "video_id": f"vid_{vid_idx:03d}",
                "title": f"{cat} Masterclass #{vid_idx} ({tags[0].title()})",
                "category": cat,
                "tags": ",".join(tags),
                "duration_sec": int(rng.integers(120, 1200)),
                "upload_date": pd.Timestamp("2026-09-01")
                + pd.Timedelta(days=int(rng.integers(0, 45))),
            }
        )
    videos_df = pd.DataFrame(video_rows)

    # 2. Generate User Interactions Dataset
    interaction_rows = []
    video_ids = videos_df["video_id"].tolist()
    # Pareto distribution so a few videos naturally become "trending"
    vid_weights = rng.pareto(a=1.5, size=n_videos)
    vid_weights /= vid_weights.sum()

    for int_idx in range(1, n_interactions + 1):
        vid_id = rng.choice(video_ids, p=vid_weights)
        dur = int(
            videos_df.loc[videos_df["video_id"] == vid_id, "duration_sec"].iloc[0]
        )
        watch_time = int(np.clip(rng.normal(loc=dur * 0.65, scale=dur * 0.25), 10, dur))

        interaction_rows.append(
            {
                "interaction_id": int_idx,
                "user_id": f"user_{rng.integers(1, 51):02d}",
                "video_id": vid_id,
                "viewed": 1,
                "liked": int(
                    rng.binomial(1, 0.25 if watch_time > dur * 0.5 else 0.05)
                ),
                "commented": int(rng.binomial(1, 0.08)),
                "watch_time_sec": watch_time,
                "timestamp": pd.Timestamp("2026-10-01")
                + pd.Timedelta(hours=int(rng.integers(0, 720))),
            }
        )
    interactions_df = pd.DataFrame(interaction_rows)

    videos_df.to_csv(f"{output_dir}/videos.csv", index=False)
    interactions_df.to_csv(f"{output_dir}/interactions.csv", index=False)


if __name__ == "__main__":
    generate_datasets()