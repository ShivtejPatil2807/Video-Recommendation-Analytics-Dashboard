import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Seed for reproducible results
np.random.seed(42)

# 5 Categories and their associated tags
CATEGORIES = {
    "Tech": ["python", "ai", "coding", "webdev", "cloud", "gadgets"],
    "Music": ["lofi", "rock", "pop", "guitar", "acoustic", "live"],
    "Gaming": ["rpg", "fps", "esports", "speedrun", "indie", "stream"],
    "Education": ["math", "physics", "history", "science", "tutorial", "lecture"],
    "Comedy": ["standup", "sketch", "parody", "satire", "improv", "prank"]
}


def generate_videos(n: int = 50) -> pd.DataFrame:
    """Generates a DataFrame of n synthetic videos across 5 categories."""
    records = []
    category_names = list(CATEGORIES.keys())
    base_date = datetime(2026, 9, 15)

    for i in range(1, n + 1):
        category = np.random.choice(category_names)
        num_tags = np.random.randint(2, 4)
        chosen_tags = np.random.choice(CATEGORIES[category], size=num_tags, replace=False)
        tags_str = "|".join(chosen_tags)
        duration_sec = int(np.random.randint(120, 1801))
        upload_date = base_date + timedelta(days=int(np.random.randint(0, 30)))

        records.append({
            "video_id": f"V{i:03d}",
            "title": f"{category} Video #{i}: {chosen_tags[0].title()}",
            "category": category,
            "tags": tags_str,
            "duration_sec": duration_sec,
            "upload_date": upload_date.strftime("%Y-%m-%d")
        })

    return pd.DataFrame(records)


def get_user_favorite_category(user_num: int) -> str:
    """Assigns a deterministic favorite category based on user number."""
    if 1 <= user_num <= 4:
        return "Tech"
    elif 5 <= user_num <= 8:
        return "Gaming"
    elif 9 <= user_num <= 12:
        return "Education"
    elif 13 <= user_num <= 16:
        return "Music"
    else:
        return "Comedy"


def generate_interactions(
    videos_df: pd.DataFrame, n_users: int = 20, n_interactions: int = 1000
) -> pd.DataFrame:
    """Generates synthetic user-video interactions with planted category preferences."""
    records = []
    base_time = datetime(2026, 10, 1, 8, 0, 0)

    for i in range(1, n_interactions + 1):
        user_num = int(np.random.randint(1, n_users + 1))
        user_id = f"U{user_num:03d}"
        fav_category = get_user_favorite_category(user_num)

        # 75% chance user watches their favorite category
        if np.random.rand() < 0.75:
            pool = videos_df[videos_df["category"] == fav_category]
        else:
            pool = videos_df

        video = pool.sample(1).iloc[0]
        duration = video["duration_sec"]

        # Users watch longer and engage more when watching their favorite category
        if video["category"] == fav_category:
            watch_ratio = np.random.uniform(0.65, 1.0)
            liked = int(np.random.rand() < 0.60)
            commented = int(np.random.rand() < 0.25)
        else:
            watch_ratio = np.random.uniform(0.10, 0.50)
            liked = int(np.random.rand() < 0.10)
            commented = int(np.random.rand() < 0.03)

        watch_time_sec = int(duration * watch_ratio)

        timestamp = base_time + timedelta(
            days=int(np.random.randint(0, 14)),
            hours=int(np.random.randint(0, 24)),
            minutes=int(np.random.randint(0, 60)),
        )

        records.append({
            "interaction_id": f"INT{i:04d}",
            "user_id": user_id,
            "video_id": video["video_id"],
            "watch_time_sec": watch_time_sec,
            "liked": liked,
            "commented": commented,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df_videos = generate_videos(n=50)
    df_videos.to_csv("data/videos.csv", index=False)
    print(f"Saved {len(df_videos)} videos to data/videos.csv")

    df_interactions = generate_interactions(df_videos, n_users=20, n_interactions=1000)
    df_interactions.to_csv("data/interactions.csv", index=False)
    print(f"Saved {len(df_interactions)} interactions to data/interactions.csv\n")
    print(df_interactions.head())