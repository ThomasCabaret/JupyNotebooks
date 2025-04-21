import pandas as pd
from collections import defaultdict

# TAG_NAMES_INCLUDED = {
#     "science-fiction", "sci-fi", "scifi", "sf", "science-fiction-fantasy",
#     "fantasy-sci-fi", "sci-fi-fantasy", "fantasy", "urban-fantasy",
#     "magical-realism", "dystopian", "dystopia", "high-fantasy",
#     "epic-fantasy", "steampunk", "post-apocalyptic", "time-travel"
# }

# TAG_NAMES_INCLUDED = {
#     "science-fiction", "sci-fi", "scifi", "sf", "dystopian", "dystopia", "steampunk", "post-apocalyptic", "time-travel"
# }

TAG_NAMES_INCLUDED = {
    "steampunk"
}

MAX_TAGS_PER_BOOK = 5
MIN_TOTAL_RATINGS_PER_USER = 30
MIN_CATEGORY_RATINGS_PER_USER = 0
PROGRESS_PRINT_INTERVAL = 1000
TOP_K = 200

def get_filtered_book_ids(tags_df, book_tags_df, books_df):
    tag_ids = tags_df.loc[tags_df["tag_name"].isin(TAG_NAMES_INCLUDED), "tag_id"]
    # Keep only the top N most frequent tags for each book
    top_tags_per_book = (
        book_tags_df.sort_values(["goodreads_book_id", "count"], ascending=[True, False])
        .groupby("goodreads_book_id")
        .head(MAX_TAGS_PER_BOOK)
    )
    # From those, keep only the rows where tag_id is in our desired set
    filtered = top_tags_per_book[top_tags_per_book["tag_id"].isin(tag_ids)]
    gr_ids = set(filtered["goodreads_book_id"])
    mapping = books_df.set_index("goodreads_book_id")["book_id"].to_dict()
    return {mapping[gid] for gid in gr_ids if gid in mapping}


def filter_ratings_by_books(ratings_df, book_id_set):
    return ratings_df[ratings_df["book_id"].isin(book_id_set)]

def filter_users_by_min_ratings(ratings_df, min_ratings):
    counts = ratings_df.groupby("user_id").size()
    good_users = counts[counts >= min_ratings].index
    return ratings_df[ratings_df["user_id"].isin(good_users)]

def user_preferences(ratings_df):
    prefs = []
    grouped = ratings_df.groupby("rating")
    rating_groups = {rating: group["book_id"].tolist() for rating, group in grouped}
    sorted_ratings = sorted(rating_groups.keys(), reverse=True)
    for i, r_high in enumerate(sorted_ratings):
        for r_low in sorted_ratings[i + 1:]:
            winners = rating_groups[r_high]
            losers = rating_groups[r_low]
            prefs.extend((w, l) for w in winners for l in losers)
    return prefs

def build_duel_matrix(ratings_df):
    duel = defaultdict(lambda: defaultdict(int))
    grouped = ratings_df.groupby("user_id")
    total = len(grouped)
    print(f"[INFO] Building duel matrix on {total} users...")
    for idx, (_, group) in enumerate(grouped, 1):
        prefs = user_preferences(group)
        for win, lose in prefs:
            duel[win][lose] += 1
        if idx % PROGRESS_PRINT_INTERVAL == 0 or idx == total:
            pct = idx * 100 / total
            print(f"[INFO] Users processed: {idx}/{total} ({pct:.1f}%)")
    books_involved = set(duel.keys()) | {b for m in duel.values() for b in m}
    print(f"[INFO] Duel matrix ready: {len(books_involved)} books involved.")
    return duel

def compute_copeland(duel):
    score = defaultdict(int)
    all_books = set(duel.keys()) | {b for m in duel.values() for b in m}
    for a in all_books:
        for b in all_books:
            if a == b:
                continue
            wins = duel[a].get(b, 0)
            losses = duel[b].get(a, 0)
            if wins > losses:
                score[a] += 1
            elif wins < losses:
                score[a] -= 1
    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return ranked[:TOP_K]

def main():
    print("[INFO] Loading CSV files...")
    tags = pd.read_csv("tags.csv")
    book_tags = pd.read_csv("book_tags.csv")
    books = pd.read_csv("books.csv")
    ratings = pd.read_csv("ratings.csv")

    print("[INFO] Filtering users by total ratings...")
    ratings = filter_users_by_min_ratings(ratings, MIN_TOTAL_RATINGS_PER_USER)
    total_users = ratings["user_id"].nunique()
    print(f"[INFO] {total_users} users remain after total filter.")

    print("[INFO] Filtering books by tag set...")
    filtered_books = get_filtered_book_ids(tags, book_tags, books)
    print(f"[INFO] {len(filtered_books)} books selected.")

    print("[INFO] Filtering ratings to those books...")
    ratings = filter_ratings_by_books(ratings, filtered_books)
    print(f"[INFO] {len(ratings)} ratings remain after book filter.")

    print("[INFO] Filtering users by category-specific ratings...")
    ratings = filter_users_by_min_ratings(ratings, MIN_CATEGORY_RATINGS_PER_USER)
    category_users = ratings["user_id"].nunique()
    print(f"[INFO] {category_users} users remain after category filter.")

    duel = build_duel_matrix(ratings)
    top = compute_copeland(duel)

    books_idx = books.set_index("book_id")

    print("\n=== Summary of Parameters and Dataset ===")
    print(f"Included tags: {sorted(TAG_NAMES_INCLUDED)}")
    print(f"MAX_TAGS_PER_BOOK: {MAX_TAGS_PER_BOOK}")
    print(f"MIN_TOTAL_RATINGS_PER_USER: {MIN_TOTAL_RATINGS_PER_USER}")
    print(f"MIN_CATEGORY_RATINGS_PER_USER: {MIN_CATEGORY_RATINGS_PER_USER}")
    print(f"TOP_K: {TOP_K}")
    print(f"Books matched to tag filter: {len(filtered_books)}")
    print(f"Users after total rating filter: {total_users}")
    print(f"Users after category rating filter: {category_users}")
    books_in_duel_matrix = len(set(duel.keys()) | {b for m in duel.values() for b in m})
    print(f"Books involved in duel matrix: {books_in_duel_matrix}")

    print("\n=== Top 200 Books by Condorcet (Copeland) ===")
    for rank, (bid, _) in enumerate(top, 1):
        if bid not in books_idx.index:
            print(f"{rank}. [Book ID {bid}] (metadata missing)")
            continue
        row = books_idx.loc[bid]
        title = row["original_title"] if pd.notnull(row["original_title"]) else row["title"]
        author = row["authors"]
        year = int(row["original_publication_year"]) if pd.notnull(row["original_publication_year"]) else "?"
        gr_id = int(row["best_book_id"] if pd.notnull(row["best_book_id"]) else row["goodreads_book_id"])
        url = f"https://www.goodreads.com/book/show/{gr_id}"
        print(f"{rank}. {title} ({year}) by {author} - {url}")

if __name__ == "__main__":
    main()
