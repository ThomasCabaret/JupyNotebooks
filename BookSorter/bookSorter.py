import pandas as pd
from collections import defaultdict

# TAG_NAMES_INCLUDED = {
#     "science-fiction", "sci-fi", "scifi", "sf", "science-fiction-fantasy",
#     "fantasy-sci-fi", "sci-fi-fantasy", "fantasy", "urban-fantasy",
#     "magical-realism", "dystopian", "dystopia", "high-fantasy",
#     "epic-fantasy", "steampunk", "post-apocalyptic", "time-travel"
# }

TAG_NAMES_INCLUDED = {
    "science-fiction", "sci-fi", "scifi", "sf", "dystopian", "dystopia", "steampunk", "post-apocalyptic", "time-travel"
}

#TAG_NAMES_INCLUDED = {
#    "steampunk"
#}

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

import numpy as np
from scipy.optimize import minimize
import time
import sys

def compute_bradley_terry_improved(duel, anchor_idx=None, top_k=None):
    """
    Computes Bradley-Terry scores using Maximum Likelihood Estimation with anchoring.

    Args:
        duel (dict): Dictionary representing pairwise comparisons.
                     Format: duel[winner_id][loser_id] = win_count
        anchor_idx (int, optional): Index of the item to anchor score to 0.
                                    Defaults to DEFAULT_ANCHOR_IDX.
        top_k (int, optional): Number of top items to return.
                               Defaults to DEFAULT_TOP_K.

    Returns:
        list: List of tuples (item_id, score, std_dev), sorted by score descending.
              Returns empty list if processing fails early.
    """
    # --- Configuration Constants ---
    DEFAULT_ANCHOR_IDX = 0
    DEFAULT_TOP_K = 200
    # Optimizer settings
    OPTIMIZER_METHOD = "L-BFGS-B"
    OPTIMIZER_GTOL = 1e-5  # Default gradient tolerance for L-BFGS-B convergence
    # Set disp=True to let L-BFGS-B print progress, replacing custom callback
    OPTIMIZER_OPTIONS = {'disp': True, 'gtol': OPTIMIZER_GTOL, 'maxiter': 1000}
    # Gradient check settings (useful for debugging, disable for performance)
    GRAD_CHECK_ENABLED = False # Set to True to perform gradient check
    GRAD_CHECK_EPSILON = 1.49e-08 # Step size for finite difference in check_grad
    GRAD_CHECK_THRESHOLD = 1e-5 # Maximum acceptable error for check_grad
    # Fisher info stability threshold
    FISHER_STABILITY_THRESHOLD = 1e-10

    # --- Initialization ---
    start_total_time = time.time()
    if anchor_idx is None:
        anchor_idx = DEFAULT_ANCHOR_IDX
    if top_k is None:
        top_k = DEFAULT_TOP_K

    print("[Init] Collecting items...")
    try:
        items_set = set(duel.keys()) | {b for opponents in duel.values() for b in opponents}
        items = sorted(list(items_set))
        id2idx = {bid: i for i, bid in enumerate(items)}
        idx2id = {i: bid for bid, i in id2idx.items()}
        n = len(items)
        print(f"[Init] {n} items collected.")
        if n == 0:
            print("[Error] No items found in duel data.")
            return []
    except Exception as e:
        print(f"[Error] Failed during item collection: {e}")
        return []

    if not (0 <= anchor_idx < n):
        print(f"[Warn] Invalid anchor_idx {anchor_idx}, using 0.")
        anchor_idx = 0

    # --- Edge Building ---
    print("[Build] Building unique edges...")
    edges = []
    added_pairs = set()
    try:
        for ai, opponents in duel.items():
            # ai might not be in id2idx if it only lost? Check needed?
            # The initial items_set should cover all involved items.
            i = id2idx[ai]
            for bj, w_ij in opponents.items():
                if bj not in id2idx:
                    # This happens if bj only lost and never won (was never a key in duel)
                    # print(f"[Debug] Skipping edge involving unknown item: {bj}")
                    continue
                j = id2idx[bj]
                if i == j:
                    continue
                # Create canonical pair (smaller_idx, larger_idx)
                pair = (min(i, j), max(i, j))
                if pair in added_pairs:
                    continue

                # Get reverse win count safely
                w_ji = duel.get(bj, {}).get(ai, 0)
                total_wins = w_ij + w_ji

                if total_wins > 0:
                    # Store consistently: (idx1, idx2, wins1, wins2) where idx1 < idx2
                    idx1, idx2 = pair
                    if idx1 == i: # Original (i, j) matches (idx1, idx2)
                        edges.append((idx1, idx2, w_ij, w_ji))
                    else: # Original (i, j) was (idx2, idx1)
                        # w_ij was wins for idx2, w_ji was wins for idx1
                        edges.append((idx1, idx2, w_ji, w_ij))
                    added_pairs.add(pair)
        print(f"[Build] {len(edges)} unique duel edges created.")
        if not edges:
             print("[Warn] No valid edges created. Cannot optimize.")
             return []
    except Exception as e:
        print(f"[Error] Failed during edge building: {e}")
        return []

    # --- Helper Functions (Anchored) ---
    def reconstruct_theta(theta_short):
        # Inserts 0.0 at anchor_idx to get the full theta vector
        return np.insert(theta_short, anchor_idx, 0.0)

    # NLL function (NO CLIPPING)
    def nll_anchor(theta_short):
        theta_full = reconstruct_theta(theta_short)
        s = 0.0
        for i, j, w_ij, w_ji in edges:
            # i is always < j in edges list construction above
            diff = theta_full[i] - theta_full[j]
            # log1p(exp(x)) is numerically unstable for large x, but log(1+exp(x)) = x + log(1+exp(-x))
            # log1p(exp(-x)) is stable.
            # Term1 = w_ij * log(P(i beats j)) = w_ij * log(1 / (1+exp(-diff))) = -w_ij * log(1+exp(-diff))
            # Term2 = w_ji * log(P(j beats i)) = w_ji * log(1 / (1+exp(diff)))  = -w_ji * log(1+exp(diff))
            # NLL = - (Term1 + Term2)
            # s -= Term1 + Term2
            s -= -w_ij * np.log1p(np.exp(-diff)) # Stable calculation for w_ij * log(P(i beats j))
            s -= -w_ji * np.log1p(np.exp(diff))  # Stable calculation for w_ji * log(P(j beats i))
        return s

    # Gradient function (NO CLIPPING)
    def grad_anchor(theta_short):
        theta_full = reconstruct_theta(theta_short)
        g_full = np.zeros(n)
        for i, j, w_ij, w_ji in edges:
             # i is always < j
             diff = theta_full[i] - theta_full[j]
             # p = P(i beats j) = 1 / (1 + exp(-diff))
             # Sigmoid function is numerically stable
             p = 1 / (1 + np.exp(-diff))

             # Gradient update based on derivatives derived earlier:
             # dNLL/d(theta_i) = -w_ij*(1-p) + w_ji*p
             # dNLL/d(theta_j) = +w_ij*(1-p) - w_ji*p
             common_term_1 = w_ij * (1 - p) # = w_ij * P(j beats i)
             common_term_2 = w_ji * p       # = w_ji * P(i beats j)

             g_full[i] -= common_term_1
             g_full[i] += common_term_2

             g_full[j] += common_term_1
             g_full[j] -= common_term_2

        # Return gradient without the anchor component
        g_short = np.delete(g_full, anchor_idx)
        return g_short

    # --- Gradient Check (Optional) ---
    if GRAD_CHECK_ENABLED:
        print("[Check] Performing gradient check...")
        # Use zeros or a small random point for checking
        # check_point = np.zeros(n - 1)
        check_point = np.random.normal(0, 0.1, size=n - 1)
        try:
            error = check_grad(nll_anchor, grad_anchor, check_point, epsilon=GRAD_CHECK_EPSILON)
            print(f"[Check] Gradient check error (lower is better): {error:.2e}")
            if error > GRAD_CHECK_THRESHOLD:
                print(f"[Error] Gradient check error {error:.2e} exceeds threshold {GRAD_CHECK_THRESHOLD:.2e}. Check grad_anchor implementation.")
                # Decide whether to stop or continue with warning
                # return [] # Option to stop if gradient is wrong
                print("[Warn] Continuing optimization despite high gradient error.")
            else:
                 print("[Check] Gradient check passed.")
        except Exception as e:
            print(f"[Error] Failed during gradient check: {e}")
            # return [] # Option to stop

    # --- Optimization ---
    # Use zeros as the standard starting point
    x0_short = np.zeros(n - 1)

    # Check initial gradient norm
    try:
        initial_grad = grad_anchor(x0_short)
        initial_grad_norm = np.linalg.norm(initial_grad)
        print(f"[Check] Initial gradient norm at x0: {initial_grad_norm:.4e}")
    except Exception as e:
        print(f"[Error] Failed to calculate initial gradient: {e}")
        return []

    print(f"[Opt] Starting {OPTIMIZER_METHOD} optimization...")
    opt_start_time = time.time()
    try:
        res = minimize(
            nll_anchor,
            x0_short,
            method=OPTIMIZER_METHOD,
            jac=grad_anchor,
            options=OPTIMIZER_OPTIONS
            # No callback needed, L-BFGS-B uses options={'disp': True}
        )
    except Exception as e:
        print(f"[Error] Optimization failed with exception: {e}")
        return []
    opt_end_time = time.time()

    print(f"[Opt] Optimization finished in {opt_end_time - opt_start_time:.2f}s")
    print(f"[Opt] Success: {res.success}")
    print(f"[Opt] Message: {res.message}")

    # Check final gradient norm
    try:
        # res.jac should contain the gradient at the returned solution res.x
        final_grad_norm = np.linalg.norm(res.jac)
        print(f"[Opt] Final gradient norm: {final_grad_norm:.4e}")
        if not res.success and final_grad_norm > OPTIMIZER_GTOL * 10: # Heuristic check
             print(f"[Warn] Optimization failed and final gradient norm ({final_grad_norm:.2e}) is significantly larger than tolerance ({OPTIMIZER_GTOL:.1e}). Results might be unreliable.")
    except Exception as e:
         print(f"[Warn] Could not compute final gradient norm: {e}")


    # Reconstruct full theta vector
    theta = reconstruct_theta(res.x)

    # --- Uncertainty Calculation ---
    print("[Info] Calculating Fisher info...")
    fisher_diag = np.zeros(n)
    try:
        for i, j, w_ij, w_ji in edges:
            # i is always < j
            diff = theta[i] - theta[j]
            # Use stable sigmoid calculation
            p = 1 / (1 + np.exp(-diff))
            # v = p * (1-p) can be numerically unstable if p is near 0 or 1.
            # However, p*(1-p) = exp(-diff) / (1+exp(-diff))^2
            # This might not be significantly more stable than direct p*(1-p)
            v = max(p * (1 - p), np.finfo(float).eps) # Ensure variance is at least machine epsilon

            cnt = w_ij + w_ji # Total comparisons for this pair
            fisher_diag[i] += cnt * v
            fisher_diag[j] += cnt * v

        std = np.full(n, np.nan) # Initialize with NaN
        unstable_count = 0
        for i in range(n):
            if fisher_diag[i] > FISHER_STABILITY_THRESHOLD:
                std[i] = np.sqrt(1.0 / fisher_diag[i])
            else:
                unstable_count += 1
        print(f"[Info] Fisher info calculation complete. Unstable variances (Fisher diag <= {FISHER_STABILITY_THRESHOLD:.1e}): {unstable_count}/{n}")
    except Exception as e:
        print(f"[Error] Failed during Fisher info calculation: {e}")
        # Continue without std deviations or return []? Let's continue.
        std = np.full(n, np.nan)

    # --- Output Formatting ---
    print(f"[Done] Sorting and returning top {top_k} ranked items.")
    try:
        out = []
        for i in range(n):
            item_id = idx2id.get(i, f"Unknown_Idx_{i}")
            score = theta[i]
            uncertainty = std[i] # Will be NaN if calculation failed or unstable
            out.append((item_id, score, uncertainty))

        # Sort by score descending, handle potential NaNs in score? (Shouldn't happen)
        out.sort(key=lambda x: x[1], reverse=True)
    except Exception as e:
        print(f"[Error] Failed during output formatting: {e}")
        return []

    total_duration = time.time() - start_total_time
    print(f"[Total] Function completed in {total_duration:.2f}s.")

    return out[:top_k]

import networkx as nx

def filter_largest_strongly_connected_component(duel):
    """
    Filters the duel dict to retain only the largest strongly connected component.
    Returns a new duel dict in the same format, fully compatible with the rest of the pipeline.
    """
    G = nx.DiGraph()
    for winner, losers in duel.items():
        for loser in losers:
            G.add_edge(winner, loser)

    print(f"[Graph] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    scc = list(nx.strongly_connected_components(G))
    scc_sorted = sorted(scc, key=len, reverse=True)
    largest_scc = scc_sorted[0]

    print(f"[SCC] Components: {len(scc)}, Largest size: {len(largest_scc)}")

    if len(largest_scc) == G.number_of_nodes():
        print("[SCC] Graph is already fully strongly connected.")
        return duel

    filtered_duel = {}
    for i in largest_scc:
        if i not in duel:
            continue
        inner = {j: duel[i][j] for j in duel[i] if j in largest_scc}
        if inner:
            filtered_duel[i] = inner

    retained_graph = G.subgraph(largest_scc).copy()
    density = nx.density(retained_graph)
    avg_degree = sum(dict(retained_graph.degree()).values()) / retained_graph.number_of_nodes()

    print(f"[Filtered] Retained nodes: {len(filtered_duel)}")
    print(f"[Filtered] Density: {density:.4f}, Avg degree: {avg_degree:.2f}")

    return filtered_duel

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
    duel_filtered = filter_largest_strongly_connected_component(duel)
    top = compute_bradley_terry_improved(duel_filtered)

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

    print("\n=== Top 200 Books by Bradley-Terry ===")
    for rank, (bid, score, stddev) in enumerate(top, 1):
        if bid not in books_idx.index:
            print(f"{rank}. [Book ID {bid}] (metadata missing)")
            continue
        row = books_idx.loc[bid]
        title = row["original_title"] if pd.notnull(row["original_title"]) else row["title"]
        author = row["authors"]
        year = int(row["original_publication_year"]) if pd.notnull(row["original_publication_year"]) else "?"
        gr_id = int(row["best_book_id"] if pd.notnull(row["best_book_id"]) else row["goodreads_book_id"])
        url = f"https://www.goodreads.com/book/show/{gr_id}"
        print(f"{rank}. {title} ({year}) by {author} - score={score:.4f} +/- {stddev:.4f} - {url}")

if __name__ == "__main__":
    main()
