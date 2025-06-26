import itertools
import networkx as nx
import matplotlib.pyplot as plt

# Unique preference bimatrices P3 and P4
def get_bimatrices():
    # Example 3x3 (cycle)
    P3 = [
        [(0,1), (2,0), (1,0)],
        [(1,0), (2,2), (0,1)],
        [(0,2), (2,1), (1,2)],
    ]
    # Example 4x4 (isolated cycle Knuth)
    P4 = [
        [(0, 2), (2, 1), (1, 3), (3, 0)],
        [(3, 0), (0, 2), (2, 1), (1, 3)],
        [(1, 3), (3, 0), (0, 2), (2, 1)],
        [(2, 1), (1, 3), (3, 0), (0, 2)]
    ]
    # convert to zero-based ranks
    def to_zero(P):
        return [[(x-1,y-1) for x,y in row] for row in P]
    return to_zero(P3), to_zero(P4)

# extract men and women preference lists from a bimatrix
def extract_preferences_from_bimatrix(P):
    n = len(P)
    pref_m = []
    for i in range(n):
        row = [(j, P[i][j][0]) for j in range(n)]  # man's rank of each woman
        row.sort(key=lambda t: t[1])
        pref_m.append([j for j, _ in row])
    pref_w = []
    for j in range(n):
        col = [(i, P[i][j][1]) for i in range(n)]  # woman's rank of each man
        col.sort(key=lambda t: t[1])
        pref_w.append([i for i, _ in col])
    return pref_m, pref_w

# generate all complete matchings
def generate_all_matchings(n):
    return list(itertools.permutations(range(n)))

# find all blocking pairs in a matching
def find_blocking_pairs(matching, pref_m, pref_w):
    blocking = []
    inv = {matching[i]: i for i in range(len(matching))}
    for m in range(len(matching)):
        w_curr = matching[m]
        for w in pref_m[m]:
            if w == w_curr:
                break
            m_prev = inv[w]
            if pref_w[w].index(m) < pref_w[w].index(m_prev):
                blocking.append((m, w))
    return blocking

# resolve a single blocking pair
def resolve_blocking_pair(matching, pair):
    m, w = pair
    new = list(matching)
    w_old = matching[m]
    m_prev = matching.index(w)
    new[m] = w
    new[m_prev] = w_old
    return tuple(new)

# build directed state graph of matchings
def build_state_graph(pref_m, pref_w):
    n = len(pref_m)
    G = nx.DiGraph()
    allm = generate_all_matchings(n)
    G.add_nodes_from(allm)
    for m in allm:
        for pair in find_blocking_pairs(m, pref_m, pref_w):
            G.add_edge(m, resolve_blocking_pair(m, pair))
    return G

# compute node types: red, green, yellow, gray
def compute_node_types(G):
    sources = {n for n in G.nodes() if G.in_degree(n)==0 and G.out_degree(n)>0}
    sinks   = {n for n in G.nodes() if G.out_degree(n)==0}
    # ancestors of sinks => yellow
    rev = G.reverse(copy=False)
    yellow = set()
    for s in sinks:
        yellow.update(nx.descendants(rev, s))
    yellow.difference_update(sources.union(sinks))
    types = {}
    for n in G.nodes():
        if n in sources:
            types[n] = 'red'
        elif n in sinks:
            types[n] = 'green'
        elif n in yellow:
            types[n] = 'yellow'
        else:
            types[n] = 'gray'
    return types

# draw with matplotlib and export both png and dot
def draw_and_export(G, case):
    # classification
    types = compute_node_types(G)
    # matplotlib
    pos = nx.kamada_kawai_layout(G)
    node_colors = [types[n] for n in G.nodes()]
    labels = {n: str(n) for n in G.nodes()}
    plt.figure(figsize=(8,6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='->', arrowsize=10,
        connectionstyle='arc3,rad=0.1')
    plt.axis('off'); plt.tight_layout()
    plt.savefig(f"state_graph_{case}.png", dpi=300); plt.close()
    # dot export
    dot = nx.nx_pydot.to_pydot(G)
    for dnode in dot.get_nodes():
        name = dnode.get_name().strip('"')
        if name:
            color = types.get(eval(name), 'gray')
            dnode.set_style('filled')
            dnode.set_fillcolor(color)
    dot.write(f"state_graph_{case}.dot")

def search_in_non_isomorphic_profiles(checker_function):
    """
    Searches through non-redundant 4x4 preference profiles for one that
    matches a specific property, determined by the checker_function.
    """
    N = 4
    profiles_checked = 0
    # --- Canonization setup ---
    canonical_f0_prefs = [0, 1, 2, 3]
    canonical_h0_prefs_list = [
        [0, 1, 2, 3], [1, 0, 2, 3], [1, 2, 0, 3], [1, 2, 3, 0]
    ]
    all_permutations = list(itertools.permutations(range(N)))
    
    # --- Progress Indicator Setup ---
    # Total search space is 4 (for H0's lists) * (24^6) (for the other 6 lists)
    total_profiles_to_check = 4 * (len(all_permutations)**6)
    # Update progress every 1,000,000 checks for readability
    update_frequency = 1_000_000
    
    print(f"Starting search through a space of ~{total_profiles_to_check / 1e9:.2f} billion non-redundant profiles.")
    print("This will take an extremely long time.")
    # --- Main Iteration Loop ---
    for h0_prefs in canonical_h0_prefs_list:
        iterator_for_remaining_lists = itertools.product(all_permutations, repeat=6)
        
        for remaining_lists_config in iterator_for_remaining_lists:
            profiles_checked += 1
            if profiles_checked % update_frequency == 0:
                progress_percent = (profiles_checked / total_profiles_to_check) * 100
                print(f"  ... Progress: {progress_percent:.4f}% ({profiles_checked:,} profiles checked)")
            # --- Assemble and Check the Profile ---
            h1_prefs, h2_prefs, h3_prefs = remaining_lists_config[0:3]
            f1_prefs, f2_prefs, f3_prefs = remaining_lists_config[3:6]
            
            final_men_prefs = [list(h0_prefs), list(h1_prefs), list(h2_prefs), list(h3_prefs)]
            final_women_prefs = [list(canonical_f0_prefs), list(f1_prefs), list(f2_prefs), list(f3_prefs)]
            if checker_function(final_men_prefs, final_women_prefs):
                print(f"\nSUCCESS: Found a profile with the desired property after checking {profiles_checked:,} instances.")
                return final_men_prefs, final_women_prefs
    print("\nFull search space exhausted. No profile with the desired property was found.")
    return None

def has_isolated_cycle(pref_m, pref_w):
    """
    Returns True if the directed state graph built from (pref_m, pref_w)
    contains a cycle among matchings that cannot reach any stable matching.
    pref_m: list of lists, pref_m[i] is the list of women (indices) in
            order of preference for man i.
    pref_w: list of lists, pref_w[j] is the list of men in order of
            preference for woman j.
    """
    # 1. Build the full state graph of matchings
    G = build_state_graph(pref_m, pref_w)
    # 2. Identify stable nodes (no outgoing edges)
    stable = [n for n in G.nodes() if G.out_degree(n) == 0]
    if not stable:
        return False
    # 3. Find all nodes that CAN reach a stable node
    #    by reversing the graph and collecting descendants of each stable
    RG = G.reverse(copy=False)
    can_reach_stable = set(stable)
    for s in stable:
        can_reach_stable |= nx.descendants(RG, s)
    # 4. The "isolated" nodes are those that cannot reach any stable node
    isolated = set(G.nodes()) - can_reach_stable
    if not isolated:
        return False
    # 5. Check for any directed cycle within that isolated subgraph
    sub = G.subgraph(isolated)
    try:
        next(nx.simple_cycles(sub))
        return True
    except StopIteration:
        return False

if __name__ == '__main__':
    P3, P4 = get_bimatrices()
    for name, bimatrix in [('P3', P3), ('P4', P4)]:
        pref_m, pref_w = extract_preferences_from_bimatrix(bimatrix)
        G = build_state_graph(pref_m, pref_w)
        draw_and_export(G, name)
        print(f"Graph exported for {name}.")


# if __name__ == '__main__':
#     # =========================================================================
#     # NOTE : The following search is a "brute-force" demonstration to find
#     # a profile with a rare property among hundreds of millions of candidates.
#     # The 'has_isolated_cycle' function is computationally expensive as it
#     # builds a graph of 24 nodes for each candidate.
#     # This search will run for an extremely long time (potentially days/weeks).
#     # =========================================================================
# 
#     # 1. Call the search function, passing the property checker as an argument.
#     found_profile_tuple = search_in_non_isomorphic_profiles(has_isolated_cycle)
#     
#     # 2. Print the preference matrices if a result is found.
#     if found_profile_tuple:
#         men_preferences, women_preferences = found_profile_tuple
#         print("\n" + "="*40)
#         print("           FOUND MATCHING PROFILE")
#         print("="*40)
#         print("\nMen's Preferences (H0-H3):")
#         for i, p_list in enumerate(men_preferences):
#             # Format output for readability, e.g., F1 > F0 > ...
#             pref_str = " > ".join([f"F{woman}" for woman in p_list])
#             print(f"  H{i}: {pref_str}")
#             
#         print("\nWomen's Preferences (F0-F3):")
#         for i, p_list in enumerate(women_preferences):
#             pref_str = " > ".join([f"H{man}" for man in p_list])
#             print(f"  F{i}: {pref_str}")
#         print("\n" + "="*40)
#     else:
#         print("\nSearch concluded without finding a matching profile.")
