import sys
import os
from datetime import datetime
import time
from autograd import grad
import autograd.numpy as np
import pygame
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import torch
import threading
import enum

class SolverType(enum.Enum):
    BASIN_HOPPING = 1
    TORCH_ADAM = 2
    SIMPLE_GRADIENT_DESCENT = 3

# Constants
SCREEN_SIZE = (1000, 800)
BACKGROUND_COLOR = (30, 30, 30)
CONTOUR_COLOR = (255, 255, 255)
POINT_COLOR = (255, 0, 0)
POINT_RADIUS = 3
MARGIN = 50
MIN_DISTANCE = 0.01
FPS = 30

# Global numerical constants
EPS = 1e-12            # To avoid division by zero
EPS_BARRIER = 1e-8     # For barrier potential stability
LARGE_NUMBER = 1e6     # Large constant (unused here)

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
clock = pygame.time.Clock()

# Interaction helpers
#---------------------------------------------------------------
def wait_for_keypress():
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()  # Properly exit the program on window close
            elif event.type == pygame.KEYDOWN:
                waiting = False

#---------------------------------------------------------------
def ASSERT(condition, message="Assertion failed"):
    if not condition:
        raise ValueError(message)

#-------------------------------------------------------------------------
def transform(point, scale, offset_x, offset_y):
    return int(point[0] * scale + offset_x), int(-point[1] * scale + offset_y)

#-------------------------------------------------------------------------
def inverse_transform(screen_point, scale, offset_x, offset_y):
    return np.array([(screen_point[0] - offset_x) / scale, -(screen_point[1] - offset_y) / scale])

#-------------------------------------------------------------------------
def compute_similarity_transform(A, B, P, Q):
    v_source = B - A; v_dest = Q - P; scale = np.linalg.norm(v_dest) / np.linalg.norm(v_source)
    angle = np.arctan2(v_dest[1], v_dest[0]) - np.arctan2(v_source[1], v_source[0])
    cos_a = np.cos(angle); sin_a = np.sin(angle)
    R = np.array([[cos_a, -sin_a],[sin_a, cos_a]])
    t = P - scale * R.dot(A)
    T = np.array([[scale * cos_a, -scale * sin_a, t[0]],[scale * sin_a, scale * cos_a, t[1]],[0,0,1]])
    return T

#-------------------------------------------------------------------------
def apply_transform(T, points):
    points = np.atleast_2d(points); ones = np.ones((points.shape[0], 1))
    ph = np.hstack([points, ones])
    transformed = (T @ ph.T).T
    return transformed[:, :2]

#-------------------------------------------------------------------------
def compute_scaling_and_offset(contours, screen_size, margin):
    all_points = np.vstack(contours)
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    width, height = max_x - min_x, max_y - min_y
    scale = min((screen_size[0] - 2 * margin) / max(width, 1),
                (screen_size[1] - 2 * margin) / max(height, 1))
    offset_x = screen_size[0] / 2 - ((max_x + min_x) / 2) * scale
    offset_y = screen_size[1] / 2 + ((max_y + min_y) / 2) * scale
    return scale, offset_x, offset_y

#-------------------------------------------------------------------------
def rotate_point(point, angle, center):
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return center + rotation_matrix @ (point - center)

#-------------------------------------------------------------------------
def distance_point_to_segment(p, a, b):
    ab = b - a
    ap = p - a
    ab_norm_sq = np.dot(ab, ab)
    proj = np.dot(ap, ab) / (ab_norm_sq + EPS)
    proj_clamped = np.clip(proj, 0.0, 1.0)
    nearest = a + proj_clamped * ab
    return np.linalg.norm(p - nearest)

#-------------------------------------------------------------------------
def crossing_distance(seg1, seg2):
    # seg1: (p, q), seg2: (r, s)
    p, q = seg1
    r, s = seg2
    r_vec = q - p
    s_vec = s - r
    r_cross_s = r_vec[0] * s_vec[1] - r_vec[1] * s_vec[0]
    inv_r_cross_s = 1.0 / (r_cross_s + EPS)
    qmp = r - p
    t = (qmp[0] * s_vec[1] - qmp[1] * s_vec[0]) * inv_r_cross_s
    u = (qmp[0] * r_vec[1] - qmp[1] * r_vec[0]) * inv_r_cross_s
    pen1 = np.minimum(t, 1 - t) * np.linalg.norm(r_vec)
    pen2 = np.minimum(u, 1 - u) * np.linalg.norm(s_vec)
    return np.minimum(np.maximum(-pen1, -pen2), 100.0) #avoid ultra high values for close to parallel segments

#-------------------------------------------------------------------------
def check_distances(points, min_distance, include_closing_segment=False):
    pts = np.array(points)
    n = pts.shape[0]
    violations = []
    # print("=== Checking point-to-segment distances ===")
    # Point-to-segment distances (non-adjacent)
    for i in range(n - 1):
        for j in range(n):
            if j != i and j != i + 1:
                d = distance_point_to_segment(pts[j], pts[i], pts[i+1]) - min_distance
                violations.append(d)
                # print(f"Point-to-segment violation added: {d:.6f}")
    if include_closing_segment:
        for j in range(n):
            if j != n - 1 and j != 0:
                d = distance_point_to_segment(pts[j], pts[-1], pts[0]) - min_distance
                violations.append(d)
                # print(f"Closing segment point violation added: {d:.6f}")
    # print("=== Checking segment-to-segment distances ===")
    # Crossing distances for segments
    segments = [(pts[i], pts[i+1]) for i in range(n - 1)]
    if include_closing_segment:
        segments.append((pts[-1], pts[0]))
    m = len(segments)
    for i in range(m):
        for j in range(i + 1, m):
            # Skip adjacent segments using indices:
            if j == i + 1 or (include_closing_segment and i == 0 and j == m - 1):
                continue
            d = crossing_distance(segments[i], segments[j]) - min_distance
            violations.append(d)
            # print(f"Segment-to-segment violation added: {d:.6f}")
    # print("=== Checking boundary constraints ===")
    # Boundary violation: each coordinate must satisfy |coordinate| <= 2.
    boundary = 2 - np.max(np.abs(pts))
    violations.append(boundary)
    # print(f"Boundary violation added: {boundary:.6f}")
    return np.min(np.array(violations))

#-------------------------------------------------------------------------
def safe_exp(x, max_exp=500):
    return np.exp(np.clip(x, -max_exp, max_exp))

#-------------------------------------------------------------------------
def scaled_sigmoid(x, amplitude, min_distance):
    alpha = 24/min_distance
    base = amplitude/(1+safe_exp(alpha*x))
    L = 1000.0
    return base + L * np.maximum(-x, 0)

#-------------------------------------------------------------------------
def barrier_potential(contours, min_distance, barrier_amplitude):
    g = min(check_distances(contour, min_distance, include_closing_segment=True) for contour in contours)
    return scaled_sigmoid(g, barrier_amplitude, min_distance)

#-------------------------------------------------------------------------
# Voderberg
def create_contour(X, Y, theta):
    N = np.array([0, 1])
    S = np.array([0, -1])
    A = rotate_point(S, theta, N)
    main_BR = np.vstack([X, A, Y])
    main_BR_excl_A = np.vstack([X])
    main_BR_excl_Y = np.vstack([X, A])
    main_BR_180 = -main_BR[::-1]  # 180-degree rotation and reverse order
    main_R = np.vstack([N, main_BR_180, main_BR, S])
    main_L_rev = np.array([rotate_point(p, -theta, N) for p in np.vstack([main_BR_180, main_BR_excl_A])])
    main_L = main_L_rev[::-1]
    main_contour = np.vstack([main_R, main_L])

    left_L_rev = np.array([rotate_point(p, -2.0*theta, N) for p in np.vstack([main_BR_180, main_BR_excl_Y])])
    left_L = left_L_rev[::-1]
    Y_teta = np.array([rotate_point(p, -theta, N) for p in Y])
    left_contour = np.vstack([N, main_L_rev, S, Y_teta, left_L])

    #The right piece contour is just 180 of the main one, irrelevant in the current version of the probleme

    return [main_contour, left_contour]

#-------------------------------------------------------------------------
# Improved Voderberg with surrounding number 2
def create_contour_srn2(theta, X, P, Q, Y, B):
    N = np.array([0, 1])
    S = np.array([0, -1])
    A = rotate_point(S, theta, N) + (S - B)
    Pm = P + (B - A) # P image by A B translation
    Qm = Q + (B - A)

    Qm_180 = -Qm  # 180-degree rotation and reverse order
    Y_180 = -Y[::-1]
    Q_180 = -Q
    A_180 = -A
    P_180 = -P
    X_180 = -X[::-1]
    Pm_180 = -Pm
    B_180 = -B
    main_R_for_L = np.vstack([Qm_180, Y_180, Q_180, A_180, P_180, X_180, X, P])
    main_R_post_A = np.vstack([Q, Y, Qm, B, Pm])
    main_R = np.vstack([Pm_180, B_180, main_R_for_L, A, main_R_post_A])
    main_L = np.array([rotate_point(p, -theta, B_180) + (N - B_180) for p in main_R_for_L])
    main_L_rev = main_L[::-1]
    main_contour = np.vstack([N, main_R, S, main_L_rev])

    main_L_for_partial_LL = np.vstack([Pm_180, N, main_L, S])
    main_LL_partial = np.array([rotate_point(p, -theta, B_180) + (N - B_180) for p in main_L_for_partial_LL])
    main_LL_partial_rev = main_LL_partial[::-1]
    main_R_post_A_LL_transformed = np.array([rotate_point(p, -theta, B_180) + (N - B_180) for p in main_R_post_A])
    left_contour = np.vstack([N, main_L, S, main_R_post_A_LL_transformed, main_LL_partial_rev])

    return [main_contour, left_contour]

#-------------------------------------------------------------------------
def contact_length(theta, P, Q, B):
    N = np.array([0,1])
    S = np.array([0,-1])
    A = rotate_point(S, theta, N) + (S - B)
    return np.linalg.norm(P - A) + np.linalg.norm(A - Q)

#-------------------------------------------------------------------------
def draw_contours(contours):
    screen.fill(BACKGROUND_COLOR)
    scale, offset_x, offset_y = compute_scaling_and_offset(contours, SCREEN_SIZE, MARGIN)
    all_points = np.vstack(contours)
    min_x, max_x = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_y, max_y = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    grid_color = (200, 200, 200)
    for x in range(int(np.floor(min_x)), int(np.ceil(max_x)) + 1):
        pygame.draw.line(screen, grid_color, transform((x, min_y - 1), scale, offset_x, offset_y),
                         transform((x, max_y + 1), scale, offset_x, offset_y), 1)
    for y in range(int(np.floor(min_y)), int(np.ceil(max_y)) + 1):
        pygame.draw.line(screen, grid_color, transform((min_x - 1, y), scale, offset_x, offset_y),
                         transform((max_x + 1, y), scale, offset_x, offset_y), 1)
    axis_color = (255, 255, 255)
    pygame.draw.line(screen, axis_color, transform((min_x - 1, 0), scale, offset_x, offset_y),
                     transform((max_x + 1, 0), scale, offset_x, offset_y), 2)
    pygame.draw.line(screen, axis_color, transform((0, min_y - 1), scale, offset_x, offset_y),
                     transform((0, max_y + 1), scale, offset_x, offset_y), 2)
    for contour in contours:
        transformed_points = [transform(point, scale, offset_x, offset_y) for point in contour]
        pygame.draw.polygon(screen, CONTOUR_COLOR, transformed_points, width=1)
        for i, point in enumerate(contour):
            color = CONTOUR_COLOR
            if np.allclose(point, [0, 1]):
                color = (255, 0, 0)
            elif np.allclose(point, [0, -1]):
                color = (0, 0, 255)
            pygame.draw.circle(screen, color, transformed_points[i], POINT_RADIUS)
            pygame.draw.circle(screen, (255, 255, 0), transformed_points[i], int(MIN_DISTANCE * scale), 1)
    pygame.display.flip()

#-------------------------------------------------------------------------
def draw_debug_point():
    scale, offset_x, offset_y = SCREEN_SIZE[0] / 4.0, SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2
    font = pygame.font.SysFont("Arial", 20)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                running = False
        screen.fill(BACKGROUND_COLOR)
        a, b = np.array([-1, 0]), np.array([1, 0])
        pygame.draw.line(screen, (255, 255, 255), transform(a, scale, offset_x, offset_y), 
                         transform(b, scale, offset_x, offset_y), 2)
        mouse_real = inverse_transform(pygame.mouse.get_pos(), scale, offset_x, offset_y)
        pygame.draw.circle(screen, POINT_COLOR, transform(mouse_real, scale, offset_x, offset_y), POINT_RADIUS)
        dist = distance_point_to_segment(mouse_real, a, b)
        interest = dist - MIN_DISTANCE
        f_interest = scaled_sigmoid(interest, 100.0, MIN_DISTANCE)
        text_surface = font.render(
            "Coords: ({:.4f}, {:.4f})  Dist: {:.4f}  (Dist-MIN): {:.4f}  f(Dist-MIN): {:.4f}".format(
                mouse_real[0], mouse_real[1], dist, interest, f_interest), True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        pygame.display.flip()
        clock.tick(FPS)

#-------------------------------------------------------------------------
def draw_debug_segment():
    scale, offset_x, offset_y = SCREEN_SIZE[0] / 4.0, SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2
    font = pygame.font.SysFont("Arial", 20)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                running = False
        screen.fill(BACKGROUND_COLOR)
        seg1_start, seg1_end = np.array([-1, 0]), np.array([1, 0])
        pygame.draw.line(screen, (255, 255, 255), transform(seg1_start, scale, offset_x, offset_y),
                         transform(seg1_end, scale, offset_x, offset_y), 2)
        seg2_start = np.array([0, 2])
        seg2_end = inverse_transform(pygame.mouse.get_pos(), scale, offset_x, offset_y)
        pygame.draw.line(screen, (0, 255, 0), transform(seg2_start, scale, offset_x, offset_y),
                         transform(seg2_end, scale, offset_x, offset_y), 2)
        cross_dist = crossing_distance((seg1_start, seg1_end), (seg2_start, seg2_end))
        pygame.draw.circle(screen, POINT_COLOR, transform(seg2_end, scale, offset_x, offset_y), POINT_RADIUS)
        text_surface = font.render("Coords: ({:.4f}, {:.4f})  Crossing: {:.4f}".format(
            seg2_end[0], seg2_end[1], cross_dist), True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        pygame.display.flip()
        clock.tick(FPS)

#-------------------------------------------------------------------------
def save_vars_with_metadata(filepath, vars_array, metadata=None):
    with open(filepath, 'w') as f:
        if metadata is not None:
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
        f.write("# VARS_START\n")
        for i, val in enumerate(vars_array):
            f.write(f"{i}: {val:.17g}\n")
        f.write("# VARS_END\n")

#-------------------------------------------------------------------------
def auto_save(vars_array, base_obj=0, barrier_val=0, combined_val=0, iteration=0):
    dirname = "voderberg_optimisation_data"
    os.makedirs(dirname, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{timestamp}_iter{iteration:04d}.txt"
    path = os.path.join(dirname, filename)
    metadata = {
        "objective": f"{base_obj:.6f}",
        "barrier": f"{barrier_val:.6f}",
        "combined": f"{combined_val:.6f}",
        "timestamp": datetime.now().isoformat(timespec='seconds')
    }
    save_vars_with_metadata(path, vars_array, metadata)

#-------------------------------------------------------------------------
def load_vars_with_metadata(filepath):
    vars_list = []
    reading_vars = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("# VARS_START"):
                reading_vars = True
            elif line.startswith("# VARS_END"):
                reading_vars = False
            elif reading_vars:
                _, val = line.split(":", 1)
                vars_list.append(float(val.strip()))
    return np.array(vars_list, dtype=np.float64)

#-------------------------------------------------------------------------
def acquisition_mode(background_image_path):
    aspect = SCREEN_SIZE[0] / SCREEN_SIZE[1]
    true_h = 3.0
    true_w = true_h * aspect
    visible_zone = np.array([[-true_w/2, -1.5],[true_w/2, 1.5]])
    all_points = visible_zone.reshape((-1,2))
    scale, offset_x, offset_y = compute_scaling_and_offset([all_points], SCREEN_SIZE, MARGIN)
    bg_img = pygame.image.load(background_image_path).convert_alpha()
    img_width, img_height = bg_img.get_size()
    px_w = int(true_w * scale)
    px_h = int(true_h * scale)
    rect_x = offset_x - (true_w/2) * scale
    rect_y = offset_y - 1.5 * scale
    scale_factor = min(px_w / img_width, px_h / img_height)
    new_w = int(img_width * scale_factor)
    new_h = int(img_height * scale_factor)
    bg_img = pygame.transform.smoothscale(bg_img, (new_w, new_h))
    bg_surf = pygame.Surface(SCREEN_SIZE, flags=pygame.SRCALPHA)
    bg_offset_x = rect_x + (px_w - new_w) // 2
    bg_offset_y = rect_y + (px_h - new_h) // 2
    bg_surf.blit(bg_img, (bg_offset_x, bg_offset_y))
    bg_surf.set_alpha(128)
    points = []
    acquiring = True
    while acquiring:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    acquiring = False
                    break
                elif event.key == pygame.K_z and (event.mod & pygame.KMOD_CTRL):
                    if points:
                        points.pop()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                true_point = inverse_transform(event.pos, scale, offset_x, offset_y)
                points.append(true_point)
        screen.fill(BACKGROUND_COLOR)
        screen.blit(bg_surf, (0,0))
        if points:
            if len(points) > 1:
                pygame.draw.lines(screen, (255,255,255), False, [transform(pt, scale, offset_x, offset_y) for pt in points], 2)
            for pt in points:
                pygame.draw.circle(screen, (255,0,0), transform(pt, scale, offset_x, offset_y), POINT_RADIUS)
        pygame.display.flip()
        pygame.time.wait(10)
    if len(points) < 2:
        return np.array([], dtype=np.float64)
    N = points[0]
    S = points[-1]
    T = compute_similarity_transform(N, S, np.array([0,1]), np.array([0,-1]))
    transformed_points = apply_transform(T, np.array(points))
    return transformed_points[1:-1]

# Global variable to hold the current basin hopping iteration number.
current_bh_iter = 0
#-------------------------------------------------------------------------
def main():
    init_file = "voderberg_srn2_vars.init"
    if os.path.isfile(init_file):
        initial_srn2_vars = load_vars_with_metadata(init_file)
        theta = initial_srn2_vars[0]
        X = initial_srn2_vars[1:15].reshape((7,2))
        P = initial_srn2_vars[15:17]
        Q = initial_srn2_vars[17:19]
        Y = initial_srn2_vars[19:25].reshape((3,2))
        B = initial_srn2_vars[25:27]
    else:
        # Voderberg SRN2 acquisition conventions
        # N X P A Q Y B S
        VSRN2_keyPoints = acquisition_mode("./VoderbergSRN2Patron.png")
        X = VSRN2_keyPoints[0:7]
        P = VSRN2_keyPoints[7]
        A = VSRN2_keyPoints[8]
        Q = VSRN2_keyPoints[9]
        Y = VSRN2_keyPoints[10:13]
        B = VSRN2_keyPoints[13]
        Bm = -B
        v_BmA = A - Bm
        v_ref = np.array([0, -2])
        theta = np.arctan2(v_BmA[1], v_BmA[0]) - np.arctan2(v_ref[1], v_ref[0])
        initial_srn2_vars = np.concatenate((
            [theta],
            X.flatten(),
            P,
            Q,
            Y.flatten(),
            B
        ))
        auto_save(initial_srn2_vars)
    srn2_contours = create_contour_srn2(theta, X, P, Q, Y, B)
    draw_contours(srn2_contours)
    print("Points loaded or set")
    wait_for_keypress()

    BARRIER_AMPLITUDE = 100.0

    # Combined objective: original objective (here: -theta) plus barrier potential
    def combined_objective_srn2(vars):
        theta = vars[0]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        X = vars[1:15].reshape((7,2))
        P = vars[15:17]
        Q = vars[17:19]
        Y = vars[19:25].reshape((3,2))
        B = vars[25:27]
        base_obj = -contact_length(theta, P, Q, B)  # maximizing via minimizing -obj
        contours = create_contour_srn2(theta, X, P, Q, Y, B)
        barrier_val = barrier_potential(contours, MIN_DISTANCE, BARRIER_AMPLITUDE)
        return base_obj + barrier_val

    combined_grad_srn2 = grad(combined_objective_srn2)

    # Callback that logs each iteration's parameters and draws the contour,
    def optimization_callback_srn2(vars, free_text=""):
        pygame.event.pump()  # Allow Pygame to process window events (fixes window freezing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
        theta = vars[0]
        X = vars[1:15].reshape((7,2))
        P = vars[15:17]
        Q = vars[17:19]
        Y = vars[19:25].reshape((3,2))
        B = vars[25:27]
        base_obj = -theta
        contours = create_contour_srn2(theta, X, P, Q, Y, B)
        barrier_val = barrier_potential(contours, MIN_DISTANCE, BARRIER_AMPLITUDE)
        combined_val = base_obj + barrier_val
        caption_str = f"{free_text} Iteration {optimization_callback_srn2.iteration}: Objective {base_obj:.6f}, Barrier {barrier_val:.6f}, Combined {combined_val:.6f}"
        pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'caption': caption_str}))
        optimization_callback_srn2.iteration += 1
        draw_contours(contours)
        # wait_for_keypress()
        auto_save(vars, base_obj, barrier_val, combined_val, optimization_callback_srn2.iteration)
        time.sleep(0.01)
    optimization_callback_srn2.iteration = 0

# OLD INITIALISATION FOR BASIC VODERBERG
#     num_X = 2
#     num_Y = 2
#     BARRIER_AMPLITUDE = 100.0
#
#     initial_theta = np.pi / 20
#     initial_X = np.array([[0.42, -0.44], [0.42, -0.64]])
#     initial_Y = np.array([[0.23, -0.98], [0.13, -0.98]])
# 
#     initial_vars = np.concatenate(([initial_theta],
#                                     initial_X.flatten(),
#                                     initial_Y.flatten()))
# 
#     # Combined objective: original objective (here: -theta) plus barrier potential
#     def combined_objective(vars):
#         theta = vars[0]
#         theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
#         X = vars[1:1+2*num_X].reshape((num_X, 2))
#         Y = vars[1+2*num_X:].reshape((num_Y, 2))
#         base_obj = -theta  # maximizing theta via minimizing -theta
#         contours = create_contour(X, Y, theta)
#         barrier_val = barrier_potential(contours, MIN_DISTANCE, BARRIER_AMPLITUDE)
#         return base_obj + barrier_val
# 
#     combined_grad = grad(combined_objective)
# 
#     # Callback that logs each iteration's parameters and draws the contour,
#     def optimization_callback(vars, free_text=""):
#         pygame.event.pump()  # Allow Pygame to process window events (fixes window freezing)
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()
#         theta = vars[0]
#         X = vars[1:1+2*num_X].reshape((num_X, 2))
#         Y = vars[1+2*num_X:].reshape((num_Y, 2))
#         base_obj = -theta
#         contours = create_contour(X, Y, theta)
#         barrier_val = barrier_potential(contours, MIN_DISTANCE, BARRIER_AMPLITUDE)
#         combined_val = base_obj + barrier_val
#         caption_str = f"{free_text} Iteration {optimization_callback.iteration}: Objective {base_obj:.6f}, Barrier {barrier_val:.6f}, Combined {combined_val:.6f}"
#         pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'caption': caption_str}))
#         optimization_callback.iteration += 1
#         draw_contours(contours)
#         # wait_for_keypress()
#         auto_save(vars, base_obj, barrier_val, combined_val, optimization_callback.iteration)
#         time.sleep(0.01)
#     optimization_callback.iteration = 0

    # Draw initial contour and wait for keypress
    optimization_callback_srn2(initial_srn2_vars)
    print("Optimisation callback initial test. Press a key to optimize")
    wait_for_keypress()

    # Threaded solver function
    SELECTED_SOLVER = SolverType.SIMPLE_GRADIENT_DESCENT
    def run_solver():
#        if SELECTED_SOLVER == SolverType.BASIN_HOPPING:  #-----------------------------------
#             def local_callback(xk):
#                 # This callback is called by the local L-BFGS-B minimizer within basin hopping.
#                 # It now accepts only one argument (xk) and uses the global current_bh_iter
#                 # to prepend free text indicating the current basin hopping iteration.
#                 optimization_callback(xk, free_text=f"BASEHOPPING (iteration {current_bh_iter}) ")
#             minimizer_kwargs = {
#                 "method": "L-BFGS-B",
#                 "jac": combined_grad,
#                 "callback": local_callback,  # Use the local callback that expects one argument.
#                 "options": {"disp": True, "eps": 1e-12, "maxiter": 10}
#             }
#             def bh_callback(x, f, accept):
#                 global current_bh_iter
#                 current_bh_iter += 1
#                 # This callback is called at each basin hopping iteration.
#                 # It injects free text with the basin hopping iteration info,
#                 # and then calls the optimization_callback to display it.
#                 optimization_callback(x, free_text=f"BASEHOPPING (iteration {current_bh_iter}, global: f: {f:.6f}, accepted: {accept}) ")
#             basinhopping(
#                 func=combined_objective,
#                 x0=initial_vars,
#                 minimizer_kwargs=minimizer_kwargs,
#                 niter=100,
#                 stepsize=0.05,
#                 disp=True,
#                 callback=bh_callback
#             )
# NEEDS A TORCH REFACTOR OF EVERYTHING
#         elif SELECTED_SOLVER == SolverType.TORCH_ADAM: #-----------------------------------
#             current_vars = torch.tensor(initial_vars, requires_grad=True, dtype=torch.float32)
#             optimizer = torch.optim.Adam([current_vars], lr=0.01)
#             max_iter = 2000
#             for _ in range(max_iter):
#                 for event in pygame.event.get():
#                     if event.type == pygame.QUIT:
#                         pygame.quit()
#                         sys.exit()
#                 optimizer.zero_grad()
#                 loss = combined_objective(current_vars)
#                 loss.backward()
#                 optimizer.step()
        #                 optimization_callback(current_vars.detach().numpy())
        if SELECTED_SOLVER == SolverType.SIMPLE_GRADIENT_DESCENT:  # -----------------------------------
            current_vars = initial_srn2_vars.copy()
            step = 0.002
            max_iter = 2000
            obj_history = []
            prev_avg = None
            for _ in range(max_iter):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                grad_val = combined_grad_srn2(current_vars)
                delta = -grad_val
                # Clip the updates:
                # For the angle (theta), a change dtheta results in a point movement of ~2*|dtheta|,
                # so we clip delta[0] to +/-(step/2).
                delta[0] = np.clip(delta[0], -step / 2, step / 2)
                # For the remaining point coordinates, clip directly to +/-step.
                delta[1:] = np.clip(delta[1:], -step, step)
                # Add small noise after clipping (noise magnitude is of order step/10)
                noise = np.random.uniform(-step/10, step/10, delta.shape)
                delta = delta + noise
                current_vars = current_vars + delta
                # Record the current combined objective value.
                current_obj = combined_objective_srn2(current_vars)
                obj_history.append(current_obj)
                # Every 4 iterations, compare the average of the last 4 objective values to the previous block's average.
                if len(obj_history) == 4:
                    avg_obj = np.mean(obj_history)
                    if prev_avg is not None:
                        # If the new average has not improved (i.e. decreased) by at least 1% compared to the previous block,
                        # then reduce the step size by half.
                        if avg_obj > prev_avg * 0.99:
                            step /= 2
                    prev_avg = avg_obj
                    obj_history = []
                free_text = f"SMART_GRADIENT_DESCENT (step: {step:.6f}"
                if prev_avg is not None:
                    free_text += f", last avg: {prev_avg:.6f}"
                free_text += ")"
                optimization_callback_srn2(current_vars, free_text=free_text)

    # Run solver in a separate thread
    solver_thread = threading.Thread(target=run_solver, daemon=True)
    solver_thread.start()

    # Keep Pygame responsive
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.USEREVENT:
                pygame.display.set_caption(event.caption)
        time.sleep(0.01)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
#    draw_debug_point()
#    pygame.quit()
    main()
