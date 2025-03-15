import sys
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
SCREEN_SIZE = (800, 800)
BACKGROUND_COLOR = (30, 30, 30)
CONTOUR_COLOR = (255, 255, 255)
POINT_COLOR = (255, 0, 0)
POINT_RADIUS = 3
MARGIN = 50
MIN_DISTANCE = 0.05
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
    L = 10.0
    return base + L * np.maximum(-x, 0)

#-------------------------------------------------------------------------
def barrier_potential(contours, min_distance, barrier_amplitude):
    g = min(check_distances(contour, min_distance, include_closing_segment=True) for contour in contours)
    return scaled_sigmoid(g, barrier_amplitude, min_distance)

#-------------------------------------------------------------------------
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
def main():
    num_X = 2
    num_Y = 2
    BARRIER_AMPLITUDE = 100.0

    initial_theta = np.pi / 20
    initial_X = np.array([[0.42, -0.44], [0.42, -0.64]])
    initial_Y = np.array([[0.23, -0.98], [0.13, -0.98]])

    initial_vars = np.concatenate(([initial_theta],
                                    initial_X.flatten(),
                                    initial_Y.flatten()))

    # Combined objective: original objective (here: -theta) plus barrier potential
    def combined_objective(vars):
        theta = vars[0]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        X = vars[1:1+2*num_X].reshape((num_X, 2))
        Y = vars[1+2*num_X:].reshape((num_Y, 2))
        base_obj = -theta  # maximizing theta via minimizing -theta
        contours = create_contour(X, Y, theta)
        barrier_val = barrier_potential(contours, MIN_DISTANCE, BARRIER_AMPLITUDE)
        return base_obj + barrier_val

    combined_grad = grad(combined_objective)

    # Callback that logs each iteration's parameters and draws the contour,
    def optimization_callback(vars):
        pygame.event.pump()  # Allow Pygame to process window events (fixes window freezing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        theta = vars[0]
        X = vars[1:1+2*num_X].reshape((num_X, 2))
        Y = vars[1+2*num_X:].reshape((num_Y, 2))
        base_obj = -theta
        contours = create_contour(X, Y, theta)
        barrier_val = min(barrier_potential(contour, MIN_DISTANCE, BARRIER_AMPLITUDE) for contour in contours)
        combined_val = base_obj + barrier_val
        caption_str = f"Iteration {optimization_callback.iteration}: Objective {base_obj:.6f}, Barrier {barrier_val:.6f}, Combined {combined_val:.6f}"
        pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'caption': caption_str}))
        optimization_callback.iteration += 1
        draw_contours(contours)
        # wait_for_keypress()
        time.sleep(0.01)
    optimization_callback.iteration = 0

    # Draw initial contour and wait for keypress
    optimization_callback(initial_vars)
    wait_for_keypress()

    # Threaded solver function
    SELECTED_SOLVER = SolverType.BASIN_HOPPING
    def run_solver():
        if SELECTED_SOLVER == SolverType.BASIN_HOPPING: #-----------------------------------
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "jac": combined_grad,
                "callback": optimization_callback,
                "options": {"disp": True, "eps": 1e-12, "maxiter": 10}
            }
            def accept_test(f_new, x_new, f_old, x_old):
                return f_new <= f_old  # Only accept new solution if it's better (or equal)
            basinhopping(
                func=combined_objective,
                x0=initial_vars,
                minimizer_kwargs=minimizer_kwargs,
                niter=100,
                stepsize=0.05,
                disp=True,
                accept_test=accept_test  # Prevents regression
            )
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
        elif SELECTED_SOLVER == SolverType.SIMPLE_GRADIENT_DESCENT: #-----------------------------------
            current_vars = initial_vars.copy()
            max_iter = 2000
            for _ in range(max_iter):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                grad_val = combined_grad(current_vars)
                delta = -grad_val
                delta = np.clip(delta, -0.001, 0.001)
                current_vars = current_vars + delta
                optimization_callback(current_vars)

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
