import sys
import time
from autograd import grad
import autograd.numpy as np
import pygame
from scipy.optimize import minimize

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
def rotate_point(point, angle, center=(0, 0)):
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
def scaled_sigmoid(x, amplitude, min_distance):
    alpha = 24 / min_distance  # Ensure transition happens over MIN_DISTANCE/2
    return amplitude / (1 + np.exp(alpha * x))

#-------------------------------------------------------------------------
def barrier_potential(points, min_distance, barrier_amplitude):
    g = check_distances(points, min_distance, include_closing_segment=True)
    return scaled_sigmoid(g, barrier_amplitude, min_distance)  # Log barrier only when constraints are violated

#-------------------------------------------------------------------------
def create_contour(X, Y, theta):
    N = np.array([0, 1])
    S = np.array([0, -1])

    A = rotate_point(S, theta)

    Z = np.vstack([X, A, Y])
    Z_cut = np.vstack([X, A])
    Z_prime = -Z[::-1]  # 180-degree rotation and reverse order

    base_contour = np.vstack([N, Z_prime, Z, S])

    W = np.array([rotate_point(p, -theta, N) for p in np.vstack([Z_prime, Z_cut])])
    W = W[::-1]

    final_contour = np.vstack([base_contour, W])

    return final_contour

#-------------------------------------------------------------------------
# def perimeter(contour):
#     distances = np.linalg.norm(np.diff(np.vstack([contour, contour[0]]), axis=0), axis=1)
#     return np.sum(distances)

#-------------------------------------------------------------------------
# def objective(variables, num_X, num_Y):
#     theta = variables[0]
#     X = variables[1:1 + 2*num_X].reshape((num_X, 2))
#     Y = variables[1 + 2*num_X:].reshape((num_Y, 2))
#     contour = create_contour(X, Y, theta)
#     return -perimeter(contour)

#-------------------------------------------------------------------------
def objective_theta(variables):
    theta = variables[0]
    return -theta  # maximizing theta by minimizing -theta

#-------------------------------------------------------------------------
# def constraint(variables, num_X, num_Y):
#     theta = variables[0]
#     X = variables[1:1 + 2*num_X].reshape((num_X, 2))
#     Y = variables[1 + 2*num_X:].reshape((num_Y, 2))
#     contour = create_contour(X, Y, theta)
#     return check_distances(contour, MIN_DISTANCE, include_closing_segment=True)

#-------------------------------------------------------------------------
def draw_contour(contour):
    screen.fill(BACKGROUND_COLOR)

    # Compute bounding box
    min_x = np.min(contour[:, 0])
    max_x = np.max(contour[:, 0])
    min_y = np.min(contour[:, 1])
    max_y = np.max(contour[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    # Avoid division by zero if width or height is zero
    if width == 0:
        width = 1
    if height == 0:
        height = 1

    # Compute scaling factor and offsets to fit in the screen with margins
    scale = min((SCREEN_SIZE[0] - 2 * MARGIN) / width,
                (SCREEN_SIZE[1] - 2 * MARGIN) / height)

    offset_x = SCREEN_SIZE[0] / 2 - ((max_x + min_x) / 2) * scale
    offset_y = SCREEN_SIZE[1] / 2 + ((max_y + min_y) / 2) * scale  # Flip Y-axis

    def transform(point):
        """Transform a point from mathematical coordinates to screen coordinates."""
        x_screen = int(point[0] * scale + offset_x)
        y_screen = int(-point[1] * scale + offset_y)  # Flip Y-axis
        return (x_screen, y_screen)

    # Draw grid
    grid_color = (200, 200, 200)  # Light gray
    for x in range(int(np.floor(min_x)), int(np.ceil(max_x)) + 1):
        pygame.draw.line(screen, grid_color, transform((x, min_y - 1)), transform((x, max_y + 1)), 1)
    for y in range(int(np.floor(min_y)), int(np.ceil(max_y)) + 1):
        pygame.draw.line(screen, grid_color, transform((min_x - 1, y)), transform((max_x + 1, y)), 1)

    # Draw axes
    axis_color = (255, 255, 255)  # White
    pygame.draw.line(screen, axis_color, transform((min_x - 1, 0)), transform((max_x + 1, 0)), 2)  # X-axis
    pygame.draw.line(screen, axis_color, transform((0, min_y - 1)), transform((0, max_y + 1)), 2)  # Y-axis

    # Transform contour points
    transformed_points = [transform(point) for point in contour]

    # Draw contour
    pygame.draw.polygon(screen, CONTOUR_COLOR, transformed_points, width=1)

    # Draw contour points and circles around them
    for i, point in enumerate(contour):
        color = CONTOUR_COLOR
        if np.allclose(point, [0, 1]):  # North point (0,1)
            color = (255, 0, 0)  # Red
        elif np.allclose(point, [0, -1]):  # South point (0,-1)
            color = (0, 0, 255)  # Blue

        pygame.draw.circle(screen, color, transformed_points[i], POINT_RADIUS)

        # Draw a circle of radius MIN_DISTANCE around each point (converted to screen scale)
        pygame.draw.circle(screen, (255, 255, 0), transformed_points[i], int(MIN_DISTANCE * scale), 1)

    pygame.display.flip()

    # Process pygame events to keep the window responsive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

#-------------------------------------------------------------------------
def draw_debug_point():
    # Assume a fixed real coordinate system of [-2, 2] for both x and y.
    scale = (SCREEN_SIZE[0] - 2 * MARGIN) / 4.0  # 4 = width of the interval [-2,2]
    offset_x = SCREEN_SIZE[0] / 2
    offset_y = SCREEN_SIZE[1] / 2

    def transform(point):
        x_screen = int(point[0] * scale + offset_x)
        y_screen = int(-point[1] * scale + offset_y)  # Invert y-axis
        return (x_screen, y_screen)

    def inverse_transform(screen_point):
        x = (screen_point[0] - offset_x) / scale
        y = -(screen_point[1] - offset_y) / scale
        return np.array([x, y])

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

        # Define the segment from (-1, 0) to (1, 0)
        a = np.array([-1, 0])
        b = np.array([1, 0])
        pygame.draw.line(screen, (255, 255, 255), transform(a), transform(b), 2)

        # Get the mouse position in real coordinates
        mouse_screen = pygame.mouse.get_pos()
        mouse_real = inverse_transform(mouse_screen)

        # Draw the mouse point
        pygame.draw.circle(screen, POINT_COLOR, transform(mouse_real), POINT_RADIUS)

        # Compute the distance from the mouse to the segment
        dist = distance_point_to_segment(mouse_real, a, b)
        interest = dist - MIN_DISTANCE
        f_interest = scaled_sigmoid(interest, 100.0, MIN_DISTANCE)
        
        # Display the mouse coordinates, the distance, (distance - MIN_DISTANCE) and f(distance - MIN_DISTANCE)
        text_surface = font.render("Coords: ({:.4f}, {:.4f})  Dist: {:.4f}  (Dist-MIN): {:.4f}  f(Dist-MIN): {:.4f}".format(
            mouse_real[0], mouse_real[1], dist, interest, f_interest), True, (255,255,255))
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

#-------------------------------------------------------------------------
def draw_debug_segment():
    scale = (SCREEN_SIZE[0] - 2 * MARGIN) / 4.0  # real coordinate system: [-2, 2]
    offset_x = SCREEN_SIZE[0] / 2
    offset_y = SCREEN_SIZE[1] / 2

    def transform(point):
        x_screen = int(point[0] * scale + offset_x)
        y_screen = int(-point[1] * scale + offset_y)
        return (x_screen, y_screen)

    def inverse_transform(screen_point):
        x = (screen_point[0] - offset_x) / scale
        y = -(screen_point[1] - offset_y) / scale
        return np.array([x, y])

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

        # Fixed segment from (-1, 0) to (1, 0)
        seg1_start = np.array([-1, 0])
        seg1_end = np.array([1, 0])
        pygame.draw.line(screen, (255, 255, 255), transform(seg1_start), transform(seg1_end), 2)

        # Dynamic segment: fixed point (0, 2) to mouse pointer
        seg2_start = np.array([0, 2])
        mouse_screen = pygame.mouse.get_pos()
        seg2_end = inverse_transform(mouse_screen)
        pygame.draw.line(screen, (0, 255, 0), transform(seg2_start), transform(seg2_end), 2)

        # Compute crossing distance between the two segments
        cross_dist = crossing_distance((seg1_start, seg1_end), (seg2_start, seg2_end))

        # Draw circle at the dynamic segment's endpoint (mouse position)
        pygame.draw.circle(screen, POINT_COLOR, transform(seg2_end), POINT_RADIUS)

        # Display the real coordinates of the mouse and the crossing distance
        text_surface = font.render("Coords: ({:.4f}, {:.4f})  Crossing: {:.4f}".format(seg2_end[0], seg2_end[1], cross_dist), True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

#-------------------------------------------------------------------------
def main():
    num_X = 1
    num_Y = 1
    BARRIER_AMPLITUDE = 100.0

    initial_theta = np.pi / 7
    initial_X = np.array([[0.42, -0.64]])
    initial_Y = np.array([[0.23, -0.98]])

    initial_vars = np.concatenate(([initial_theta],
                                    initial_X.flatten(),
                                    initial_Y.flatten()))

    # Debug block: Check crossing distance between two non-crossing segments
    #seg1 = (np.array([-1.0, 0.0]), np.array([1.0, 0.0]))
    #seg2 = (np.array([0.0, 1.0]), np.array([0.0, 2.0]))

    #crossing_dist = crossing_distance(seg1, seg2)
    #print("Debug: Crossing distance between non-crossing segments:", crossing_dist)
    #wait_for_keypress()

    # Combined objective: original objective (here: -theta) plus barrier potential
    def combined_objective(vars):
        theta = vars[0]
        X = vars[1:1+2*num_X].reshape((num_X, 2))
        Y = vars[1+2*num_X:].reshape((num_Y, 2))
        base_obj = -theta  # maximizing theta via minimizing -theta
        contour = create_contour(X, Y, theta)
        barrier_val = barrier_potential(contour, MIN_DISTANCE, BARRIER_AMPLITUDE)
        return base_obj + barrier_val

    combined_grad = grad(combined_objective)

    # Callback that logs each iteration's parameters and draws the contour,
    # printing objective, barrier, and combined values.
    def optimization_callback(vars):
        theta = vars[0]
        X = vars[1:1+2*num_X].reshape((num_X, 2))
        Y = vars[1+2*num_X:].reshape((num_Y, 2))
        base_obj = -theta
        contour = create_contour(X, Y, theta)
        # Compute the inequality constraint value
        ineq_value = check_distances(contour, MIN_DISTANCE, include_closing_segment=True)
        print("Constraint inequality value (should be = 0 when feasible): {:.6f}".format(ineq_value))
        # Compute barrier potential
        barrier_val = barrier_potential(contour, MIN_DISTANCE, BARRIER_AMPLITUDE)
        combined_val = base_obj + barrier_val
        print("Iteration {}: theta = {:.6f}, X = {}, Y = {}".format(
            optimization_callback.iteration, theta, X, Y))
        print("Objective: {:.6f}, Barrier: {:.6f}, Combined: {:.6f}".format(
            base_obj, barrier_val, combined_val))
        optimization_callback.iteration += 1
        draw_contour(contour)
        # wait_for_keypress()
        time.sleep(0.1)  # Pause for 0.1 second
    optimization_callback.iteration = 0

    # Draw initial contour and wait for keypress
    optimization_callback(initial_vars)

    result = minimize(
        fun=combined_objective,
        x0=initial_vars,
        jac=combined_grad,
        method='L-BFGS-B',
        callback=optimization_callback,
        options={'disp': True, 'eps': 1e-12, 'maxiter': 200}
    )

    wait_for_keypress()
    pygame.quit()

if __name__ == "__main__":
#    draw_debug_point()
#    pygame.quit()
    main()
