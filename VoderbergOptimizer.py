import sys
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
    proj = np.dot(ap, ab) / np.dot(ab, ab) if np.dot(ab, ab) > 0 else 0
    if 0 <= proj <= 1:
        projection = a + proj * ab
        return np.linalg.norm(p - projection)
    return float('inf')

#-------------------------------------------------------------------------
def segment_crossing_violation(a, b, c, d, tol=0.01):
    r = b - a
    s = d - c
    r_cross_s = r[0] * s[1] - r[1] * s[0]
    if np.isclose(r_cross_s, 0.0):  # Segments are nearly parallel (no crossing)
        return 1e6
    qmp = c - a
    t = (qmp[0] * s[1] - qmp[1] * s[0]) / r_cross_s
    u = (qmp[0] * r[1] - qmp[1] * r[0]) / r_cross_s
    # Measure penetration for any t, u values (even outside [0,1])
    penetration = 0.5 - min(abs(t - 0.5), abs(u - 0.5))
    return -penetration  # NEGATIVE when crossing occurs (violated), positive otherwise

#-------------------------------------------------------------------------
def check_distances(points, min_distance, include_closing_segment=False):
    points = np.array(points)
    n = len(points)
    min_violation = np.inf

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points[i] - points[j]) - min_distance
            if dist < min_violation:
                min_violation = dist

    for i in range(n - 1):
        for j in range(n):
            if j != i and j != i + 1:
                dist = distance_point_to_segment(points[j], points[i], points[i + 1]) - min_distance
                if dist < min_violation:
                    min_violation = dist

    if include_closing_segment:
        for j in range(n):
            if j != n - 1 and j != 0:
                dist = distance_point_to_segment(points[j], points[-1], points[0]) - min_distance
                if dist < min_violation:
                    min_violation = dist

    segments = []
    for i in range(n - 1):
        segments.append((points[i], points[i + 1]))
    if include_closing_segment:
        segments.append((points[-1], points[0]))

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if (np.allclose(segments[i][0], segments[j][0]) or 
                np.allclose(segments[i][0], segments[j][1]) or 
                np.allclose(segments[i][1], segments[j][0]) or 
                np.allclose(segments[i][1], segments[j][1])):
                continue
            violation = segment_crossing_violation(segments[i][0], segments[i][1],
                                                     segments[j][0], segments[j][1])
            if violation < min_violation:
                min_violation = violation

    # Add boundary condition: points must have |coordinate| <= 2.
    # If the maximum absolute coordinate is above 2, return a negative value 
    # equal to (2 - max(|coordinate|)).
    boundary_violation = 2 - np.max(np.abs(points))
    if boundary_violation < min_violation:
        min_violation = boundary_violation

    return min_violation

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
def perimeter(contour):
    distances = np.linalg.norm(np.diff(np.vstack([contour, contour[0]]), axis=0), axis=1)
    return np.sum(distances)

#-------------------------------------------------------------------------
def objective(variables, num_X, num_Y):
    theta = variables[0]
    X = variables[1:1 + 2*num_X].reshape((num_X, 2))
    Y = variables[1 + 2*num_X:].reshape((num_Y, 2))

    contour = create_contour(X, Y, theta)
    return -perimeter(contour)

#-------------------------------------------------------------------------
def objective_theta(variables):
    theta = variables[0]
    return -theta  # maximizing theta by minimizing -theta

#-------------------------------------------------------------------------
def constraint(variables, num_X, num_Y):
    theta = variables[0]
    X = variables[1:1 + 2*num_X].reshape((num_X, 2))
    Y = variables[1 + 2*num_X:].reshape((num_Y, 2))

    contour = create_contour(X, Y, theta)
    return check_distances(contour, MIN_DISTANCE, include_closing_segment=True)

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

    # Draw contour points
    for i, point in enumerate(contour):
        color = CONTOUR_COLOR
        if np.allclose(point, [0, 1]):  # North point (0,1)
            color = (255, 0, 0)  # Red
        elif np.allclose(point, [0, -1]):  # South point (0,-1)
            color = (0, 0, 255)  # Blue

        pygame.draw.circle(screen, color, transformed_points[i], POINT_RADIUS)

    pygame.display.flip()

    # Process pygame events to keep the window responsive
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

#-------------------------------------------------------------------------
def main():
    num_X = 1
    num_Y = 1

    initial_theta = np.pi / 7
    initial_X = np.array([[0.42, -0.64]])
    initial_Y = np.array([[0.23, -0.98]])

    initial_vars = np.concatenate(([initial_theta], initial_X.flatten(), initial_Y.flatten()))

    draw_contour(create_contour(
        initial_vars[1:1 + 2*num_X].reshape((num_X, 2)),
        initial_vars[1 + 2*num_X:].reshape((num_Y, 2)),
        initial_vars[0]
    ))
    wait_for_keypress()

    constraints = [{'type': 'ineq', 'fun': lambda vars: constraint(vars, num_X, num_Y)}]

    # Autograd computes the gradient of the objective function
    objective_grad = grad(objective_theta)

    # Callback that logs each iteration's parameters and draws the contour
    def optimization_callback(vars):
        contour = create_contour(
            vars[1:1 + 2*num_X].reshape((num_X, 2)),
            vars[1 + 2*num_X:].reshape((num_Y, 2)),
            vars[0]
        )
        violation = check_distances(contour, MIN_DISTANCE, include_closing_segment=True)
        print("Iteration {}: theta = {:.6f}, X = {}, Y = {}, Violation = {:.6f}".format(
            optimization_callback.iteration, vars[0],
            vars[1:1+2*num_X], vars[1+2*num_X:], violation
        ))
        optimization_callback.iteration += 1
        draw_contour(contour)
        wait_for_keypress()
    optimization_callback.iteration = 0

    result = minimize(
        fun=lambda vars: objective_theta(vars),
        x0=initial_vars,
        method='SLSQP',
        jac=lambda vars: objective_grad(vars),  # Provide explicit gradient
        constraints=constraints,
        callback=optimization_callback,
        options={'disp': True, 'eps': 1e-12, 'maxiter': 200}
    )

    wait_for_keypress()
    pygame.quit()

if __name__ == "__main__":
    main()
