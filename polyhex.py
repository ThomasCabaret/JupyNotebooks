import sys
import math
import time
import pygame
import itertools
from collections import deque

# Interaction helpers
def wait_for_keypress():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                running = False

# Maths helpers
def axial_to_cartesian(coord, edge_length=1):
    i, j = coord
    x = edge_length * (1.5 * i)
    y = edge_length * (math.sqrt(3) * (j + i / 2))
    angle = math.radians(-30)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    return (new_x, new_y)

#---------------------------------------------------------------
def convert_path_to_cartesian(path, edge_length=1):
    return [axial_to_cartesian(pt, edge_length) for pt in path]

# Draw helpers
def draw_multicolor_segment(surface, start, end, colors, width=1):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    # Compute unit perpendicular vector
    ux = -dy / length
    uy = dx / length
    offsets = [0, width, 2*width]
    for color, off in zip(colors, offsets):
        if color is None:
            continue
        start_off = (start[0] + ux * off, start[1] + uy * off)
        end_off = (end[0] + ux * off, end[1] + uy * off)
        pygame.draw.line(surface, color, start_off, end_off, width)

global_min_distance = 10**9

#############################################################################################
class PolyhexCandidate:
    def __init__(self, size, idx_b_pole, a_offset, b_offset, a_flip, b_flip):
        self.size = size
        self.angles = [None] * size
        self.idx_b_pole = idx_b_pole
        self.a_offset = a_offset
        self.b_offset = b_offset
        self.a_flip = a_flip
        self.b_flip = b_flip
        # rest done in initSearchState (cannot be called here as it can fail)

    #---------------------------------------------------------------
    def __str__(self):
        return (f"PolyhexCandidate(size={self.size}, idx_b_pole={self.idx_b_pole}, "
                f"a_offset={self.a_offset}, b_offset={self.b_offset}, "
                f"a_flip={self.a_flip}, b_flip={self.b_flip}, angles={self.angles})")

    # Neighbor directions for hex grid (axial coordinates)
    _neighbors = [
        (1, 0),    # 0: east
        (0, 1),    # 1: southeast
        (-1, 1),   # 2: southwest
        (-1, 0),   # 3: west
        (0, -1),   # 4: northwest
        (1, -1)    # 5: northeast
    ]

    #---------------------------------------------------------------
    def print_graph(self):
        matrix = [[0] * self.size for _ in range(self.size)]
        for i, neighbors in self.graph.items():
            for j in neighbors:
                matrix[i][j] = 1
        for row in matrix:
            print(" ".join(str(cell) for cell in row))

    #---------------------------------------------------------------
    def build_graph(self):
        graph = {i: set() for i in range(self.size)}
        # Mapping A (same as in draw_schema)
        start_a, end_a = 0, self.idx_b_pole  # Start and end of section A
        start_a_proj = (start_a + self.a_offset) % self.size if not self.a_flip else (end_a + self.a_offset) % self.size
        end_a_proj = (end_a + self.a_offset) % self.size if not self.a_flip else (start_a + self.a_offset) % self.size
        for i in range(1, self.idx_b_pole): # We do not connect bondaries to their projection
            if self.a_flip:
                j = (start_a_proj - i) % self.size
            else:
                j = (start_a_proj + i) % self.size
            graph[start_a + i].add(j)
            graph[j].add(start_a + i)
        # Mapping B (same logic)
        start_b, end_b = self.idx_b_pole, 0  # Start and end of section B
        start_b_proj = (start_b + self.b_offset) % self.size if not self.b_flip else (end_b + self.b_offset) % self.size
        end_b_proj = (end_b + self.b_offset) % self.size if not self.b_flip else (start_b + self.b_offset) % self.size
        for i in range(1, self.size - self.idx_b_pole):
            if self.b_flip:
                j = (start_b_proj - i) % self.size
            else:
                j = (start_b_proj + i) % self.size
            graph[start_b + i].add(j)
            graph[j].add(start_b + i)
        return graph

    #---------------------------------------------------------------
    def get_components(self):
        visited = [False] * self.size
        components = []
        for i in range(self.size):
            if not visited[i]:
                comp = []
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for neighbor in self.graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)
                components.append(comp)
        return components

    #---------------------------------------------------------------
    def propagate_component(self, comp, changes):
        queue = deque([i for i in comp if self.angles[i] is not None])
        while queue:
            v = queue.popleft()
            for u in self.graph[v]:
                if u not in comp:
                    continue
                expected = -self.angles[v]
                if self.angles[u] is None:
                    self.angles[u] = expected
                    changes.append(u)
                    queue.append(u)
                elif self.angles[u] != expected:
                    return False
        return True
    
    #---------------------------------------------------------------
    def propagate_fixed(self):
        for comp in self.components:
            changes = []
            if not self.propagate_component(comp, changes):
                #for i in changes:
                #    self.angles[i] = None
                return False
        return True

    #---------------------------------------------------------------
    def can_loop(self, target=6):
        current_sum = sum(a for a in self.angles if a is not None)
        remaining_nones = self.angles.count(None)
        # The maximum possible sum we can get by setting all None to 1
        max_possible_sum = current_sum + remaining_nones
        # The minimum possible sum we can get by setting all None to -1
        min_possible_sum = current_sum - remaining_nones
        return min_possible_sum <= target <= max_possible_sum

    #---------------------------------------------------------------
    def is_valid_loop(self):
        global global_min_distance
        start = (0, 0)
        direction = 0
        current = start
        for turn in self.angles:
            if turn not in (1, -1):
                raise ValueError("Turn values must be 1 or -1.")
            direction = (direction + turn) % 6
            move = PolyhexCandidate._neighbors[direction]
            current = (current[0] + move[0], current[1] + move[1])
        distance = abs(current[0] - start[0]) + abs(current[1] - start[1])
        if distance < global_min_distance:
            global_min_distance = distance  # Update global if new distance is smaller
        return (current == (0, 0)) and (direction == 0)

    #---------------------------------------------------------------
    def is_self_intersecting(self):
        start = (0, 0)
        visited = {start}
        direction = 0
        current = start
        for i, turn in enumerate(self.angles):
            if turn is None:
                return False
            direction = (direction + turn) % 6
            move = PolyhexCandidate._neighbors[direction]
            current = (current[0] + move[0], current[1] + move[1])
            if i < len(self.angles) - 1:
                if current in visited:
                    return True
            else:
                if current != start and current in visited:
                    return True
            visited.add(current)
        return False

    # Helper methods added to PolyhexCandidate
    #---------------------------------------------------------------
    def compute_mapping_range(self, count, offset, flip, base_nonflip, base_flip):
        mapping = []
        for i in range(count):
            if not flip:
                mapping.append((base_nonflip + offset + i) % self.size)
            else:
                mapping.append((base_flip + offset - i) % self.size)
        return mapping

    #---------------------------------------------------------------
    def compute_mappings(self):
        mappedA = self.compute_mapping_range(self.idx_b_pole + 1, self.a_offset, self.a_flip, 0, self.idx_b_pole)
        mappedB = self.compute_mapping_range(self.size - self.idx_b_pole + 1, self.b_offset, self.b_flip, self.idx_b_pole, 0)
        return mappedA, mappedB

    #---------------------------------------------------------------
    def draw_edges_helper(self, screen, points, mappedA, mappedB):
        for i in range(len(points) - 1):
            next_i = i + 1
            base_color = (255, 0, 0) if (1 <= next_i <= self.idx_b_pole) else (0, 0, 255)
            mappingA_color = (255, 0, 0) if (i in mappedA and next_i in mappedA) else None
            mappingB_color = (0, 0, 255) if (i in mappedB and next_i in mappedB) else None
            draw_multicolor_segment(screen, points[i], points[next_i],
                                    colors=[base_color, mappingA_color, mappingB_color],
                                    width=6)

    #---------------------------------------------------------------
    def draw(self, edge_length=40, window_size=(800, 600)):
        pygame.display.set_caption(str(self))
        screen = pygame.display.set_mode(window_size)
        screen.fill((255, 255, 255))
        candidate_start = (0, 0)
        candidate_path = [candidate_start]
        direction = 0
        current = candidate_start
        for turn in self.angles[:-1]:
            direction = (direction + turn) % 6
            move = PolyhexCandidate._neighbors[direction]
            current = (current[0] + move[0], current[1] + move[1])
            candidate_path.append(current)
        #fixed_point = (-1, 0)
        #full_path = [fixed_point] + candidate_path + [fixed_point]
        cartesian_path = [axial_to_cartesian(pt, edge_length) for pt in candidate_path]
        xs = [pt[0] for pt in cartesian_path]
        ys = [pt[1] for pt in cartesian_path]
        offset_x = (window_size[0] - (max(xs) - min(xs))) / 2 - min(xs)
        offset_y = (window_size[1] - (max(ys) - min(ys))) / 2 - min(ys)
        adjusted_points = [(pt[0] + offset_x, pt[1] + offset_y) for pt in cartesian_path]
        # Draw edges:
        mappedA, mappedB = self.compute_mappings()
        self.draw_edges_helper(screen, adjusted_points, mappedA, mappedB)
        # Draw labels:
        font = pygame.font.SysFont("Consolas", 14, bold=True)
        for idx, pos in enumerate(adjusted_points):
            pygame.draw.circle(screen, (0, 0, 0), (int(pos[0]), int(pos[1])), 4)
            if idx < len(self.angles) and self.angles[idx] is not None:
                label = "{} ({})".format(idx, self.angles[idx])
                text_surface = font.render(label, True, (0, 0, 0))
                # Position the text so that its midleft is at (pos[0] + shift, pos[1])
                text_rect = text_surface.get_rect(midleft=(int(pos[0]) + 5, int(pos[1])))
                screen.blit(text_surface, text_rect)
        pygame.display.flip()

    #---------------------------------------------------------------
    def draw_schema(self, window_size=(800, 800), radius=300):
        self.print_graph()
        pygame.display.set_caption(str(self))
        screen = pygame.display.set_mode(window_size)
        screen.fill((255, 255, 255))
        center = (window_size[0] // 2, window_size[1] // 2)
        positions = []
        for i in range(self.size):
            angle_rad = math.radians(-90 + (360.0 / self.size) * i)
            x = center[0] + radius * math.cos(angle_rad)
            y = center[1] + radius * math.sin(angle_rad)
            positions.append((x, y))
        # Draw edges:
        mappedA, mappedB = self.compute_mappings()
        for i in range(self.size):
            next_i = (i + 1) % self.size
            base_color = (255, 0, 0) if (1 <= next_i <= self.idx_b_pole) else (0, 0, 255)
            mappingA_color = (255, 0, 0) if (i in mappedA and next_i in mappedA) else None
            mappingB_color = (0, 0, 255) if (i in mappedB and next_i in mappedB) else None
            draw_multicolor_segment(screen, positions[i], positions[next_i],
                                    colors=[base_color, mappingA_color, mappingB_color],
                                    width=6)
        # Draw mapping connection lines:
        # Mapping A: draw a red line from each vertex (0 to idx_b_pole) to its mapped vertex.
        for i in range(self.idx_b_pole):
            if self.a_flip:
                dest = (self.idx_b_pole + self.a_offset - i) % self.size
            else:
                dest = (self.a_offset + i) % self.size
            # Check that the connection exists in the graph before drawing
            if dest in self.graph[i]:
                pygame.draw.line(screen, (255, 0, 0), positions[i], positions[dest], 1)
        # Mapping B: draw a blue line from each vertex (starting at idx_b_pole) to its mapped vertex.
        for i in range(self.size - self.idx_b_pole):
            origin = (self.idx_b_pole + i) % self.size
            if self.b_flip:
                dest = (self.b_offset - i) % self.size
            else:
                dest = (self.idx_b_pole + self.b_offset + i) % self.size
            # Check that the connection exists in the graph before drawing
            if dest in self.graph[origin]:
                pygame.draw.line(screen, (0, 0, 255), positions[origin], positions[dest], 1)
        # Draw labels:
        font = pygame.font.SysFont("Consolas", 14, bold=True)
        for i, pos in enumerate(positions):
            pygame.draw.circle(screen, (0, 0, 0), (int(pos[0]), int(pos[1])), 5)
            if self.angles[i] is not None:
                label = str(self.angles[i])
                text_surface = font.render(label, True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(int(pos[0]), int(pos[1]) - 10))
                screen.blit(text_surface, text_rect)
        pygame.display.flip()

    #---------------------------------------------------------------
    def initSearchState(self):
        # Build graph and deduce its connected subgraphs
        self.graph = self.build_graph()
        self.components = self.get_components()
        # Ensure all angles start as None
        self.angles = [None] * self.size
        # Set initial fixed angles
        self.angles[0] = 1
        self.angles[self.idx_b_pole % self.size] = 1
        self.angles[self.a_offset % self.size] = 1
        self.angles[(self.idx_b_pole + self.a_offset) % self.size] = 1
        self.angles[self.b_offset % self.size] = 1
        self.angles[(self.idx_b_pole + self.b_offset) % self.size] = 1
        if not self.propagate_fixed():
            return False
        for comp in self.components:
            if not any(self.angles[i] is not None for i in comp):
                continue
            colors = {}
            for v in comp:
                if v not in colors:
                    colors[v] = 1
                    stack = [v]
                    while stack:
                        u = stack.pop()
                        for w in self.graph[u]:
                            if w in comp:
                                if w not in colors:
                                    colors[w] = -colors[u]
                                    stack.append(w)
                                elif colors[w] == colors[u]:
                                    return False
        return True

    #---------------------------------------------------------------
    def backtrack_components(self, comp_index=0):
        if not self.can_loop() or self.is_self_intersecting():
            return
        if comp_index == len(self.components):
            if self.is_valid_loop():
                print("Valid candidate:", self.angles)
                self.draw()
                wait_for_keypress();
            return
        comp = self.components[comp_index]
        if any(self.angles[i] is None for i in comp):
            for i in comp:
                if self.angles[i] is None:
                    node = i
                    break
            for value in [1, -1]:
                self.angles[node] = value
                local_changes = [node]
                if self.propagate_component(comp, local_changes):
                    self.backtrack_components(comp_index + 1)
                for j in local_changes:
                    self.angles[j] = None
        else:
            self.backtrack_components(comp_index + 1)

#############################################################################################

if __name__ == '__main__':
    pygame.init()
#     print("Pygame initialized")
#     candidate = PolyhexCandidate(20, 8, 13, 13, False, False)
#     initStatus = candidate.initSearchState()
#     print("Initialization Status:", initStatus)
#     candidate.draw_schema()
#     wait_for_keypress()
    MAX_SIZE = 200
    start_time = time.time()
    print("Starting loop")
    for size in range(100, MAX_SIZE + 1, 2):
        elapsed_time = time.time() - start_time
        header = f"size={size} starting. {elapsed_time:.2f}s from init. "
        print(header, end="", flush=True)
        global_min_distance = 10**9
        total = size // 2
        for idx_b_pole in range(1, total):
            percentage = (idx_b_pole / total) * 100
            sys.stdout.write(f"\r{header}{percentage:.1f}% {global_min_distance}             ")
            sys.stdout.flush()
            for a_offset in range(1, size):
                for b_offset in range(1, size):
                    for a_flip in [False, True]:
                        for b_flip in [False, True]:
                            # print(f"size={size}, idx_b_pole={idx_b_pole}, a_offset={a_offset}, b_offset={b_offset}, a_flip={a_flip}, b_flip={b_flip}")
                            candidate = PolyhexCandidate(size, idx_b_pole, a_offset, b_offset, a_flip, b_flip)
                            if not candidate.initSearchState():
                                continue
                            candidate.backtrack_components()
                            # candidate.draw_schema()
                            # wait_for_keypress()
        print("")
