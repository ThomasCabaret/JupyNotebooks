import math
import pygame
import itertools
from collections import deque

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

    # Neighbor directions for hex grid (axial coordinates)
    _neighbors = [
        (1, 0),    # 0: east
        (0, 1),    # 1: southeast
        (-1, 1),   # 2: southwest
        (-1, 0),   # 3: west
        (0, -1),   # 4: northwest
        (1, -1)    # 5: northeast
    ]

    @staticmethod
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

    def convert_path_to_cartesian(self, path, edge_length=1):
        return [PolyhexCandidate.axial_to_cartesian(pt, edge_length) for pt in path]

    def build_graph(self):
        graph = {i: set() for i in range(self.size)}
        # Mapping A (same as in draw_schema)
        start_a, end_a = 0, self.idx_b_pole  # Start and end of section A
        start_a_proj = (start_a + self.a_offset) % self.size if not self.a_flip else (end_a + self.a_offset) % self.size
        end_a_proj = (end_a + self.a_offset) % self.size if not self.a_flip else (start_a + self.a_offset) % self.size
        for i in range(1, self.idx_b_pole): # We do not connect bondaries to their projection
            if self.a_flip:
                j = (end_a_proj - i) % self.size
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
                j = (end_b_proj - i) % self.size
            else:
                j = (start_b_proj + i) % self.size
            graph[start_b + i].add(j)
            graph[j].add(start_b + i)
        return graph

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
    
    def propagate_fixed(self):
        for comp in self.components:
            changes = []
            if not self.propagate_component(comp, changes):
                #for i in changes:
                #    self.angles[i] = None
                return False
        return True

    def is_valid_loop(self):
        start = (0, 0)
        direction = 0
        current = start
        for turn in self.angles:
            if turn not in (1, -1):
                raise ValueError("Turn values must be 1 or -1.")
            direction = (direction + turn) % 6
            move = PolyhexCandidate._neighbors[direction]
            current = (current[0] + move[0], current[1] + move[1])
        return (current == (0, 0)) and (direction == 0)

    def is_self_intersecting(self):
        start = (0, 0)
        candidate_path = [start]
        direction = 0
        current = start
        for i, turn in enumerate(self.angles):
            direction = (direction + turn) % 6
            move = PolyhexCandidate._neighbors[direction]
            current = (current[0] + move[0], current[1] + move[1])
            if i < len(self.angles) - 1:
                if current in candidate_path:
                    return True
            else:
                if current != start and current in candidate_path:
                    return True
            candidate_path.append(current)
        return False

    def draw(self, edge_length=40, window_size=(800, 600)):
        fixed_point = (-1, 0)
        candidate_start = (0, 0)
        candidate_path = [candidate_start]
        direction = 0
        current = candidate_start
        for turn in self.angles[:-1]:
            direction = (direction + turn) % 6
            move = PolyhexCandidate._neighbors[direction]
            current = (current[0] + move[0], current[1] + move[1])
            candidate_path.append(current)
        full_path = [fixed_point] + candidate_path + [fixed_point]
        valid = self.is_valid_loop()
        cartesian_path = [PolyhexCandidate.axial_to_cartesian(pt, edge_length) for pt in full_path]
        xs = [pt[0] for pt in cartesian_path]
        ys = [pt[1] for pt in cartesian_path]
        offset_x = (window_size[0] - (max(xs) - min(xs))) / 2 - min(xs)
        offset_y = (window_size[1] - (max(ys) - min(ys))) / 2 - min(ys)

        def draw_dotted_line(surf, color, start_pos, end_pos, dot_spacing=5, dot_radius=2):
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            dist = math.hypot(dx, dy)
            if dist == 0:
                return
            steps = int(dist / dot_spacing)
            for i in range(steps + 1):
                t = i / steps
                x = start_pos[0] + t * dx
                y = start_pos[1] + t * dy
                pygame.draw.circle(surf, color, (int(x), int(y)), dot_radius)

        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Polyhex Candidate")
        font = pygame.font.Font(None, 20)
        screen.fill((255, 255, 255))
        adjusted_points = [(pt[0] + offset_x, pt[1] + offset_y) for pt in cartesian_path]

        # Draw edges: red for edges from index 0 to idx_b_pole, blue for the rest
        for i in range(len(adjusted_points) - 1):
            if 1 <= i <= self.idx_b_pole:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            pygame.draw.line(screen, color, adjusted_points[i], adjusted_points[i+1], 2)

        # Draw vertices with labels if assigned
        for idx in range(len(candidate_path)):
            pos = adjusted_points[idx + 1]
            pygame.draw.circle(screen, (0, 0, 0), (int(pos[0]), int(pos[1])), 4)
            label = "{} ({})".format(idx, self.angles[idx])
            text_surface = font.render(label, True, (0, 0, 0))
            screen.blit(text_surface, (pos[0] + 5, pos[1] - 5))

        if not valid:
            draw_dotted_line(screen, (128, 128, 128), adjusted_points[-1], adjusted_points[0])
        pygame.display.flip()

    def draw_schema(self, window_size=(800, 800), radius=300):
        screen = pygame.display.set_mode(window_size)
        screen.fill((255, 255, 255))
        center = (window_size[0] // 2, window_size[1] // 2)
        positions = []
        for i in range(self.size):
            angle_rad = math.radians(-90 + (360.0 / self.size) * i)
            x = center[0] + radius * math.cos(angle_rad)
            y = center[1] + radius * math.sin(angle_rad)
            positions.append((x, y))
        # Compute mapped vertices for mapping A and mapping B (preserving order)
        mappedA = []
        for i in range(self.idx_b_pole + 1):
            if self.a_flip:
                mappedA.append((self.idx_b_pole + self.a_offset - i) % self.size)
            else:
                mappedA.append((i + self.a_offset) % self.size)
        mappedB = []
        for i in range(self.size - self.idx_b_pole + 1):
            if self.b_flip:
                mappedB.append((self.b_offset - i) % self.size)
            else:
                mappedB.append((self.idx_b_pole + self.b_offset + i) % self.size)
        # Draw polygon edges as tricolor segments:
        #   - Base: red if next vertex in [1, idx_b_pole], blue otherwise.
        #   - Mapping A track: red if both endpoints appear in mappedA.
        #   - Mapping B track: blue if both endpoints appear in mappedB.
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
        for i in range(self.idx_b_pole + 1):
            if self.a_flip:
                dest = (self.idx_b_pole + self.a_offset - i) % self.size
            else:
                dest = (i + self.a_offset) % self.size
            # Check that the connection exists in the graph before drawing
            if dest in self.graph[i]:
                pygame.draw.line(screen, (255, 0, 0), positions[i], positions[dest], 1)
        # Mapping B: draw a blue line from each vertex (starting at idx_b_pole) to its mapped vertex.
        for i in range(self.size - self.idx_b_pole + 1):
            origin = (self.idx_b_pole + i) % self.size
            if self.b_flip:
                dest = (self.b_offset - i) % self.size
            else:
                dest = (self.idx_b_pole + self.b_offset + i) % self.size
            # Check that the connection exists in the graph before drawing
            if dest in self.graph[origin]:
                pygame.draw.line(screen, (0, 0, 255), positions[origin], positions[dest], 1)
        # Draw vertices and labels if assigned.
        font = pygame.font.Font(None, 50)
        for i, pos in enumerate(positions):
            pygame.draw.circle(screen, (0, 0, 0), (int(pos[0]), int(pos[1])), 5)
            if self.angles[i] is not None:
                label = str(self.angles[i])
                text_surface = font.render(label, True, (0, 0, 0), (255, 255, 255))
                text_rect = text_surface.get_rect(center=(int(pos[0]), int(pos[1]) - 10))
                screen.blit(text_surface, text_rect)
        pygame.display.flip()

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

    def backtrack_components(self, comp_index=0):
        if comp_index == len(self.components):
            print("Valid candidate:", self.angles)
            self.draw()
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
                if propagate_component(self, comp, local_changes):
                    self.backtrack_components(comp_index + 1)
                for j in local_changes:
                    self.angles[j] = None
        else:
            self.backtrack_components(comp_index + 1)

def wait_for_keypress():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                running = False

if __name__ == '__main__':
    pygame.init()
#     print("Pygame initialized")
#     candidate = PolyhexCandidate(20, 8, 13, 13, False, False)
#     initStatus = candidate.initSearchState()
#     print("Initialization Status:", initStatus)
#     candidate.draw_schema()
#     wait_for_keypress()
    MAX_SIZE = 100
    print("Starting loop")
    for size in range(6, MAX_SIZE + 1, 2):
        for idx_b_pole in range(1, size):
            for a_offset in range(1, size):
                for b_offset in range(1, size):
                    for a_flip in [False, True]:
                        for b_flip in [False, True]:
                            print(f"size={size}, idx_b_pole={idx_b_pole}, a_offset={a_offset}, b_offset={b_offset}, a_flip={a_flip}, b_flip={b_flip}")
                            candidate = PolyhexCandidate(size, idx_b_pole, a_offset, b_offset, a_flip, b_flip)
                            if not candidate.initSearchState():
                                continue
                            candidate.draw_schema()
                            wait_for_keypress()
