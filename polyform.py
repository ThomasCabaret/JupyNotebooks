import sys
import math
import time
import pygame
import itertools
from enum import Enum
from collections import deque

# Interaction helpers
#---------------------------------------------------------------
def wait_for_keypress():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                running = False

#---------------------------------------------------------------
def ASSERT(condition, message="Assertion failed"):
    if not condition:
        raise ValueError(message)

#---------------------------------------------------------------
global_min_distance = 10**9

# Maths helpers
#---------------------------------------------------------------
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
# Either distance vertex to vertex in a triangular grid, or cell to cell in an hex grid
def axial_distance(axial1, axial2):
    q1, r1 = axial1
    q2, r2 = axial2
    return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) // 2

#---------------------------------------------------------------
# Neighbor directions for hex grid (axial coordinates)
axial_neighbors = [
    (1, 0),    # 0: east
    (0, 1),    # 1: southeast
    (-1, 1),   # 2: southwest
    (-1, 0),   # 3: west
    (0, -1),   # 4: northwest
    (1, -1)    # 5: northeast
]

#---------------------------------------------------------------
def axial_move(current, direction):
    move = axial_neighbors[direction]
    return (current[0] + move[0], current[1] + move[1])

#---------------------------------------------------------------
def convert_path_to_cartesian(path, edge_length=1):
    return [axial_to_cartesian(pt, edge_length) for pt in path]

# Draw helpers
#---------------------------------------------------------------
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

#############################################################################################

class PolyformType(Enum):
    POLYHEX = "POLYHEX"
    POLYTRI = "POLYTRI"
    POLYSQR = "POLYSQR"

class PolyformCandidate:
    def __init__(self, polyform_type: PolyformType, size, idx_b_pole, a_offset, b_offset, a_flip, b_flip):
        self.size = size
        self.angles = [None] * size
        self.components_angles = [0] * size
        self.idx_b_pole = idx_b_pole
        self.a_offset = a_offset
        self.b_offset = b_offset
        self.a_flip = a_flip
        self.b_flip = b_flip
        self.polyform_type = polyform_type
        # rest done in initSearchState (cannot be called here as it can fail)

    #---------------------------------------------------------------
    def __str__(self):
        return (f"PolyformCandidate(size={self.size}, idx_b_pole={self.idx_b_pole}, "
                f"a_offset={self.a_offset}, b_offset={self.b_offset}, "
                f"a_flip={self.a_flip}, b_flip={self.b_flip}, angles={self.angles})")

    #---------------------------------------------------------------
    def print_graph(self):
        matrix = [[0] * self.size for _ in range(self.size)]
        for i, neighbors in self.graph.items():
            for j in neighbors:
                matrix[i][j] = 1
        for row in matrix:
            print(" ".join(str(cell) for cell in row))

    #---------------------------------------------------------------
    def valid_turns(self):
        if self.polyform_type == PolyformType.POLYHEX:
            return {1, -1}
        elif self.polyform_type == PolyformType.POLYTRI:
            return {2, 1, 0, -1, -2}
        raise ValueError(f"Unknown polyform type: {self.polyform_type}")

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
            if not start_a + i == j: # no reverse if on itself
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
            if not start_b + i == j: # no reverse if on itself
                graph[j].add(start_b + i)
        return graph

    #---------------------------------------------------------------
    def build_components(self):
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
    def sort_components(self):
        if not self.components:
            return
        # Work on a copy of the components
        unsorted = self.components[:]
        sorted_components = []
        # Choose the first component: largest size, tie-breaker by smallest minimum element.
        first = max(unsorted, key=lambda comp: (len(comp), -min(comp)))
        sorted_components.append(first)
        unsorted.remove(first)
        # Create a set of all elements in the sorted components.
        sorted_set = set(first)
        # Define a function to count the number of "touches" for a component.
        def touch_count(comp):
            count = 0
            for x in comp:
                if (((x - 1) % self.size) in sorted_set) or (((x + 1) % self.size) in sorted_set):
                    count += 1
            return count
        # Process remaining components.
        while unsorted:
            # Select the component with the highest touch count.
            # In case of a tie, choose the one with the smallest minimum element.
            next_comp = max(unsorted, key=lambda comp: (touch_count(comp), -min(comp)))
            sorted_components.append(next_comp)
            unsorted.remove(next_comp)
            # Update the set of elements with the new component.
            sorted_set.update(next_comp)
        # Update self.components with the sorted list.
        self.components = sorted_components

    #---------------------------------------------------------------
    def compute_components_angles(self):
        components_angles = []
        for comp in self.components:
            # Record indices that were None and changed by this method
            changes = []
            # If the component is completely unset, assign the first element the value 2.
            if all(self.angles[i] is None for i in comp):
                self.angles[comp[0]] = 2
                changes.append(comp[0])
            # Propagate the alternating values in the component.
            # This will fill in all None entries according to the rule: neighbor gets -value.
            self.propagate_component(comp, changes)
            # Compute the sum for the component (ignoring indices that are still None).
            comp_sum = sum(self.angles[i] for i in comp if self.angles[i] is not None)
            components_angles.append(comp_sum)
            # Rollback only the changes we made (restore None for indices that were originally unset)
            for idx in changes:
                self.angles[idx] = None
        self.components_angles = components_angles

    #---------------------------------------------------------------
    def propagate_component(self, comp, changes):
        queue = deque([i for i in comp if self.angles[i] is not None])
        while queue:
            v = queue.popleft()
            for u in self.graph[v]:
                if u not in comp: # useless if no bug
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
    # compute the path forced by a sequence of angles and returns
    # the crow fly distance from start to end
    # convention: if called with all angles we discard the last as it's enough to go back to origin
    def component_crow_fly_distance(self, start_index):
        n = self.size
        virtual_start = (-1, 0)
        direction = 0
        current = axial_move(virtual_start, direction) # should be (0, 0)
        i = start_index
        steps = 0  # Safety counter to prevent infinite loops
        while self.angles[i] is not None and steps < n-1:  # n-1 because full loop distance should be counted as loop minus one angle distance
            turn = self.angles[i]
            direction = (direction + turn) % 6
            current = axial_move(current, direction)
            i = (i + 1) % n  # Move to the next index in a circular manner
            steps += 1  # Increment safety counter
        # Compute crow-fly (axial) distance from virtual start (-1,0) to final position
        return axial_distance(virtual_start, current)

    #---------------------------------------------------------------
    # Smart can_loop considering all components
    def can_loop(self, target=6):
        global global_min_distance
        # Turns check
        current_sum = sum(a for a in self.angles if a is not None)
        remaining_nones = self.angles.count(None)
        # The max min possible sum we can get by setting all None to 1 or -1 # TODO optim: take into account component constraints
        #max_possible_sum = current_sum + 2*remaining_nones  # The max angle is 2...
        #min_possible_sum = current_sum - 2*remaining_nones
        remaining_max_abs_angle = 0
        for comp, comp_angle in zip(self.components, self.components_angles):
            if self.angles[comp[0]] is None:
                remaining_max_abs_angle += comp_angle
        max_possible_sum = current_sum + remaining_max_abs_angle
        min_possible_sum = current_sum - remaining_max_abs_angle
        if not (min_possible_sum <= target <= max_possible_sum):
            return False
        # Moves check
        n = self.size
        component_distances = []
        # If no None values, there's only one component (the entire sequence)
        if remaining_nones == 0:
            cfd = self.component_crow_fly_distance(0) #do not consider the last turn in the path
            return cfd == 0
        # Other cases
        for i in range(n):
            prev_index = (i - 1) % n  # Circular indexing for previous index
            if self.angles[i] is not None and self.angles[prev_index] is None:
                component_distances.append(self.component_crow_fly_distance(i))
        ASSERT(component_distances, "No components")
        largest_distance = max(component_distances)
        sum_other_distances = sum(component_distances) - largest_distance
        free_moves = remaining_nones - len(component_distances)
        ASSERT(free_moves >= 0, "Negative free_moves")
        if largest_distance > sum_other_distances + free_moves:
            return False
        if remaining_nones < global_min_distance:
            #self.draw()
            #wait_for_keypress()
            global_min_distance = remaining_nones  # Update global if new distance is smaller
        return True

    #---------------------------------------------------------------
    def is_valid_loop(self):
        global global_min_distance
        start = (0, 0)
        direction = 0
        current = start
        for turn in self.angles:
            direction = (direction + turn) % 6
            move = axial_neighbors[direction]
            current = (current[0] + move[0], current[1] + move[1])
        #distance = abs(current[0] - start[0]) + abs(current[1] - start[1])
        #if distance < global_min_distance:
        #    global_min_distance = distance  # Update global if new distance is smaller
        return (current == (0, 0)) and (direction == 0)

    #---------------------------------------------------------------
    def is_component_self_intersecting(self, start_index):
        n = self.size
        visited = {(0, 0)}
        direction = 0
        current = (0, 0)
        i = start_index
        steps = 0  # Safety counter to ensure we don't loop indefinitely
        while self.angles[i] is not None and steps < n-1: # we do not consider the last when called on the whole loop
            turn = self.angles[i]
            direction = (direction + turn) % 6
            move = axial_neighbors[direction]
            current = (current[0] + move[0], current[1] + move[1])
            if current in visited:
                return True
            visited.add(current)
            i = (i + 1) % n  # Move to the next index in a circular manner
            steps += 1  # Increment safety counter
        return False

    #---------------------------------------------------------------
    def is_self_intersecting(self):
        n = self.size
        has_none = any(angle is None for angle in self.angles)
        if not has_none:
            return self.is_component_self_intersecting(0)
        for i in range(n):
            prev_index = (i - 1) % n  # Correctly handle previous index in a circular way
            if self.angles[i] is not None and self.angles[prev_index] is None:
                if self.is_component_self_intersecting(i):
                    return True
        return False

    #---------------------------------------------------------------
    # for testing and debug
    def count_self_intersections(self):
        n = self.size
        i = 0
        first_block = True
        intersection_count = 0
        while i < n:
            while i < n and self.angles[i] is None:
                i += 1
            if i >= n:
                break
            block_start = (0, 0)
            visited = {block_start}
            direction = 0
            current = block_start
            allowed_return = first_block
            while i < n and self.angles[i] is not None:
                turn = self.angles[i]
                direction = (direction + turn) % 6
                move = axial_neighbors[direction]
                current = (current[0] + move[0], current[1] + move[1])
                if current in visited:
                    if not (allowed_return and current == block_start and i == n-1):
                        intersection_count += 1
                visited.add(current)
                i += 1
            first_block = False
        return intersection_count

    # Helper methods added to PolyformCandidate
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
            mappingA_color = (255, 20, 147) if (i in mappedA and next_i in mappedA) else None
            mappingB_color = (0, 200, 255) if (i in mappedB and next_i in mappedB) else None
            draw_multicolor_segment(screen, points[i], points[next_i],
                                    colors=[base_color, mappingA_color, mappingB_color],
                                    width=3)

    #---------------------------------------------------------------
    def get_longest_non_none_block_indices(self):
        n = self.size
        best_start = None
        best_length = 0
        current_start = None
        current_length = 0
        # double the array virtually to handle wrap-around
        for i in range(2 * n):
            val = self.angles[i % n]
            if val is not None:
                if current_start is None:
                    current_start = i
                current_length += 1
                if current_length > best_length:
                    best_length = current_length
                    best_start = current_start
                if current_length == n:
                    break
            else:
                current_start = None
                current_length = 0
        if best_start is None:
            return (0, 0)
        best_length = min(best_length, n)
        return (best_start % n, best_length)

    #---------------------------------------------------------------
    def draw(self, edge_length=40, window_size=(800, 600)):
        pygame.display.set_caption(str(self))
        screen = pygame.display.set_mode(window_size)
        screen.fill((255, 255, 255))
        # Instead of starting at vertex 0, start at the block defined by get_longest_non_none_block_indices
        block_start, block_length = self.get_longest_non_none_block_indices()
        #fixed_point = (-1, 0)
        candidate_start = (0, 0)
        candidate_path = [candidate_start]
        candidate_indices = [block_start]  # store the original index from self.angles
        #print("INIT", candidate_path, candidate_indices)
        direction = 0
        current = candidate_start
        i = (block_start + 1) % self.size
        for _ in range(block_length-1):   # probably a cleaner way to write that
            turn = self.angles[i]
            direction = (direction + turn) % 6
            move = axial_neighbors[direction]
            current = (current[0] + move[0], current[1] + move[1])
            candidate_path.append(current)
            candidate_indices.append(i)
            #print("INCR", candidate_path, candidate_indices)
            i = (i + 1) % self.size
        #print(candidate_path)
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
        for j, pos in enumerate(adjusted_points):
            pygame.draw.circle(screen, (0, 0, 0), (int(pos[0]), int(pos[1])), 4)
            idx = (candidate_indices[j] + 1) % self.size  # I do not get why +1 is needed here but it is
            #print(j, idx, candidate_indices)
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
        self.components = self.build_components()
        # Another tentative heuristic
        self.compute_components_angles()
        # Heuristic to be tested and tuned
        # CANNOT USE SORT WITH COMPONENT ANGLES OPTIM
        # self.sort_components()
        # self.components.sort(key=lambda comp: (-len(comp), min(comp)))
        # Ensure all angles start as None
        self.angles = [None] * self.size
        # Set initial fixed angles
        self.angles[0] = 2   # TODO ###############################################"
        self.angles[self.idx_b_pole % self.size] = 2
        self.angles[self.a_offset % self.size] = 2
        self.angles[(self.idx_b_pole + self.a_offset) % self.size] = 2
        self.angles[self.b_offset % self.size] = 2
        self.angles[(self.idx_b_pole + self.b_offset) % self.size] = 2
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
        if self.is_self_intersecting() or not self.can_loop():
            return
        if comp_index == len(self.components):
            #if self.is_valid_loop():
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
            for value in self.valid_turns():
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
    # TEST
#     testCandidate = PolyformCandidate(PolyformType.POLYTRI, 9, 5, 0, 0, False, True)
#     if testCandidate.initSearchState():
#         testCandidate.backtrack_components()
#     print("End TEST")
#     wait_for_keypress()
    START_SIZE = 3  # Min for polyhex would be 6
    MAX_SIZE = 10000
    start_time = time.time()
    print("Starting loop")
    for size in range(START_SIZE, MAX_SIZE + 1, 1): # For polyhex 2 by 2 is possible
        elapsed_time = time.time() - start_time
        header = f"size={size} starting. {elapsed_time:.2f}s from init. "
        print(header, end="", flush=True)
        global_min_distance = 10**9
        total = size // 2
        for idx_b_pole in range(total - 1, 0, -1): # Starting by ballanced split
            percentage = (idx_b_pole / total) * 100
            sys.stdout.write(f"\r{header}{percentage:.1f}% {global_min_distance}             ")
            sys.stdout.flush()
            for a_offset in range(0, size):
                for b_offset in range(0, size):
                    if a_offset == 0 and b_offset == 0:   # FILTER --------
                        continue
                    #if not (a_offset == 0 or b_offset == 0):  # FILTER --------
                    #    continue
                    for a_flip in [False, True]:
                        for b_flip in [False, True]:
                            #print(f"size={size}, idx_b_pole={idx_b_pole}, a_offset={a_offset}, b_offset={b_offset}, a_flip={a_flip}, b_flip={b_flip}")
                            candidate = PolyformCandidate(PolyformType.POLYTRI, size, idx_b_pole, a_offset, b_offset, a_flip, b_flip)
                            if not candidate.initSearchState():
                                continue
                            candidate.backtrack_components()
                            # candidate.draw_schema()
                            # wait_for_keypress()
        print("")

# STATS
# with at most one offset at 0,  critical angles at 2, count intersection 0 or 1
# can loop angle optim
# Starting loop
# size=3 starting. 0.00s from init.
# size=4 starting. 0.00s from init. 50.0% 1000000000
# size=5 starting. 0.00s from init. 50.0% 1000000000
# size=6 starting. 0.00s from init. 33.3% 1000000000
# size=7 starting. 0.01s from init. 33.3% 3
# size=8 starting. 0.01s from init. 25.0% 4
# size=9 starting. 0.02s from init. 25.0% 3
# size=10 starting. 0.03s from init. 20.0% 3
# size=11 starting. 0.05s from init. 20.0% 3
# size=12 starting. 0.07s from init. 16.7% 3
# size=13 starting. 0.12s from init. 16.7% 2
# size=14 starting. 0.17s from init. 14.3% 3
# size=15 starting. 0.24s from init. 14.3% 3
# size=16 starting. 0.34s from init. 12.5% 2
# size=17 starting. 0.47s from init. 12.5% 2
# size=18 starting. 0.63s from init. 11.1% 2
# size=19 starting. 0.86s from init. 11.1% 1
# size=20 starting. 1.14s from init. 10.0% 2
# size=21 starting. 1.56s from init. 10.0% 2
# size=22 starting. 2.06s from init. 9.1% 1
# size=23 starting. 2.79s from init. 9.1% 2
# size=24 starting. 3.85s from init. 8.3% 2
# size=25 starting. 5.29s from init. 8.3% 1
# size=26 starting. 7.38s from init. 7.7% 1
# size=27 starting. 11.05s from init. 7.7% 2
# size=28 starting. 15.96s from init. 7.1% 1
# size=29 starting. 23.84s from init. 7.1% 1
# size=30 starting. 38.61s from init. 66.7% 2
# removed bug from canloop 2*
# Starting loop
# size=3 starting. 0.00s from init.
# size=4 starting. 0.00s from init. 50.0% 1000000000
# size=5 starting. 0.00s from init. 50.0% 1000000000
# size=6 starting. 0.00s from init. 33.3% 1000000000
# size=7 starting. 0.01s from init. 33.3% 3
# size=8 starting. 0.01s from init. 25.0% 4
# size=9 starting. 0.02s from init. 25.0% 3
# size=10 starting. 0.03s from init. 20.0% 2
# size=11 starting. 0.05s from init. 20.0% 3
# size=12 starting. 0.07s from init. 16.7% 3
# size=13 starting. 0.12s from init. 16.7% 2
# size=14 starting. 0.17s from init. 14.3% 2
# size=15 starting. 0.24s from init. 14.3% 2
# size=16 starting. 0.35s from init. 12.5% 2
# size=17 starting. 0.49s from init. 12.5% 2
# size=18 starting. 0.67s from init. 11.1% 2
# size=19 starting. 0.92s from init. 11.1% 1
# size=20 starting. 1.24s from init. 10.0% 2
# size=21 starting. 1.75s from init. 10.0% 2
# size=22 starting. 2.41s from init. 9.1% 1
# size=23 starting. 3.48s from init. 9.1% 2
# size=24 starting. 5.01s from init. 8.3% 2
# size=25 starting. 7.77s from init. 8.3% 1
# size=26 starting. 11.58s from init. 7.7% 1
# size=27 starting. 19.69s from init. 7.7% 2
# size=28 starting. 30.88s from init. 7.1% 1
# size=29 starting. 55.90s from init. 7.1% 1
# size=30 starting. 92.12s from init. 73.3% 2
# with latest optims + component order that seems to be a little worst
# Starting loop
# size=3 starting. 0.00s from init.
# size=4 starting. 0.00s from init. 50.0% 1000000000
# size=5 starting. 0.00s from init. 50.0% 1000000000
# size=6 starting. 0.00s from init. 33.3% 1000000000
# size=7 starting. 0.00s from init. 33.3% 3
# size=8 starting. 0.00s from init. 25.0% 4
# size=9 starting. 0.01s from init. 25.0% 3
# size=10 starting. 0.02s from init. 20.0% 2
# size=11 starting. 0.03s from init. 20.0% 3
# size=12 starting. 0.05s from init. 16.7% 3
# size=13 starting. 0.07s from init. 16.7% 2
# size=14 starting. 0.10s from init. 14.3% 2
# size=15 starting. 0.15s from init. 14.3% 2
# size=16 starting. 0.22s from init. 12.5% 2
# size=17 starting. 0.31s from init. 12.5% 2
# size=18 starting. 0.42s from init. 11.1% 2
# size=19 starting. 0.60s from init. 11.1% 2
# size=20 starting. 0.82s from init. 10.0% 2
# size=21 starting. 1.21s from init. 10.0% 2
# size=22 starting. 1.70s from init. 9.1% 2
# size=23 starting. 2.55s from init. 9.1% 2
# size=24 starting. 3.83s from init. 8.3% 2
# size=25 starting. 6.07s from init. 8.3% 2
# size=26 starting. 9.43s from init. 7.7% 2
# size=27 starting. 16.22s from init. 7.7% 2
# size=28 starting. 26.13s from init. 7.1% 2
# size=29 starting. 47.08s from init. 7.1% 2
# size=30 starting. 80.85s from init. 60.0% 2
# with at least one offset at 0
# pygame-ce 2.5.3 (SDL 2.30.12, Python 3.12.5)
# Starting loop
# size=6 starting. 0.00s from init. 66.7% 4
# size=8 starting. 0.00s from init. 75.0% 4
# size=10 starting. 0.00s from init. 80.0% 3
# size=12 starting. 0.01s from init. 83.3% 2
# size=14 starting. 0.01s from init. 85.7% 2
# size=16 starting. 0.03s from init. 87.5% 2
# size=18 starting. 0.05s from init. 88.9% 2
# size=20 starting. 0.09s from init. 90.0% 1
# size=22 starting. 0.16s from init. 90.9% 2
# size=24 starting. 0.30s from init. 91.7% 2
# size=26 starting. 0.54s from init. 92.3% 1
# size=28 starting. 1.01s from init. 92.9% 1
# size=30 starting. 1.92s from init. 93.3% 1
# size=32 starting. 3.61s from init. 93.8% 1
# size=34 starting. 6.91s from init. 94.1% 1
# size=36 starting. 13.28s from init. 94.4% 1
# size=38 starting. 25.84s from init. 94.7% 1
# size=40 starting. 50.37s from init. 95.0% 1
# size=42 starting. 98.34s from init. 95.2% 1
# size=44 starting. 190.18s from init. 95.5% 1
# size=46 starting. 367.47s from init. 95.7% 1
# size=48 starting. 703.24s from init. 95.8% 1
# size=50 starting. 1345.33s from init. 96.0% 1
# size=52 starting. 2566.51s from init. 96.2% 1
# size=54 starting. 4888.26s from init. 3.7%
# without offset at 0
# size=104 starting. 0.00s from init. 98.1% 2
# size=106 starting. 929.35s from init. 98.1% 2
# size=108 starting. 1757.69s from init. 98.1% 2
# size=110 starting. 3409.16s from init. 96.4% 2
