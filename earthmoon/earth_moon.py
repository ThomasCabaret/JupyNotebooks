import argparse
import ctypes
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyglet
from pyglet.gl import *
from PIL import Image

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
TEXTURE_EARTH_PATH = os.path.join("textures", "earth.jpg")
TEXTURE_MOON_PATH = os.path.join("textures", "moon.jpg")

WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 800

DEFAULT_YAW_DEG = 35.0
DEFAULT_PITCH_DEG = -20.0
DEFAULT_DISTANCE = 8.0

EARTH_RADIUS = 1.0
MOON_RADIUS = 0.27
MOON_ORBIT_RADIUS = 4.0

EARTH_SPIN_DEG_PER_SEC = 15.0
MOON_SPIN_DEG_PER_SEC = 4.0
MOON_ORBIT_DEG_PER_SEC = 8.0

SPHERE_SLICES = 64
SPHERE_STACKS = 32

SHOW_ORBIT_PLANE = True 

# ------------------------------------------------------------
# Matrix Helpers (Standard Row-Major for logic, Transpose for GL)
# ------------------------------------------------------------
def mat4_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)

def mat4_translate(x: float, y: float, z: float) -> np.ndarray:
    m = mat4_identity()
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m

def mat4_rotate_y(deg: float) -> np.ndarray:
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = mat4_identity()
    m[0, 0], m[0, 2] = c, s
    m[2, 0], m[2, 2] = -s, c
    return m

def perspective(fovy_deg: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = mat4_identity()
    m[0, 0:3], m[1, 0:3], m[2, 0:3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)
    return m

def to_gl(m: np.ndarray):
    # Standard OpenGL expects column-major. Transpose our row-major result.
    return m.T.flatten().astype(np.float32)

# ------------------------------------------------------------
# Camera
# ------------------------------------------------------------
@dataclass
class CameraOrbit:
    yaw_deg: float = DEFAULT_YAW_DEG
    pitch_deg: float = DEFAULT_PITCH_DEG
    distance: float = DEFAULT_DISTANCE

    def clamp(self):
        self.pitch_deg = max(-89.0, min(89.0, self.pitch_deg))
        self.distance = max(1.5, min(50.0, self.distance))

    def get_view_matrix(self) -> np.ndarray:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        x = self.distance * math.cos(pitch) * math.sin(yaw)
        y = self.distance * math.sin(pitch)
        z = self.distance * math.cos(pitch) * math.cos(yaw)
        return look_at(np.array([x, y, z]), np.array([0, 0, 0]), np.array([0, 1, 0]))

# ------------------------------------------------------------
# Mesh Building
# ------------------------------------------------------------
def build_sphere(radius: float, slices: int, stacks: int):
    pos, nor, uv, idx = [], [], [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            nx, ny, nz = math.sin(phi)*math.cos(theta), math.cos(phi), math.sin(phi)*math.sin(theta)
            pos.extend([radius*nx, radius*ny, radius*nz])
            nor.extend([nx, ny, nz])
            uv.extend([j/slices, i/stacks])
    for i in range(stacks):
        for j in range(slices):
            p1, p2 = i*(slices+1)+j, (i+1)*(slices+1)+j
            idx.extend([p1, p2, p1+1, p1+1, p2, p2+1])
    return np.array(pos, np.float32), np.array(nor, np.float32), np.array(uv, np.float32), np.array(idx, np.uint32)

def build_disc(radius: float, segments: int):
    pos, nor, uv, idx = [0.0,0.0,0.0], [0.0,1.0,0.0], [0.5,0.5], []
    for i in range(segments + 1):
        ang = 2.0 * math.pi * i / segments
        px, pz = radius * math.cos(ang), radius * math.sin(ang)
        pos.extend([px, 0.0, pz])
        nor.extend([0.0, 1.0, 0.0])
        uv.extend([0.5 + 0.5*math.cos(ang), 0.5 + 0.5*math.sin(ang)])
        if i < segments: idx.extend([0, i+1, i+2])
    return np.array(pos, np.float32), np.array(nor, np.float32), np.array(uv, np.float32), np.array(idx, np.uint32)

def build_orbit_circle(radius: float, segments: int):
    pos, nor, uv, idx = [], [], [], []
    for i in range(segments + 1):
        ang = 2.0 * math.pi * i / segments
        px, pz = radius * math.cos(ang), radius * math.sin(ang)
        pos.extend([px, 0.0, pz])
        nor.extend([0.0, 1.0, 0.0])
        uv.extend([i/segments, 0.5])
        idx.append(i)
    return np.array(pos, np.float32), np.array(nor, np.float32), np.array(uv, np.float32), np.array(idx, np.uint32)

# ------------------------------------------------------------
# Shaders
# ------------------------------------------------------------
VERT_SRC = """#version 330 core
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_nor;
layout(location=2) in vec2 a_uv;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_nor;
out vec2 v_uv;
void main(){
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    v_nor = normalize(mat3(u_model) * a_nor);
    v_uv = a_uv;
}"""

FRAG_SRC = """#version 330 core
in vec3 v_nor;
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec3 u_light;
uniform float u_alpha;
out vec4 f_col;
void main(){
    vec4 tex = texture(u_tex, v_uv);
    float d = max(dot(v_nor, normalize(u_light)), 0.0);
    f_col = vec4(tex.rgb * (0.2 + 0.8 * d), tex.a * u_alpha);
}"""

class GLMesh:
    def __init__(self, p, n, u, i, tid, mode=GL_TRIANGLES):
        self.tid, self.count, self.mode = tid, i.size, mode
        self.vao = GLuint(); glGenVertexArrays(1, ctypes.byref(self.vao)); glBindVertexArray(self.vao.value)
        vbo = GLuint(); glGenBuffers(1, ctypes.byref(vbo)); glBindBuffer(GL_ARRAY_BUFFER, vbo.value)
        data = np.hstack([p.reshape(-1,3), n.reshape(-1,3), u.reshape(-1,2)]).astype(np.float32)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data.ctypes.data, GL_STATIC_DRAW)
        ebo = GLuint(); glGenBuffers(1, ctypes.byref(ebo)); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo.value)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, i.nbytes, i.ctypes.data, GL_STATIC_DRAW)
        for loc, sz, off in [(0,3,0), (1,3,12), (2,2,24)]:
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, sz, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(off))
        glBindVertexArray(0)

    def draw(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tid)
        glBindVertexArray(self.vao.value)
        glDrawElements(self.mode, self.count, GL_UNSIGNED_INT, 0)

# ------------------------------------------------------------
# Main App
# ------------------------------------------------------------
class SolarApp(pyglet.window.Window):
    def __init__(self):
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "Earth-Moon System", resizable=True,
                         config=pyglet.gl.Config(major_version=3, minor_version=3, depth_size=24, double_buffer=True))
        self.camera = CameraOrbit()
        self.t_earth = self.t_moon = self.t_orb = 0.0
        
        glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE); glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glFrontFace(GL_CW)

        self.prog = self._load_prog()
        self.u_mvp = glGetUniformLocation(self.prog, b"u_mvp")
        self.u_model = glGetUniformLocation(self.prog, b"u_model")
        self.u_light = glGetUniformLocation(self.prog, b"u_light")
        self.u_alpha = glGetUniformLocation(self.prog, b"u_alpha")

        self.tex_e = self._load_tex(TEXTURE_EARTH_PATH)
        self.tex_m = self._load_tex(TEXTURE_MOON_PATH)

        self.tex_magenta = self._load_solid_tex(255, 0, 255, 255)
        self.tex_cyan = self._load_solid_tex(0, 255, 255, 255)
        
        self.earth = GLMesh(*build_sphere(EARTH_RADIUS, SPHERE_SLICES, SPHERE_STACKS), self.tex_e)
        self.moon = GLMesh(*build_sphere(MOON_RADIUS, SPHERE_SLICES, SPHERE_STACKS), self.tex_m)
        self.plane = GLMesh(*build_disc(MOON_ORBIT_RADIUS, 64), self.tex_cyan)
        self.orbit = GLMesh(*build_orbit_circle(MOON_ORBIT_RADIUS, 256), self.tex_magenta, GL_LINE_STRIP)

        self.proj = perspective(60.0, self.width/max(1, self.height), 0.1, 1000.0)
        pyglet.clock.schedule_interval(self.update, 1/60.0)

    def _load_tex(self, path):
        img = Image.open(path).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)
        tid = GLuint(); glGenTextures(1, ctypes.byref(tid)); glBindTexture(GL_TEXTURE_2D, tid.value)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
        glGenerateMipmap(GL_TEXTURE_2D)
        return tid.value

    def _load_solid_tex(self, r, g, b, a):
        tid = GLuint(); glGenTextures(1, ctypes.byref(tid)); glBindTexture(GL_TEXTURE_2D, tid.value)
        px = (ctypes.c_ubyte * 4)(r, g, b, a)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        return tid.value

    def _load_prog(self):
        def c(src, t):
            s = glCreateShader(t)
            src_bytes = src.encode('utf-8')
            src_ptr = ctypes.cast(ctypes.c_char_p(src_bytes), ctypes.POINTER(ctypes.c_char))
            glShaderSource(s, 1, ctypes.byref(src_ptr), None)
            glCompileShader(s)
            return s
        p = glCreateProgram()
        glAttachShader(p, c(VERT_SRC, GL_VERTEX_SHADER))
        glAttachShader(p, c(FRAG_SRC, GL_FRAGMENT_SHADER))
        glLinkProgram(p)
        return p

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        self.proj = perspective(60.0, width/max(1, height), 0.1, 1000.0)

    def update(self, dt):
        self.t_earth = (self.t_earth + EARTH_SPIN_DEG_PER_SEC * dt) % 360
        self.t_moon = (self.t_moon + MOON_SPIN_DEG_PER_SEC * dt) % 360
        self.t_orb = (self.t_orb + MOON_ORBIT_DEG_PER_SEC * dt) % 360

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.camera.yaw_deg -= dx * 0.2
        self.camera.pitch_deg += dy * 0.2
        self.camera.clamp()

    def on_mouse_scroll(self, x, y, sx, sy):
        self.camera.distance *= (0.9 ** sy)
        self.camera.clamp()

    def on_draw(self):
        self.clear(); glUseProgram(self.prog)
        view = self.camera.get_view_matrix()
        vp = self.proj @ view
        glUniform3f(self.u_light, 1.0, 0.5, 1.0)
        
        # Earth
        m_e = mat4_rotate_y(self.t_earth)
        self._render(self.earth, m_e, vp @ m_e, 1.0)

        # Moon
        m_m = mat4_rotate_y(self.t_orb) @ mat4_translate(MOON_ORBIT_RADIUS, 0, 0) @ mat4_rotate_y(self.t_moon)
        self._render(self.moon, m_m, vp @ m_m, 1.0)

        # Orbit Circle (very visible)
        glLineWidth(4.0)
        self._render(self.orbit, mat4_identity(), vp, 1.0)
        glLineWidth(1.0)

        # Orbit Plane (much clearer)
        if SHOW_ORBIT_PLANE:
            glDisable(GL_CULL_FACE)
            self._render(self.plane, mat4_identity(), vp, 0.55)
            glEnable(GL_CULL_FACE)

    def _render(self, obj, m, mvp, alpha):
        glUniform1f(self.u_alpha, alpha)
        glUniformMatrix4fv(self.u_model, 1, GL_FALSE, (GLfloat * 16)(*to_gl(m)))
        glUniformMatrix4fv(self.u_mvp, 1, GL_FALSE, (GLfloat * 16)(*to_gl(mvp)))
        obj.draw()

if __name__ == "__main__":
    app = SolarApp()
    pyglet.app.run()
