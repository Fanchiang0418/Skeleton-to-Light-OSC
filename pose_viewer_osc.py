# pose_viewer_osc.py
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.animation as animation
import numpy as np
import ast
import os
import glob
from PIL import Image, ImageTk
from pythonosc.udp_client import SimpleUDPClient
import numpy.linalg as LA

# ===================== 可調參數 =====================
OSC_IP = "10.77.5.158"   # TouchDesigner 在同一台：127.0.0.1；別台請改成對方內網 IP
OSC_PORT = 8000
FPS_DEFAULT = 25       # 若你的資料 FPS 為 25，就用 25（會影響事件/平滑的時間尺度）
EMA_ALPHA = 0.3        # 平滑係數（0~1，越大越跟手）
IMAGE_DISPLAY_SIZE = (400, 400)
# ===================================================

# 關節連線（與你原本相同；去掉腳趾彼此連線）
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
    (15, 21), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29),
    (27, 31), (24, 26), (26, 28), (28, 30), (28, 32)
]
FOOT_INDICES = [27, 28, 29, 30, 31, 32]
HIP_INDICES = [23, 24]

# ----------------- 共同函式 -----------------
def ema(prev, x, a=EMA_ALPHA):
    return a*x + (1-a)*prev if prev is not None else x

def load_pose_data(filepath):
    """讀 CSV：每列 p0..p32 皆為 '(x,y,z)' 字串；回傳 ndarray (T, 33, 3)"""
    df = pd.read_csv(filepath)
    frames = []
    for _, row in df.iterrows():
        pts = [ast.literal_eval(row[f"p{i}"]) for i in range(33)]
        frames.append(np.array(pts, dtype=float))
    return np.array(frames)

def normalize_by_first_frame(all_frames):
    """
    以第一幀做地板/中心對齊：把髖中心移到原點、y 軸翻正，並讓整段最低點為 y=0。
    """
    if len(all_frames) == 0:
        return all_frames

    first = all_frames[0]
    floor_y = np.max(first[FOOT_INDICES, 1])
    hip_center = np.mean(first[HIP_INDICES], axis=0)
    tx, ty, tz = hip_center[0], floor_y, hip_center[2]
    T = []
    for f in all_frames:
        g = f.copy()
        g[:, 0] -= tx
        g[:, 1] -= ty
        g[:, 2] -= tz
        g[:, 1] *= -1.0          # y 翻正
        T.append(g)
    arr = np.asarray(T)
    arr[:, :, 1] -= arr[:, :, 1].min()  # 讓最低點為 0
    return arr

def precompute_features(J, fps):
    """
    通用特徵：全身能量、水平朝向角速度（以 x-z PCA 主軸近似）、平均高度與其速度。
    回傳 feats 與門檻設定 cfg。
    """
    T = J.shape[0]
    dt = 1.0 / fps

    # 速度與能量
    V = np.gradient(J, dt, axis=0)               # (T,33,3)
    speed = LA.norm(V, axis=2)                   # (T,33)
    energy = speed.sum(axis=1)                   # (T,)

    # x-z 主軸 → yaw 與角速度
    yaw = []
    for t in range(T):
        xz = J[t, :, [0, 2]]
        xz -= xz.mean(axis=0, keepdims=True)
        C = np.cov(xz.T)
        vals, vecs = LA.eigh(C)
        v1 = vecs[:, np.argmax(vals)]
        yaw.append(np.arctan2(v1[1], v1[0]))
    yaw = np.unwrap(np.array(yaw))
    yaw_rate = np.gradient(yaw, dt)

    # 平均高度與其速度（抓跳躍）
    com_y  = J[:, :, 1].mean(axis=1)
    com_vy = np.gradient(com_y, dt)

    # 正規化範圍與事件門檻
    e_lo, e_hi = np.percentile(energy, [10, 95])
    y_lo, y_hi = np.percentile(com_y,   [10, 90])
    cfg = {
        "e_lo": e_lo, "e_hi": e_hi,
        "yaw_lo": -2.5, "yaw_hi": 2.5,          # 合理角速範圍
        "JUMP_VY": (y_hi - y_lo) * 3.0,         # 上衝速度門檻
        "JUMP_DY": (y_hi - y_lo) * 0.15,        # 抬升高度門檻
        "SPIN_W": 1.6,                          # 快速旋轉門檻 (rad/s)
        "COOLDOWN": 0.6                         # 事件冷卻
    }
    feats = {"energy": energy, "yaw_rate": yaw_rate, "com_y": com_y, "com_vy": com_vy}
    return feats, cfg

def n01(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0, 1))

# ----------------- 主程式（Tk App） -----------------
class PoseVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D/3D Pose Visualizer (Fixed Floor)")
        self.root.geometry("1100x700")
        self.ani = None

        # 版面
        top_frame = tk.Frame(root); top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        main_container = tk.Frame(root); main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.plot_frame = tk.Frame(main_container, relief=tk.SUNKEN, borderwidth=1)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_frame = tk.Frame(main_container, relief=tk.SUNKEN, borderwidth=1, width=IMAGE_DISPLAY_SIZE[0])
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False); self.image_frame.pack_propagate(False)

        # 元件
        self.select_button = tk.Button(top_frame, text="Select CSV File and Start", command=self.on_select_file)
        self.select_button.pack(side=tk.LEFT)
        self.info_label = tk.Label(top_frame, text="Please select a file to begin."); self.info_label.pack(side=tk.LEFT, padx=10)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(self.image_frame, text="Image Display Area", bg='gray')
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # OSC & 特徵狀態
        self.osc = SimpleUDPClient(OSC_IP, OSC_PORT)
        self.fps = FPS_DEFAULT
        self.feats = None
        self.cfg = None
        self._sm_intensity = None
        self._sm_speed = None
        self._sm_hue01 = None
        self._yaw_accum = 0.0
        self._t = 0.0
        self._last_jump_ts = -1e9
        self._last_spin_ts = -1e9

    # ---- 載檔並預處理 ----
    def on_select_file(self):
        filepath = filedialog.askopenfilename(
            title="Select a CSV file", initialdir=".",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not filepath:
            return

        csv_filename = os.path.basename(filepath)
        track_name = os.path.splitext(csv_filename)[0]
        image_dir = os.path.join(os.path.dirname(filepath), track_name)
        if not os.path.isdir(image_dir):
            messagebox.showwarning("Path Error", f"Could not find image folder:\n{image_dir}")
            return

        raw_frames = load_pose_data(filepath)
        self.all_frames = normalize_by_first_frame(raw_frames)
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if len(self.all_frames) != len(self.image_files):
            messagebox.showwarning("Count Mismatch", f"CSV frames ({len(self.all_frames)}) vs. images ({len(self.image_files)}).")

        # 預先計算特徵（供 OSC 輸出）
        self.fps = FPS_DEFAULT
        self.feats, self.cfg = precompute_features(self.all_frames, self.fps)
        self._sm_intensity = self._sm_speed = self._sm_hue01 = None
        self._yaw_accum = 0.0; self._t = 0.0
        self._last_jump_ts = -1e9; self._last_spin_ts = -1e9

        self.info_label.config(text=f"Loaded: {csv_filename} ({len(self.all_frames)} frames)")
        self.start_animation()

    # ---- 開始動畫 ----
    def start_animation(self):
        if self.ani:
            self.ani.event_source.stop()

        self.ax.clear()

        all_points = self.all_frames.reshape(-1, 3)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

        x_range = x_max - x_min; z_range = z_max - z_min
        r = max(x_range, z_range) / 2.0
        mid_x = (x_max + x_min) / 2; mid_z = (z_max + z_min) / 2

        self.ax.set_xlim(mid_x - r, mid_x + r)
        self.ax.set_ylim(mid_z - r, mid_z + r)
        self.ax.set_zlim(0, y_max * 1.1)

        self.ax.set_xlabel("X-axis"); self.ax.set_ylabel("Z-axis (Depth)"); self.ax.set_zlabel("Y-axis (Height)")
        self.ax.view_init(elev=15, azim=-75)

        self.scatter = self.ax.scatter([], [], [], c="red", marker="o", s=20)
        self.lines = [self.ax.plot([], [], [], "b-")[0] for _ in POSE_CONNECTIONS]
        self.frame_text = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)

        interval_ms = int(1000.0 / self.fps)
        self.ani = animation.FuncAnimation(self.fig, self._update_animation,
                                           frames=len(self.all_frames), interval=interval_ms,
                                           blit=False, repeat=True)
        self.canvas.draw()

    # ---- 每幀更新（含 OSC 輸出）----
    def _update_animation(self, frame_num):
        pts = self.all_frames[frame_num]
        self.scatter._offsets3d = (pts[:, 0], pts[:, 2], pts[:, 1])
        for i, (a, b) in enumerate(POSE_CONNECTIONS):
            pa, pb = pts[a], pts[b]
            self.lines[i].set_data([pa[0], pb[0]], [pa[2], pb[2]])
            self.lines[i].set_3d_properties([pa[1], pb[1]])
        self.frame_text.set_text(f"Frame: {frame_num+1}/{len(self.all_frames)}")

        if frame_num < len(self.image_files):
            img_path = self.image_files[frame_num]
            try:
                img = Image.open(img_path)
                img.thumbnail(IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                self.image_label.config(image=photo); self.image_label.image = photo
            except Exception:
                self.image_label.config(text=f"Could not load image\n{os.path.basename(img_path)}")

        # ===== 骨架 → 特徵 → OSC =====
        if self.feats is not None:
            E = n01(self.feats["energy"][frame_num], self.cfg["e_lo"], self.cfg["e_hi"])
            yaw_rate = float(np.clip(self.feats["yaw_rate"][frame_num], self.cfg["yaw_lo"], self.cfg["yaw_hi"]))
            self._yaw_accum += yaw_rate * (1.0 / self.fps)
            hue_deg = (self._yaw_accum * 180.0 / np.pi) % 360.0
            hue01 = hue_deg / 360.0

            intensity = 0.2 + 0.7 * E
            speed = 0.6 + 1.0 * E

            self._sm_intensity = ema(self._sm_intensity, intensity)
            self._sm_speed     = ema(self._sm_speed,     speed)
            self._sm_hue01     = ema(self._sm_hue01,     hue01)

            # 連續參數
            self.osc.send_message("/intensity", float(self._sm_intensity))
            self.osc.send_message("/speed",     float(self._sm_speed))
            self.osc.send_message("/hue",       float(hue_deg))  # 以度數送出

            # 事件：跳躍（上衝 + 抬升）
            if self.feats["com_vy"][frame_num] > self.cfg["JUMP_VY"] and (self._t - self._last_jump_ts) > self.cfg["COOLDOWN"]:
                j0 = max(0, frame_num - int(0.8 * self.fps))
                baseline = float(np.median(self.feats["com_y"][j0:frame_num+1]))
                if (self.feats["com_y"][frame_num] - baseline) > self.cfg["JUMP_DY"]:
                    self.osc.send_message("/event/jump", 1.0)
                    self._last_jump_ts = self._t

            # 事件：快速旋轉（正/負代表方向）
            if abs(yaw_rate) > self.cfg["SPIN_W"] and (self._t - self._last_spin_ts) > self.cfg["COOLDOWN"]:
                self.osc.send_message("/event/spin", float(np.sign(yaw_rate)))
                self._last_spin_ts = self._t

            self._t += (1.0 / self.fps)

        self.canvas.draw()
        return self.scatter

# ----------------- 入口 -----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PoseVisualizerApp(root)
    root.mainloop()
