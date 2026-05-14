"""
HIT137 Assignment 3 
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import random
import os


# ─────────────────────────────────────────────
# 1.  IMAGE PROCESSOR  (OpenCV logic)
# ─────────────────────────────────────────────
class ImageProcessor:
    """Handles all OpenCV image loading, scaling, and alteration logic."""

    NUM_DIFFERENCES = 5
    MIN_REGION = 40   # minimum px side for a difference patch
    MAX_REGION = 90   # maximum px side for a difference patch

    def __init__(self):
        self.original_cv = None   # BGR ndarray – full-size original
        self.modified_cv = None   # BGR ndarray – full-size modified copy
        self.difference_regions = []   # list of (x, y, w, h) in display coords

        self._display_w = 600
        self._display_h = 450
        self._scale_x = 1.0
        self._scale_y = 1.0

    # ── public ──────────────────────────────
    def load_image(self, path: str) -> bool:
        """Load an image from *path* and generate the modified clone."""
        img = cv2.imread(path)
        if img is None:
            return False
        self.original_cv = img.copy()
        self.modified_cv, self.difference_regions = self._generate_differences(img.copy())
        return True

    def get_display_images(self):
        """Return (original_tk, modified_tk) scaled for display."""
        orig_disp = self._scale(self.original_cv)
        mod_disp  = self._scale(self.modified_cv)
        return self._to_tk(orig_disp), self._to_tk(mod_disp)

    def draw_circle_on_images(self, region_idx: int, colour: str):
        """
        Draw a circle on **scaled** copies of both images and return new
        PhotoImage objects.  colour is 'red' or 'blue'.
        """
        bgr = (0, 0, 255) if colour == 'red' else (255, 0, 0)

        orig_disp = self._scale(self.original_cv)
        mod_disp  = self._scale(self.modified_cv)

        rx, ry, rw, rh = self.difference_regions[region_idx]
        cx, cy = rx + rw // 2, ry + rh // 2
        radius = max(rw, rh) // 2 + 10
        thickness = 3

        cv2.circle(orig_disp, (cx, cy), radius, bgr, thickness)
        cv2.circle(mod_disp,  (cx, cy), radius, bgr, thickness)
        return self._to_tk(orig_disp), self._to_tk(mod_disp)

    def display_size(self):
        return self._display_w, self._display_h

    # ── private ─────────────────────────────
    def _scale(self, img):
        """Return a copy scaled to the fixed display size, preserving aspect ratio."""
        h, w = img.shape[:2]
        scale = min(self._display_w / w, self._display_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        self._scale_x = scale
        self._scale_y = scale
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Pad to exact display size
        canvas = np.zeros((self._display_h, self._display_w, 3), dtype=np.uint8)
        y_off = (self._display_h - new_h) // 2
        x_off = (self._display_w - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = scaled
        return canvas

    @staticmethod
    def _to_tk(bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil)

    def _generate_differences(self, img):
        """
        Plant exactly NUM_DIFFERENCES non-overlapping alterations into *img*.
        Returns the modified image and a list of (x,y,w,h) regions in
        **display** coordinates.
        """
        h, w = img.shape[:2]
        placed = []   # (x,y,w,h) in original image coords

        alteration_methods = [
            self._colour_shift,
            self._brightness_patch,
            self._blur_patch,
            self._invert_patch,
            self._noise_patch,
            self._darken_patch,
        ]

        attempts = 0
        while len(placed) < self.NUM_DIFFERENCES and attempts < 1000:
            attempts += 1
            rw = random.randint(self.MIN_REGION, self.MAX_REGION)
            rh = random.randint(self.MIN_REGION, self.MAX_REGION)
            rx = random.randint(0, max(0, w - rw - 1))
            ry = random.randint(0, max(0, h - rh - 1))

            if not self._overlaps(rx, ry, rw, rh, placed):
                method = random.choice(alteration_methods)
                method(img, rx, ry, rw, rh)
                placed.append((rx, ry, rw, rh))

        # Convert placed regions to display coords
        scale = min(self._display_w / w, self._display_h / h)
        x_off = (self._display_w - int(w * scale)) // 2
        y_off = (self._display_h - int(h * scale)) // 2

        disp_regions = []
        for (rx, ry, rw, rh) in placed:
            dx = int(rx * scale) + x_off
            dy = int(ry * scale) + y_off
            dw = int(rw * scale)
            dh = int(rh * scale)
            disp_regions.append((dx, dy, dw, dh))

        return img, disp_regions

    @staticmethod
    def _overlaps(rx, ry, rw, rh, placed, margin=10):
        for (ox, oy, ow, oh) in placed:
            if not (rx + rw + margin < ox or ox + ow + margin < rx or
                    ry + rh + margin < oy or oy + oh + margin < ry):
                return True
        return False

    # ── alteration methods ───────────────────
    @staticmethod
    def _colour_shift(img, x, y, w, h):
        region = img[y:y+h, x:x+w].astype(np.int16)
        shift = np.array([random.randint(30, 70), random.randint(30, 70),
                          random.randint(30, 70)], dtype=np.int16)
        if random.random() > 0.5:
            shift = -shift
        region = np.clip(region + shift, 0, 255).astype(np.uint8)
        img[y:y+h, x:x+w] = region

    @staticmethod
    def _brightness_patch(img, x, y, w, h):
        region = img[y:y+h, x:x+w].astype(np.int16)
        delta = random.choice([-60, -50, 50, 60])
        img[y:y+h, x:x+w] = np.clip(region + delta, 0, 255).astype(np.uint8)

    @staticmethod
    def _blur_patch(img, x, y, w, h):
        region = img[y:y+h, x:x+w]
        ksize = random.choice([11, 15, 19])
        img[y:y+h, x:x+w] = cv2.GaussianBlur(region, (ksize, ksize), 0)

    @staticmethod
    def _invert_patch(img, x, y, w, h):
        img[y:y+h, x:x+w] = cv2.bitwise_not(img[y:y+h, x:x+w])

    @staticmethod
    def _noise_patch(img, x, y, w, h):
        region = img[y:y+h, x:x+w].astype(np.int16)
        noise = np.random.randint(-40, 40, region.shape, dtype=np.int16)
        img[y:y+h, x:x+w] = np.clip(region + noise, 0, 255).astype(np.uint8)
        
    @staticmethod
    def _darken_patch(img, x, y, w, h):
        region = img[y:y+h, x:x+w].astype(np.int16)
        img[y:y+h, x:x+w] = np.clip(region - 35, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 2.  GAME STATE  (pure logic, no GUI)
# ─────────────────────────────────────────────
class GameState:
    """Tracks score, mistakes, and which differences have been found."""

    MAX_MISTAKES = 3

    def __init__(self):
        self.total_score = 0      # cumulative found across images
        self.mistakes = 0
        self.found = []           # list of booleans per current image
        self.locked = False       # True when mistake limit reached

    def new_image(self, num_differences: int):
        self.mistakes = 0
        self.found = [False] * num_differences
        self.locked = False

    def register_hit(self, idx: int):
        if not self.found[idx]:
            self.found[idx] = True
            self.total_score += 1

    def register_miss(self):
        self.mistakes += 1
        if self.mistakes >= self.MAX_MISTAKES:
            self.locked = True

    def remaining(self):
        return sum(1 for f in self.found if not f)

    def all_found(self):
        return all(self.found)

    def is_locked(self):
        return self.locked


# ─────────────────────────────────────────────
# 3.  GUI  (Tkinter – inherits from tk.Tk)
# ─────────────────────────────────────────────
class SpotTheDifferenceApp(tk.Tk):
    """
    Main application window.
    Inherits from tk.Tk (demonstrating inheritance & polymorphism by
    overriding geometry management and adding domain-specific behaviour).
    """

    CANVAS_W = 600
    CANVAS_H = 450
    BG       = "#0f172a"
    PANEL_BG = "#16213e"
    ACCENT   = "#e94560"
    TEXT_FG  = "#eaeaea"
    FOUND_CLR = "#00b894"
    MISS_CLR  = "#e17055"

    def __init__(self):
        super().__init__()
        self.title("Spot The Difference Game - HIT137 Assignment 3")
        self.configure(bg=self.BG)
        self.resizable(False, False)

        self._processor = ImageProcessor()
        self._state     = GameState()

        # These hold the current PhotoImage objects (must be kept alive)
        self._orig_photo = None
        self._mod_photo  = None

        # Circle overlays drawn on canvases: {region_idx: (orig_item, mod_item)}
        self._circle_items = {}

        self._build_ui()

    # ── UI construction ─────────────────────
    def _build_ui(self):
        # ── top bar ──────────────────────────
        top = tk.Frame(self, bg=self.PANEL_BG, pady=8)
        top.pack(fill=tk.X)

        tk.Label(top, text="🔍 FIND THE 5 DIFFERENCE", font=("Courier New", 18, "bold"),
                 bg=self.PANEL_BG, fg=self.ACCENT).pack(side=tk.LEFT, padx=16)

        self._score_var = tk.StringVar(value="Score: 0")
        tk.Label(top, textvariable=self._score_var, font=("Courier New", 13),
                 bg=self.PANEL_BG, fg=self.TEXT_FG).pack(side=tk.RIGHT, padx=16)

        # ── image canvases ───────────────────
        canvas_frame = tk.Frame(self, bg=self.BG)
        canvas_frame.pack(padx=10, pady=(8, 4))

        tk.Label(canvas_frame, text="ORIGINAL", font=("Courier New", 10, "bold"),
                 bg=self.BG, fg="#888").grid(row=0, column=0, pady=(0, 2))
        tk.Label(canvas_frame, text="MODIFIED  (click here!)", font=("Courier New", 10, "bold"),
                 bg=self.BG, fg=self.ACCENT).grid(row=0, column=1, pady=(0, 2))

        self._orig_canvas = tk.Canvas(canvas_frame, width=self.CANVAS_W,
                                      height=self.CANVAS_H, bg="#111", highlightthickness=0)
        self._orig_canvas.grid(row=1, column=0, padx=(0, 6))

        self._mod_canvas = tk.Canvas(canvas_frame, width=self.CANVAS_W,
                                     height=self.CANVAS_H, bg="#111", highlightthickness=0,
                                     cursor="crosshair")
        self._mod_canvas.grid(row=1, column=1, padx=(6, 0))
        self._mod_canvas.bind("<Button-1>", self._on_canvas_click)

        # ── status bar ───────────────────────
        status_frame = tk.Frame(self, bg=self.PANEL_BG, pady=6)
        status_frame.pack(fill=tk.X, pady=(4, 0))

        self._remaining_var = tk.StringVar(value="Remaining: –")
        tk.Label(status_frame, textvariable=self._remaining_var,
                 font=("Courier New", 12), bg=self.PANEL_BG, fg=self.FOUND_CLR
                 ).pack(side=tk.LEFT, padx=14)

        self._mistakes_var = tk.StringVar(value="Mistakes: 0 / 3")
        tk.Label(status_frame, textvariable=self._mistakes_var,
                 font=("Courier New", 12), bg=self.PANEL_BG, fg=self.MISS_CLR
                 ).pack(side=tk.LEFT, padx=14)

        self._msg_var = tk.StringVar(value="Welcome! Load an image and find the hidden differences.")
        tk.Label(status_frame, textvariable=self._msg_var,
                 font=("Courier New", 11, "italic"), bg=self.PANEL_BG, fg="#aaa"
                 ).pack(side=tk.LEFT, padx=14)

        # ── buttons ──────────────────────────
        btn_frame = tk.Frame(self, bg=self.BG, pady=8)
        btn_frame.pack()

        btn_cfg = dict(font=("Courier New", 11, "bold"), relief=tk.FLAT,
                       padx=14, pady=6, cursor="hand2")

        self._load_btn = tk.Button(btn_frame, text="📂  Choose Image",
                                   bg=self.ACCENT, fg="white",
                                   command=self._load_image, **btn_cfg)
        self._load_btn.pack(side=tk.LEFT, padx=8)

        self._reveal_btn = tk.Button(btn_frame, text="👁  Reveal All",
                                     bg="#444", fg="#bbb",
                                     command=self._reveal_all, **btn_cfg)
        self._reveal_btn.pack(side=tk.LEFT, padx=8)
        self._reveal_btn.config(state=tk.DISABLED)

    # ── actions ─────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        if not self._processor.load_image(path):
            messagebox.showerror("Image Error", "Unable to load image. Please select a valid image file.")
            return

        self._state.new_image(ImageProcessor.NUM_DIFFERENCES)
        self._circle_items.clear()
        self._refresh_canvases()
        self._update_status()
        self._reveal_btn.config(state=tk.NORMAL, bg="#2d3436", fg=self.TEXT_FG)
        self._msg_var.set("Click on the MODIFIED image to find differences!")

    def _on_canvas_click(self, event):
        if self._processor.original_cv is None:
            return
        if self._state.is_locked():
            self._msg_var.set("❌ Too many mistakes! Load a new image.")
            return

        cx, cy = event.x, event.y
        regions = self._processor.difference_regions

        # Check proximity to each region
        for i, (rx, ry, rw, rh) in enumerate(regions):
            if self._state.found[i]:
                continue
            # proximity: within circle of radius max(rw,rh)//2 + tolerance
            tolerance = max(rw, rh) // 2 + 15
            dist = ((cx - (rx + rw//2))**2 + (cy - (ry + rh//2))**2) ** 0.5
            if dist <= tolerance:
                self._state.register_hit(i)
                self._draw_circle(i, 'red')
                self._update_status()
                if self._state.all_found():
                    self._msg_var.set("🎉 All found! Load another image to keep going.")
                    messagebox.showinfo("Congratulations!",
                                        f"You found all 5 differences!\n"
                                        f"Total score: {self._state.total_score}")
                else:
                    self._msg_var.set(f"✅ Found one! {self._state.remaining()} remaining.")
                return

        # Miss
        self._state.register_miss()
        self._update_status()
        if self._state.is_locked():
            self._msg_var.set("❌ 3 mistakes reached. Use Reveal or load a new image.")
            messagebox.showwarning("Too Many Mistakes",
                                   f"You made 3 mistakes!\n"
                                   f"Differences found so far: "
                                   f"{ImageProcessor.NUM_DIFFERENCES - self._state.remaining()}/5\n"
                                   f"Press 'Reveal All' or load a new image.")
        else:
            self._msg_var.set(f"❌ Miss! {self._state.MAX_MISTAKES - self._state.mistakes} tries left.")

    def _reveal_all(self):
        if self._processor.original_cv is None:
            return
        for i in range(ImageProcessor.NUM_DIFFERENCES):
            if not self._state.found[i]:
                self._draw_circle(i, 'blue')
        self._state.locked = True
        self._msg_var.set("All hidden differences have been revealed. Load another image to play again.")

    # ── helpers ─────────────────────────────
    def _refresh_canvases(self):
        """Redraw both canvases with the current (un-circled) images."""
        orig_tk, mod_tk = self._processor.get_display_images()
        self._orig_photo = orig_tk
        self._mod_photo  = mod_tk
        self._orig_canvas.delete("all")
        self._mod_canvas.delete("all")
        self._orig_canvas.create_image(0, 0, anchor=tk.NW, image=self._orig_photo)
        self._mod_canvas.create_image(0, 0, anchor=tk.NW, image=self._mod_photo)

    def _draw_circle(self, region_idx: int, colour: str):
        """
        Draw a persistent circle overlay on both canvases for the given region.
        Uses canvas oval items so we don't need to regenerate the PhotoImage
        for every circle.
        """
        rx, ry, rw, rh = self._processor.difference_regions[region_idx]
        cx, cy = rx + rw // 2, ry + rh // 2
        radius = max(rw, rh) // 2 + 10
        outline = 'red' if colour == 'red' else 'blue'
        width = 3

        o_item = self._orig_canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            outline=outline, width=width)
        m_item = self._mod_canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            outline=outline, width=width)
        self._circle_items[region_idx] = (o_item, m_item)

    def _update_status(self):
        self._remaining_var.set(f"Remaining: {self._state.remaining()}")
        self._mistakes_var.set(f"Mistakes: {self._state.mistakes} / {self._state.MAX_MISTAKES}")
        self._score_var.set(f"Score: {self._state.total_score}")


# ─────────────────────────────────────────────
# 4.  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = SpotTheDifferenceApp()
    app.mainloop()
