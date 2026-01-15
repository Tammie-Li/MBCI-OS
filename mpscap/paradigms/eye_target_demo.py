from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

try:
    import pygame
except Exception as _pg_err:  # noqa: N812
    pygame = None
    _PG_ERROR = _pg_err
else:
    _PG_ERROR = None

from ..core.drivers.eye_tracker import TobiiEyeTracker


class _Target:
    def __init__(self, x: int, y: int, radius: int, label: int) -> None:
        self.x = x
        self.y = y
        self.radius = radius
        self.label = label
        self.gaze_time = 0.0
        self.done = False
        self.active = False

    def update(self, gaze_px: Optional[Tuple[int, int]], dwell_sec: float, dt: float) -> bool:
        if self.done:
            return False
        if gaze_px is None:
            self.gaze_time = max(0.0, self.gaze_time - dt * 0.5)
            self.active = False
            return False
        dx = gaze_px[0] - self.x
        dy = gaze_px[1] - self.y
        dist = math.hypot(dx, dy)
        self.active = dist <= self.radius
        if self.active:
            self.gaze_time += dt
        else:
            self.gaze_time = max(0.0, self.gaze_time - dt * 0.5)
        if self.gaze_time >= dwell_sec:
            self.done = True
            return True
        return False


def _generate_targets(
    count: int,
    mode: str,
    screen_w: int,
    screen_h: int,
    radius: int,
) -> List[_Target]:
    mode = (mode or "random").lower()
    padding = radius * 2 + 20
    min_x, max_x = padding, screen_w - padding
    min_y, max_y = padding, screen_h - padding
    targets: List[_Target] = []

    if mode == "grid":
        cols = int(count**0.5) + 1
        rows = (count + cols - 1) // cols
        cell_w = (screen_w - 2 * padding) // max(1, cols)
        cell_h = (screen_h - 2 * padding) // max(1, rows)
        for idx in range(count):
            col = idx % cols
            row = idx // cols
            x = padding + col * cell_w + cell_w // 2
            y = padding + row * cell_h + cell_h // 2
            targets.append(_Target(x, y, radius, idx + 1))
    elif mode == "circle":
        center_x, center_y = screen_w // 2, screen_h // 2
        ring_r = min(screen_w, screen_h) // 3
        for idx, angle in enumerate([i * 2 * math.pi / max(1, count) for i in range(count)]):
            x = int(center_x + ring_r * math.cos(angle))
            y = int(center_y + ring_r * math.sin(angle))
            targets.append(_Target(x, y, radius, idx + 1))
    elif mode == "triangle":
        row = 0
        while len(targets) < count:
            num_in_row = row + 1
            start_x = screen_w // 2 - (num_in_row - 1) * (radius * 2 + 10)
            for col in range(num_in_row):
                if len(targets) >= count:
                    break
                x = start_x + col * (radius * 2 + 10)
                y = screen_h // 4 + row * (radius * 2 + 40)
                targets.append(_Target(x, y, radius, len(targets) + 1))
            row += 1
    else:
        attempts = 0
        while len(targets) < count and attempts < 1200:
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
            ok = True
            for t in targets:
                if math.hypot(x - t.x, y - t.y) < (radius * 2 + 20):
                    ok = False
                    break
            if ok:
                targets.append(_Target(x, y, radius, len(targets) + 1))
            attempts += 1
    return targets[:count]


def _draw_scene(
    screen: "pygame.Surface",
    targets: List[_Target],
    gaze_px: Optional[Tuple[int, int]],
    elapsed: float,
    dwell_sec: float,
) -> None:
    screen.fill((32, 32, 32))
    for x in range(0, screen.get_width(), 50):
        pygame.draw.line(screen, (60, 60, 60), (x, 0), (x, screen.get_height()))
    for y in range(0, screen.get_height(), 50):
        pygame.draw.line(screen, (60, 60, 60), (0, y), (screen.get_width(), y))

    for t in targets:
        if t.done:
            color = (80, 160, 80)
        elif t.active:
            color = (255, 120, 120)
        else:
            color = (100, 150, 255)
        pygame.draw.circle(screen, color, (t.x, t.y), t.radius)
        label_font = pygame.font.SysFont("simhei", 26)
        txt = label_font.render(str(t.label), True, (255, 255, 255))
        screen.blit(txt, txt.get_rect(center=(t.x, t.y)))

        progress = min(1.0, t.gaze_time / max(0.001, dwell_sec))
        if progress > 0 and not t.done:
            rect = (t.x - t.radius - 6, t.y - t.radius - 6, 2 * (t.radius + 6), 2 * (t.radius + 6))
            pygame.draw.arc(
                screen,
                (0, 220, 0),
                rect,
                -math.pi / 2,
                -math.pi / 2 + 2 * math.pi * progress,
                width=4,
            )

    info_font = pygame.font.SysFont("simhei", 22)
    info_txt = info_font.render(
        f"剩余目标: {len([t for t in targets if not t.done])} | 注视阈值 {dwell_sec:.2f}s | 耗时 {elapsed:.1f}s",
        True,
        (255, 255, 255),
    )
    screen.blit(info_txt, (16, 16))

    if gaze_px:
        pygame.draw.circle(screen, (255, 220, 0), gaze_px, 6)
    pygame.display.flip()


def _save_logs(
    save_dir: Optional[str],
    layout: str,
    target_count: int,
    dwell_time_sec: float,
    selection_times: List[Tuple[int, float]],
    gaze_trace: List[Tuple[float, int, int]],
    targets: List[_Target],
) -> None:
    base = Path(save_dir) if save_dir else Path.home() / "mpscap_eye"
    out_dir = base / "eye_target"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = time.strftime("%Y%m%d_%H%M%S")
    summary = out_dir / f"{ts_tag}_summary.txt"
    gaze_file = out_dir / f"{ts_tag}_gaze.csv"

    try:
        with open(summary, "w", encoding="utf-8") as fh:
            fh.write(f"layout={layout}\n")
            fh.write(f"target_count={target_count}\n")
            fh.write(f"dwell_time_sec={dwell_time_sec}\n")
            fh.write("target_positions:\n")
            for t in targets:
                fh.write(f"{t.label},{t.x},{t.y}\n")
            fh.write("selection_times(label,seconds_since_start):\n")
            for label, sec in selection_times:
                fh.write(f"{label},{sec:.3f}\n")
    except Exception as e:
        print(f"[EyeTarget] 写入 summary 失败: {e}")

    try:
        with open(gaze_file, "w", encoding="utf-8") as fh:
            fh.write("timestamp,x,y\n")
            for ts, x, y in gaze_trace:
                fh.write(f"{ts:.6f},{x},{y}\n")
    except Exception as e:
        print(f"[EyeTarget] 写入 gaze 轨迹失败: {e}")


def run_eye_target_demo(
    target_count: int = 10,
    dwell_time_sec: float = 0.8,
    layout: str = "random",
    trigger_cb: Optional[Callable[[int], None]] = None,
    save_dir: Optional[str] = None,
    screen_size: Tuple[int, int] = (1280, 720),
) -> None:
    """眼动目标消除范式，参考 examples/paradigm/EyeTrackerOnly.py，使用 Tobii 驱动获取注视点。"""
    if pygame is None:
        print(f"[EyeTarget] pygame 未安装或加载失败: {_PG_ERROR}")
        return

    try:
        eye = TobiiEyeTracker(filter_len=5)
    except Exception as e:
        print(f"[EyeTarget] 无法初始化眼动仪: {e}")
        return

    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("眼动目标消除")
    clock = pygame.time.Clock()
    radius = 50
    targets = _generate_targets(max(1, target_count), layout, screen_size[0], screen_size[1], radius)
    selection_times: List[Tuple[int, float]] = []
    gaze_trace: List[Tuple[float, int, int]] = []
    start_ts = time.time()

    try:
        running = True
        while running:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            gaze_px = None
            try:
                gx, gy = eye.latest_xy()
            except Exception:
                gx, gy = -1.0, -1.0
            if gx >= 0 and gy >= 0:
                gaze_px = (int(gx * screen_size[0]), int(gy * screen_size[1]))
                gaze_trace.append((time.time(), gaze_px[0], gaze_px[1]))

            for t in targets:
                if t.update(gaze_px, dwell_time_sec, dt):
                    selection_times.append((t.label, time.time() - start_ts))
                    if trigger_cb:
                        try:
                            trigger_cb(int(t.label))
                        except Exception:
                            pass

            if all(t.done for t in targets):
                running = False

            _draw_scene(screen, targets, gaze_px, time.time() - start_ts, dwell_time_sec)

        # 完成后短暂显示“任务完成”
        done_font = pygame.font.SysFont("simhei", 40)
        screen.fill((20, 20, 20))
        msg = done_font.render("任务完成，可关闭窗口", True, (255, 255, 255))
        screen.blit(msg, msg.get_rect(center=(screen_size[0] // 2, screen_size[1] // 2)))
        pygame.display.flip()
        time.sleep(1.5)
    finally:
        _save_logs(save_dir, layout, target_count, dwell_time_sec, selection_times, gaze_trace, targets)
        try:
            eye.stop()
        except Exception:
            pass
        pygame.quit()



