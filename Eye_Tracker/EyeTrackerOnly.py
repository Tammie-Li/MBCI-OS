import pygame
import random
import time
import os
import sys
import numpy as np
from core import TobiiPy, ROIDetector
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置 matplotlib 后端为非交互式后端，避免弹出窗口干扰 Pygame
import matplotlib
matplotlib.use('Agg')

# 设置支持中文的字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 初始化 Pygame
pygame.init()

# 屏幕参数
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("眼动目标消除系统 v5.0")

# 偏移参数（可独立调节）
ADJUSTMENT = {
    "x_offset_ratio": 2.4,  # 水平偏移比例（正值向右，负值向左）
    "y_offset_ratio": 2.0,  # 垂直偏移比例（正值向下，负值向上）
    "deadzone_radius": 60  # 中心缓冲区域半径
}

# 颜色配置
COLORS = {
    "background": (40, 40, 40),
    "grid": (60, 60, 60),
    "target": (100, 150, 255),
    "active": (255, 100, 100),
    "text": (255, 255, 255),
    "progress": (0, 255, 0)
}

# 高级参数
TARGET_RADIUS = 50 #目标大小
GAZE_THRESHOLD = 0.5 #注视阈值
TARGET_COUNT = 10  # 固定目标数量

# 数据保存路径
DATA_FOLDER = r"C:\Users\86187\Desktop\Eye_tracker\experiment_data"
TRACK_FOLDER = os.path.join(DATA_FOLDER, "轨迹")
INFO_FOLDER = os.path.join(DATA_FOLDER, "数据信息")

for folder in [TRACK_FOLDER, INFO_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)


class Target:
    def __init__(self, x, y, number):
        self.x = x
        self.y = y
        self.number = number
        self.radius = TARGET_RADIUS
        self.gaze_time = 0.0
        self.last_update = time.time()
        self.active = False
        self.has_been_recognized = False

        # 计算动态偏移量
        self.detect_center = (
            self.x + int(self.radius * ADJUSTMENT["x_offset_ratio"]),
            self.y + int(self.radius * ADJUSTMENT["y_offset_ratio"])
        )

    def update_gaze(self, gaze_pos):
        dx = gaze_pos[0] - self.detect_center[0]
        dy = gaze_pos[1] - self.detect_center[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5

        now = time.time()
        dt = now - self.last_update
        self.last_update = now

        # 分层检测机制
        if distance <= ADJUSTMENT["deadzone_radius"]:  # 中心缓冲区域
            self.gaze_time += dt * 1.5
        elif distance <= self.radius:
            self.gaze_time += dt
        else:
            self.gaze_time = max(0, self.gaze_time - dt * 3)

        self.active = distance <= self.radius
        if self.gaze_time >= GAZE_THRESHOLD:
            self.has_been_recognized = True
            return True
        return False

    def draw(self):
        color = COLORS["active"] if self.active else COLORS["target"]
        pygame.draw.circle(screen, color, (self.x, self.y), self.radius)

        # 进度环
        progress = min(self.gaze_time / GAZE_THRESHOLD, 1.0)
        if progress > 0:
            rect = (self.x - self.radius - 10, self.y - self.radius - 10,
                    2 * (self.radius + 10), 2 * (self.radius + 10))
            start_angle = -90
            end_angle = 360 * progress - 90
            pygame.draw.arc(screen, COLORS["progress"], rect,
                            start_angle / 180 * np.pi, end_angle / 180 * np.pi, 4)

        # 数字标签
        font = pygame.font.SysFont('simhei', 32, bold=True)
        text = font.render(str(self.number), True, COLORS["text"])
        text_rect = text.get_rect(center=(self.x, self.y - 2))
        screen.blit(text, text_rect)


def generate_targets(count, mode='random'):
    targets = []
    padding = TARGET_RADIUS * 2 + 20
    min_x = padding
    max_x = SCREEN_WIDTH - padding
    min_y = padding
    max_y = SCREEN_HEIGHT - padding

    if mode == 'random':
        max_attempts = 1000  # 最大尝试次数
        attempts = 0
        while len(targets) < count and attempts < max_attempts:
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
            valid = True
            for target in targets:
                dist = ((x - target.x) ** 2 + (y - target.y) ** 2) ** 0.5
                if dist < 2 * TARGET_RADIUS + 20:  # 确保目标不重叠
                    valid = False
                    break
            if valid:
                targets.append(Target(x, y, len(targets) + 1))
            attempts += 1

        if len(targets) < count:
            print("警告：未能在最大尝试次数内生成足够的目标。")

    elif mode == 'grid':
        cols = int(count ** 0.5) + 1
        rows = (count + cols - 1) // cols
        cell_width = (SCREEN_WIDTH - 2 * padding) // cols
        cell_height = (SCREEN_HEIGHT - 2 * padding) // rows
        for i in range(count):
            col = i % cols
            row = i // cols
            x = padding + col * cell_width + cell_width // 2
            y = padding + row * cell_height + cell_height // 2
            targets.append(Target(x, y, i + 1))

    elif mode == 'circle':
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        radius = min(SCREEN_WIDTH, SCREEN_HEIGHT) // 3
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for i in range(count):
            angle = angles[i]
            x = int(center_x + radius * np.cos(angle))
            y = int(center_y + radius * np.sin(angle))
            targets.append(Target(x, y, i + 1))

    elif mode == 'triangle':
        row = 0
        while len(targets) < count:
            num_in_row = row + 1
            start_x = SCREEN_WIDTH // 2 - (num_in_row - 1) * 60
            for col in range(num_in_row):
                if len(targets) >= count:
                    break
                x = start_x + col * 120
                y = SCREEN_HEIGHT // 4 + row * 100
                targets.append(Target(x, y, len(targets) + 1))
            row += 1

    return targets[:count]


class GameState:
    def __init__(self):
        self.current_mode = 'menu'
        self.targets = []
        self.initial_targets = []  # 保存初始目标位置
        self.current_number = 1
        self.order = 'asc'
        self.start_time = 0
        self.tobii = None
        self.game_mode = 'random'
        self.init_eyetracker()
        self.gaze_coordinates = []  # 保存原始注视坐标信息
        self.correct_count = 0  # 正确消除目标的数量
        self.total_recognition_count = 0  # 总识别次数
        self.target_selection_times = []  # 记录每个目标被选中的时间

    def init_eyetracker(self):
        print("正在初始化眼动仪...")
        try:
            self.tobii = TobiiPy(filter=5)
            time.sleep(1)
            print("眼动仪初始化成功")
        except Exception as e:
            print(f"眼动仪错误：{str(e)}")
            pygame.quit()
            sys.exit()

    def show_menu(self):
        screen.fill(COLORS["background"])
        font = pygame.font.SysFont('simhei', 48)
        small_font = pygame.font.SysFont('simhei', 32)

        mode_buttons = [
            {"text": "随机排列", "pos": (SCREEN_WIDTH // 4, 200), "mode": "random"},
            {"text": "网格排列", "pos": (SCREEN_WIDTH // 2, 200), "mode": "grid"},
            {"text": "圆形排列", "pos": (SCREEN_WIDTH * 3 // 4, 200), "mode": "circle"},
            {"text": "三角形排列", "pos": (SCREEN_WIDTH // 2, 300), "mode": "triangle"}
        ]

        order_buttons = [
            {"text": "升序模式", "pos": (SCREEN_WIDTH // 2 - 180, 550), "order": "asc"},
            {"text": "降序模式", "pos": (SCREEN_WIDTH // 2 + 180, 550), "order": "desc"}
        ]

        selected_mode = None

        while True:
            mouse_pos = pygame.mouse.get_pos()
            screen.fill(COLORS["background"])

            # 绘制标题
            title = font.render("选择目标排列模式", True, COLORS["text"])
            title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 80))
            screen.blit(title, title_rect)

            # 模式选择按钮
            for btn in mode_buttons:
                button_width = 220
                button_height = 60
                rect = pygame.Rect(0, 0, button_width, button_height)
                rect.center = btn["pos"]
                color = COLORS["active"] if rect.collidepoint(mouse_pos) else COLORS["target"]
                pygame.draw.rect(screen, color, rect, border_radius=10)
                text = small_font.render(btn["text"], True, COLORS["text"])
                text_rect = text.get_rect(center=btn["pos"])
                screen.blit(text, text_rect)

            # 顺序选择按钮
            if selected_mode:
                for btn in order_buttons:
                    button_width = 180
                    button_height = 50
                    rect = pygame.Rect(0, 0, button_width, button_height)
                    rect.center = btn["pos"]
                    color = COLORS["active"] if rect.collidepoint(mouse_pos) else COLORS["target"]
                    pygame.draw.rect(screen, color, rect, border_radius=8)
                    text = small_font.render(btn["text"], True, COLORS["text"])
                    text_rect = text.get_rect(center=btn["pos"])
                    screen.blit(text, text_rect)

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_game()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        # 检测模式选择
                        for btn in mode_buttons:
                            button_width = 220
                            button_height = 60
                            rect = pygame.Rect(0, 0, button_width, button_height)
                            rect.center = btn["pos"]
                            if rect.collidepoint(event.pos):
                                selected_mode = btn["mode"]

                        # 检测顺序选择
                        if selected_mode:
                            for btn in order_buttons:
                                button_width = 180
                                button_height = 50
                                rect = pygame.Rect(0, 0, button_width, button_height)
                                rect.center = btn["pos"]
                                if rect.collidepoint(event.pos):
                                    self.order = btn["order"]
                                    self.game_mode = selected_mode
                                    self.start_game()
                                    return

    def start_game(self):
        # 重置屏幕大小，确保每次开始界面状态一致
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.targets = generate_targets(TARGET_COUNT, self.game_mode)
        self.initial_targets = self.targets.copy()  # 保存初始目标
        self.current_number = 1 if self.order == 'asc' else len(self.targets)
        self.start_time = time.time()
        self.current_mode = 'game'
        self.gaze_coordinates = []  # 清空注视坐标信息
        self.correct_count = 0  # 重置正确消除目标的数量
        self.total_recognition_count = 0  # 重置总识别次数
        self.target_selection_times = []  # 清空目标选中时间记录
        self.main_loop()

    def main_loop(self):
        clock = pygame.time.Clock()
        while self.current_mode == 'game':
            dt = clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_game()

            gaze_pos = self.tobii.gazepos
            # 直接记录原始注视坐标
            if gaze_pos[0] > 0 and gaze_pos[1] > 0:
                self.gaze_coordinates.append(gaze_pos)

            current_target = next(
                (t for t in self.targets if t.number == self.current_number), None)

            targets_to_remove = []
            for target in self.targets:
                recognized = target.update_gaze(gaze_pos if gaze_pos[0] > 0 and gaze_pos[1] > 0 else (0, 0))
                if recognized:
                    self.total_recognition_count += 1
                    if target.number == self.current_number:
                        elapsed_time = time.time() - self.start_time
                        self.target_selection_times.append((target.number, elapsed_time))
                        targets_to_remove.append(target)
                        self.correct_count += 1
                        self.current_number += 1 if self.order == 'asc' else -1

            for target in targets_to_remove:
                if target in self.targets:
                    self.targets.remove(target)

            if not self.targets:
                self.save_experiment_data()
                self.show_result()
                return

            screen.fill(COLORS["background"])
            self.draw_grid()

            for target in self.targets:
                target.draw()

            # 显示信息
            elapsed = time.time() - self.start_time
            self.draw_text(f"时间: {elapsed:.1f}s", (20, 20))
            self.draw_text(f"模式: {self.game_mode}", (SCREEN_WIDTH - 200, 20))
            self.draw_text(f"总识别次数: {self.total_recognition_count}", (20, 50))

            pygame.display.flip()

    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH, 50):
            pygame.draw.line(screen, COLORS["grid"], (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, 50):
            pygame.draw.line(screen, COLORS["grid"], (0, y), (SCREEN_WIDTH, y))

    def draw_text(self, text, pos, size=32):
        font = pygame.font.SysFont('simhei', size)
        surface = font.render(text, True, COLORS["text"])
        screen.blit(surface, pos if isinstance(pos, tuple) else surface.get_rect(center=pos))

    def save_experiment_data(self):
        elapsed = time.time() - self.start_time
        accuracy = self.correct_count / self.total_recognition_count if self.total_recognition_count > 0 else 0

        # 保存注视坐标信息
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        data_file = os.path.join(INFO_FOLDER, f"{timestamp}_experiment.txt")
        with open(data_file, 'w') as f:
            f.write(f"任务模式: {self.game_mode}\n")
            f.write(f"任务顺序: {self.order}\n")
            f.write(f"花费时间: {elapsed:.2f}秒\n")
            f.write(f"正确消除目标数量: {self.correct_count}\n")
            f.write(f"总识别次数: {self.total_recognition_count}\n")
            f.write(f"识别正确率: {accuracy * 100:.2f}%\n")
            f.write(f"采用的检测阈值: {GAZE_THRESHOLD}秒\n")

            # 写入目标位置信息
            f.write("\n目标位置信息:\n")
            for target in self.initial_targets:
                f.write(f"目标序号: {target.number}, X坐标: {target.x}, Y坐标: {target.y}\n")

            # 写入目标选中时间信息
            f.write("\n目标选中时间信息（相对于任务开始）:\n")
            for number, t in self.target_selection_times:
                f.write(f"目标序号: {number}, 选中时间: {t:.2f}秒\n")

            # 写入注视坐标信息
            f.write("\n实时注视坐标信息:\n")
            for x, y in self.gaze_coordinates:
                f.write(f"{x},{y}\n")

    def show_result(self):
        elapsed = time.time() - self.start_time
        accuracy = self.correct_count / self.total_recognition_count if self.total_recognition_count > 0 else 0
        screen.fill(COLORS["background"])
        font = pygame.font.SysFont('simhei', 48)
        small_font = pygame.font.SysFont('simhei', 32)

        result_text = font.render("任务完成", True, COLORS["text"])
        result_rect = result_text.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(result_text, result_rect)

        time_text = small_font.render(f"花费时间: {elapsed:.2f}秒", True, COLORS["text"])
        time_rect = time_text.get_rect(center=(SCREEN_WIDTH // 2, 200))
        screen.blit(time_text, time_rect)

        accuracy_text = small_font.render(f"识别正确率: {accuracy * 100:.2f}%", True, COLORS["text"])
        accuracy_rect = accuracy_text.get_rect(center=(SCREEN_WIDTH // 2, 300))
        screen.blit(accuracy_text, accuracy_rect)

        retry_button = pygame.Rect(0, 0, 200, 60)
        retry_button.center = (SCREEN_WIDTH // 2, 450)
        pygame.draw.rect(screen, COLORS["target"], retry_button, border_radius=10)
        retry_text = small_font.render("重新开始", True, COLORS["text"])
        retry_text_rect = retry_text.get_rect(center=retry_button.center)
        screen.blit(retry_text, retry_text_rect)

        quit_button = pygame.Rect(0, 0, 200, 60)
        quit_button.center = (SCREEN_WIDTH // 2, 550)
        pygame.draw.rect(screen, COLORS["target"], quit_button, border_radius=10)
        quit_text = small_font.render("退 出", True, COLORS["text"])
        quit_text_rect = quit_text.get_rect(center=quit_button.center)
        screen.blit(quit_text, quit_text_rect)

        pygame.display.flip()

        # 绘制并保存轨迹图
        self.draw_gaze_trajectory()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quit_game()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if retry_button.collidepoint(event.pos):
                            self.show_menu()
                            return
                        elif quit_button.collidepoint(event.pos):
                            self.quit_game()

    def draw_gaze_trajectory(self):
        if not self.gaze_coordinates:
            return

        x_coords, y_coords = zip(*self.gaze_coordinates)
        # 计算坐标的均值
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)

        # 计算平移量
        x_shift = SCREEN_WIDTH / 2 - x_mean
        y_shift = SCREEN_HEIGHT / 2 - y_mean

        # 对坐标进行平移
        x_coords = [x + x_shift for x in x_coords]
        y_coords = [y + y_shift for y in y_coords]

        plt.figure(figsize=(10, 6))
        plt.plot(x_coords, y_coords, '-o', markersize=2, color='blue', alpha=0.7)
        plt.xlim(0, SCREEN_WIDTH)
        plt.ylim(0, SCREEN_HEIGHT)
        plt.gca().invert_yaxis()
        plt.title("眼动轨迹图")
        plt.xlabel("X 坐标")
        plt.ylabel("Y 坐标")

        # 保存轨迹图
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        track_file = os.path.join(TRACK_FOLDER, f"{timestamp}_gaze_trajectory.png")
        plt.savefig(track_file)
        plt.close()

    def filter_gaze_data(self, gaze_pos):
        if gaze_pos[0] > 0 and gaze_pos[1] > 0:
            return gaze_pos
        return None

    def quit_game(self):
        # 检查 TobiiPy 对象是否有 close 方法
        if hasattr(self.tobii, 'close'):
            self.tobii.close()
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = GameState()
    game.show_menu()
    