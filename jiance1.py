"""
药片检测计数系统 - 优化版
与训练端UI统一风格，修复加解密逻辑
"""

import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox, Listbox, Scrollbar, simpledialog, ttk, StringVar, IntVar
import numpy as np
from ultralytics import YOLO
import os
import threading
import time
import json
import shutil
from pathlib import Path
import hashlib
import logging
from datetime import datetime
from queue import Queue
import glob
from PIL import Image, ImageDraw, ImageFont
import pygame  # 用于播放提示音
import pyttsx3  # 用于语音合成
import random
import winsound  # Windows提示音
import subprocess
import tempfile
from pydub import AudioSegment
from pydub.playback import play
import io

# 配置日志 - 控制输出频率
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置外观
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


# ========== 音频管理器（修复版） ==========
class AudioManager:
    """音频管理器，使用事件回调确保连续播报"""

    def __init__(self):
        self.engine = None
        self.initialized = False
        self.message_queue = []  # 消息队列
        self.is_speaking = False  # 是否正在播报
        self._init_audio()
        self.speech_start_time = 0  # 记录语音开始时间
        self.last_log_time = 0  # 记录上次日志时间

    def _init_audio(self):
        """初始化音频系统"""
        try:
            # 初始化pygame用于播放提示音
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

            # 初始化语音合成引擎
            self.engine = pyttsx3.init()

            # 设置语音属性（中文支持）
            voices = self.engine.getProperty('voices')

            # 尝试设置中文语音
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.name.lower() or '中文' in voice.name:
                    self.engine.setProperty('voice', voice.id)
                    break
                elif 'Microsoft Huihui' in voice.name or 'Microsoft Zira' in voice.name:
                    # Windows系统中文语音
                    self.engine.setProperty('voice', voice.id)

            # 设置语速和音量
            self.engine.setProperty('rate', 160)  # 稍慢一点的语速
            self.engine.setProperty('volume', 0.9)  # 音量

            # 连接结束事件
            self.engine.connect('finished-utterance', self._on_speech_end)

            self.initialized = True
            logger.info("音频系统初始化成功")

        except Exception as e:
            logger.error(f"音频系统初始化失败: {e}")
            self.initialized = False

    def _on_speech_end(self, name, completed):
        """语音播报结束回调"""
        self.is_speaking = False

        # 如果队列中有消息，播放下一条
        if self.message_queue:
            time.sleep(0.5)  # 给一点间隔时间
            self._process_next_message()

    def _process_next_message(self):
        """处理下一条消息"""
        if self.message_queue and not self.is_speaking:
            next_message = self.message_queue.pop(0)
            self._speak_direct(next_message)

    def _speak_direct(self, text):
        """直接播报文本（内部方法）"""
        if not self.initialized or not self.engine:
            return

        try:
            self.is_speaking = True
            self.speech_start_time = time.time()
            self.engine.say(text, text)
            self.engine.runAndWait()
            current_time = time.time()
            # 控制日志输出频率（每5秒输出一次）
            if current_time - self.last_log_time > 5:
                logger.info(f"语音播报: {text}")
                self.last_log_time = current_time
        except Exception as e:
            logger.error(f"语音播报失败: {e}")
            self.is_speaking = False

    def play_beep(self):
        """播放提示音"""
        try:
            # 使用winsound播放标准提示音
            winsound.Beep(1000, 300)  # 频率1000Hz，持续300ms
            current_time = time.time()
            # 控制日志输出频率
            if current_time - self.last_log_time > 5:
                logger.info("播放提示音")
                self.last_log_time = current_time
        except Exception as e:
            logger.error(f"播放提示音失败: {e}")

    def speak(self, text):
        """语音播报文本 - 使用队列机制，允许相同内容连续播报"""
        if not self.initialized or not self.engine:
            logger.warning("音频系统未初始化")
            return

        if not text or len(text.strip()) == 0:
            return

        # 将消息加入队列
        self.message_queue.append(text)

        # 控制日志输出频率
        current_time = time.time()
        if current_time - self.last_log_time > 5:
            logger.info(f"消息加入队列: {text} (队列长度: {len(self.message_queue)})")
            self.last_log_time = current_time

        # 如果当前没有在播报，立即开始
        if not self.is_speaking:
            self._process_next_message()
        else:
            # 控制日志输出频率
            if current_time - self.last_log_time > 5:
                logger.info(f"当前正在播报，消息排队等待")

    def stop(self):
        """停止音频系统"""
        try:
            if self.engine:
                self.engine.stop()
            pygame.mixer.quit()
            self.message_queue.clear()
            self.is_speaking = False
            logger.info("音频系统已停止")
        except Exception as e:
            logger.error(f"停止音频系统失败: {e}")


# ========== 模型加解密处理器（与训练端一致） ==========
class RPModelHandler:
    """RP模型加密解密处理器 - 与训练端保持一致"""

    HEADER = b"PILL_MODEL_RP_2026"  # 保持与训练端相同的文件头
    KEY = 0x5A

    @staticmethod
    def encrypt_model(pt_path, rp_path):
        """加密模型文件"""
        try:
            with open(pt_path, 'rb') as f:
                model_data = f.read()
                md5 = hashlib.md5(model_data).digest()

            # 简单异或加密
            encrypted_data = bytes([b ^ RPModelHandler.KEY for b in model_data])

            with open(rp_path, 'wb') as f:
                f.write(RPModelHandler.HEADER + md5 + encrypted_data)

            logger.info(f"模型加密成功: {pt_path} -> {rp_path}")
            return True

        except Exception as e:
            logger.error(f"模型加密失败: {e}")
            return False

    @staticmethod
    def decrypt_model(rp_path, pt_path):
        """解密模型文件"""
        try:
            with open(rp_path, 'rb') as f:
                header = f.read(16)
                if header != RPModelHandler.HEADER:
                    logger.error("无效的模型文件头")
                    return False

                md5 = f.read(16)
                encrypted_data = f.read()

            # 解密
            model_data = bytes([b ^ RPModelHandler.KEY for b in encrypted_data])

            # 校验完整性
            if hashlib.md5(model_data).digest() != md5:
                logger.error("模型文件校验失败")
                return False

            with open(pt_path, 'wb') as f:
                f.write(model_data)

            logger.info(f"模型解密成功: {rp_path} -> {pt_path}")
            return True

        except Exception as e:
            logger.error(f"模型解密失败: {e}")
            return False


class CameraThread(threading.Thread):
    """摄像头线程，使用队列安全传递帧"""

    def __init__(self, camera_index, width, height):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.running = True
        self.frame_queue = Queue(maxsize=1)
        self.camera = None
        self.last_log_time = 0  # 控制日志输出频率

    def run(self):
        """线程主函数"""
        try:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

            for backend in backends:
                try:
                    self.camera = cv2.VideoCapture(self.camera_index, backend)
                    if self.camera.isOpened():
                        break
                except:
                    continue

            if not self.camera or not self.camera.isOpened():
                logger.error(f"无法打开摄像头 {self.camera_index}")
                return

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)

            logger.info(f"摄像头 {self.camera_index} 启动成功")

            while self.running:
                ret, frame = self.camera.read()
                if ret:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    self.frame_queue.put(frame.copy())

                cv2.waitKey(1)

        except Exception as e:
            logger.error(f"摄像头线程错误: {e}")
        finally:
            self.stop()

    def get_frame(self):
        """获取最新帧"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None

    def stop(self):
        """停止线程"""
        self.running = False
        if self.camera:
            self.camera.release()
        current_time = time.time()
        if current_time - self.last_log_time > 10:
            logger.info(f"摄像头 {self.camera_index} 已停止")
            self.last_log_time = current_time


class DraggablePreview:
    """可拖动的摄像头预览窗口"""

    def __init__(self, parent, width, height):
        self.parent = parent
        self.width = width
        self.height = height

        # 创建顶层窗口作为悬浮窗
        self.window = ctk.CTkToplevel(parent)
        self.window.title("摄像头预览")
        self.window.geometry(f"{width}x{height}")
        self.window.overrideredirect(True)
        self.window.attributes('-topmost', True)
        self.window.configure(fg_color="#f0f0f0")

        # 设置初始位置
        parent.update_idletasks()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        if parent_width > width and parent_height > height:
            x = parent.winfo_x() + parent_width - width - 20
            y = parent.winfo_y() + parent_height - height - 20
            self.window.geometry(f"{width}x{height}+{x}+{y}")

        # 标题栏
        title_frame = ctk.CTkFrame(self.window, height=30, fg_color="#4a90e2")
        title_frame.pack(fill="x", padx=0, pady=0)

        ctk.CTkLabel(
            title_frame,
            text="📷 摄像头预览",
            text_color="white",
            font=("Arial", 12, "bold")
        ).pack(side="left", padx=10, pady=5)

        # 关闭按钮
        close_btn = ctk.CTkButton(
            title_frame,
            text="×",
            width=30,
            height=30,
            fg_color="transparent",
            hover_color="#e81123",
            command=self.hide
        )
        close_btn.pack(side="right", padx=5, pady=0)

        # 预览画布
        self.canvas = ctk.CTkCanvas(
            self.window,
            width=width,
            height=height - 30,
            bg="#000000",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True, padx=0, pady=0)

        # 绑定拖动事件
        title_frame.bind("<ButtonPress-1>", self.start_drag)
        title_frame.bind("<B1-Motion>", self.on_drag)

        # 保存拖动变量
        self.drag_data = {"x": 0, "y": 0}

        # 默认隐藏
        self.window.withdraw()

    def start_drag(self, event):
        """开始拖动"""
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def on_drag(self, event):
        """处理拖动"""
        delta_x = event.x - self.drag_data["x"]
        delta_y = event.y - self.drag_data["y"]

        x = self.window.winfo_x() + delta_x
        y = self.window.winfo_y() + delta_y

        self.window.geometry(f"+{x}+{y}")

    def update_preview(self, frame):
        """更新预览图像"""
        if frame is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.width, self.height - 30))

                pil_img = Image.fromarray(frame_resized)
                preview_photo = ctk.CTkImage(light_image=pil_img, size=(self.width, self.height - 30))

                self.canvas.delete("all")
                # 注意：CTkCanvas不支持直接显示CTkImage，这里保持原方式
                from PIL import ImageTk
                tk_img = ImageTk.PhotoImage(image=pil_img)
                self.canvas.create_image(0, 0, image=tk_img, anchor="nw")
                self.canvas.tk_img = tk_img  # 保持引用
            except Exception as e:
                logger.error(f"更新预览错误: {e}")

    def show(self):
        """显示预览窗口"""
        self.window.deiconify()
        self.window.lift()

    def hide(self):
        """隐藏预览窗口"""
        self.window.withdraw()

    def toggle(self):
        """切换显示/隐藏"""
        if self.window.state() == "normal":
            self.hide()
        else:
            self.show()

    def destroy(self):
        """销毁窗口"""
        if self.window:
            self.window.destroy()


class WatermarkGenerator:
    """水印生成器 - 支持中文和位置拖动"""

    def __init__(self):
        self.custom_text = "药片检测系统"
        self.max_pill_count = 0
        self.max_pill_frame = 0
        self.frame_count = 0
        self.current_pill_count = 0  # 添加当前片数变量

        # 目标片数设置
        self.target_pills = 0
        self.target_reached = False
        self.target_start_time = 0
        self.target_stable_seconds = 0
        self.success_message_shown = False
        self.notification_triggered = False  # 防止重复触发
        self.success_timestamp = ""  # 成功时间戳

        # 水印位置 (x, y) - 百分比坐标 (0.0-1.0)
        self.position_x = 0.02  # 2% from left
        self.position_y = 0.02  # 2% from top

        # 字体设置
        self._init_fonts()

        # 是否正在拖动
        self.dragging = False
        self.drag_start_pos = (0, 0)

        # 控制日志输出频率
        self.last_log_time = 0

    def _init_fonts(self):
        """初始化字体 - 解决中文乱码问题"""
        try:
            # 尝试加载系统中文字体
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
            ]

            for font_path in font_paths:
                if os.path.exists(font_path):
                    # 大字体用于主标题
                    self.title_font = ImageFont.truetype(font_path, 24)
                    # 小字体用于详细信息
                    self.info_font = ImageFont.truetype(font_path, 20)
                    self.small_font = ImageFont.truetype(font_path, 16)
                    logger.info(f"已加载字体: {font_path}")
                    return

            # 如果没有找到系统字体，使用默认字体
            self.title_font = ImageFont.load_default()
            self.info_font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
            logger.warning("未找到中文字体，使用默认字体")

        except Exception as e:
            logger.error(f"加载字体失败: {e}")
            self.title_font = ImageFont.load_default()
            self.info_font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    def set_custom_text(self, text):
        """设置自定义文本"""
        self.custom_text = text

    def set_target_pills(self, count):
        """设置目标片数"""
        self.target_pills = count
        self.target_reached = False
        self.success_message_shown = False
        self.notification_triggered = False
        self.success_timestamp = ""

    def set_position(self, x_percent, y_percent):
        """设置水印位置（百分比坐标）"""
        self.position_x = max(0.0, min(1.0, x_percent))
        self.position_y = max(0.0, min(1.0, y_percent))

    def update_stats(self, pill_count):
        """更新统计信息 - 必须严格等于目标片数"""
        self.frame_count += 1
        self.current_pill_count = pill_count  # 更新当前片数

        # 更新最大片数
        if pill_count > self.max_pill_count:
            self.max_pill_count = pill_count
            self.max_pill_frame = self.frame_count

        # 检查目标片数（必须严格等于目标片数）
        if self.target_pills > 0:
            if pill_count == self.target_pills:  # 严格等于
                if not self.target_reached:
                    # 第一次达到目标片数
                    self.target_reached = True
                    self.target_start_time = time.time()
                    self.target_stable_seconds = random.randint(2, 5)  # 随机2-5秒，缩短时间
                    self.notification_triggered = False  # 重置触发标志
                    self.success_message_shown = False
                    current_time = time.time()
                    if current_time - self.last_log_time > 10:
                        logger.info(f"达到目标片数 {self.target_pills}，需要稳定 {self.target_stable_seconds} 秒")
                        self.last_log_time = current_time
                elif not self.success_message_shown and not self.notification_triggered:
                    # 检查是否稳定足够时间
                    elapsed = time.time() - self.target_start_time
                    if elapsed >= self.target_stable_seconds:
                        # 稳定时间足够，可以触发通知
                        current_time = time.time()
                        if current_time - self.last_log_time > 10:
                            logger.info(f"稳定时间到达 {elapsed:.1f}秒 >= {self.target_stable_seconds}秒，触发通知")
                            self.last_log_time = current_time
                        return True  # 触发成功提示
            else:
                # 片数不等于目标片数，重置状态
                if self.target_reached and not self.success_message_shown:
                    current_time = time.time()
                    if current_time - self.last_log_time > 10:
                        logger.info(f"片数变化: {pill_count} != {self.target_pills}，重置稳定计时")
                        self.last_log_time = current_time
                self.target_reached = False
                self.success_message_shown = False
                self.notification_triggered = False
                self.target_start_time = 0
                self.success_timestamp = ""
        else:
            # 目标片数为0，禁用此功能
            self.target_reached = False
            self.success_message_shown = False
            self.notification_triggered = False
            self.success_timestamp = ""

        return False

    def mark_notification_triggered(self):
        """标记通知已触发"""
        self.notification_triggered = True
        self.success_message_shown = True
        self.success_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = time.time()
        if current_time - self.last_log_time > 10:
            logger.info(f"通知已触发，成功时间: {self.success_timestamp}")
            self.last_log_time = current_time

    def reset_stats(self):
        """重置统计"""
        self.max_pill_count = 0
        self.max_pill_frame = 0
        self.frame_count = 0
        self.current_pill_count = 0
        self.target_reached = False
        self.success_message_shown = False
        self.notification_triggered = False
        self.success_timestamp = ""
        current_time = time.time()
        if current_time - self.last_log_time > 10:
            logger.info("水印统计已重置")
            self.last_log_time = current_time

    def start_drag(self, x, y, frame_width, frame_height):
        """开始拖动水印"""
        self.dragging = True
        self.drag_start_pos = (x, y)

    def update_drag(self, x, y, frame_width, frame_height):
        """更新拖动位置"""
        if self.dragging:
            dx = x - self.drag_start_pos[0]
            dy = y - self.drag_start_pos[1]

            # 转换为百分比坐标
            dx_percent = dx / frame_width
            dy_percent = dy / frame_height

            self.position_x = max(0.0, min(1.0, self.position_x + dx_percent))
            self.position_y = max(0.0, min(1.0, self.position_y + dy_percent))

            self.drag_start_pos = (x, y)

    def end_drag(self):
        """结束拖动"""
        self.dragging = False

    def add_watermark(self, frame, show_drag_rect=False):
        """为帧添加水印（支持中文）"""
        try:
            height, width = frame.shape[:2]

            # 将OpenCV图像转换为PIL图像以便绘制中文
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img, 'RGBA')

            # 获取当前时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 水印信息 - 修正：使用当前片数而不是最大片数
            info_lines = [
                f"时间: {current_time}",
                f"当前片数: {self.current_pill_count}",
                f"最高片数: {self.max_pill_count} (第{self.max_pill_frame}帧)",
            ]

            # 添加目标片数信息
            if self.target_pills > 0:
                info_lines.append(f"目标片数: {self.target_pills}")
                if self.target_reached and not self.success_message_shown:
                    elapsed = time.time() - self.target_start_time
                    remaining = max(0, self.target_stable_seconds - elapsed)
                    info_lines.append(f"稳定倒计时: {remaining:.1f}秒")
                elif self.success_message_shown:
                    info_lines.append(f"{self.custom_text}{self.target_pills}片发药成功")
                    info_lines.append(f"成功时间: {self.success_timestamp}")

            info_lines.append(f"{self.custom_text}")

            # 计算水印位置（像素坐标）
            pos_x = int(width * self.position_x)
            pos_y = int(height * self.position_y)

            # 计算水印背景大小
            max_line_width = 0
            total_height = 0
            line_heights = []

            for i, line in enumerate(info_lines):
                if i == 0:  # 时间行
                    bbox = draw.textbbox((0, 0), line, font=self.info_font)
                elif i >= len(info_lines) - 2:  # 最后两行
                    bbox = draw.textbbox((0, 0), line, font=self.title_font)
                else:  # 其他信息行
                    bbox = draw.textbbox((0, 0), line, font=self.small_font)

                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]

                max_line_width = max(max_line_width, line_width)
                line_heights.append(line_height)
                total_height += line_height + 5  # 5px行间距

            # 添加水印背景（半透明黑色）
            bg_width = max_line_width + 20
            bg_height = total_height + 20

            # 确保水印在图像范围内
            if pos_x + bg_width > width:
                pos_x = width - bg_width - 10
            if pos_y + bg_height > height:
                pos_y = height - bg_height - 10
            if pos_x < 0:
                pos_x = 10
            if pos_y < 0:
                pos_y = 10

            # 绘制半透明背景
            bg_color = (0, 0, 0, 180)  # 半透明黑色
            draw.rectangle(
                [pos_x, pos_y, pos_x + bg_width, pos_y + bg_height],
                fill=bg_color
            )

            # 绘制拖动指示框（如果正在拖动）
            if show_drag_rect:
                draw.rectangle(
                    [pos_x - 2, pos_y - 2, pos_x + bg_width + 2, pos_y + bg_height + 2],
                    outline=(255, 0, 0, 255),
                    width=2
                )
                # 绘制拖动提示文本
                draw.text((pos_x + 5, pos_y + bg_height + 5),
                          "拖动水印位置",
                          font=self.small_font,
                          fill=(255, 0, 0, 255))

            # 绘制文本
            current_y = pos_y + 10
            for i, line in enumerate(info_lines):
                if i == 0:  # 时间行
                    font = self.info_font
                    color = (255, 255, 0, 255)  # 黄色
                elif i >= len(info_lines) - 2:  # 最后两行
                    font = self.title_font
                    if "发药成功" in line:
                        color = (0, 255, 0, 255)  # 成功消息用绿色
                    else:
                        color = (255, 255, 255, 255)  # 白色
                elif self.target_pills > 0 and i == len(info_lines) - 4:  # 目标片数行
                    font = self.small_font
                    if self.target_reached and not self.success_message_shown:
                        color = (255, 165, 0, 255)  # 橙色（稳定中）
                    elif self.success_message_shown:
                        color = (0, 255, 0, 255)  # 绿色（成功）
                    else:
                        color = (255, 255, 255, 255)  # 白色
                elif self.target_reached and not self.success_message_shown and i == len(info_lines) - 3:  # 倒计时行
                    font = self.small_font
                    color = (255, 165, 0, 255)  # 橙色
                else:  # 其他信息行
                    font = self.small_font
                    color = (200, 200, 200, 255)  # 浅灰色

                draw.text((pos_x + 10, current_y), line, font=font, fill=color)
                current_y += line_heights[i] + 5

            # 转换回OpenCV图像
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            return frame

        except Exception as e:
            logger.error(f"添加水印错误: {e}")
            return frame


class WatermarkPositionDialog(ctk.CTkToplevel):
    """水印位置设置对话框 - 可调整大小版本"""

    def __init__(self, parent, watermark_generator):
        super().__init__(parent)
        self.watermark = watermark_generator
        self.parent = parent

        self.title("水印位置设置")
        self.geometry("450x400")  # 增加窗口大小
        self.minsize(450, 400)  # 设置最小大小
        self.resizable(True, True)  # 允许调整大小

        # 使对话框模态
        self.transient(parent)
        self.grab_set()

        # 配置网格权重
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._setup_ui()

    def _setup_ui(self):
        """设置对话框UI"""
        # 创建主容器
        main_container = ctk.CTkScrollableFrame(self)
        main_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        main_container.grid_columnconfigure(0, weight=1)

        # 标题
        ctk.CTkLabel(
            main_container,
            text="⚙️ 水印位置设置",
            font=("Arial", 16, "bold")
        ).pack(pady=(0, 20))

        # 位置设置框架
        pos_frame = ctk.CTkFrame(main_container)
        pos_frame.pack(fill="x", padx=10, pady=10)

        # X轴位置
        x_frame = ctk.CTkFrame(pos_frame)
        x_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(x_frame, text="水平位置 (左← →右):", font=("Arial", 12)).pack(anchor="w", pady=(0, 5))

        x_slider_frame = ctk.CTkFrame(x_frame)
        x_slider_frame.pack(fill="x", pady=5)
        x_slider_frame.grid_columnconfigure(0, weight=1)
        x_slider_frame.grid_columnconfigure(1, weight=0)

        self.x_slider = ctk.CTkSlider(
            x_slider_frame,
            from_=0,
            to=100,
            command=self._update_position,
            number_of_steps=100
        )
        self.x_slider.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="ew")
        self.x_slider.set(self.watermark.position_x * 100)

        self.x_value_label = ctk.CTkLabel(x_slider_frame, text=f"{self.watermark.position_x * 100:.1f}%", width=60)
        self.x_value_label.grid(row=0, column=1, pady=5)

        # Y轴位置
        y_frame = ctk.CTkFrame(pos_frame)
        y_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(y_frame, text="垂直位置 (上← →下):", font=("Arial", 12)).pack(anchor="w", pady=(0, 5))

        y_slider_frame = ctk.CTkFrame(y_frame)
        y_slider_frame.pack(fill="x", pady=5)
        y_slider_frame.grid_columnconfigure(0, weight=1)
        y_slider_frame.grid_columnconfigure(1, weight=0)

        self.y_slider = ctk.CTkSlider(
            y_slider_frame,
            from_=0,
            to=100,
            command=self._update_position,
            number_of_steps=100
        )
        self.y_slider.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="ew")
        self.y_slider.set(self.watermark.position_y * 100)

        self.y_value_label = ctk.CTkLabel(y_slider_frame, text=f"{self.watermark.position_y * 100:.1f}%", width=60)
        self.y_value_label.grid(row=0, column=1, pady=5)

        # 预设位置按钮
        preset_frame = ctk.CTkFrame(main_container)
        preset_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(preset_frame, text="预设位置:", font=("Arial", 12)).pack(anchor="w", pady=(0, 5))

        # 第一行按钮
        btn_frame1 = ctk.CTkFrame(preset_frame)
        btn_frame1.pack(fill="x", pady=5)

        presets1 = [
            ("↖ 左上", 0.02, 0.02),
            ("↗ 右上", 0.98, 0.02),
        ]

        for text, x, y in presets1:
            btn = ctk.CTkButton(
                btn_frame1,
                text=text,
                width=100,
                command=lambda x=x, y=y: self._set_preset_position(x, y)
            )
            btn.pack(side="left", padx=5)

        # 第二行按钮
        btn_frame2 = ctk.CTkFrame(preset_frame)
        btn_frame2.pack(fill="x", pady=5)

        presets2 = [
            ("↙ 左下", 0.02, 0.98),
            ("↘ 右下", 0.98, 0.98),
            ("◎ 居中", 0.5, 0.5),
        ]

        for text, x, y in presets2:
            btn = ctk.CTkButton(
                btn_frame2,
                text=text,
                width=100,
                command=lambda x=x, y=y: self._set_preset_position(x, y)
            )
            btn.pack(side="left", padx=5)

        # 手动输入位置
        manual_frame = ctk.CTkFrame(main_container)
        manual_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(manual_frame, text="手动输入位置:", font=("Arial", 12)).pack(anchor="w", pady=(0, 5))

        input_frame = ctk.CTkFrame(manual_frame)
        input_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(input_frame, text="X:", width=20).pack(side="left", padx=(5, 2))
        self.x_entry = ctk.CTkEntry(input_frame, width=80, placeholder_text="0-100")
        self.x_entry.pack(side="left", padx=2)
        ctk.CTkLabel(input_frame, text="%").pack(side="left", padx=(0, 10))

        ctk.CTkLabel(input_frame, text="Y:", width=20).pack(side="left", padx=(10, 2))
        self.y_entry = ctk.CTkEntry(input_frame, width=80, placeholder_text="0-100")
        self.y_entry.pack(side="left", padx=2)
        ctk.CTkLabel(input_frame, text="%").pack(side="left", padx=(0, 10))

        ctk.CTkButton(
            input_frame,
            text="应用",
            width=60,
            command=self._apply_manual_position
        ).pack(side="left", padx=10)

        # 拖动模式复选框
        self.drag_mode_var = ctk.BooleanVar(value=False)
        drag_frame = ctk.CTkFrame(main_container)
        drag_frame.pack(fill="x", padx=10, pady=10)

        drag_check = ctk.CTkCheckBox(
            drag_frame,
            text="启用鼠标拖动模式",
            variable=self.drag_mode_var,
            command=self._toggle_drag_mode,
            font=("Arial", 12)
        )
        drag_check.pack(anchor="w", pady=5)

        ctk.CTkLabel(
            drag_frame,
            text="提示：启用后可在视频画布上直接拖动水印",
            font=("Arial", 10),
            text_color="#666666"
        ).pack(anchor="w", pady=(0, 5))

        # 当前坐标显示
        self.pos_label = ctk.CTkLabel(
            main_container,
            text=f"当前位置: X={self.watermark.position_x * 100:.1f}%, Y={self.watermark.position_y * 100:.1f}%",
            font=("Arial", 11, "bold")
        )
        self.pos_label.pack(pady=10)

        # 操作按钮框架
        button_frame = ctk.CTkFrame(main_container)
        button_frame.pack(fill="x", padx=10, pady=(10, 0))

        ctk.CTkButton(
            button_frame,
            text="💾 保存设置",
            command=self._save_settings,
            width=100,
            height=35
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="🔄 重置位置",
            command=self._reset_position,
            width=100,
            height=35
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            button_frame,
            text="关闭",
            command=self.destroy,
            width=100,
            height=35
        ).pack(side="right", padx=5)

    def _update_position(self, value=None):
        """更新位置"""
        x_percent = self.x_slider.get() / 100.0
        y_percent = self.y_slider.get() / 100.0

        self.watermark.set_position(x_percent, y_percent)
        self.x_value_label.configure(text=f"{x_percent * 100:.1f}%")
        self.y_value_label.configure(text=f"{y_percent * 100:.1f}%")
        self.pos_label.configure(
            text=f"当前位置: X={x_percent * 100:.1f}%, Y={y_percent * 100:.1f}%"
        )

        # 更新手动输入框
        self.x_entry.delete(0, "end")
        self.x_entry.insert(0, f"{x_percent * 100:.1f}")
        self.y_entry.delete(0, "end")
        self.y_entry.insert(0, f"{y_percent * 100:.1f}")

    def _set_preset_position(self, x, y):
        """设置预设位置"""
        self.x_slider.set(x * 100)
        self.y_slider.set(y * 100)
        self._update_position()

    def _apply_manual_position(self):
        """应用手动输入的位置"""
        try:
            x_str = self.x_entry.get().strip()
            y_str = self.y_entry.get().strip()

            if x_str and y_str:
                x_percent = float(x_str) / 100.0
                y_percent = float(y_str) / 100.0

                # 限制范围
                x_percent = max(0.0, min(1.0, x_percent))
                y_percent = max(0.0, min(1.0, y_percent))

                self.x_slider.set(x_percent * 100)
                self.y_slider.set(y_percent * 100)
                self._update_position()

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    def _reset_position(self):
        """重置位置到默认"""
        self.watermark.set_position(0.02, 0.02)
        self.x_slider.set(2.0)
        self.y_slider.set(2.0)
        self._update_position()

    def _save_settings(self):
        """保存设置"""
        self.parent._save_watermark_config()
        messagebox.showinfo("成功", "水印设置已保存")

    def _toggle_drag_mode(self):
        """切换拖动模式"""
        if self.drag_mode_var.get():
            self.parent.enable_watermark_drag(True)
        else:
            self.parent.enable_watermark_drag(False)


class PillDetectorApp(ctk.CTk):
    """药片检测计数系统 - 主应用类"""

    def __init__(self):
        super().__init__()

        # 应用基础设置
        self.title("药片检测计数系统")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # 核心变量
        self.camera_thread = None
        self.current_model = None
        self.current_model_name = ""
        self.models = {}  # 模型配置 {name: path}
        self.detecting = False
        self.recording = False
        self.video_writer = None
        self.save_dir = ""
        self.current_frame = None
        self.temp_models_dir = Path.home() / "PillDetectorTemp"
        self.temp_models_dir.mkdir(exist_ok=True)

        # 控制日志输出频率
        self.last_detection_log_time = 0  # 记录上次检测日志时间
        self.last_frame_count = 0  # 记录上次帧数
        self.frame_counter = 0  # 帧计数器

        # 音频管理器（修复版）
        self.audio_manager = AudioManager()

        # 水印生成器
        self.watermark = WatermarkGenerator()
        self.watermark_drag_enabled = False

        # 水印位置对话框
        self.watermark_dialog = None

        # 目标片数相关
        self.target_pills_var = IntVar(value=0)

        # 可拖动预览窗口
        self.preview_window = None

        # 配置文件路径
        self.config_dir = Path.home() / "PillDetectorConfig"
        self.config_dir.mkdir(exist_ok=True)
        self.models_config_path = self.config_dir / "models.json"
        self.watermark_config_path = self.config_dir / "watermark.json"
        self.target_config_path = self.config_dir / "target.json"

        # 初始化UI - 必须先初始化UI再加载配置
        self._setup_ui()

        # 加载已有配置
        self._load_models_config()
        self._load_watermark_config()
        self._load_target_config()

        # 绑定键盘事件
        self._bind_keyboard_events()

        # 绑定鼠标事件
        self._bind_mouse_events()

        # 启动时自动刷新模型列表
        self.after(100, self._auto_load_models)

        # 设置退出时清理
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        """设置UI界面 - 与训练端统一风格"""
        # 创建分页面框架
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # 检测页面
        self.detect_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.detect_frame, text="🔍 实时检测")
        self._setup_detect_page()

        # 模型管理页面
        self.models_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.models_frame, text="📦 模型管理")
        self._setup_models_page()

        # 设置页面
        self.settings_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.settings_frame, text="⚙️ 系统设置")
        self._setup_settings_page()

    def _setup_detect_page(self):
        """设置检测页面"""
        # 使用网格布局
        self.detect_frame.grid_columnconfigure(0, weight=3)
        self.detect_frame.grid_columnconfigure(1, weight=1)
        self.detect_frame.grid_rowconfigure(0, weight=1)

        # ========== 左侧：视频预览区 ==========
        left_frame = ctk.CTkFrame(self.detect_frame)
        left_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1)

        # 顶部控制区
        top_ctrl_frame = ctk.CTkFrame(left_frame, height=70)
        top_ctrl_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # 摄像头控制
        cam_frame = ctk.CTkFrame(top_ctrl_frame)
        cam_frame.pack(side="left", padx=5, pady=2)

        ctk.CTkLabel(cam_frame, text="摄像头：").pack(side="left", padx=(5, 2), pady=2)

        # 检测可用摄像头
        self.available_cameras = self._detect_cameras()
        cam_options = [f"摄像头 {i}" for i in self.available_cameras] if self.available_cameras else ["无可用摄像头"]
        self.cam_combo = ctk.CTkComboBox(
            cam_frame,
            values=cam_options,
            state="normal" if cam_options else "disabled",
            width=120
        )
        self.cam_combo.pack(side="left", padx=2, pady=2)
        if cam_options:
            self.cam_combo.set(cam_options[0])

        self.cam_btn = ctk.CTkButton(
            cam_frame,
            text="打开摄像头",
            command=self._toggle_camera,
            width=100
        )
        self.cam_btn.pack(side="left", padx=5, pady=2)

        # 显示/隐藏预览按钮
        self.preview_toggle_btn = ctk.CTkButton(
            cam_frame,
            text="📷 显示预览",
            command=self._toggle_preview_window,
            width=100,
            state="disabled"
        )
        self.preview_toggle_btn.pack(side="left", padx=5, pady=2)

        # 当前模型显示
        model_display_frame = ctk.CTkFrame(top_ctrl_frame)
        model_display_frame.pack(side="right", padx=5, pady=2)

        ctk.CTkLabel(model_display_frame, text="当前模型：").pack(side="left", padx=(5, 2), pady=2)
        self.current_model_label = ctk.CTkLabel(
            model_display_frame,
            text="未加载",
            font=("Arial", 10, "bold"),
            text_color="#4a90e2"
        )
        self.current_model_label.pack(side="left", padx=2, pady=2)

        # 主视频画布
        self.video_canvas = ctk.CTkCanvas(
            left_frame,
            width=800,
            height=600,
            bg="#000000",
            highlightthickness=2,
            highlightbackground="#cccccc"
        )
        self.video_canvas.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        # 计数显示区
        count_frame = ctk.CTkFrame(left_frame)
        count_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(
            count_frame,
            text="🎯 检测结果",
            font=("Arial", 12, "bold")
        ).pack(side="left", padx=(10, 20), pady=10)

        ctk.CTkLabel(count_frame, text="药片数量：").pack(side="left", padx=5, pady=10)
        self.count_label = ctk.CTkLabel(
            count_frame,
            text="0",
            font=("Arial", 24, "bold"),
            text_color="red"
        )
        self.count_label.pack(side="left", padx=5, pady=10)

        # ========== 右侧：控制面板 ==========
        right_frame = ctk.CTkFrame(self.detect_frame)
        right_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)

        # 检测控制区
        detect_ctrl_frame = ctk.CTkFrame(right_frame)
        detect_ctrl_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            detect_ctrl_frame,
            text="🎛️ 检测控制",
            font=("Arial", 14, "bold")
        ).pack(pady=(0, 10))

        # 检测按钮
        self.detect_btn = ctk.CTkButton(
            detect_ctrl_frame,
            text="▶️ 开始检测",
            command=self._toggle_detection,
            state="disabled",
            height=40
        )
        self.detect_btn.pack(fill="x", pady=5)

        # 置信度阈值
        ctk.CTkLabel(detect_ctrl_frame, text="置信度阈值：").pack(anchor="w", pady=(10, 0))

        self.conf_var = ctk.DoubleVar(value=0.5)
        conf_slider = ctk.CTkSlider(
            detect_ctrl_frame,
            from_=0.1,
            to=0.9,
            variable=self.conf_var,
            number_of_steps=8,
            command=self._update_conf_label
        )
        conf_slider.pack(fill="x", pady=5)

        self.conf_label = ctk.CTkLabel(detect_ctrl_frame, text="0.50")
        self.conf_label.pack(pady=5)

        # 目标片数设置区
        target_frame = ctk.CTkFrame(right_frame)
        target_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            target_frame,
            text="🎯 目标片数设置",
            font=("Arial", 14, "bold")
        ).pack(pady=(0, 10))

        # 目标片数输入
        target_input_frame = ctk.CTkFrame(target_frame)
        target_input_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(target_input_frame, text="目标片数：").pack(side="left", padx=(5, 2), pady=5)
        self.target_entry = ctk.CTkEntry(
            target_input_frame,
            textvariable=self.target_pills_var,
            width=80,
            placeholder_text="0表示禁用"
        )
        self.target_entry.pack(side="left", padx=2, pady=5)

        # 绑定事件，处理空值
        self.target_entry.bind('<FocusOut>', lambda e: self._validate_target_entry())
        self.target_entry.bind('<Return>', lambda e: self._set_target_pills())

        ctk.CTkButton(
            target_input_frame,
            text="设置",
            width=60,
            command=self._set_target_pills
        ).pack(side="left", padx=5, pady=5)

        # 目标状态显示
        self.target_status_label = ctk.CTkLabel(
            target_frame,
            text="目标片数: 未设置",
            font=("Arial", 10),
            text_color="gray"
        )
        self.target_status_label.pack(pady=5)

        # 录像控制区
        record_frame = ctk.CTkFrame(right_frame)
        record_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            record_frame,
            text="📹 录像控制",
            font=("Arial", 14, "bold")
        ).pack(pady=(0, 10))

        # 水印自定义文本
        ctk.CTkLabel(record_frame, text="水印文字：").pack(anchor="w", pady=(5, 0))
        self.watermark_text_var = StringVar(value=self.watermark.custom_text)
        self.watermark_entry = ctk.CTkEntry(
            record_frame,
            textvariable=self.watermark_text_var,
            placeholder_text="输入水印文字"
        )
        self.watermark_entry.pack(fill="x", pady=5)

        self.watermark_entry.bind('<Return>', lambda e: self._update_watermark_text())
        self.watermark_entry.bind('<FocusOut>', lambda e: self._update_watermark_text())

        # 水印位置设置按钮
        ctk.CTkButton(
            record_frame,
            text="📍 设置水印位置",
            command=self._open_watermark_position_dialog,
            width=120
        ).pack(pady=5)

        # 录像目录选择
        self.record_dir_label = ctk.CTkLabel(record_frame, text="未选择录像目录")
        self.record_dir_label.pack(pady=5)

        ctk.CTkButton(
            record_frame,
            text="📁 选择目录",
            command=self._select_record_dir,
            width=120
        ).pack(pady=5)

        # 录像按钮
        self.record_btn = ctk.CTkButton(
            record_frame,
            text="⏺️ 开始录像",
            command=self._toggle_recording,
            state="disabled",
            height=40
        )
        self.record_btn.pack(fill="x", pady=5)

        # 截图按钮
        self.capture_btn = ctk.CTkButton(
            record_frame,
            text="📸 保存截图",
            command=self._capture_frame,
            state="disabled",
            height=40
        )
        self.capture_btn.pack(fill="x", pady=5)

        # 状态显示
        status_frame = ctk.CTkFrame(right_frame)
        status_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            status_frame,
            text="📊 状态信息",
            font=("Arial", 14, "bold")
        ).pack(pady=(0, 10))

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="请先加载模型并打开摄像头",
            font=("Arial", 10)
        )
        self.status_label.pack(anchor="w", pady=5)

    def _setup_models_page(self):
        """设置模型管理页面"""
        self.models_frame.grid_columnconfigure(0, weight=1)
        self.models_frame.grid_rowconfigure(0, weight=1)

        # 创建可滚动容器
        scroll_container = ctk.CTkScrollableFrame(self.models_frame)
        scroll_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # 标题
        ctk.CTkLabel(
            scroll_container,
            text="📦 模型管理",
            font=("Arial", 16, "bold")
        ).pack(pady=(0, 20))

        # 模型导入区
        import_frame = ctk.CTkFrame(scroll_container)
        import_frame.pack(fill="x", padx=10, pady=(0, 20))

        ctk.CTkLabel(
            import_frame,
            text="导入模型",
            font=("Arial", 12, "bold")
        ).pack(pady=(10, 5))

        ctk.CTkLabel(import_frame, text="支持格式：.rp（加密模型）").pack(pady=5)

        ctk.CTkButton(
            import_frame,
            text="📥 导入加密模型",
            command=self._import_encrypted_model,
            height=40
        ).pack(pady=10)

        ctk.CTkButton(
            import_frame,
            text="🔓 导入普通模型",
            command=self._import_plain_model,
            height=40
        ).pack(pady=5)

        # 模型列表区
        list_frame = ctk.CTkFrame(scroll_container)
        list_frame.pack(fill="x", padx=10, pady=(0, 20))

        ctk.CTkLabel(
            list_frame,
            text="已加载模型",
            font=("Arial", 12, "bold")
        ).pack(pady=(10, 5))

        # 模型列表容器
        list_container = ctk.CTkFrame(list_frame)
        list_container.pack(fill="both", expand=True, padx=10, pady=10)
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)

        # 滚动条和列表框
        scrollbar = Scrollbar(list_container)
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.models_listbox = Listbox(
            list_container,
            width=40,
            height=15,
            yscrollcommand=scrollbar.set,
            selectbackground="#4a90e2",
            selectforeground="white"
        )
        self.models_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.models_listbox.yview)

        # 绑定事件
        self.models_listbox.bind('<<ListboxSelect>>', self._on_model_select)

        # 模型操作按钮
        btn_frame = ctk.CTkFrame(scroll_container)
        btn_frame.pack(fill="x", padx=10, pady=(0, 20))

        buttons = [
            ("🚀 使用选中", self._use_selected_model, 120),
            ("✏️ 重命名", self._rename_model, 120),
            ("🗑️ 删除模型", self._delete_model, 120),
        ]

        for text, command, width in buttons:
            btn = ctk.CTkButton(
                btn_frame,
                text=text,
                command=command,
                width=width
            )
            btn.pack(side="left", padx=5, pady=5)

    def _setup_settings_page(self):
        """设置系统设置页面"""
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(0, weight=1)

        # 创建可滚动容器
        scroll_container = ctk.CTkScrollableFrame(self.settings_frame)
        scroll_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # 标题
        ctk.CTkLabel(
            scroll_container,
            text="⚙️ 系统设置",
            font=("Arial", 16, "bold")
        ).pack(pady=(0, 20))

        # 临时文件清理
        temp_frame = ctk.CTkFrame(scroll_container)
        temp_frame.pack(fill="x", padx=10, pady=(0, 20))

        ctk.CTkLabel(
            temp_frame,
            text="临时文件管理",
            font=("Arial", 12, "bold")
        ).pack(pady=(10, 5))

        ctk.CTkLabel(
            temp_frame,
            text=f"临时目录：{self.temp_models_dir}",
            font=("Arial", 10)
        ).pack(pady=5)

        ctk.CTkButton(
            temp_frame,
            text="🧹 清理临时文件",
            command=self._clean_temp_files,
            width=150
        ).pack(pady=10)

        # 配置管理
        config_frame = ctk.CTkFrame(scroll_container)
        config_frame.pack(fill="x", padx=10, pady=(0, 20))

        ctk.CTkLabel(
            config_frame,
            text="配置文件",
            font=("Arial", 12, "bold")
        ).pack(pady=(10, 5))

        ctk.CTkLabel(
            config_frame,
            text=f"配置目录：{self.config_dir}",
            font=("Arial", 10)
        ).pack(pady=5)

        ctk.CTkButton(
            config_frame,
            text="💾 保存当前配置",
            command=self._save_all_configs,
            width=150
        ).pack(pady=10)

    def _bind_keyboard_events(self):
        """绑定键盘事件"""
        shortcuts = [
            ("<space>", self._toggle_detection),
            ("<r>", self._toggle_recording),
            ("<s>", self._capture_frame),
            ("<Escape>", self._toggle_camera),
            ("<Control-w>", self._toggle_watermark_drag),
        ]

        for key, command in shortcuts:
            self.bind(key, lambda e, cmd=command: cmd())

    def _bind_mouse_events(self):
        """绑定鼠标事件"""
        self.video_canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.video_canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

    def _detect_cameras(self):
        """检测可用摄像头"""
        available = []
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
                    cap.release()
            except:
                continue
        return available

    def _load_models_config(self):
        """加载模型配置"""
        try:
            if self.models_config_path.exists():
                with open(self.models_config_path, 'r', encoding='utf-8') as f:
                    self.models = json.load(f)
                logger.info(f"已加载 {len(self.models)} 个模型配置")
        except Exception as e:
            logger.error(f"加载模型配置失败: {e}")

    def _load_watermark_config(self):
        """加载水印配置"""
        try:
            if self.watermark_config_path.exists():
                with open(self.watermark_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.watermark.custom_text = config.get('custom_text', '药片检测系统')
                    self.watermark.set_position(
                        config.get('position_x', 0.02),
                        config.get('position_y', 0.02)
                    )
                    if hasattr(self, 'watermark_text_var'):
                        self.watermark_text_var.set(self.watermark.custom_text)
                logger.info("已加载水印配置")
        except Exception as e:
            logger.error(f"加载水印配置失败: {e}")

    def _load_target_config(self):
        """加载目标片数配置"""
        try:
            if self.target_config_path.exists():
                with open(self.target_config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    target_value = config.get('target_pills', 0)
                    if target_value == "":
                        target_value = 0
                    self.target_pills_var.set(int(target_value))
                    self.watermark.set_target_pills(int(target_value))

                    # 更新状态显示
                    if int(target_value) > 0:
                        self.target_status_label.configure(
                            text=f"目标片数: {int(target_value)}",
                            text_color="orange"
                        )
                logger.info("已加载目标片数配置")
        except Exception as e:
            logger.error(f"加载目标片数配置失败: {e}")

    def _validate_target_entry(self):
        """验证目标片数输入"""
        try:
            value = self.target_entry.get()
            if value.strip() == "":
                self.target_pills_var.set(0)
        except:
            pass

    def _save_models_config(self):
        """保存模型配置"""
        try:
            with open(self.models_config_path, 'w', encoding='utf-8') as f:
                json.dump(self.models, f, indent=4, ensure_ascii=False)
            logger.info("模型配置已保存")
        except Exception as e:
            logger.error(f"保存模型配置失败: {e}")

    def _save_watermark_config(self):
        """保存水印配置"""
        try:
            config = {
                'custom_text': self.watermark.custom_text,
                'position_x': self.watermark.position_x,
                'position_y': self.watermark.position_y
            }
            with open(self.watermark_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            logger.info("水印配置已保存")
        except Exception as e:
            logger.error(f"保存水印配置失败: {e}")

    def _save_target_config(self):
        """保存目标片数配置"""
        try:
            config = {
                'target_pills': self.target_pills_var.get()
            }
            with open(self.target_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            logger.info("目标片数配置已保存")
        except Exception as e:
            logger.error(f"保存目标片数配置失败: {e}")

    def _save_all_configs(self):
        """保存所有配置"""
        self._save_models_config()
        self._save_watermark_config()
        self._save_target_config()
        messagebox.showinfo("成功", "所有配置已保存")
        logger.info("所有配置已保存")

    def _auto_load_models(self):
        """自动加载模型列表"""
        try:
            self._update_models_list()

            # 如果有模型，自动选择第一个
            if self.models:
                self.models_listbox.select_set(0)
                self.models_listbox.event_generate("<<ListboxSelect>>")
                self.status_label.configure(text=f"已加载 {len(self.models)} 个模型，选中第一个模型")
                logger.info(f"自动加载了 {len(self.models)} 个模型")

        except Exception as e:
            logger.error(f"自动加载模型列表失败: {e}")

    def _update_models_list(self):
        """更新模型列表"""
        self.models_listbox.delete(0, "end")
        for name in self.models.keys():
            self.models_listbox.insert("end", name)

    def _import_encrypted_model(self):
        """导入加密模型"""
        file_path = filedialog.askopenfilename(
            title="选择加密模型文件",
            filetypes=[("RP模型文件", "*.rp"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        model_name = simpledialog.askstring("模型名称", "请输入模型显示名称：")
        if not model_name:
            return

        if model_name in self.models:
            if not messagebox.askyesno("提示", f"名称「{model_name}」已存在，是否覆盖？"):
                return

        self.models[model_name] = file_path
        self._update_models_list()
        self._save_models_config()

        self.status_label.configure(text=f"加密模型已导入: {model_name}")
        messagebox.showinfo("成功", f"加密模型「{model_name}」已导入")

    def _import_plain_model(self):
        """导入普通模型"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )

        if not file_path:
            return

        model_name = simpledialog.askstring("模型名称", "请输入模型显示名称：")
        if not model_name:
            return

        if model_name in self.models:
            if not messagebox.askyesno("提示", f"名称「{model_name}」已存在，是否覆盖？"):
                return

        self.models[model_name] = file_path
        self._update_models_list()
        self._save_models_config()

        self.status_label.configure(text=f"普通模型已导入: {model_name}")
        messagebox.showinfo("成功", f"普通模型「{model_name}」已导入")

    def _on_model_select(self, event):
        """模型选择事件"""
        selection = self.models_listbox.curselection()
        if selection:
            model_name = self.models_listbox.get(selection[0])
            self.status_label.configure(text=f"选中模型: {model_name}")

    def _use_selected_model(self):
        """使用选中的模型"""
        selection = self.models_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请选择一个模型")
            return

        model_name = self.models_listbox.get(selection[0])
        model_path = self.models[model_name]

        try:
            # 如果是加密模型（.rp后缀），先解密
            if model_path.endswith('.rp'):
                temp_path = self.temp_models_dir / f"{model_name}_temp.pt"
                if not RPModelHandler.decrypt_model(model_path, str(temp_path)):
                    messagebox.showerror("错误", "模型解密失败")
                    return
                model_path = str(temp_path)

            # 加载模型
            self.current_model = YOLO(model_path)
            self.current_model_name = model_name
            self.current_model_label.configure(text=model_name)

            # 启用检测相关按钮
            self.cam_btn.configure(state="normal")
            self.detect_btn.configure(state="normal")
            self.capture_btn.configure(state="normal")

            self.status_label.configure(text=f"模型已加载: {model_name}")
            messagebox.showinfo("成功", f"模型「{model_name}」已加载")

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            messagebox.showerror("错误", f"加载模型失败: {e}")

    def _rename_model(self):
        """重命名模型"""
        selection = self.models_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请选择一个模型")
            return

        old_name = self.models_listbox.get(selection[0])
        new_name = simpledialog.askstring("重命名", "请输入新名称：", initialvalue=old_name)

        if new_name and new_name != old_name:
            if new_name in self.models:
                messagebox.showwarning("提示", "名称已存在")
                return

            self.models[new_name] = self.models.pop(old_name)
            self._update_models_list()
            self._save_models_config()

            if self.current_model_name == old_name:
                self.current_model_name = new_name
                self.current_model_label.configure(text=new_name)

            self.status_label.configure(text=f"模型已重命名: {old_name} → {new_name}")

    def _delete_model(self):
        """删除模型"""
        selection = self.models_listbox.curselection()
        if not selection:
            messagebox.showwarning("提示", "请选择一个模型")
            return

        model_name = self.models_listbox.get(selection[0])

        if not messagebox.askyesno("确认", f"确定删除「{model_name}」吗？"):
            return

        del self.models[model_name]
        self._update_models_list()
        self._save_models_config()

        if self.current_model_name == model_name:
            self.current_model = None
            self.current_model_name = ""
            self.current_model_label.configure(text="未加载")
            self.cam_btn.configure(state="disabled")
            self.detect_btn.configure(state="disabled")
            self.capture_btn.configure(state="disabled")

        self.status_label.configure(text=f"模型已删除: {model_name}")

    def _clean_temp_files(self):
        """清理临时文件"""
        try:
            count = 0
            for file in self.temp_models_dir.glob("*"):
                if file.is_file():
                    file.unlink()
                    count += 1

            messagebox.showinfo("成功", f"已清理 {count} 个临时文件")
            logger.info(f"已清理 {count} 个临时文件")
        except Exception as e:
            logger.error(f"清理临时文件失败: {e}")
            messagebox.showerror("错误", f"清理失败: {e}")

    def _open_watermark_position_dialog(self):
        """打开水印位置设置对话框"""
        if self.watermark_dialog is None or not self.watermark_dialog.winfo_exists():
            self.watermark_dialog = WatermarkPositionDialog(self, self.watermark)
            self.watermark_dialog.protocol("WM_DELETE_WINDOW", self._on_watermark_dialog_close)
        else:
            self.watermark_dialog.focus()

    def _on_watermark_dialog_close(self):
        """水印对话框关闭事件"""
        self.watermark_dialog = None
        self._save_watermark_config()

    def _set_target_pills(self):
        """设置目标片数"""
        try:
            target_str = self.target_entry.get().strip()
            if target_str == "":
                target = 0
            else:
                target = int(target_str)

            if target < 0:
                messagebox.showwarning("警告", "目标片数不能为负数")
                return

            self.watermark.set_target_pills(target)
            self._save_target_config()

            if target > 0:
                self.target_status_label.configure(
                    text=f"目标片数: {target}",
                    text_color="orange"
                )
                self.status_label.configure(text=f"已设置目标片数: {target}")
                logger.info(f"设置目标片数: {target}")
            else:
                self.target_status_label.configure(
                    text="目标片数: 未设置",
                    text_color="gray"
                )
                self.status_label.configure(text="已禁用目标片数检测")
                logger.info("禁用目标片数检测")

        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    def _trigger_success_notification(self):
        """触发成功通知"""
        try:
            # 播放提示音
            self.audio_manager.play_beep()

            # 获取当前时间
            current_time_str = datetime.now().strftime("%H:%M:%S")

            # 语音播报 - 使用队列机制，可以连续播报
            success_message = f"{self.watermark.custom_text}{self.target_pills_var.get()}片发药成功"
            logger.info(f"准备播报成功消息: {success_message}")

            # 语音播报
            self.audio_manager.speak(success_message)

            # 标记水印已触发通知
            self.watermark.mark_notification_triggered()

            # 显示状态信息
            full_message = f"✅ {success_message} - {current_time_str}"
            self.status_label.configure(text=full_message)
            logger.info(f"发药成功: {full_message}")

        except Exception as e:
            logger.error(f"触发成功通知失败: {e}")

    def _toggle_camera(self):
        """打开/关闭摄像头"""
        if self.camera_thread and self.camera_thread.is_alive():
            # 关闭摄像头
            self.camera_thread.stop()
            self.camera_thread = None
            self.cam_btn.configure(text="打开摄像头")
            self.detect_btn.configure(state="disabled", text="▶️ 开始检测")
            self.record_btn.configure(state="disabled", text="⏺️ 开始录像")
            self.capture_btn.configure(state="disabled")
            self.preview_toggle_btn.configure(state="disabled", text="📷 显示预览")
            self.detecting = False
            self.recording = False

            # 关闭预览窗口
            if self.preview_window:
                self.preview_window.hide()

            # 停止录像
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            # 清空画布
            self.video_canvas.delete("all")
            self.count_label.configure(text="0")

            # 重置水印统计
            self.watermark.reset_stats()

            # 重置帧计数器
            self.frame_counter = 0
            self.last_frame_count = 0

            self.status_label.configure(text="摄像头已关闭")
            logger.info("摄像头已关闭")

        else:
            # 打开摄像头
            if not self.current_model:
                messagebox.showwarning("警告", "请先加载模型")
                return

            try:
                cam_idx = int(self.cam_combo.get().split()[1])
            except:
                cam_idx = 0

            self.camera_thread = CameraThread(cam_idx, 800, 600)
            self.camera_thread.start()

            self.cam_btn.configure(text="关闭摄像头")
            self.detect_btn.configure(state="normal")
            self.record_btn.configure(state="normal")
            self.capture_btn.configure(state="normal")
            self.preview_toggle_btn.configure(state="normal", text="📷 隐藏预览")

            # 重置水印统计
            self.watermark.reset_stats()

            # 重置帧计数器
            self.frame_counter = 0
            self.last_frame_count = 0

            # 创建预览窗口
            if not self.preview_window:
                self.preview_window = DraggablePreview(self, 250, 200)
                self.preview_window.hide()

            self.status_label.configure(text="摄像头已打开")

            # 启动视频更新
            self.after(100, self._update_video)
            logger.info(f"摄像头 {cam_idx} 已打开")

    def _toggle_preview_window(self):
        """切换预览窗口显示/隐藏"""
        if self.preview_window:
            if self.preview_toggle_btn.cget("text") == "📷 显示预览":
                self.preview_window.show()
                self.preview_toggle_btn.configure(text="📷 隐藏预览")
            else:
                self.preview_window.hide()
                self.preview_toggle_btn.configure(text="📷 显示预览")

    def _update_video(self):
        """更新视频画面"""
        if self.camera_thread and self.camera_thread.is_alive():
            frame = self.camera_thread.get_frame()
            if frame is not None:
                try:
                    self.current_frame = frame.copy()
                    self.frame_counter += 1

                    # 更新预览窗口
                    if self.preview_window:
                        self.preview_window.update_preview(frame)

                    # 执行检测
                    pill_count = 0
                    if self.detecting and self.current_model:
                        results = self.current_model(frame, conf=self.conf_var.get())
                        if results and len(results) > 0:
                            boxes = results[0].boxes
                            pill_count = len(boxes)

                            # 控制检测日志输出频率（每10秒输出一次）
                            current_time = time.time()
                            if current_time - self.last_detection_log_time > 10:
                                logger.info(f"检测到 {pill_count} 片药片 (置信度阈值: {self.conf_var.get():.2f})")
                                self.last_detection_log_time = current_time

                            # 更新水印统计 - 检查是否达到目标片数（严格等于）
                            target_reached = self.watermark.update_stats(pill_count)
                            if target_reached:
                                # 触发成功通知
                                self._trigger_success_notification()

                            # 绘制检测框
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = box.conf[0]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # 如果没有检测，也要更新水印的当前片数为0
                        self.watermark.update_stats(0)

                        # 控制帧率日志输出（每30秒输出一次）
                        current_time = time.time()
                        if current_time - self.last_detection_log_time > 30:
                            fps = (self.frame_counter - self.last_frame_count) / 30
                            logger.info(f"摄像头运行中，当前帧率: {fps:.1f} FPS")
                            self.last_frame_count = self.frame_counter
                            self.last_detection_log_time = current_time

                    # 更新计数标签
                    self.count_label.configure(text=str(pill_count))

                    # 显示水印
                    show_drag_rect = self.watermark_drag_enabled
                    watermarked_frame = self.watermark.add_watermark(frame.copy(), show_drag_rect)

                    # 如果是录制中，使用带水印的帧
                    if self.recording and self.video_writer:
                        self.video_writer.write(watermarked_frame)

                    # 显示到画布
                    frame_to_show = watermarked_frame

                    # 调整大小并显示
                    frame_rgb = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (800, 600))

                    pil_img = Image.fromarray(frame_resized)
                    from PIL import ImageTk
                    video_photo = ImageTk.PhotoImage(image=pil_img)

                    self.video_canvas.delete("all")
                    self.video_canvas.create_image(0, 0, image=video_photo, anchor="nw")
                    self.video_canvas.photo = video_photo

                except Exception as e:
                    current_time = time.time()
                    if current_time - self.last_detection_log_time > 10:
                        logger.error(f"更新视频错误: {e}")
                        self.last_detection_log_time = current_time

            # 继续更新
            self.after(50, self._update_video)

    def _toggle_detection(self):
        """开始/停止检测"""
        if not self.detecting:
            self.detecting = True
            self.detect_btn.configure(text="⏸️ 停止检测")
            self.status_label.configure(text=f"正在使用「{self.current_model_name}」检测...")
            logger.info("检测已开始")
        else:
            self.detecting = False
            self.detect_btn.configure(text="▶️ 开始检测")
            self.status_label.configure(text="检测已停止")
            logger.info("检测已停止")

    def _update_conf_label(self, value):
        """更新置信度标签"""
        self.conf_label.configure(text=f"{value:.2f}")

    def _update_watermark_text(self):
        """更新水印文字"""
        text = self.watermark_text_var.get()
        self.watermark.set_custom_text(text)
        self.status_label.configure(text=f"水印文字已更新: {text}")
        self._save_watermark_config()

    def _toggle_watermark_drag(self):
        """切换水印拖动模式"""
        self.watermark_drag_enabled = not self.watermark_drag_enabled
        status = "启用" if self.watermark_drag_enabled else "禁用"
        self.status_label.configure(text=f"水印拖动模式已{status}")
        logger.info(f"水印拖动模式已{status}")

    def enable_watermark_drag(self, enabled):
        """启用/禁用水印拖动"""
        self.watermark_drag_enabled = enabled

    def _on_mouse_down(self, event):
        """鼠标按下事件"""
        if self.watermark_drag_enabled and self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
            self.watermark.start_drag(event.x, event.y, width, height)

    def _on_mouse_drag(self, event):
        """鼠标拖动事件"""
        if self.watermark_drag_enabled and self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
            self.watermark.update_drag(event.x, event.y, width, height)

    def _on_mouse_up(self, event):
        """鼠标释放事件"""
        if self.watermark_drag_enabled:
            self.watermark.end_drag()
            self._save_watermark_config()
            self.status_label.configure(
                text=f"水印位置已更新: X={self.watermark.position_x * 100:.1f}%, Y={self.watermark.position_y * 100:.1f}%")

    def _select_record_dir(self):
        """选择录像目录"""
        dir_path = filedialog.askdirectory(title="选择录像目录")
        if dir_path:
            self.save_dir = dir_path
            self.record_dir_label.configure(text=f"录像目录: {Path(dir_path).name}")
            self.status_label.configure(text=f"录像目录已设置: {dir_path}")

    def _toggle_recording(self):
        """开始/停止录像"""
        if not self.save_dir:
            self._select_record_dir()
            if not self.save_dir:
                return

        if not self.recording:
            # 重置水印统计
            self.watermark.reset_stats()

            # 更新水印文字
            self._update_watermark_text()

            # 开始录像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_save_path = Path(self.save_dir) / f"detection_{timestamp}.mp4"

            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    str(self.video_save_path),
                    fourcc,
                    20.0,
                    (800, 600)
                )

                self.recording = True
                self.record_btn.configure(text="⏹️ 停止录像")
                self.status_label.configure(text=f"正在录像: {self.video_save_path.name}")
                logger.info(f"开始录像: {self.video_save_path}")

            except Exception as e:
                logger.error(f"开始录像失败: {e}")
                messagebox.showerror("错误", f"开始录像失败: {e}")
        else:
            # 停止录像
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.record_btn.configure(text="⏺️ 开始录像")
            self.status_label.configure(text=f"录像已保存: {self.video_save_path.name}")
            logger.info(f"录像已保存: {self.video_save_path}")

    def _capture_frame(self):
        """保存当前帧"""
        if self.current_frame is None:
            messagebox.showwarning("提示", "没有可用的视频帧")
            return

        # 绘制检测结果
        frame = self.current_frame.copy()
        if self.detecting and self.current_model:
            results = self.current_model(frame, conf=self.conf_var.get())
            if results and len(results) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 添加水印
        frame = self.watermark.add_watermark(frame)

        # 保存文件
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("所有文件", "*.*")],
            title="保存截图"
        )

        if file_path:
            cv2.imwrite(file_path, frame)
            self.status_label.configure(text=f"截图已保存: {Path(file_path).name}")
            messagebox.showinfo("成功", f"截图已保存: {file_path}")
            logger.info(f"截图已保存: {file_path}")

    def _on_closing(self):
        """关闭窗口时清理资源"""
        # 停止摄像头
        if self.camera_thread:
            self.camera_thread.stop()

        # 停止录像
        if self.video_writer:
            self.video_writer.release()

        # 清理预览窗口
        if self.preview_window:
            self.preview_window.destroy()

        # 停止音频系统
        if self.audio_manager:
            self.audio_manager.stop()

        # 清理临时文件
        self._clean_temp_files()

        # 保存所有配置
        self._save_all_configs()

        # 关闭窗口
        self.destroy()
        logger.info("应用程序已关闭")


def main():
    """主函数"""
    # Windows高DPI适配
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    try:
        app = PillDetectorApp()
        app.mainloop()
    except Exception as e:
        logger.error(f"应用程序错误: {e}")
        messagebox.showerror("错误", f"应用程序启动失败:\n{e}")
        raise


if __name__ == "__main__":
    # 安装依赖提示
    print("=" * 50)
    print("药片检测计数系统 - 启动检查")
    print("=" * 50)
    print("如果需要语音播报功能，请安装以下依赖：")
    print("pip install pyttsx3 pygame")
    print("pip install pydub  # 音频处理")
    print()

    # 尝试导入必要的库
    try:
        import pygame
        import pyttsx3

        print("✅ 音频依赖已满足")
    except ImportError as e:
        print(f"⚠️ 缺少音频依赖: {e}")
        print("音频提示功能可能无法正常工作")

    print("=" * 50)
    print()

    main()