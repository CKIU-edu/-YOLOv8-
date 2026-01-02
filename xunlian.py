"""
药片计数标注训练系统 - 优化版
摄像头预览改为可拖动的悬浮窗
"""

import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox, Listbox, Scrollbar, simpledialog, ttk
from datetime import datetime
from pathlib import Path
import threading
import hashlib
import shutil
import random
import os
import sys
import json
import glob
from queue import Queue
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入numpy（OpenCV需要）
try:
    import numpy as np
except ImportError:
    print("错误: 需要安装numpy库")
    print("请运行: pip install numpy")
    sys.exit(1)

# 尝试导入深度学习库（可选的）
try:
    import torch
    from ultralytics import YOLO

    DL_AVAILABLE = True
except ImportError:
    torch = None
    YOLO = None
    DL_AVAILABLE = False
    logger.warning("深度学习库未安装，训练功能将不可用")

# GPU优化配置（可选）
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# 预设参数模板
DEFAULT_TEMPLATES = {
    "通用模板": {
        "epochs": 100,
        "batch": 16,
        "conf_thres": 0.5,
        "iou_thres": 0.5,
        "patience": 20,
        "optimizer": "Adam",
        "lr0": 0.001,
        "lrf": 0.0001,
        "weight_decay": 0.001,
        "hsv_h": 0.05,
        "hsv_s": 0.2,
        "hsv_v": 0.2,
        "degrees": 10.0,
        "translate": 0.1,
        "fliplr": 0.5
    },
    "小目标模板": {
        "epochs": 150,
        "batch": 8,
        "conf_thres": 0.4,
        "iou_thres": 0.4,
        "patience": 30,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.00005,
        "weight_decay": 0.0005,
        "hsv_h": 0.1,
        "hsv_s": 0.3,
        "hsv_v": 0.3,
        "degrees": 5.0,
        "translate": 0.05,
        "fliplr": 0.3
    },
    "高精准模板": {
        "epochs": 200,
        "batch": 16,
        "conf_thres": 0.7,
        "iou_thres": 0.6,
        "patience": 40,
        "optimizer": "SGD",
        "lr0": 0.0001,
        "lrf": 0.00001,
        "weight_decay": 0.001,
        "hsv_h": 0.02,
        "hsv_s": 0.1,
        "hsv_v": 0.1,
        "degrees": 3.0,
        "translate": 0.03,
        "fliplr": 0.2
    }
}

# 模板保存路径
TEMPLATE_DIR = Path.home() / "PillTrainerTemplates"
TEMPLATE_DIR.mkdir(exist_ok=True)

# 基础配置
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# 常量定义
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
PREVIEW_WIDTH = 250
PREVIEW_HEIGHT = 200
VAL_SPLIT_RATIO = 0.2
MIN_BOX_SIZE = 10


class RPModelHandler:
    """RP模型加密解密处理器"""

    HEADER = b"PILL_MODEL_RP_2026"
    KEY = 0x5A

    @staticmethod
    def encrypt_model(pt_path, rp_path):
        """加密模型文件"""
        try:
            with open(pt_path, 'rb') as f:
                model_data = f.read()
                md5 = hashlib.md5(model_data).digest()

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

            model_data = bytes([b ^ RPModelHandler.KEY for b in encrypted_data])

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
        logger.info(f"摄像头 {self.camera_index} 已停止")


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
        self.window.overrideredirect(True)  # 移除窗口边框
        self.window.attributes('-topmost', True)  # 保持在顶层
        self.window.configure(fg_color="#f0f0f0")

        # 设置初始位置（右下角）
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

        # 默认显示
        self.window.deiconify()

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

                from PIL import Image, ImageTk
                pil_img = Image.fromarray(frame_resized)
                preview_photo = ImageTk.PhotoImage(image=pil_img)

                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=preview_photo, anchor="nw")
                self.canvas.photo = preview_photo
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


class PillTrainer(ctk.CTk):
    """主应用类"""

    def __init__(self):
        super().__init__()

        self.title("药片计数标注训练系统")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # 核心变量
        self.camera_thread = None
        self.dataset_dir = ""
        self.current_frame = None
        self.annotations = []
        self.photo = None

        # 批量标注变量
        self.image_list = []
        self.current_image_idx = -1
        self.current_image_path = ""
        self.drawing = False
        self.start_x = 0
        self.start_y = 0

        # 模板管理
        self.current_template = "通用模板"
        self.custom_templates = self._load_custom_templates()

        # 可用摄像头列表
        self.available_cameras = self._detect_cameras()

        # 可拖动预览窗口
        self.preview_window = None

        # 初始化UI
        self._setup_ui()

        # 绑定键盘事件
        self._bind_keyboard_events()

        # 设置退出时清理
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        """设置UI界面"""
        # 创建分页面框架
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # 主页面
        self.main_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.main_frame, text="标注管理")
        self._setup_main_page()

        # 设置页面
        self.settings_frame = ctk.CTkFrame(self.notebook)
        self.notebook.add(self.settings_frame, text="训练设置")
        self._setup_settings_page()

    def _setup_main_page(self):
        """设置主页面 - 优化布局"""
        # 使用网格布局
        self.main_frame.grid_columnconfigure(0, weight=3)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # ========== 左侧：标注预览区 ==========
        left_frame = ctk.CTkFrame(self.main_frame)
        left_frame.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="nsew")
        left_frame.grid_columnconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1)

        # 顶部控制区 - 重新设计避免遮挡
        top_ctrl_frame = ctk.CTkFrame(left_frame, height=70)
        top_ctrl_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        top_ctrl_frame.grid_columnconfigure(0, weight=1)

        # 创建三行布局
        top_row1 = ctk.CTkFrame(top_ctrl_frame)
        top_row1.pack(fill="x", padx=5, pady=2)

        top_row2 = ctk.CTkFrame(top_ctrl_frame)
        top_row2.pack(fill="x", padx=5, pady=2)

        # 第一行：摄像头控制
        cam_frame = ctk.CTkFrame(top_row1)
        cam_frame.pack(side="left", padx=5, pady=2)

        ctk.CTkLabel(cam_frame, text="摄像头：").pack(side="left", padx=(5, 2), pady=2)

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

        self.capture_btn = ctk.CTkButton(
            cam_frame,
            text="拍照",
            command=self._capture_photo,
            state="disabled",
            width=80
        )
        self.capture_btn.pack(side="left", padx=5, pady=2)

        # 显示/隐藏预览按钮
        self.preview_toggle_btn = ctk.CTkButton(
            cam_frame,
            text="📷 显示预览",
            command=self._toggle_preview_window,
            width=100,
            state="disabled"
        )
        self.preview_toggle_btn.pack(side="left", padx=5, pady=2)

        # 第二行：数据集控制
        data_frame = ctk.CTkFrame(top_row2)
        data_frame.pack(side="left", padx=5, pady=2)

        ctk.CTkLabel(data_frame, text="数据集：").pack(side="left", padx=(5, 2), pady=2)

        self.dataset_entry = ctk.CTkEntry(data_frame, width=300)
        self.dataset_entry.pack(side="left", padx=2, pady=2)

        ctk.CTkButton(
            data_frame,
            text="选择",
            command=self._select_dataset_dir,
            width=60
        ).pack(side="left", padx=5, pady=2)

        # 主标注画布
        self.canvas = ctk.CTkCanvas(
            left_frame,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            bg="#f0f0f0",
            highlightthickness=2,
            highlightbackground="#cccccc"
        )
        self.canvas.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

        # 绑定画布事件
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # 标注控制区
        anno_ctrl_frame = ctk.CTkFrame(left_frame)
        anno_ctrl_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        # 标注控制按钮
        button_configs = [
            ("📂 加载图片", self._load_images, 100),
            ("◀️ 上一张", self._prev_image, 90),
            ("▶️ 下一张", self._next_image, 90),
            ("💾 保存标注", self._save_annotations, 100),
            ("🗑️ 删除框", self._delete_last_anno, 100),
            ("🧹 清空", self._clear_annotations, 80),
        ]

        for text, command, width in button_configs:
            btn = ctk.CTkButton(
                anno_ctrl_frame,
                text=text,
                command=command,
                width=width
            )
            btn.pack(side="left", padx=2, pady=5)

        # 图片信息显示
        self.image_info_label = ctk.CTkLabel(
            left_frame,
            text="状态：未加载图片",
            font=("Arial", 10)
        )
        self.image_info_label.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="w")

        # ========== 右侧：文件管理区 ==========
        right_frame = ctk.CTkFrame(self.main_frame)
        right_frame.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)

        # 文件列表标题
        ctk.CTkLabel(
            right_frame,
            text="📁 图片列表",
            font=("Arial", 14, "bold")
        ).grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # 文件列表容器
        list_container = ctk.CTkFrame(right_frame)
        list_container.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        list_container.grid_rowconfigure(0, weight=1)
        list_container.grid_columnconfigure(0, weight=1)

        # 滚动条和列表框
        scrollbar = Scrollbar(list_container)
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.file_listbox = Listbox(
            list_container,
            width=35,
            height=25,
            yscrollcommand=scrollbar.set,
            selectbackground="#4a90e2",
            selectforeground="white"
        )
        self.file_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.config(command=self.file_listbox.yview)

        # 绑定事件
        self.file_listbox.bind('<<ListboxSelect>>', self._on_file_select)
        self.file_listbox.bind('<Double-1>', self._on_file_double_click)

        # 文件操作按钮区
        file_btn_frame = ctk.CTkFrame(right_frame)
        file_btn_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")

        file_buttons = [
            ("🔄 刷新", self._refresh_file_list, 80),
            ("🗑️ 删除", self._delete_selected_file, 80),
            ("🚀 训练", self._start_training, 80),
        ]

        for text, command, width in file_buttons:
            btn = ctk.CTkButton(
                file_btn_frame,
                text=text,
                command=command,
                width=width
            )
            btn.pack(side="left", padx=2, pady=5)

        # 状态显示
        self.status_label = ctk.CTkLabel(
            right_frame,
            text="就绪",
            font=("Arial", 10)
        )
        self.status_label.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="w")

    def _setup_settings_page(self):
        """设置训练参数页面"""
        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(0, weight=1)

        # 创建可滚动容器
        scroll_container = ctk.CTkScrollableFrame(self.settings_frame)
        scroll_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # 模板选择区
        template_frame = ctk.CTkFrame(scroll_container)
        template_frame.pack(fill="x", padx=10, pady=(0, 20))

        ctk.CTkLabel(template_frame, text="参数模板：").pack(side="left", padx=(10, 5), pady=10)

        all_templates = list(DEFAULT_TEMPLATES.keys()) + list(self.custom_templates.keys())
        self.template_combo = ctk.CTkComboBox(
            template_frame,
            values=all_templates,
            command=self._on_template_change,
            width=150
        )
        self.template_combo.pack(side="left", padx=5, pady=10)
        self.template_combo.set("通用模板")

        ctk.CTkButton(
            template_frame,
            text="保存模板",
            command=self._save_custom_template,
            width=100
        ).pack(side="left", padx=5, pady=10)

        ctk.CTkButton(
            template_frame,
            text="删除模板",
            command=self._delete_custom_template,
            width=100
        ).pack(side="left", padx=5, pady=10)

        # 基础参数
        self._create_param_section(scroll_container, "基础参数", [
            ("训练轮数：", "epochs_entry", "100"),
            ("批次大小：", "batch_entry", "16"),
            ("置信度阈值：", "conf_entry", "0.5"),
            ("IOU阈值：", "iou_entry", "0.5"),
            ("早停耐心值：", "patience_entry", "20"),
            ("优化器：", "optimizer_combo", ["Adam", "AdamW", "SGD", "RMSprop"]),
        ])

        # 学习率参数
        self._create_param_section(scroll_container, "学习率参数", [
            ("初始学习率：", "lr0_entry", "0.001"),
            ("最终学习率：", "lrf_entry", "0.0001"),
            ("权重衰减：", "weight_decay_entry", "0.001"),
        ])

        # 数据增强参数
        self._create_param_section(scroll_container, "数据增强参数", [
            ("色相增强：", "hsv_h_entry", "0.05"),
            ("饱和度增强：", "hsv_s_entry", "0.2"),
            ("明度增强：", "hsv_v_entry", "0.2"),
            ("旋转角度：", "degrees_entry", "10.0"),
            ("平移系数：", "translate_entry", "0.1"),
            ("左右翻转：", "fliplr_entry", "0.5"),
        ])

        # 设备选择
        device_frame = ctk.CTkFrame(scroll_container)
        device_frame.pack(fill="x", padx=10, pady=(0, 20))

        ctk.CTkLabel(device_frame, text="训练设备：").pack(side="left", padx=(10, 20), pady=10)

        self.device_var = ctk.StringVar(value="GPU" if DL_AVAILABLE and torch and torch.cuda.is_available() else "CPU")

        if DL_AVAILABLE and torch and torch.cuda.is_available():
            ctk.CTkRadioButton(
                device_frame,
                text="GPU训练",
                variable=self.device_var,
                value="GPU"
            ).pack(side="left", padx=20, pady=10)

        ctk.CTkRadioButton(
            device_frame,
            text="CPU训练",
            variable=self.device_var,
            value="CPU"
        ).pack(side="left", padx=20, pady=10)

        # 加载默认模板
        self._load_template("通用模板")

    def _create_param_section(self, parent, title, params):
        """创建参数部分"""
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=10, pady=(0, 20))

        ctk.CTkLabel(frame, text=title, font=("Arial", 12, "bold")).grid(
            row=0, column=0, columnspan=4, padx=10, pady=10, sticky="w"
        )

        for i, (label, name, default) in enumerate(params):
            row = i // 2 + 1
            col = (i % 2) * 2

            ctk.CTkLabel(frame, text=label).grid(
                row=row, column=col, padx=(10, 5), pady=5, sticky="e"
            )

            if isinstance(default, list):  # 下拉框
                widget = ctk.CTkComboBox(frame, values=default, width=150)
                widget.set(default[0])
                setattr(self, name, widget)
            else:  # 输入框
                widget = ctk.CTkEntry(frame, width=150)
                widget.insert(0, default)
                setattr(self, name, widget)

            widget.grid(row=row, column=col + 1, padx=5, pady=5, sticky="w")

    def _bind_keyboard_events(self):
        """绑定键盘事件"""
        shortcuts = [
            ("<Left>", self._prev_image),
            ("<Right>", self._next_image),
            ("<Up>", self._delete_last_anno),
            ("<Down>", self._save_annotations),
            ("<s>", self._save_annotations),
            ("<Delete>", self._delete_last_anno),
            ("<Escape>", self._clear_annotations),
        ]

        for key, command in shortcuts:
            self.bind(key, lambda e, cmd=command: cmd())

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

    def _toggle_camera(self):
        """打开/关闭摄像头"""
        if self.camera_thread and self.camera_thread.is_alive():
            # 关闭摄像头
            self.camera_thread.stop()
            self.camera_thread = None
            self.cam_btn.configure(text="打开摄像头")
            self.capture_btn.configure(state="disabled")
            self.preview_toggle_btn.configure(state="disabled", text="📷 显示预览")

            # 关闭预览窗口
            if self.preview_window:
                self.preview_window.hide()

            self.status_label.configure(text="摄像头已关闭")
            logger.info("摄像头已关闭")
        else:
            # 打开摄像头
            if not self.available_cameras:
                messagebox.showerror("错误", "未检测到可用摄像头！")
                return

            try:
                cam_idx = int(self.cam_combo.get().split()[1])
            except:
                cam_idx = 0

            self.camera_thread = CameraThread(cam_idx, PREVIEW_WIDTH, PREVIEW_HEIGHT)
            self.camera_thread.start()

            self.cam_btn.configure(text="关闭摄像头")
            self.capture_btn.configure(state="normal")
            self.preview_toggle_btn.configure(state="normal", text="📷 隐藏预览")

            # 创建预览窗口
            if not self.preview_window:
                self.preview_window = DraggablePreview(self, PREVIEW_WIDTH, PREVIEW_HEIGHT + 30)
                self.preview_window.show()

            self.status_label.configure(text="摄像头已打开")

            # 启动预览更新
            self.after(100, self._update_preview)
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

    def _update_preview(self):
        """更新摄像头预览"""
        if self.camera_thread and self.camera_thread.is_alive():
            frame = self.camera_thread.get_frame()
            if frame is not None:
                try:
                    # 更新预览窗口
                    if self.preview_window:
                        self.preview_window.update_preview(frame)

                    # 保存当前帧用于拍照
                    self.current_frame = frame.copy()
                except Exception as e:
                    logger.error(f"预览更新错误: {e}")

            # 继续更新
            self.after(50, self._update_preview)

    def _capture_photo(self):
        """拍照保存"""
        if not self.dataset_dir:
            messagebox.showwarning("警告", "请先选择数据集目录！")
            return

        if self.current_frame is None:
            messagebox.showwarning("警告", "没有可用的摄像头帧！")
            return

        try:
            img_dir = Path(self.dataset_dir) / "images" / "train"
            img_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_name = f"pill_{timestamp}.jpg"
            img_path = img_dir / img_name

            cv2.imwrite(str(img_path), self.current_frame)

            self._refresh_file_list()

            self.status_label.configure(text=f"已拍照: {img_name}")
            messagebox.showinfo("成功", f"图片已保存至:\n{img_path}")
            logger.info(f"图片已保存: {img_path}")

        except Exception as e:
            logger.error(f"拍照失败: {e}")
            messagebox.showerror("错误", f"保存图片失败: {e}")

    def _select_dataset_dir(self):
        """选择数据集目录"""
        dir_path = filedialog.askdirectory(title="选择数据集目录")
        if dir_path:
            self.dataset_dir = dir_path
            self.dataset_entry.delete(0, "end")
            self.dataset_entry.insert(0, dir_path)

            try:
                for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
                    Path(dir_path, subdir).mkdir(parents=True, exist_ok=True)

                self._refresh_file_list()
                self.status_label.configure(text=f"数据集: {dir_path}")
                logger.info(f"数据集目录已选择: {dir_path}")

            except Exception as e:
                logger.error(f"创建目录结构失败: {e}")
                messagebox.showerror("错误", f"创建目录结构失败: {e}")

    def _load_images(self):
        """加载数据集图片"""
        if not self.dataset_dir:
            messagebox.showwarning("警告", "请先选择数据集目录！")
            return

        try:
            img_dir = Path(self.dataset_dir) / "images" / "train"
            self.image_list = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))

            if not self.image_list:
                messagebox.showinfo("提示", "未找到图片，请先拍照或导入图片！")
                return

            self.current_image_idx = 0
            self._load_image_by_idx(0)

            self._refresh_file_list()

            logger.info(f"已加载 {len(self.image_list)} 张图片")

        except Exception as e:
            logger.error(f"加载图片失败: {e}")
            messagebox.showerror("错误", f"加载图片失败: {e}")

    def _load_image_by_idx(self, idx):
        """加载指定索引的图片"""
        if 0 <= idx < len(self.image_list):
            try:
                self.annotations.clear()
                self.canvas.delete("all")

                self.current_image_path = self.image_list[idx]

                img = cv2.imread(self.current_image_path)
                if img is None:
                    raise ValueError(f"无法读取图片: {self.current_image_path}")

                img_resized = cv2.resize(img, (CAMERA_WIDTH, CAMERA_HEIGHT))
                self._update_main_canvas(img_resized)

                self._load_annotations(img.shape[:2])

                self.current_image_idx = idx

                info_text = f"标注: {len(self.annotations)} 个框 | {idx + 1}/{len(self.image_list)}: {Path(self.current_image_path).name}"
                self.image_info_label.configure(text=info_text)
                self.status_label.configure(text=f"当前标注: {len(self.annotations)} 个框")

                self._select_file_in_list(self.current_image_path)

                logger.info(f"已加载图片: {self.current_image_path}")

            except Exception as e:
                logger.error(f"加载图片失败: {e}")
                messagebox.showerror("错误", f"加载图片失败: {e}")

    def _update_main_canvas(self, img):
        """更新主画布显示"""
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            from PIL import Image, ImageTk
            pil_img = Image.fromarray(img_rgb)
            self.photo = ImageTk.PhotoImage(image=pil_img)

            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

            self._draw_annotations()

        except Exception as e:
            logger.error(f"更新画布失败: {e}")

    def _draw_annotations(self):
        """绘制所有标注框"""
        self.canvas.delete("anno_rect")
        self.canvas.delete("anno_text")

        for i, (x1, y1, x2, y2) in enumerate(self.annotations):
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="red",
                width=2,
                tags="anno_rect"
            )

            self.canvas.create_text(
                x1 + 5, y1 + 15,
                text=str(i + 1),
                fill="white",
                font=("Arial", 10, "bold"),  # 添加字体设置
                tags="anno_text"
            )

    def _load_annotations(self, img_shape):
        """加载已有标注"""
        if not self.current_image_path:
            return

        try:
            label_path = Path(self.dataset_dir) / "labels" / "train" / (Path(self.current_image_path).stem + ".txt")

            if not label_path.exists():
                return

            with open(label_path, 'r') as f:
                lines = f.readlines()

            img_h, img_w = img_shape[:2]

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cx = float(parts[1]) * CAMERA_WIDTH
                        cy = float(parts[2]) * CAMERA_HEIGHT
                        bw = float(parts[3]) * CAMERA_WIDTH
                        bh = float(parts[4]) * CAMERA_HEIGHT

                        x1 = int(cx - bw / 2)
                        y1 = int(cy - bh / 2)
                        x2 = int(cx + bw / 2)
                        y2 = int(cy + bh / 2)

                        self.annotations.append((x1, y1, x2, y2))
                    except ValueError:
                        continue

            logger.info(f"已加载 {len(self.annotations)} 个标注框")

        except Exception as e:
            logger.error(f"加载标注失败: {e}")

    def _on_canvas_click(self, event):
        """画布点击事件"""
        if not self.current_image_path:
            return

        self.drawing = True
        self.start_x, self.start_y = event.x, event.y

    def _on_canvas_drag(self, event):
        """画布拖动事件"""
        if self.drawing:
            self.canvas.delete("temp_rect")
            self.canvas.create_rectangle(
                self.start_x, self.start_y,
                event.x, event.y,
                outline="yellow",
                width=2,
                tags="temp_rect"
            )

    def _on_canvas_release(self, event):
        """画布释放事件"""
        if self.drawing:
            self.drawing = False

            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)

            if abs(x2 - x1) > MIN_BOX_SIZE and abs(y2 - y1) > MIN_BOX_SIZE:
                self.annotations.append((x1, y1, x2, y2))
                self._draw_annotations()
                self.status_label.configure(text=f"当前标注: {len(self.annotations)} 个框")

            self.canvas.delete("temp_rect")

    def _save_annotations(self):
        """保存标注"""
        if not self.current_image_path or not self.dataset_dir:
            messagebox.showwarning("警告", "请先加载图片！")
            return

        if not self.annotations:
            if not messagebox.askyesno("确认", "没有标注框，是否保存空文件？"):
                return

        try:
            label_dir = Path(self.dataset_dir) / "labels" / "train"
            label_dir.mkdir(parents=True, exist_ok=True)

            label_path = label_dir / (Path(self.current_image_path).stem + ".txt")

            img = cv2.imread(self.current_image_path)
            if img is None:
                raise ValueError("无法读取原始图片")

            img_h, img_w = img.shape[:2]

            with open(label_path, 'w') as f:
                for x1, y1, x2, y2 in self.annotations:
                    cx = (x1 + x2) / 2 / CAMERA_WIDTH * img_w
                    cy = (y1 + y2) / 2 / CAMERA_HEIGHT * img_h
                    bw = (x2 - x1) / CAMERA_WIDTH * img_w
                    bh = (y2 - y1) / CAMERA_HEIGHT * img_h

                    cx /= img_w
                    cy /= img_h
                    bw /= img_w
                    bh /= img_h

                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            self.status_label.configure(text=f"标注已保存: {label_path.name}")
            messagebox.showinfo("成功", f"标注已保存！共 {len(self.annotations)} 个框")
            logger.info(f"标注已保存: {label_path}, 共 {len(self.annotations)} 个框")

        except Exception as e:
            logger.error(f"保存标注失败: {e}")
            messagebox.showerror("错误", f"保存标注失败: {e}")

    def _delete_last_anno(self):
        """删除最后一个标注框"""
        if self.annotations:
            self.annotations.pop()
            self._draw_annotations()
            self.status_label.configure(text=f"当前标注: {len(self.annotations)} 个框")

    def _clear_annotations(self):
        """清空所有标注框"""
        if self.annotations:
            if messagebox.askyesno("确认", "清空所有标注框？"):
                self.annotations.clear()
                self._draw_annotations()
                self.status_label.configure(text="当前标注: 0 个框")

    def _refresh_file_list(self):
        """刷新文件列表"""
        if not self.dataset_dir:
            return

        try:
            self.file_listbox.delete(0, "end")
            img_dir = Path(self.dataset_dir) / "images" / "train"

            img_files = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))

            for img_path in img_files:
                self.file_listbox.insert("end", Path(img_path).name)

            self.status_label.configure(text=f"文件列表已刷新 ({len(img_files)} 个文件)")

        except Exception as e:
            logger.error(f"刷新文件列表失败: {e}")

    def _select_file_in_list(self, file_path):
        """在文件列表中选中指定文件"""
        filename = Path(file_path).name
        for i in range(self.file_listbox.size()):
            if self.file_listbox.get(i) == filename:
                self.file_listbox.selection_clear(0, "end")
                self.file_listbox.selection_set(i)
                self.file_listbox.see(i)
                break

    def _on_file_select(self, event):
        """文件列表选择事件"""
        selection = self.file_listbox.curselection()
        if selection and self.image_list:
            idx = selection[0]
            if 0 <= idx < len(self.image_list):
                self.current_image_idx = idx
                self._load_image_by_idx(idx)

    def _on_file_double_click(self, event):
        """文件列表双击事件"""
        self._on_file_select(event)

    def _delete_selected_file(self):
        """删除选中的文件"""
        selection = self.file_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请选择要删除的文件！")
            return

        if not messagebox.askyesno("确认", "删除选中文件及其标注？"):
            return

        try:
            idx = selection[0]
            if 0 <= idx < len(self.image_list):
                img_path = self.image_list[idx]
                if Path(img_path).exists():
                    Path(img_path).unlink()

                label_path = Path(self.dataset_dir) / "labels" / "train" / (Path(img_path).stem + ".txt")
                if label_path.exists():
                    label_path.unlink()

                self._load_images()
                self._refresh_file_list()
                self.status_label.configure(text="文件已删除")
                logger.info(f"已删除文件: {img_path}")

        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            messagebox.showerror("错误", f"删除文件失败: {e}")

    def _prev_image(self):
        """上一张图片"""
        if self.image_list and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self._load_image_by_idx(self.current_image_idx)

    def _next_image(self):
        """下一张图片"""
        if self.image_list and self.current_image_idx < len(self.image_list) - 1:
            self.current_image_idx += 1
            self._load_image_by_idx(self.current_image_idx)

    # ========== 设置页面功能 ==========

    def _load_custom_templates(self):
        """加载自定义模板"""
        templates = {}
        for file in TEMPLATE_DIR.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    templates[file.stem] = json.load(f)
            except Exception as e:
                logger.error(f"加载模板失败 {file}: {e}")
        return templates

    def _load_template(self, template_name):
        """加载模板参数"""
        if template_name in DEFAULT_TEMPLATES:
            template_data = DEFAULT_TEMPLATES[template_name]
        elif template_name in self.custom_templates:
            template_data = self.custom_templates[template_name]
        else:
            return

        # 填充参数到界面
        param_mapping = [
            (self.epochs_entry, "epochs"),
            (self.batch_entry, "batch"),
            (self.conf_entry, "conf_thres"),
            (self.iou_entry, "iou_thres"),
            (self.patience_entry, "patience"),
            (self.lr0_entry, "lr0"),
            (self.lrf_entry, "lrf"),
            (self.weight_decay_entry, "weight_decay"),
            (self.hsv_h_entry, "hsv_h"),
            (self.hsv_s_entry, "hsv_s"),
            (self.hsv_v_entry, "hsv_v"),
            (self.degrees_entry, "degrees"),
            (self.translate_entry, "translate"),
            (self.fliplr_entry, "fliplr"),
        ]

        for entry, key in param_mapping:
            entry.delete(0, "end")
            entry.insert(0, str(template_data[key]))

        self.optimizer_combo.set(template_data["optimizer"])
        self.current_template = template_name

    def _on_template_change(self, template_name):
        """模板切换事件"""
        self._load_template(template_name)

    def _save_custom_template(self):
        """保存自定义模板"""
        template_name = simpledialog.askstring("保存模板", "请输入模板名称：")
        if not template_name:
            return

        if template_name in DEFAULT_TEMPLATES:
            if not messagebox.askyesno("确认", f"模板「{template_name}」是系统模板，是否覆盖？"):
                return

        try:
            template_data = {
                "epochs": int(self.epochs_entry.get()),
                "batch": int(self.batch_entry.get()),
                "conf_thres": float(self.conf_entry.get()),
                "iou_thres": float(self.iou_entry.get()),
                "patience": int(self.patience_entry.get()),
                "optimizer": self.optimizer_combo.get(),
                "lr0": float(self.lr0_entry.get()),
                "lrf": float(self.lrf_entry.get()),
                "weight_decay": float(self.weight_decay_entry.get()),
                "hsv_h": float(self.hsv_h_entry.get()),
                "hsv_s": float(self.hsv_s_entry.get()),
                "hsv_v": float(self.hsv_v_entry.get()),
                "degrees": float(self.degrees_entry.get()),
                "translate": float(self.translate_entry.get()),
                "fliplr": float(self.fliplr_entry.get()),
            }
        except ValueError as e:
            messagebox.showerror("错误", f"参数格式错误：{e}")
            return

        try:
            template_path = TEMPLATE_DIR / f"{template_name}.json"
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=4, ensure_ascii=False)

            self.custom_templates[template_name] = template_data
            all_templates = list(DEFAULT_TEMPLATES.keys()) + list(self.custom_templates.keys())
            self.template_combo.configure(values=all_templates)
            self.template_combo.set(template_name)

            messagebox.showinfo("成功", f"模板「{template_name}」已保存！")
            logger.info(f"模板已保存: {template_name}")

        except Exception as e:
            logger.error(f"保存模板失败: {e}")
            messagebox.showerror("错误", f"保存模板失败: {e}")

    def _delete_custom_template(self):
        """删除自定义模板"""
        template_name = self.template_combo.get()

        if template_name in DEFAULT_TEMPLATES:
            messagebox.showwarning("警告", "系统模板无法删除！")
            return

        if not messagebox.askyesno("确认", f"删除模板「{template_name}」？"):
            return

        try:
            template_path = TEMPLATE_DIR / f"{template_name}.json"
            if template_path.exists():
                template_path.unlink()

            if template_name in self.custom_templates:
                del self.custom_templates[template_name]

            all_templates = list(DEFAULT_TEMPLATES.keys()) + list(self.custom_templates.keys())
            self.template_combo.configure(values=all_templates)
            self.template_combo.set("通用模板")

            messagebox.showinfo("成功", "模板已删除！")
            logger.info(f"模板已删除: {template_name}")

        except Exception as e:
            logger.error(f"删除模板失败: {e}")
            messagebox.showerror("错误", f"删除模板失败: {e}")

    def _start_training(self):
        """开始训练"""
        if not DL_AVAILABLE:
            messagebox.showerror("错误", "深度学习库未安装！请安装 PyTorch 和 ultralytics")
            return

        if not self.dataset_dir:
            messagebox.showwarning("警告", "请先选择数据集目录！")
            return

        img_dir = Path(self.dataset_dir) / "images" / "train"
        if not list(img_dir.glob("*.jpg")) and not list(img_dir.glob("*.png")):
            messagebox.showwarning("警告", "无训练数据，请先标注图片！")
            return

        try:
            params = {
                "epochs": int(self.epochs_entry.get()),
                "batch": int(self.batch_entry.get()),
                "conf_thres": float(self.conf_entry.get()),
                "iou_thres": float(self.iou_entry.get()),
                "patience": int(self.patience_entry.get()),
                "optimizer": self.optimizer_combo.get(),
                "lr0": float(self.lr0_entry.get()),
                "lrf": float(self.lrf_entry.get()),
                "weight_decay": float(self.weight_decay_entry.get()),
                "hsv_h": float(self.hsv_h_entry.get()),
                "hsv_s": float(self.hsv_s_entry.get()),
                "hsv_v": float(self.hsv_v_entry.get()),
                "degrees": float(self.degrees_entry.get()),
                "translate": float(self.translate_entry.get()),
                "fliplr": float(self.fliplr_entry.get()),
                "device": self.device_var.get(),
            }
        except ValueError as e:
            messagebox.showerror("错误", f"参数格式错误：{e}")
            return

        self._split_train_val()

        yaml_path = Path(self.dataset_dir) / "dataset.yaml"
        try:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(f"""# 药片检测数据集配置
path: {self.dataset_dir}
train: images/train
val: images/val
nc: 1
names: ['pill']
""")
        except Exception as e:
            messagebox.showerror("错误", f"创建配置文件失败：{e}")
            return

        def train_thread():
            try:
                if params["device"] == "GPU" and torch.cuda.is_available():
                    device = 0
                    device_name = torch.cuda.get_device_name(0)
                    self.status_label.configure(text=f"使用GPU训练: {device_name}")
                    logger.info(f"使用GPU: {device_name}")
                else:
                    device = "cpu"
                    self.status_label.configure(text="使用CPU训练")
                    logger.info("使用CPU训练")

                self.status_label.configure(text="加载模型中...")
                model = YOLO('yolov8n.pt')

                self.status_label.configure(text="训练中...")
                logger.info("开始训练...")

                results = model.train(
                    data=str(yaml_path),
                    epochs=params["epochs"],
                    batch=params["batch"],
                    imgsz=640,
                    device=device,
                    patience=params["patience"],
                    save=True,
                    project=str(self.dataset_dir),
                    name="pill_train",
                    exist_ok=True,
                    optimizer=params["optimizer"],
                    val=True,
                    cache=True,
                    cos_lr=True,
                    conf=params["conf_thres"],
                    iou=params["iou_thres"],
                    lr0=params["lr0"],
                    lrf=params["lrf"],
                    weight_decay=params["weight_decay"],
                    hsv_h=params["hsv_h"],
                    hsv_s=params["hsv_s"],
                    hsv_v=params["hsv_v"],
                    degrees=params["degrees"],
                    translate=params["translate"],
                    fliplr=params["fliplr"],
                    verbose=False,
                )

                best_model_path = Path(self.dataset_dir) / "pill_train" / "weights" / "best.pt"
                self.status_label.configure(text="训练完成！")

                if messagebox.askyesno("训练完成", f"训练完成！\n模型已保存至:\n{best_model_path}\n\n是否加密模型？"):
                    rp_path = best_model_path.with_suffix('.rp')
                    if RPModelHandler.encrypt_model(str(best_model_path), str(rp_path)):
                        messagebox.showinfo("成功", f"模型已加密保存为:\n{rp_path}")

                logger.info(f"训练完成，模型保存在: {best_model_path}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"训练失败: {error_msg}")
                self.status_label.configure(text=f"训练失败: {error_msg}")
                self.after(0, lambda: messagebox.showerror("错误", f"训练失败:\n{error_msg}"))

        thread = threading.Thread(target=train_thread, daemon=True)
        thread.start()

        messagebox.showinfo("提示", "训练已开始，请查看状态栏进度...")

    def _split_train_val(self):
        """拆分训练集和验证集"""
        try:
            img_dir = Path(self.dataset_dir) / "images" / "train"
            label_dir = Path(self.dataset_dir) / "labels" / "train"

            val_img_dir = Path(self.dataset_dir) / "images" / "val"
            val_label_dir = Path(self.dataset_dir) / "labels" / "val"
            val_img_dir.mkdir(parents=True, exist_ok=True)
            val_label_dir.mkdir(parents=True, exist_ok=True)

            img_files = sorted(glob.glob(str(img_dir / "*.jpg")) + glob.glob(str(img_dir / "*.png")))

            if len(img_files) < 5:
                logger.info("数据量不足，不拆分验证集")
                return

            val_count = max(2, int(len(img_files) * VAL_SPLIT_RATIO))
            val_files = random.sample(img_files, val_count)

            moved_count = 0
            for img_path in val_files:
                try:
                    dst_img = val_img_dir / Path(img_path).name
                    shutil.move(img_path, dst_img)

                    label_path = label_dir / (Path(img_path).stem + ".txt")
                    if label_path.exists():
                        dst_label = val_label_dir / label_path.name
                        shutil.move(label_path, dst_label)

                    moved_count += 1
                except Exception as e:
                    logger.error(f"移动文件失败 {img_path}: {e}")

            self.status_label.configure(text=f"已拆分 {moved_count} 张图片到验证集")
            logger.info(f"已拆分 {moved_count}/{len(img_files)} 张图片到验证集")

        except Exception as e:
            logger.error(f"拆分数据集失败: {e}")

    def _on_closing(self):
        """关闭窗口时清理资源"""
        if self.camera_thread:
            self.camera_thread.stop()

        if self.preview_window:
            self.preview_window.destroy()

        self.destroy()
        logger.info("应用程序已关闭")


def enable_dpi_awareness():
    """启用DPI感知"""
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass


def main():
    """主函数"""
    enable_dpi_awareness()

    try:
        app = PillTrainer()
        app.mainloop()
    except Exception as e:
        logger.error(f"应用程序错误: {e}")
        messagebox.showerror("错误", f"应用程序启动失败:\n{e}")
        raise


if __name__ == "__main__":
    main()