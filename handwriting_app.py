import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import torch
import cv2
import numpy as np

from src.model_crnn import ctc_decode, CRNN

src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import model_crnn

BG_APP = "#292939"
BG_CARD = "#32324a"
BG_DROPZONE = "#3a3a56"
BORDER_CARD = "#454564"
BORDER_DASH = "#50506f"
ACCENT = "#5b7cfa"
ACCENT_HOVER = "#7089ff"
ACCENT_DARK = "#33449c"
TEXT_MAIN = "#f1f2f7"
TEXT_SUB = "#b7bacf"
TEXT_MUTED = "#8b8ea6"
SUCCESS_BG = "#1e3a2b"
SUCCESS_TEXT = "#4fd482"


class RoundedButton(tk.Canvas):

    def __init__(self, parent, text, command, bg=ACCENT, hover=ACCENT_HOVER,
                 fg="white", font=("Segoe UI", 11, "bold"), padx=18, pady=10,
                 icon=""):
        super().__init__(parent, bg=parent["bg"], highlightthickness=0, bd=0,
                         cursor="hand2")
        self.command = command
        self.bg_color = bg
        self.hover_color = hover
        self.fg = fg
        self.font = font
        self.padx = padx
        self.pady = pady
        self.label_text = f"{icon}  {text}" if icon else text

        self._draw(self.bg_color)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", lambda e: self._draw(self.hover_color))
        self.bind("<Leave>", lambda e: self._draw(self.bg_color))

    def _rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _draw(self, color):
        self.delete("all")
        text_id = self.create_text(0, 0, text=self.label_text, font=self.font,
                                   fill=self.fg, anchor="nw")
        bbox = self.bbox(text_id)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        self.delete(text_id)

        w = text_w + self.padx * 2
        h = text_h + self.pady * 2
        self.config(width=w, height=h)
        self._rounded_rect(0, 0, w, h, h / 2, fill=color, outline=color)
        self.create_text(w / 2, h / 2, text=self.label_text, font=self.font,
                         fill=self.fg, anchor="center")

    def _on_click(self, event):
        if self.command:
            self.command()


class HandwritingRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng Dụng Nhận Dạng Chữ Viết Tay")
        # 1. Tăng chiều cao lên 740 để giao diện không bị chèn ép các khung khi có thanh thông số
        self.root.geometry("1200x740")
        self.root.configure(bg=BG_APP)
        self.root.minsize(980, 650)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.charset = None
        self.model_epoch = '25'  # Ghi nhận mặc định epoch 25 theo yêu cầu của bạn
        self.load_model()

        self.current_file_path = None

        self._build_header()

        main_frame = tk.Frame(root, bg=BG_APP)
        main_frame.pack(expand=True, fill='both', padx=40, pady=(10, 15))
        main_frame.columnconfigure(0, weight=1, uniform="col")
        main_frame.columnconfigure(1, weight=1, uniform="col")
        main_frame.rowconfigure(0, weight=1)

        self._build_left_panel(main_frame)
        self._build_right_panel(main_frame)

        # 2. Xây dựng thanh thông số nằm ngay phía dưới các khung chính
        self._build_info_bar()

        self._show_placeholder()

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG_APP)
        header.pack(fill='x', padx=40, pady=(24, 10))

        title_label = tk.Label(header, text="Nhận Dạng Chữ Viết Tay",
                               font=("Segoe UI", 24, "bold"), bg=BG_APP, fg=TEXT_MAIN)
        title_label.pack(anchor='center')

    def _build_left_panel(self, parent):
        card = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER_CARD,
                        highlightthickness=1, bd=0)
        card.grid(row=0, column=0, sticky='nsew', padx=(0, 12))

        top_bar = tk.Frame(card, bg=BG_CARD)
        top_bar.pack(fill='x', padx=24, pady=(20, 4))

        section_label = tk.Label(top_bar, text="Hình Ảnh",
                                 font=("Segoe UI", 13, "bold"), bg=BG_CARD, fg=TEXT_MAIN)
        section_label.pack(side='left', anchor='w')

        self.browse_btn = RoundedButton(top_bar, text="Tải ảnh lên",
                                        command=self.browse_file,
                                        icon="📁", bg=ACCENT, hover=ACCENT_HOVER,
                                        font=("Segoe UI", 10, "bold"),
                                        padx=14, pady=8)
        self.browse_btn.pack(side='right', anchor='e')

        self.upload_frame = tk.Frame(card, bg=BG_DROPZONE, highlightbackground=BORDER_DASH,
                                     highlightthickness=1, bd=0)
        self.upload_frame.pack(fill='both', expand=True, padx=24, pady=(12, 20))
        self.upload_frame.pack_propagate(False)

    def _show_placeholder(self):
        for widget in self.upload_frame.winfo_children():
            widget.destroy()

        placeholder = tk.Frame(self.upload_frame, bg=BG_DROPZONE)
        placeholder.place(relx=0.5, rely=0.5, anchor='center')

        folder_icon = tk.Label(placeholder, text="🖼️", font=("Segoe UI Emoji", 44),
                               bg=BG_DROPZONE, fg=ACCENT)
        folder_icon.pack(pady=(0, 10))

        instruction_label = tk.Label(placeholder,
                                     text="Chưa có ảnh nào được chọn",
                                     font=("Segoe UI", 12), bg=BG_DROPZONE, fg=TEXT_SUB)
        instruction_label.pack()

        format_label = tk.Label(placeholder,
                                text="Hỗ trợ định dạng .jpg, .jpeg, .png",
                                font=("Segoe UI", 10), bg=BG_DROPZONE, fg=TEXT_MUTED)
        format_label.pack(pady=(4, 0))

    def _build_right_panel(self, parent):
        card = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER_CARD,
                        highlightthickness=1, bd=0)
        card.grid(row=0, column=1, sticky='nsew', padx=(12, 0))

        result_title = tk.Label(card, text="Kết Quả Nhận Dạng",
                                font=("Segoe UI", 13, "bold"), bg=BG_CARD, fg=TEXT_MAIN)
        result_title.pack(anchor='w', padx=24, pady=(20, 12))

        self.result_frame = tk.Frame(card, bg=BG_DROPZONE, highlightbackground=BORDER_CARD,
                                     highlightthickness=1, bd=0)
        self.result_frame.pack(fill='both', expand=True, padx=24, pady=(0, 24))
        self.result_frame.pack_propagate(False)

        self._show_result_placeholder()

    def _show_result_placeholder(self):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        placeholder_label = tk.Label(self.result_frame,
                                     text="Tải ảnh lên để xem kết quả nhận dạng",
                                     font=("Segoe UI", 12), bg=BG_DROPZONE, fg=TEXT_MUTED)
        placeholder_label.place(relx=0.5, rely=0.5, anchor='center')

    def _build_info_bar(self):
        """Khung chứa thông số Model, Epoch và Độ chính xác dịch động ở dưới cùng (Căn giữa hoàn hảo)"""
        info_bar = tk.Frame(self.root, bg=BG_APP)
        info_bar.pack(fill='x', side='bottom', padx=40, pady=(0, 25))

        info_card = tk.Frame(info_bar, bg=BG_CARD, highlightbackground=BORDER_CARD,
                             highlightthickness=1, bd=0)
        info_card.pack(fill='x', ipady=15)  # Tăng nhẹ khoảng đệm trong để khung thoáng hơn

        # Thiết lập cấu hình giãn đều cho các cột và dòng
        info_card.rowconfigure(0, weight=1)
        info_card.columnconfigure(0, weight=1)
        info_card.columnconfigure(1, weight=1)
        info_card.columnconfigure(2, weight=1)

        # 1. Tên Mô hình (Căn giữa dọc + ngang)
        m_frame = tk.Frame(info_card, bg=BG_CARD)
        m_frame.grid(row=0, column=0, sticky="nsew")

        m_inner = tk.Frame(m_frame, bg=BG_CARD)
        m_inner.pack(expand=True)  # Căn giữa tuyệt đối trong m_frame
        tk.Label(m_inner, text="MÔ HÌNH", font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=TEXT_MUTED).pack()
        tk.Label(m_inner, text="CRNN", font=("Segoe UI", 14, "bold"), bg=BG_CARD, fg=ACCENT).pack(pady=(2, 0))

        # 2. Epoch (Căn giữa dọc + ngang)
        e_frame = tk.Frame(info_card, bg=BG_CARD)
        e_frame.grid(row=0, column=1, sticky="nsew")

        e_inner = tk.Frame(e_frame, bg=BG_CARD)
        e_inner.pack(expand=True)  # Căn giữa tuyệt đối trong e_frame
        tk.Label(e_inner, text="EPOCH", font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=TEXT_MUTED).pack()
        self.epoch_lbl = tk.Label(e_inner, text=self.model_epoch, font=("Segoe UI", 14, "bold"), bg=BG_CARD,
                                  fg=TEXT_MAIN)
        self.epoch_lbl.pack(pady=(2, 0))

        # 3. Độ chính xác nhận dạng dịch (Căn giữa dọc + ngang)
        c_frame = tk.Frame(info_card, bg=BG_CARD)
        c_frame.grid(row=0, column=2, sticky="nsew")

        c_inner = tk.Frame(c_frame, bg=BG_CARD)
        c_inner.pack(expand=True)  # Căn giữa tuyệt đối trong c_frame
        tk.Label(c_inner, text="ĐỘ TIN CẬY", font=("Segoe UI", 9, "bold"), bg=BG_CARD, fg=TEXT_MUTED).pack()
        self.accuracy_lbl = tk.Label(c_inner, text="0%", font=("Segoe UI", 14, "bold"), bg=BG_CARD, fg=TEXT_MUTED)
        self.accuracy_lbl.pack(pady=(2, 0))

    def load_model(self):
        checkpoints_dir = "checkpoints"
        best_model_path = None
        max_epoch_found = -1

        try:
            if os.path.exists(checkpoints_dir):
                # 1. Tìm tất cả các file có đuôi .pth
                files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]

                # 2. Quét qua từng file để đọc thử giá trị 'epoch' được lưu thực tế bên trong
                for file_name in files:
                    file_path = os.path.join(checkpoints_dir, file_name)
                    try:
                        # Chỉ load phần meta-data (weights_only=False nhưng không load vào GPU để tránh tốn tài nguyên)
                        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                        if 'epoch' in checkpoint:
                            raw_epoch = checkpoint['epoch']
                            # Đổi về dạng số nguyên để so sánh chuẩn xác
                            epoch_val = int(raw_epoch) if isinstance(raw_epoch, (int, float)) else int(str(raw_epoch))

                            # Tìm file có epoch thực tế bên trong lớn nhất
                            if epoch_val > max_epoch_found:
                                max_epoch_found = epoch_val
                                best_model_path = file_path
                    except Exception:
                        # Bỏ qua nếu file checkpoint đó bị lỗi hoặc không có cấu trúc chuẩn
                        continue

            # 3. Tiến hành load model tối ưu nhất tìm được lên thiết bị chạy (CPU/CUDA)
            if best_model_path and os.path.exists(best_model_path):
                print(
                    f"Đang tự động load model tối ưu nhất từ: {best_model_path} (Epoch lưu bên trong: {max_epoch_found})")
                checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                self.charset = checkpoint['charset']
                self.model = CRNN(vocab_size=checkpoint['vocab_size'])
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device).eval()

                # Cập nhật số hiển thị lên giao diện (cộng 1 để hiển thị thân thiện 1-indexed)
                self.model_epoch = str(max_epoch_found + 1)

                if hasattr(self, 'epoch_lbl'):
                    self.epoch_lbl.config(text=self.model_epoch)
            else:
                print("Không tìm thấy bất kỳ file model (.pth) hợp lệ nào trong thư mục checkpoints.")
        except Exception as e:
            print(f"Lỗi load model tự động: {e}")

    def preprocess_image(self, image_path):

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")

        if np.mean(img) < 127:
            img = 255 - img

        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = cv2.resize(img, (128, 32))
        img = img.astype(np.float32) / 255.0
        img = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        return img

    def predict_text(self, image_path):

        if self.model is None or self.charset is None:
            return "Chưa tải được mô hình", 0.0

        try:
            image = self.preprocess_image(image_path)
            image = image.to(self.device)

            with torch.no_grad():
                output = self.model(image)  # Shape: (Visual_Steps, Batch, Class_Nums)
                decoded_indices = ctc_decode(output)

                # ---- THUẬT TOÁN TÍNH ĐỘ CHÍNH XÁC DỰA TRÊN SOFTMAX LOGITS ----
                probs = torch.softmax(output, dim=-1)
                if probs.dim() == 3:
                    probs = probs.squeeze(1) if probs.shape[1] == 1 else probs.squeeze(0)

                max_probs, max_indices = torch.max(probs, dim=-1)
                non_blank_mask = max_indices != 0  # Bỏ qua ký tự trống thừa của CTC để tính chuẩn xác

                if non_blank_mask.any():
                    conf_score = max_probs[non_blank_mask].mean().item() * 100
                else:
                    conf_score = max_probs.mean().item() * 100

            text = ''.join([self.charset[idx - 1] for idx in decoded_indices if 0 < idx <= len(self.charset)])
            return (text if text else "Không phát hiện văn bản"), conf_score
        except Exception as e:
            return f"Lỗi: {str(e)}", 0.0

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Chọn Tệp Ảnh",
            filetypes=[
                ("Tệp ảnh", "*.jpg *.jpeg *.png"),
                ("Tất cả tệp", "*.*")
            ]
        )

        if file_path:
            self.current_file_path = file_path
            self.display_image(file_path)
            self.process_recognition(file_path)

    def display_image(self, file_path):
        try:
            for widget in self.upload_frame.winfo_children():
                widget.destroy()

            preview_box = tk.Frame(self.upload_frame, bg=BG_DROPZONE)
            preview_box.place(relx=0.5, rely=0.5, anchor='center')

            image = Image.open(file_path)
            image.thumbnail((420, 320), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            img_card = tk.Frame(preview_box, bg=BG_CARD, highlightbackground=BORDER_CARD,
                                highlightthickness=1, bd=0)
            img_card.pack(pady=(0, 10))

            img_label = tk.Label(img_card, image=photo, bg=BG_CARD)
            img_label.image = photo
            img_label.pack(padx=6, pady=6)

            file_name = os.path.basename(file_path)
            name_label = tk.Label(preview_box, text=file_name,
                                  font=("Segoe UI", 10), bg=BG_DROPZONE, fg=TEXT_SUB)
            name_label.pack()

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")

    def process_recognition(self, file_path):
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        loading_label = tk.Label(self.result_frame, text="Đang xử lý…",
                                 font=("Segoe UI", 12), bg=BG_DROPZONE, fg=TEXT_SUB)
        loading_label.place(relx=0.5, rely=0.5, anchor='center')
        self.root.update()

        # Nhận cả giá trị văn bản dịch được lẫn số phần trăm độ chính xác
        predicted_text, confidence = self.predict_text(file_path)

        loading_label.destroy()

        # Cập nhật số phần trăm độ chính xác lên thanh thông số bên dưới (Đổi sang màu xanh lá)
        self.accuracy_lbl.config(text=f"{confidence:.1f}%", fg=SUCCESS_TEXT)

        wrapper = tk.Frame(self.result_frame, bg=BG_DROPZONE)
        wrapper.pack(fill='both', expand=True, padx=16, pady=16)

        tag_label = tk.Label(wrapper, text="VĂN BẢN NHẬN DẠNG ĐƯỢC",
                             font=("Segoe UI", 9, "bold"), bg=BG_DROPZONE, fg=TEXT_MUTED)
        tag_label.pack(anchor='w', pady=(0, 8))

        text_card = tk.Frame(wrapper, bg=BG_CARD, highlightbackground=BORDER_CARD,
                             highlightthickness=1, bd=0)
        text_card.pack(fill='both', expand=True)

        result_text = tk.Text(text_card, font=("Segoe UI", 16, "bold"),
                              bg=BG_CARD, fg=TEXT_MAIN, wrap='word',
                              relief='flat', bd=0, padx=16, pady=16)
        result_text.pack(fill='both', expand=True)
        result_text.insert('1.0', predicted_text)
        result_text.config(state='disabled')


if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingRecognitionApp(root)
    root.mainloop()