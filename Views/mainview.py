from tkinter import *
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

class Mainview:
    def __init__(self, window, controller):
        self.window = window
        self.controller = controller
        self.window.title("Phần mềm nhận diện biển số xe")
        self.window.state("zoomed")
        self.window.configure(bg="white")

        self.frame_menu = Frame(self.window, bg="white")
        self.frame_menu.place(x=0, y=0, width=220, height=800)

        self.frame_main = Frame(self.window, bg="white")
        self.frame_main.place(x=200, y=0, width=1980, height=800)

        self.frame_lap = Frame(self.frame_main, bg="#D3D3D3", bd=3, relief="solid")
        self.frame_lap.place(x=10, y=50, width=1200, height=670)

        self.frame_img = Frame(self.frame_lap, bg="white", bd=3, relief="solid")
        self.frame_img.place(x=70, y=50, width=480, height=350)
        self.frame_img.bind("<Button-1>", lambda event: self.load_image())

        self.frame_img2 = Frame(self.frame_lap, bg="white", bd=3, relief="solid")
        self.frame_img2.place(x=653, y=50, width=480, height=350)

        self.label_intro = Label(self.frame_main, text="NHẬN DIỆN BIỂN SỐ XE", font=("Times New Roman", 20, "bold"), fg="#000000", bg="white")
        self.label_intro.place(x=450, y=10)

        self.label_1 = Label(self.frame_lap, text="Ảnh tải lên", font=("Times New Roman", 18, "bold"), fg="#000000", bg="#D3D3D3")
        self.label_1.place(x=250, y=405)

        self.label_2 = Label(self.frame_lap, text="Ảnh biển số đọc được", font=("Times New Roman", 18, "bold"), fg="#000000", bg="#D3D3D3")
        self.label_2.place(x=780, y=405)

        self.btn_open = Button(self.window, text="NHẬN DIỆN", font=("Times New Roman", 13), bg="#6FA8DC", fg="white", command=self.detect_plate)
        self.btn_open.place(x=765, y=460, width=100, height=45)

        self.c = Canvas(self.frame_lap, width=70, height=80, bg="#D3D3D3", highlightthickness=0)
        self.c.place(x=565, y=190)
        self.c.create_line(10, 40, 60, 40, width=6, capstyle="round", arrow=LAST, arrowshape=(20, 22, 10))

        self.label_3 = Label(self.frame_lap, text="Thông tin về biển số: ", font=("Times New Roman", 20, "bold"), fg="#000000", bg="#D3D3D3")
        self.label_3.place(x=70, y=460)

        self.label_4 = Label(self.frame_lap, text="Mã biển số: ", font=("Times New Roman", 16, "bold"), fg="#000000", bg="#D3D3D3")
        self.label_4.place(x=70, y=510)

        self.label_5 = Label(self.frame_lap, text="Tỉnh thành: ", font=("Times New Roman", 16, "bold"), fg="#000000", bg="#D3D3D3")
        self.label_5.place(x=70, y=540)

        self.label_6 = Label(self.frame_lap, text="Seri cấp phát: ", font=("Times New Roman", 16, "bold"), fg="#000000", bg="#D3D3D3")
        self.label_6.place(x=70, y=570)

        self.label_6 = Label(self.frame_lap, text="Mã cá nhân: ", font=("Times New Roman", 16, "bold"), fg="#000000", bg="#D3D3D3")
        self.label_6.place(x=70, y=600)

        self.label_img2 = Label(self.frame_img2, bg="white")
        self.label_img2.pack(fill="both", expand=True)

        self.buttons = [
            Button(self.frame_menu, text="Trang chủ", font=("Times New Roman", 14), command=self.controller.home),
            Button(self.frame_menu, text="Tải ảnh lên", font=("Times New Roman", 14), command=self.load_image),
            Button(self.frame_menu, text="Lịch sử", font=("Times New Roman", 14)),
            Button(self.frame_menu, text="Đọc biển số", font=("Times New Roman", 14)),
            Button(self.frame_menu, text="Tra cứu", font=("Times New Roman", 14)),
            Button(self.frame_menu, text="Thoát", bg="#FF6666", fg="white", font=("Times New Roman", 14, "bold"), command=self.window.quit),
            Button(self.frame_menu, text="vẽ xe tự động", font=("Times New Roman", 14) ),
        ]
        self.place_buttons()

        self.img_label = None
        self.img_tk = None
        self.file_path = None

    def place_buttons(self):
        gap = 65
        for i, btn in enumerate(self.buttons):
            btn.place(x=25, y=(i + 0.8) * gap, width=150, height=50)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            img = Image.open(file_path)
            img = img.resize((480, 350))
            self.img_tk = ImageTk.PhotoImage(img)
            self.file_path = file_path
            if self.img_label:
                self.img_label.config(image=self.img_tk)
            else:
                self.img_label = Label(self.frame_img, image=self.img_tk, bg="white")
                self.img_label.pack(expand=True)

    def detect_plate(self):
        if not self.file_path:
            messagebox.showwarning("Cảnh báo", "Chưa chọn ảnh!")
            return

        img_crop, text = self.controller.detect_plate(self.file_path)
        if img_crop:
            self.label_img2.config(image=img_crop)
            self.label_img2.image = img_crop
        if text:
            self.label_4.config(text=f"Mã biển số: {text}")


