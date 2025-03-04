import customtkinter as ctk
import sys, json, os, base64
from tkinter import messagebox

# Import các hàm từ DatabaseHooking
from DatabaseHooking import connect_db, create_tables, verify_user, create_default_users
from control_panel import open_control_panel

# --- Hàm load và save cấu hình ---
def load_config():
    config_file = "config.json"
    default = {
        "theme": "Light",
        "language": "Tiếng Việt",
        "db_host": "",
        "db_username": "",
        "db_password": "",
        "camera_type": "Webcam mặc định",
        "camera_url": "",
        "camera_types": ["Webcam mặc định", "Camera IP LAN", "Camera WiFi"]
    }
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            if config.get("db_username"):
                config["db_username"] = base64.b64decode(config["db_username"].encode()).decode()
            if config.get("db_password"):
                config["db_password"] = base64.b64decode(config["db_password"].encode()).decode()
            if "camera_types" not in config:
                config["camera_types"] = default["camera_types"]
            return config
        except Exception:
            return default
    else:
        return default

def save_config(theme, language, db_host, db_username, db_password, camera_type, camera_url, camera_types):
    config_file = "config.json"
    try:
        enc_username = base64.b64encode(db_username.encode()).decode() if db_username else ""
        enc_password = base64.b64encode(db_password.encode()).decode() if db_password else ""
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({
                "theme": theme,
                "language": language,
                "db_host": db_host,
                "db_username": enc_username,
                "db_password": enc_password,
                "camera_type": camera_type,
                "camera_url": camera_url,
                "camera_types": camera_types
            }, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print("Error saving config:", e)

config = load_config()
ctk.set_appearance_mode(config.get("theme", "Light"))
ctk.set_default_color_theme("blue")

# --- Bản dịch ---
translations = {
    "Tiếng Việt": {
        "mysql_title": "Đăng nhập MySQL",
        "mysql_label": "Đăng nhập MySQL",
        "db_host": "Máy chủ CSDL:",
        "db_username": "Tên đăng nhập CSDL:",
        "db_password": "Mật khẩu CSDL:",
        "language": "Ngôn ngữ:",
        "login": "Đăng nhập",
        "exit": "Thoát",
        "toggle_dark": "Chuyển sang Dark",
        "toggle_light": "Chuyển sang Light",
        "remember": "Nhớ tôi",
        "user_title": "Đăng nhập người dùng",
        "username": "Tên đăng nhập:",
        "password": "Mật khẩu:",
        "back": "Quay lại",
        "control_title": "Bảng điều khiển",
        "welcome": "Chào mừng",
        "manage": "Quản lý học sinh",
        "attendance": "Điểm danh",
        "export": "Xuất danh sách",
        "edit": "Chỉnh sửa học sinh",
        "quit": "Thoát",
        "no_data": "Không Có Dữ Liệu :(",
        "camera_header": "Cấu hình Camera",
        "camera_type_label": "Loại kết nối camera:",
        "camera_url_label": "Địa chỉ URL:"
    },
    "English": {
        "mysql_title": "MySQL Login",
        "mysql_label": "MySQL Login",
        "db_host": "Database Host:",
        "db_username": "Database Username:",
        "db_password": "Database Password:",
        "language": "Language:",
        "login": "Login",
        "exit": "Exit",
        "toggle_dark": "Switch to Dark",
        "toggle_light": "Switch to Light",
        "remember": "Remember me",
        "user_title": "User Login",
        "username": "Username:",
        "password": "Password:",
        "back": "Back",
        "control_title": "Control Panel",
        "welcome": "Welcome",
        "manage": "Manage Students",
        "attendance": "Attendance",
        "export": "Export List",
        "edit": "Edit Students",
        "quit": "Quit",
        "no_data": "No Data :(",
        "camera_header": "Camera Configuration",
        "camera_type_label": "Camera Type:",
        "camera_url_label": "Camera URL:"
    }
}

# Map tên camera giữa 2 ngôn ngữ (có thể tuỳ biến)
mapping_vi_to_en = {
    "Webcam mặc định": "Default Webcam",
    "Camera IP LAN": "LAN IP Camera",
    "Camera WiFi": "WiFi Camera"
}
mapping_en_to_vi = {
    "Default Webcam": "Webcam mặc định",
    "LAN IP Camera": "Camera IP LAN",
    "WiFi Camera": "Camera WiFi"
}

# --- Cửa sổ chỉnh sửa cấu hình Camera ---
class CameraConfigWindow(ctk.CTkToplevel):
    def __init__(self, parent, current_camera_types):
        super().__init__(parent)
        self.title("Chỉnh sửa cấu hình Camera")
        self.geometry("400x300")
        self.parent = parent
        # Textbox hiển thị các loại camera, mỗi dòng một loại
        self.textbox = ctk.CTkTextbox(self, width=350, height=200)
        self.textbox.pack(pady=10)
        # Điền danh sách hiện có vào textbox
        initial_text = "\n".join(current_camera_types)
        self.textbox.insert("0.0", initial_text)
        # Nút lưu và hủy
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(pady=10)
        self.save_button = ctk.CTkButton(self.button_frame, text="Lưu", command=self.save)
        self.save_button.grid(row=0, column=0, padx=10)
        self.cancel_button = ctk.CTkButton(self.button_frame, text="Hủy", command=self.destroy)
        self.cancel_button.grid(row=0, column=1, padx=10)

    def save(self):
        content = self.textbox.get("0.0", "end").strip()
        new_camera_types = [line.strip() for line in content.splitlines() if line.strip()]
        if not new_camera_types:
            messagebox.showerror("Lỗi", "Danh sách loại kết nối không được để trống.")
            return
        # Cập nhật lại trong parent
        self.parent.camera_types = new_camera_types
        self.parent.combo_camera_type.configure(values=new_camera_types)
        if self.parent.combo_camera_type.get() not in new_camera_types:
            self.parent.combo_camera_type.set(new_camera_types[0])
        self.destroy()

# ========= Form đăng nhập MySQL =========
class MySQLLoginWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.current_mode = config.get("theme", "Light")
        self.language = config.get("language", "Tiếng Việt")
        self.trans = translations[self.language]
        self.title(self.trans["mysql_title"])
        self.geometry("1200x800")
        try:
            self.state("zoomed")
        except Exception:
            pass
        self.resizable(True, True)
        # Lưu danh sách các loại kết nối camera
        self.camera_types = config.get("camera_types", ["Webcam mặc định", "Camera IP LAN", "Camera WiFi"])
        self.create_widgets()
        self.create_theme_toggle()
        # Load thông tin CSDL nếu có
        if config.get("db_host"):
            self.entry_db_host.insert(0, config["db_host"])
        if config.get("db_username"):
            self.entry_db_username.insert(0, config["db_username"])
        if config.get("db_password"):
            self.entry_db_password.insert(0, config["db_password"])
        # Load cấu hình camera nếu có
        if config.get("camera_type"):
            self.combo_camera_type.set(config["camera_type"])
        if config.get("camera_url"):
            self.entry_camera_url.insert(0, config["camera_url"])

    def create_widgets(self):
        self.label_title = ctk.CTkLabel(self, text=self.trans["mysql_label"], font=("Arial", 24))
        self.label_title.pack(pady=20)

        self.frame_form = ctk.CTkFrame(self)
        self.frame_form.pack(pady=10, padx=40, fill="both", expand=True)

        self.label_db_host = ctk.CTkLabel(self.frame_form, text=self.trans["db_host"])
        self.label_db_host.grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.entry_db_host = ctk.CTkEntry(self.frame_form, placeholder_text="localhost")
        self.entry_db_host.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.label_db_username = ctk.CTkLabel(self.frame_form, text=self.trans["db_username"])
        self.label_db_username.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.entry_db_username = ctk.CTkEntry(self.frame_form)
        self.entry_db_username.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.label_db_password = ctk.CTkLabel(self.frame_form, text=self.trans["db_password"])
        self.label_db_password.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.entry_db_password = ctk.CTkEntry(self.frame_form, show="*")
        self.entry_db_password.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        self.label_language = ctk.CTkLabel(self.frame_form, text=self.trans["language"])
        self.label_language.grid(row=3, column=0, padx=10, pady=10, sticky="e")
        self.combo_language = ctk.CTkComboBox(self.frame_form, values=["Tiếng Việt", "English"],
                                              command=self.change_language)
        self.combo_language.set(self.language)
        self.combo_language.grid(row=3, column=1, padx=10, pady=10, sticky="w")

        self.checkbox_remember = ctk.CTkCheckBox(self.frame_form, text=self.trans["remember"])
        self.checkbox_remember.grid(row=4, column=1, padx=10, pady=10, sticky="w")

        # --- Phần cấu hình Camera ---
        self.frame_camera = ctk.CTkFrame(self)
        self.frame_camera.pack(pady=10, padx=40, fill="both", expand=True)

        # Dùng key "camera_header" để đổi ngôn ngữ
        self.label_camera_header = ctk.CTkLabel(self.frame_camera, text=self.trans["camera_header"], font=("Arial", 20, "bold"))
        self.label_camera_header.grid(row=0, column=0, columnspan=3, pady=10)

        # Dùng key "camera_type_label"
        self.label_camera_type = ctk.CTkLabel(self.frame_camera, text=self.trans["camera_type_label"])
        self.label_camera_type.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.combo_camera_type = ctk.CTkComboBox(self.frame_camera, values=self.camera_types,
                                                 command=self.on_camera_type_change)
        self.combo_camera_type.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.button_camera_config = ctk.CTkButton(self.frame_camera, text="⚙️", width=30,
                                                  command=self.open_camera_config)
        self.button_camera_config.grid(row=1, column=2, padx=10, pady=10)

        # Dùng key "camera_url_label"
        self.label_camera_url = ctk.CTkLabel(self.frame_camera, text=self.trans["camera_url_label"])
        self.label_camera_url.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.entry_camera_url = ctk.CTkEntry(self.frame_camera)
        self.entry_camera_url.grid(row=2, column=1, padx=10, pady=10, sticky="w")

        # Ban đầu, nếu là webcam mặc định, disable ô URL
        if self.combo_camera_type.get() in ["Webcam mặc định", "Default Webcam"]:
            self.entry_camera_url.configure(state="disabled")
        else:
            self.entry_camera_url.configure(state="normal")

        self.frame_buttons = ctk.CTkFrame(self)
        self.frame_buttons.pack(pady=10, padx=40, fill="x")
        self.frame_buttons.grid_columnconfigure(0, weight=1)
        self.frame_buttons.grid_columnconfigure(1, weight=1)
        self.button_login = ctk.CTkButton(self.frame_buttons, text=self.trans["login"], command=self.handle_login,
                                          width=200, height=40)
        self.button_login.grid(row=0, column=0, padx=20, pady=10)
        self.button_exit = ctk.CTkButton(self.frame_buttons, text=self.trans["exit"], command=self.exit_app, width=200,
                                         height=40)
        self.button_exit.grid(row=0, column=1, padx=20, pady=10)

    def open_camera_config(self):
        CameraConfigWindow(self, self.camera_types)

    def on_camera_type_change(self, value):
        if value in ["Webcam mặc định", "Default Webcam"]:
            self.entry_camera_url.delete(0, "end")
            self.entry_camera_url.configure(state="disabled")
        else:
            self.entry_camera_url.configure(state="normal")

    def create_theme_toggle(self):
        button_text = self.trans["toggle_light"] if self.current_mode == "Dark" else self.trans["toggle_dark"]
        self.toggle_button = ctk.CTkButton(self, text=button_text, width=40, height=40, corner_radius=8,
                                           command=self.toggle_theme)
        self.toggle_button.place(relx=0.98, rely=0.02, anchor="ne")

    def change_language(self, new_lang):
        self.language = new_lang
        self.trans = translations[self.language]

        # Cập nhật các label, button, title...
        self.title(self.trans["mysql_title"])
        self.label_title.configure(text=self.trans["mysql_label"])
        self.label_db_host.configure(text=self.trans["db_host"])
        self.label_db_username.configure(text=self.trans["db_username"])
        self.label_db_password.configure(text=self.trans["db_password"])
        self.label_language.configure(text=self.trans["language"])
        self.button_login.configure(text=self.trans["login"])
        self.button_exit.configure(text=self.trans["exit"])
        self.checkbox_remember.configure(text=self.trans["remember"])

        # Cập nhật nút chuyển theme
        button_text = self.trans["toggle_light"] if self.current_mode == "Dark" else self.trans["toggle_dark"]
        self.toggle_button.configure(text=button_text)

        # Cập nhật label phần camera
        self.label_camera_header.configure(text=self.trans["camera_header"])
        self.label_camera_type.configure(text=self.trans["camera_type_label"])
        self.label_camera_url.configure(text=self.trans["camera_url_label"])

        # Chuyển đổi danh sách camera_type sang ngôn ngữ mới
        new_camera_types = []
        if new_lang == "English":
            for cam in self.camera_types:
                # Nếu cam có trong mapping_vi_to_en thì dùng giá trị map
                new_camera_types.append(mapping_vi_to_en.get(cam, cam))
        else:
            # Ngược lại, map về tiếng Việt
            for cam in self.camera_types:
                new_camera_types.append(mapping_en_to_vi.get(cam, cam))

        # Cập nhật combo_camera_type
        self.combo_camera_type.configure(values=new_camera_types)
        # Nếu giá trị hiện tại không còn hợp lệ, đặt lại
        if self.combo_camera_type.get() not in new_camera_types:
            self.combo_camera_type.set(new_camera_types[0])

        # Cập nhật lại ô URL
        self.on_camera_type_change(self.combo_camera_type.get())

    def toggle_theme(self):
        if self.current_mode == "Light":
            ctk.set_appearance_mode("Dark")
            self.current_mode = "Dark"
            self.toggle_button.configure(text=self.trans["toggle_light"])
        else:
            ctk.set_appearance_mode("Light")
            self.current_mode = "Light"
            self.toggle_button.configure(text=self.trans["toggle_dark"])

    def handle_login(self):
        db_host = self.entry_db_host.get().strip()
        db_username = self.entry_db_username.get().strip()
        db_password = self.entry_db_password.get().strip()
        language = self.language

        if not db_username or not db_password:
            if self.language == "Tiếng Việt":
                messagebox.showerror("Lỗi", "❌ Vui lòng nhập tên đăng nhập và mật khẩu CSDL.")
            else:
                messagebox.showerror("Error", "❌ Please enter database username and password.")
            return

        cnx, cursor = connect_db(db_username, db_password, db_host)
        if cnx is None:
            if self.language == "Tiếng Việt":
                messagebox.showerror("Lỗi CSDL",
                                     "❌ Kết nối CSDL thất bại!\nVui lòng kiểm tra thông tin đăng nhập MySQL hoặc thực hiện thao tác setup.")
            else:
                messagebox.showerror("Database Error",
                                     "❌ Database connection failed!\nPlease check your MySQL credentials or run the setup operation.")
            return

        create_tables(cursor)
        create_default_users(cursor, cnx)

        camera_type = self.combo_camera_type.get().strip()
        camera_url = self.entry_camera_url.get().strip()

        if self.checkbox_remember.get():
            save_config(self.current_mode, language, db_host, db_username, db_password,
                        camera_type, camera_url, self.camera_types)
        else:
            # Xoá bỏ thông tin CSDL khi không chọn "remember"
            save_config(self.current_mode, language, "", "", "",
                        camera_type, camera_url, self.camera_types)
        self.destroy()
        from control_panel import open_user_login_window
        open_user_login_window(cnx, cursor, language)

    def exit_app(self):
        save_config(self.current_mode,
                    self.combo_language.get(),
                    self.entry_db_host.get().strip(),
                    self.entry_db_username.get().strip(),
                    self.entry_db_password.get().strip(),
                    self.combo_camera_type.get().strip(),
                    self.entry_camera_url.get().strip(),
                    self.camera_types)
        self.destroy()
        sys.exit(0)

# ========= Form đăng nhập người dùng =========
class UserLoginWindow(ctk.CTk):
    def __init__(self, cnx, cursor, language):
        super().__init__()
        self.cnx = cnx
        self.cursor = cursor
        self.language = language
        self.trans = translations[self.language]
        self.current_mode = "Light"
        self.title(self.trans["user_title"])
        self.geometry("1200x800")
        try:
            self.state("zoomed")
        except Exception:
            pass
        self.resizable(True, True)
        self.create_widgets()
        self.create_theme_toggle()

    def create_widgets(self):
        self.label_title = ctk.CTkLabel(self, text=self.trans["user_title"], font=("Arial", 24))
        self.label_title.pack(pady=20)

        self.frame_form = ctk.CTkFrame(self)
        self.frame_form.pack(pady=10, padx=40, fill="both", expand=True)

        self.label_username = ctk.CTkLabel(self.frame_form, text=self.trans["username"])
        self.label_username.grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.entry_username = ctk.CTkEntry(self.frame_form)
        self.entry_username.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.label_password = ctk.CTkLabel(self.frame_form, text=self.trans["password"])
        self.label_password.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.entry_password = ctk.CTkEntry(self.frame_form, show="*")
        self.entry_password.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.frame_buttons = ctk.CTkFrame(self)
        self.frame_buttons.pack(pady=10, padx=40, fill="x")
        self.frame_buttons.grid_columnconfigure(0, weight=1)
        self.frame_buttons.grid_columnconfigure(1, weight=1)
        self.frame_buttons.grid_columnconfigure(2, weight=1)
        self.button_login = ctk.CTkButton(self.frame_buttons, text=self.trans["login"], command=self.handle_user_login,
                                          width=200, height=40)
        self.button_login.grid(row=0, column=0, padx=20, pady=10)
        self.button_back = ctk.CTkButton(self.frame_buttons, text=self.trans["back"], command=self.go_back, width=200,
                                         height=40)
        self.button_back.grid(row=0, column=1, padx=20, pady=10)
        self.button_attendance = ctk.CTkButton(self.frame_buttons, text=self.trans["attendance"], command=self.open_attendance,
                                               width=200, height=40)
        self.button_attendance.grid(row=0, column=2, padx=20, pady=10)

    def create_theme_toggle(self):
        self.toggle_button = ctk.CTkButton(self, text=self.trans["toggle_dark"], width=40, height=40, corner_radius=8,
                                           command=self.toggle_theme)
        self.toggle_button.place(relx=0.98, rely=0.02, anchor="ne")

    def toggle_theme(self):
        if self.current_mode == "Light":
            ctk.set_appearance_mode("Dark")
            self.current_mode = "Dark"
            self.toggle_button.configure(text=self.trans["toggle_light"])
        else:
            ctk.set_appearance_mode("Light")
            self.current_mode = "Light"
            self.toggle_button.configure(text=self.trans["toggle_dark"])

    def handle_user_login(self):
        user_username = self.entry_username.get().strip()
        user_password = self.entry_password.get().strip()
        if not user_username or not user_password:
            if self.language == "Tiếng Việt":
                messagebox.showerror("Lỗi", "❌ Vui lòng nhập tên đăng nhập và mật khẩu.")
            else:
                messagebox.showerror("Error", "❌ Please enter username and password.")
            return

        user_info = verify_user(self.cursor, user_username, user_password)
        if user_info is None:
            if self.language == "Tiếng Việt":
                messagebox.showerror("Đăng nhập thất bại", "❌ Tên đăng nhập hoặc mật khẩu không hợp lệ!")
            else:
                messagebox.showerror("Login Failed", "❌ Invalid username or password!")
            return

        self.destroy()
        from control_panel import open_control_panel
        open_control_panel(user_info, self.cnx, self.cursor, self.language)

    def open_attendance(self):
        try:
            import json
            with open("config.json", "r", encoding="utf-8") as f:
                config_data = json.load(f)
            camera_type = config_data.get("camera_type", "Webcam mặc định")
            camera_url = config_data.get("camera_url", "")
            camera_source = 0 if camera_type in ["Webcam mặc định", "Default Webcam"] else camera_url

            from FacialRecognition import main as run_attendance
            self.destroy()
            run_attendance(self.cnx, self.cursor, camera_source)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở chức năng điểm danh:\n{e}")

    def go_back(self):
        self.destroy()
        from GUI import MySQLLoginWindow
        win = MySQLLoginWindow()
        try:
            win.state("zoomed")
        except Exception:
            pass
        win.mainloop()

    def exit_app(self):
        save_config(self.current_mode, self.language, "", "", "",
                    "", "", [])
        self.destroy()
        sys.exit(0)

def main():
    app = MySQLLoginWindow()
    try:
        app.state("zoomed")
    except Exception:
        pass
    app.mainloop()

if __name__ == "__main__":
    main()
