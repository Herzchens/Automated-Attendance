import customtkinter as ctk
from tkinter import messagebox, filedialog
import sys, os
from .common import translations, CustomTable


##############################################
# Cửa sổ Thêm Học Sinh từ Ảnh (folder hoặc file)
##############################################
class AddStudentImageWindow(ctk.CTkToplevel):
    def __init__(self, master, cnx, cursor, language, on_success_callback=None):
        """
        Cửa sổ cho phép chọn thêm học sinh từ ảnh.
         - Chọn "Thêm từ thư mục": thêm tất cả ảnh trong thư mục (mỗi ảnh là 1 học sinh).
         - Chọn "Thêm từ file": thêm 1 hoặc nhiều file ảnh được chọn.
        """
        super().__init__(master)
        self.cnx = cnx
        self.cursor = cursor
        self.language = language
        self.on_success_callback = on_success_callback
        self.trans = translations[self.language]
        self.title("Add Student from Images" if self.language == "English" else "Thêm học sinh từ ảnh")
        self.geometry("500x400")
        self.resizable(False, False)
        self.mode_var = ctk.StringVar(value="folder")  # Mặc định chọn thư mục
        self.selected_path = ""  # Đường dẫn thư mục hoặc danh sách file đã chọn
        self.create_widgets()

    def create_widgets(self):
        # Lựa chọn chế độ thêm: Thêm từ thư mục hoặc Thêm từ file
        frame_mode = ctk.CTkFrame(self)
        frame_mode.pack(pady=10, padx=20, fill="x")

        self.radio_folder = ctk.CTkRadioButton(frame_mode,
                                               text="Thêm từ thư mục" if self.language == "Tiếng Việt" else "Add from Folder",
                                               variable=self.mode_var, value="folder")
        self.radio_folder.grid(row=0, column=0, padx=10, pady=5)

        self.radio_file = ctk.CTkRadioButton(frame_mode,
                                             text="Thêm từ file" if self.language == "Tiếng Việt" else "Add from File",
                                             variable=self.mode_var, value="file")
        self.radio_file.grid(row=0, column=1, padx=10, pady=5)

        # Nhập thông tin lớp (Lớp học)
        frame_class = ctk.CTkFrame(self)
        frame_class.pack(pady=10, padx=20, fill="x")
        self.label_class = ctk.CTkLabel(frame_class, text="Lớp:" if self.language == "Tiếng Việt" else "Class:")
        self.label_class.grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.entry_class = ctk.CTkEntry(frame_class)
        self.entry_class.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # Nút "Browse" để chọn thư mục hoặc file
        self.button_browse = ctk.CTkButton(self,
                                           text="Browse" if self.language == "English" else "Chọn",
                                           command=self.browse_path)
        self.button_browse.pack(pady=10)

        # Hiển thị đường dẫn đã chọn
        self.label_path = ctk.CTkLabel(self, text="Đường dẫn: " if self.language == "Tiếng Việt" else "Path: ")
        self.label_path.pack(pady=5)

        # Nút Thêm và Hủy
        frame_buttons = ctk.CTkFrame(self)
        frame_buttons.pack(pady=20)
        self.button_add = ctk.CTkButton(frame_buttons,
                                        text="Thêm" if self.language == "Tiếng Việt" else "Add",
                                        command=self.add_students)
        self.button_add.grid(row=0, column=0, padx=10)
        self.button_cancel = ctk.CTkButton(frame_buttons,
                                           text="Hủy" if self.language == "Tiếng Việt" else "Cancel",
                                           command=self.destroy)
        self.button_cancel.grid(row=0, column=1, padx=10)

    def browse_path(self):
        mode = self.mode_var.get()
        if mode == "folder":
            path = filedialog.askdirectory(title="Chọn thư mục" if self.language == "Tiếng Việt" else "Select Folder")
        else:
            # Cho phép chọn nhiều file ảnh
            path = filedialog.askopenfilenames(
                title="Chọn file ảnh" if self.language == "Tiếng Việt" else "Select Image Files",
                filetypes=[("Image files", "*.jpg *.jpeg *.png")])
            if path:
                path = list(path)
        if path:
            self.selected_path = path
            if self.mode_var.get() == "folder":
                self.label_path.configure(
                    text=f"Đường dẫn: {path}" if self.language == "Tiếng Việt" else f"Path: {path}")
            else:
                self.label_path.configure(
                    text=f"Đã chọn {len(path)} file" if self.language == "Tiếng Việt" else f"{len(path)} files selected")

    def add_students(self):
        class_name = self.entry_class.get().strip()
        if not class_name:
            messagebox.showerror("Error" if self.language == "English" else "Lỗi",
                                 "Please enter the class name." if self.language == "English" else "Vui lòng nhập lớp học.")
            return

        if not self.selected_path:
            messagebox.showerror("Error" if self.language == "English" else "Lỗi",
                                 "Please select a folder or file(s)." if self.language == "English" else "Vui lòng chọn thư mục hoặc file.")
            return

        mode = self.mode_var.get()
        valid_extensions = (".jpg", ".jpeg", ".png")
        added_count = 0

        try:
            if mode == "folder":
                folder = self.selected_path  # Đây là chuỗi đường dẫn thư mục
                files = [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]
                if not files:
                    messagebox.showerror("Error" if self.language == "English" else "Lỗi",
                                         "No image files found in the selected folder." if self.language == "English" else "Không tìm thấy file ảnh trong thư mục đã chọn.")
                    return
                for file in files:
                    student_name = os.path.splitext(file)[0]  # Lấy tên file không có phần mở rộng
                    image_path = os.path.join(folder, file)
                    default_status = ""  # Lưu chuỗi rỗng cho DiemDanhStatus
                    query = """INSERT INTO Students (HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh, ImagePath)
                               VALUES (%s, %s, %s, %s, %s)"""
                    self.cursor.execute(query, (student_name, class_name, default_status, None, image_path))
                    added_count += 1
            else:  # mode == "file"
                files = self.selected_path  # Danh sách các file được chọn
                for file_path in files:
                    file_name = os.path.basename(file_path)
                    student_name = os.path.splitext(file_name)[0]
                    default_status = ""
                    query = """INSERT INTO Students (HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh, ImagePath)
                               VALUES (%s, %s, %s, %s, %s)"""
                    self.cursor.execute(query, (student_name, class_name, default_status, None, file_path))
                    added_count += 1

            self.cnx.commit()
            messagebox.showinfo("Info" if self.language == "English" else "Thông báo",
                                f"Added {added_count} students successfully!" if self.language == "English" else f"Đã thêm thành công {added_count} học sinh!")
            if self.on_success_callback:
                self.on_success_callback()
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error" if self.language == "English" else "Lỗi", f"Error adding students:\n{e}")


##############################################
# Lớp AdminControlPanel
##############################################
class AdminControlPanel(ctk.CTk):
    def __init__(self, user_info, cnx, cursor, language):
        super().__init__()
        self.user_info = user_info
        self.cnx = cnx
        self.cursor = cursor
        self.language = language
        self.trans = translations[self.language]
        self.current_mode = "Light"
        self.title(self.trans["control_title"] + " - Admin")
        self.geometry("1200x800")
        try:
            self.state("zoomed")
        except Exception:
            pass
        self.resizable(True, True)
        self.create_widgets()
        self.create_theme_toggle()
        self.fetch_data()

    def create_widgets(self):
        greeting = f"{self.trans['welcome']} {self.user_info[1]} ({self.user_info[2]})"
        self.label_greeting = ctk.CTkLabel(self, text=greeting, font=("Arial", 24))
        self.label_greeting.pack(pady=20)
        self.search_frame = ctk.CTkFrame(self)
        self.search_frame.pack(pady=10)
        self.search_entry = ctk.CTkEntry(self.search_frame, placeholder_text=self.trans["search"])
        self.search_entry.pack(side="left", padx=10)
        self.search_button = ctk.CTkButton(self.search_frame, text=self.trans["search"], command=self.search_student)
        self.search_button.pack(side="left", padx=10)
        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.pack(pady=10, padx=40, fill="x")
        self.button_add = ctk.CTkButton(self.frame_controls, text=self.trans["add_student"], command=self.add_student)
        self.button_add.grid(row=0, column=0, padx=10, pady=10)
        self.button_delete = ctk.CTkButton(self.frame_controls, text=self.trans["delete_student"],
                                           command=self.delete_student)
        self.button_delete.grid(row=0, column=1, padx=10, pady=10)
        self.button_edit = ctk.CTkButton(self.frame_controls, text=self.trans["edit_student"],
                                         command=self.edit_student)
        self.button_edit.grid(row=0, column=2, padx=10, pady=10)
        self.button_cutoff = ctk.CTkButton(self.frame_controls, text=self.trans["set_cutoff"], command=self.set_cutoff)
        self.button_cutoff.grid(row=0, column=3, padx=10, pady=10)
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(pady=10, padx=40, fill="both", expand=True)
        columns = [self.trans["col_index"], self.trans["col_name"], self.trans["col_attendance"]]
        self.custom_table = CustomTable(self.table_frame, columns=columns, corner_radius=8)
        self.custom_table.pack(fill="both", expand=True)
        self.frame_buttons_bottom = ctk.CTkFrame(self)
        self.frame_buttons_bottom.pack(pady=10)
        self.button_export = ctk.CTkButton(self.frame_buttons_bottom, text=self.trans["export"], width=150,
                                           command=self.export_data)
        self.button_export.grid(row=0, column=0, padx=20, pady=10)
        self.button_logout = ctk.CTkButton(self.frame_buttons_bottom, text=self.trans["logout"], width=150,
                                           command=self.logout)
        self.button_logout.grid(row=0, column=1, padx=20, pady=10)
        self.button_quit = ctk.CTkButton(self.frame_buttons_bottom, text=self.trans["quit"], width=150,
                                         command=self.quit_app)
        self.button_quit.grid(row=0, column=2, padx=20, pady=10)

    def add_student(self):
        # Mở cửa sổ thêm học sinh từ ảnh (folder hoặc file)
        from .admin_panel import AddStudentImageWindow  # import cục bộ để tránh vòng lặp import
        AddStudentImageWindow(self, self.cnx, self.cursor, self.language, on_success_callback=self.fetch_data)

    def set_cutoff(self):
        messagebox.showinfo("Info",
                            "Set cutoff feature not implemented yet." if self.language == "English" else "Chức năng cài đặt hạn chót chưa được triển khai.")

    def create_theme_toggle(self):
        btn_text = self.trans["toggle_light"] if self.current_mode == "Dark" else self.trans["toggle_dark"]
        self.toggle_button = ctk.CTkButton(self, text=btn_text, width=40, height=40, corner_radius=8,
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

    def fetch_data(self):
        # Lấy danh sách học sinh của tất cả các lớp, sắp xếp theo tên (A->Z)
        query = "SELECT id, HoVaTen, DiemDanhStatus, ThoiGianDiemDanh FROM Students ORDER BY HoVaTen ASC"
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching data:\n{e}")
            return
        self.custom_table.clear_rows()
        if not rows:
            self.custom_table.pack_forget()
            self.watermark_label = ctk.CTkLabel(self.table_frame, text=self.trans["no_data"], font=("Arial", 48),
                                                fg_color="transparent")
            self.watermark_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "watermark_label"):
                self.watermark_label.destroy()
            self.custom_table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                attendance = row[3] if row[3] is not None else row[2]
                self.custom_table.add_row((idx, row[1], attendance))

    def search_student(self):
        search_term = self.search_entry.get().strip().lower()
        if not search_term:
            return
        self.custom_table.clear_rows()
        query = "SELECT id, HoVaTen, DiemDanhStatus, ThoiGianDiemDanh FROM Students WHERE LOWER(HoVaTen) LIKE %s ORDER BY HoVaTen ASC"
        self.cursor.execute(query, (f"%{search_term}%",))
        rows = self.cursor.fetchall()
        if not rows:
            self.custom_table.pack_forget()
            self.watermark_label = ctk.CTkLabel(self.table_frame, text=self.trans["no_data"], font=("Arial", 48),
                                                fg_color="transparent")
            self.watermark_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "watermark_label"):
                self.watermark_label.destroy()
            self.custom_table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                attendance = row[3] if row[3] is not None else row[2]
                self.custom_table.add_row((idx, row[1], attendance))

    def export_data(self):
        messagebox.showinfo("Info", "Export List clicked!")

    def logout(self):
        self.destroy()
        from . import open_user_login_window
        open_user_login_window(self.cnx, self.cursor, self.language)

    def quit_app(self):
        self.destroy()
        sys.exit(0)

    def get_selected_student(self):
        idx = self.custom_table.selected_row_index
        if idx is None:
            return None
        return self.custom_table.rows_data[idx - 1]

    def edit_student(self):
        student = self.get_selected_student()
        if not student:
            messagebox.showerror("Error",
                                 "Please select a student to edit." if self.language == "English" else "Vui lòng chọn học sinh để chỉnh sửa.")
            return
        messagebox.showinfo("Info",
                            f"Edit student {student[1]}." if self.language == "English" else f"Chỉnh sửa học sinh {student[1]} được chọn.")

    def delete_student(self):
        student = self.get_selected_student()
        if not student:
            messagebox.showerror("Error",
                                 "Please select a student to delete." if self.language == "English" else "Vui lòng chọn học sinh để xoá.")
            return
        confirm = messagebox.askyesno("Confirm",
                                      "Are you sure you want to delete this student?" if self.language == "English" else "Bạn có chắc muốn xoá học sinh này?")
        if confirm:
            messagebox.showinfo("Info",
                                f"Student {student[1]} deleted." if self.language == "English" else f"Học sinh {student[1]} đã được xoá.")
            self.fetch_data()
