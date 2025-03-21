import customtkinter as ctk
from tkinter import messagebox, simpledialog, filedialog
import time, random, datetime, re
from translator import translations
from DatabaseHooking import add_student, update_student, remove_student, add_students_batch, update_cutoff_time

#######################################
# UI COMPONENTS – CÁC THÀNH PHẦN GIAO DIỆN
#######################################

class CustomTable(ctk.CTkScrollableFrame):
    def __init__(self, parent, columns, column_weights=None, row_height=40, **kwargs):
        super().__init__(parent, **kwargs)
        self.columns = columns
        self.row_height = row_height
        if column_weights is None:
            if len(columns) == 3:
                column_weights = [1, 2, 1]
            else:
                column_weights = [1] * len(columns)
        self.column_weights = column_weights
        self.selected_row_index = None
        self.rows_data = []
        self.row_frames = []

        for col in range(len(columns)):
            self.grid_columnconfigure(col, weight=self.column_weights[col], minsize=100)

        for i, header_text in enumerate(columns):
            anchor_style = "w" if i == 1 else "center"
            header = ctk.CTkLabel(self, text=header_text, font=("Arial", 14, "bold"), anchor=anchor_style)
            header.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
        self.current_row = 1

    def add_row(self, row_data):
        self.rows_data.append(row_data)
        row_frame = ctk.CTkFrame(self, fg_color="transparent")
        row_frame.grid(row=self.current_row, column=0, columnspan=len(self.columns), sticky="nsew", padx=1, pady=1)
        for i in range(len(self.columns)):
            row_frame.grid_columnconfigure(i, weight=self.column_weights[i], minsize=100)
        for i, data in enumerate(row_data):
            anchor_style = "w" if i == 1 else "center"
            cell = ctk.CTkLabel(row_frame, text=str(data), font=("Arial", 14), anchor=anchor_style)
            cell.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            cell.bind("<Button-1>", lambda e, index=self.current_row: self.select_row(index))
        row_frame.bind("<Button-1>", lambda e, index=self.current_row: self.select_row(index))
        self.row_frames.append(row_frame)
        self.current_row += 1

    def clear_rows(self):
        for frame in self.row_frames:
            frame.destroy()
        self.row_frames = []
        self.rows_data = []
        self.current_row = 1
        self.selected_row_index = None

    def select_row(self, index):
        if self.selected_row_index is not None and 0 <= self.selected_row_index - 1 < len(self.row_frames):
            prev_frame = self.row_frames[self.selected_row_index - 1]
            prev_frame.configure(fg_color="transparent")
        self.selected_row_index = index
        selected_frame = self.row_frames[index - 1]
        selected_frame.configure(fg_color="#a3d2ca")


class AddStudentImageWindow(ctk.CTkToplevel):
    """
    Cửa sổ thêm học sinh từ ảnh.
    Người dùng chỉ nhập tên, lớp và chọn ảnh.
    Các trường UID, Ngày sinh và Giới tính được tự tạo mặc định.
    """
    def __init__(self, parent, cnx, cursor, language, on_success_callback=None):
        super().__init__(parent)
        self.cnx = cnx
        self.cursor = cursor
        self.language = language
        self.on_success_callback = on_success_callback
        self.trans = translations[self.language]
        self.title("Thêm học sinh từ ảnh" if self.language == "Tiếng Việt" else "Add Student from Images")
        self.geometry("500x300")
        self.resizable(False, False)

        self.label_name = ctk.CTkLabel(self, text="Tên học sinh:" if self.language == "Tiếng Việt" else "Student Name:")
        self.label_name.pack(pady=5)
        self.entry_name = ctk.CTkEntry(self)
        self.entry_name.pack(pady=5)

        self.label_class = ctk.CTkLabel(self, text="Lớp:" if self.language == "Tiếng Việt" else "Class:")
        self.label_class.pack(pady=5)
        self.entry_class = ctk.CTkEntry(self)
        self.entry_class.pack(pady=5)

        self.button_browse = ctk.CTkButton(
            self,
            text="Chọn ảnh" if self.language == "Tiếng Việt" else "Browse Image",
            command=self.browse_image
        )
        self.button_browse.pack(pady=5)
        self.label_image = ctk.CTkLabel(self, text="(Chưa chọn ảnh)")
        self.label_image.pack(pady=5)

        self.button_add = ctk.CTkButton(
            self,
            text="Thêm" if self.language == "Tiếng Việt" else "Add",
            command=self.add_student
        )
        self.button_add.pack(pady=10)
        self.selected_image = ""

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh" if self.language == "Tiếng Việt" else "Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.selected_image = file_path
            self.label_image.configure(text=file_path)

    def add_student(self):
        name = self.entry_name.get().strip()
        class_name = self.entry_class.get().strip()
        if not name or not class_name or not self.selected_image:
            messagebox.showerror(
                "Lỗi" if self.language == "Tiếng Việt" else "Error",
                "Vui lòng nhập đầy đủ thông tin" if self.language == "Tiếng Việt" else "Please fill all fields"
            )
            return
        try:
            uid = f"{int(time.time())}{random.randint(100,999)}"
            default_birthdate = datetime.date.today().strftime("%Y-%m-%d")
            default_gender = "Nam"
            add_student(self.cursor, self.cnx, uid, name, default_birthdate, class_name, default_gender, self.selected_image)
            messagebox.showinfo(
                "Info",
                "Thêm học sinh thành công" if self.language == "Tiếng Việt" else "Student added successfully"
            )
            if self.on_success_callback:
                self.on_success_callback()
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))


def add_student_ui(parent, cnx, cursor, language, on_success_callback=None):
    """Mở cửa sổ thêm học sinh từ ảnh."""
    AddStudentImageWindow(parent, cnx, cursor, language, on_success_callback=on_success_callback)


def edit_student_ui(parent, cnx, cursor, language, student, on_success_callback=None):
    """
    Hiển thị hộp thoại nhập tên và lớp mới của học sinh và gọi hàm update_student.
    student: tuple chứa thông tin học sinh (id, HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh)
    """
    trans = translations[language]
    new_name = simpledialog.askstring(
        "Edit Student",
        "Enter new name:" if language == "English" else "Nhập tên mới:",
        initialvalue=student[1]
    )
    if new_name is None:
        return
    new_class = simpledialog.askstring(
        "Edit Student",
        "Enter new class:" if language == "English" else "Nhập lớp mới:",
        initialvalue=student[2]
    )
    if new_class is None:
        return
    try:
        update_student(cursor, cnx, student[0], HoVaTen=new_name, Lop=new_class)
        messagebox.showinfo(
            "Info",
            "Student updated successfully." if language == "English" else "Học sinh đã được cập nhật thành công."
        )
        if on_success_callback:
            on_success_callback()
    except Exception as e:
        messagebox.showerror("Error", f"Lỗi cập nhật học sinh:\n{e}")


def remove_student_ui(parent, cnx, cursor, language, student, on_success_callback=None):
    """
    Xác nhận xoá học sinh và gọi hàm remove_student.
    """
    trans = translations[language]
    confirm = messagebox.askyesno(
        "Confirm",
        "Are you sure you want to delete this student?" if language == "English" else "Bạn có chắc muốn xoá học sinh này?"
    )
    if confirm:
        try:
            remove_student(cursor, cnx, student[0])
            messagebox.showinfo(
                "Info",
                "Student removed successfully." if language == "English" else "Học sinh đã được xoá."
            )
            if on_success_callback:
                on_success_callback()
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi xoá học sinh:\n{e}")


def add_students_batch_ui(parent, cnx, cursor, language, on_success_callback=None):
    """
    Hiển thị hộp thoại chọn thư mục chứa ảnh và gọi hàm batch add.
    Mỗi ảnh có tên định dạng: Họ_Tên_Lớp (ví dụ: Nguyen_Van_A_12A).
    """
    trans = translations[language]
    folder = filedialog.askdirectory(title=trans.get("choose_folder", "Chọn thư mục chứa ảnh"))
    if not folder:
        return
    try:
        count = add_students_batch(cursor, cnx, language, folder)
        if on_success_callback:
            on_success_callback()
    except Exception as e:
        messagebox.showerror("Error", str(e))


def edit_user_operation(cursor, cnx, account, language):
    """
    Hiển thị hộp thoại chỉnh sửa tài khoản người dùng.
    account: tuple chứa thông tin tài khoản (id, username, role).
    """
    trans = translations[language]
    new_username = simpledialog.askstring(
        "Edit User",
        "Enter new username:" if language == "English" else "Nhập tên đăng nhập mới:",
        initialvalue=account[1]
    )
    if new_username is None:
        return False
    new_role = simpledialog.askstring(
        "Edit User",
        "Enter new role (superuser/admin/moderator/user):" if language == "English" else "Nhập quyền mới (superuser/admin/moderator/user):",
        initialvalue=account[2]
    )
    if new_role is None:
        return False
    try:
        query = "UPDATE Users SET username=%s, role=%s WHERE id=%s"
        cursor.execute(query, (new_username, new_role, account[0]))
        cnx.commit()
        messagebox.showinfo(
            "Info",
            "User updated successfully." if language == "English" else "Tài khoản đã được cập nhật thành công."
        )
        return True
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return False


def delete_user_operation(cursor, cnx, account, language):
    """
    Xác nhận xoá tài khoản người dùng và thực hiện xoá.
    Không cho phép xoá tài khoản superuser duy nhất.
    """
    trans = translations[language]
    if account[2].lower() == "superuser":
        cursor.execute("SELECT COUNT(*) FROM Users WHERE LOWER(role) = 'superuser'")
        count = cursor.fetchone()[0]
        if count <= 1:
            messagebox.showerror(
                "Error",
                "Cannot delete the only superuser account." if language == "English" else "Không thể xoá tài khoản superuser duy nhất."
            )
            return False
    confirm = messagebox.askyesno(
        "Confirm",
        "Are you sure you want to delete this account?" if language == "English" else "Bạn có chắc muốn xoá tài khoản này?"
    )
    if confirm:
        try:
            query = "DELETE FROM Users WHERE id=%s"
            cursor.execute(query, (account[0],))
            cnx.commit()
            messagebox.showinfo(
                "Info",
                "User deleted successfully." if language == "English" else "Tài khoản đã được xoá thành công."
            )
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False
    return False

##########################################
# GIAO DIỆN CÀI ĐẶT HẠN CHÓT (CUT-OFF TIME) - THEO PHONG CÁCH GMT
##########################################

class CutoffTimeWindowGMT(ctk.CTkToplevel):
    """
    Cửa sổ UI cho cài đặt hạn chót với lựa chọn GMT và nhập thời gian theo định dạng HH:MM.
    Người dùng chọn múi giờ (GMT) từ dropdown và nhập giờ phút (HH:MM).
    Sau khi nhấn nút Submit, hàm update_cutoff_time() sẽ được gọi để cập nhật DB.
    """
    def __init__(self, parent, cnx, cursor, language):
        super().__init__(parent)
        self.cnx = cnx
        self.cursor = cursor
        self.language = language
        self.trans = translations[self.language]

        self.title(self.trans.get("set_cutoff", "Set Cutoff Time"))
        self.geometry("400x200")
        self.resizable(False, False)

        # Frame cho lựa chọn GMT
        self.frame_gmt = ctk.CTkFrame(self)
        self.frame_gmt.pack(pady=10)
        self.label_gmt = ctk.CTkLabel(self.frame_gmt, text=self.trans.get("gmt", "GMT:"), font=("Arial", 12))
        self.label_gmt.grid(row=0, column=0, padx=5)
        # Tạo danh sách các giá trị GMT từ -12 đến +14
        self.gmt_values = [f"GMT{offset:+d}" for offset in range(-12, 15)]
        self.combo_gmt = ctk.CTkComboBox(self.frame_gmt, values=self.gmt_values, state="readonly", width=120)
        self.combo_gmt.set("GMT+0")
        self.combo_gmt.grid(row=0, column=1, padx=5)

        # Frame cho nhập Hạn chót (HH:MM)
        self.frame_cutoff = ctk.CTkFrame(self)
        self.frame_cutoff.pack(pady=10)
        self.label_cutoff = ctk.CTkLabel(self.frame_cutoff, text=self.trans.get("cutoff_time", "Cutoff Time (HH:MM):"), font=("Arial", 12))
        self.label_cutoff.grid(row=0, column=0, padx=5)
        self.entry_cutoff = ctk.CTkEntry(self.frame_cutoff, width=120)
        self.entry_cutoff.grid(row=0, column=1, padx=5)

        # Nút Submit
        self.submit_button = ctk.CTkButton(self, text=self.trans.get("submit", "Submit"), command=self.submit_cutoff)
        self.submit_button.pack(pady=10)

    def submit_cutoff(self):
        gmt = self.combo_gmt.get()  # Ví dụ "GMT+0"
        cutoff = self.entry_cutoff.get().strip()  # Giờ phút dưới dạng HH:MM

        # Kiểm tra định dạng HH:MM bằng regex
        if not re.fullmatch(r"\d{2}:\d{2}", cutoff):
            messagebox.showerror("Error", self.trans.get("invalid_time_format", "Invalid time format. Please enter HH:MM."))
            return

        try:
            # Gọi hàm cập nhật hạn chót từ DatabaseHooking.
            update_cutoff_time(self.cnx, self.cursor, gmt, cutoff)
            messagebox.showinfo("Info", self.trans.get("cutoff_set_success", "Cutoff time set successfully."))
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error setting cutoff time:\n{e}")
