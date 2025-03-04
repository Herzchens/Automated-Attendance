# control_panel/common.py
import customtkinter as ctk
from tkinter import messagebox, simpledialog, filedialog
import csv

translations = {
    "Tiếng Việt": {
        "control_title": "Bảng điều khiển",
        "welcome": "Chào mừng",
        "manage": "Quản lý học sinh",
        "attendance": "Điểm danh",
        "export": "Xuất danh sách",
        "edit": "Chỉnh sửa học sinh",
        "quit": "Thoát",
        "logout": "Đăng xuất",
        "no_data": "Không Có Dữ Liệu :(",
        "toggle_dark": "Chuyển sang Dark",
        "toggle_light": "Chuyển sang Light",
        "col_index": "Số Thứ Tự",
        "col_name": "Tên Học Sinh",
        "col_attendance": "Điểm Danh",
        "search": "Tìm kiếm",
        "reason": "Lý do",
        "add_student": "Thêm học sinh",
        "delete_student": "Xoá học sinh",
        "edit_student": "Chỉnh sửa học sinh",
        "set_cutoff": "Cài đặt hạn chót",
        "add_user": "Thêm tài khoản",
        "delete_user": "Xoá tài khoản",
        "edit_user": "Chỉnh sửa tài khoản",
        "user_accounts_tab": "Quản lý tài khoản",
        "students_tab": "Quản lý học sinh"
    },
    "English": {
        "control_title": "Control Panel",
        "welcome": "Welcome",
        "manage": "Manage Students",
        "attendance": "Attendance",
        "export": "Export List",
        "edit": "Edit Students",
        "quit": "Quit",
        "logout": "Log Out",
        "no_data": "No Data :(",
        "toggle_dark": "Switch to Dark",
        "toggle_light": "Switch to Light",
        "col_index": "STT",
        "col_name": "Student Name",
        "col_attendance": "Attendance",
        "search": "Search",
        "reason": "Reason",
        "add_student": "Add Student",
        "delete_student": "Delete Student",
        "edit_student": "Edit Student",
        "set_cutoff": "Set Cutoff Time",
        "add_user": "Add User",
        "delete_user": "Delete User",
        "edit_user": "Edit User",
        "user_accounts_tab": "User Account Management",
        "students_tab": "Student Management"
    }
}


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


###############################
# STUDENT OPERATIONS
###############################

class AddStudentImageWindow(ctk.CTkToplevel):
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

        self.button_browse = ctk.CTkButton(self, text="Chọn ảnh" if self.language == "Tiếng Việt" else "Browse Image",
                                           command=self.browse_image)
        self.button_browse.pack(pady=5)
        self.label_image = ctk.CTkLabel(self, text="(Chưa chọn ảnh)")
        self.label_image.pack(pady=5)

        self.button_add = ctk.CTkButton(self, text="Thêm" if self.language == "Tiếng Việt" else "Add",
                                        command=self.add_student)
        self.button_add.pack(pady=10)
        self.selected_image = ""

    def browse_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh" if self.language == "Tiếng Việt" else "Select Image",
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.selected_image = file_path
            self.label_image.configure(text=file_path)

    def add_student(self):
        name = self.entry_name.get().strip()
        class_name = self.entry_class.get().strip()
        if not name or not class_name or not self.selected_image:
            messagebox.showerror("Lỗi" if self.language == "Tiếng Việt" else "Error",
                                 "Vui lòng nhập đầy đủ thông tin" if self.language == "Tiếng Việt" else "Please fill all fields")
            return
        try:
            from DatabaseHooking import add_student
            add_student(self.cursor, self.cnx, name, class_name, self.selected_image)
            messagebox.showinfo("Info",
                                "Thêm học sinh thành công" if self.language == "Tiếng Việt" else "Student added successfully")
            if self.on_success_callback:
                self.on_success_callback()
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))


def edit_student_operation(parent, cnx, cursor, language, student):
    trans = translations[language]
    new_name = simpledialog.askstring("Edit Student",
                                      "Enter new name:" if language == "English" else "Nhập tên mới:",
                                      initialvalue=student[1])
    if new_name is None:
        return False
    new_class = simpledialog.askstring("Edit Student",
                                       "Enter new class:" if language == "English" else "Nhập lớp mới:",
                                       initialvalue=student[2])
    if new_class is None:
        return False
    try:
        from DatabaseHooking import update_student
        update_student(cursor, cnx, student[0], HoVaTen=new_name, Lop=new_class)
        messagebox.showinfo("Info",
                            "Student updated successfully." if language == "English" else "Học sinh đã được cập nhật thành công.")
        return True
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return False


def delete_student_operation(cursor, cnx, student, language):
    trans = translations[language]
    confirm = messagebox.askyesno("Confirm",
                                  "Are you sure you want to delete this student?" if language == "English" else "Bạn có chắc muốn xoá học sinh này?")
    if confirm:
        try:
            from DatabaseHooking import remove_student
            remove_student(cursor, cnx, student[0])
            messagebox.showinfo("Info",
                                "Student deleted successfully." if language == "English" else "Học sinh đã được xoá thành công.")
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False
    return False


def export_students_list(cursor, language):
    try:
        cursor.execute("SELECT id, HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh FROM Students ORDER BY id")
        rows = cursor.fetchall()
        if not rows:
            messagebox.showinfo("Info", "No data to export." if language == "English" else "Không có dữ liệu để xuất.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")],
                                                 title="Save as" if language == "English" else "Lưu dưới dạng")
        if file_path:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "HoVaTen", "Lop", "DiemDanhStatus", "ThoiGianDiemDanh"])
                writer.writerows(rows)
            messagebox.showinfo("Info",
                                "Exported successfully." if language == "English" else "Xuất dữ liệu thành công.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def set_cutoff_operation(language):
    cutoff_time = simpledialog.askstring("Set Cutoff",
                                         "Enter cutoff time (HH:MM):" if language == "English" else "Nhập thời gian hạn chót (HH:MM):")
    if cutoff_time:
        messagebox.showinfo("Info",
                            f"Cutoff time set to {cutoff_time}" if language == "English" else f"Đã cài đặt hạn chót là {cutoff_time}")
###############################
# USER ACCOUNT OPERATIONS
###############################
def edit_user_operation(cursor, cnx, account, language):
    trans = translations[language]
    new_username = simpledialog.askstring("Edit User",
                                           "Enter new username:" if language=="English" else "Nhập tên đăng nhập mới:",
                                           initialvalue=account[1])
    if new_username is None:
        return False
    new_role = simpledialog.askstring("Edit User",
                                      "Enter new role (superuser/admin/moderator/user):" if language=="English"
                                      else "Nhập quyền mới (superuser/admin/moderator/user):",
                                      initialvalue=account[2])
    if new_role is None:
        return False
    try:
        query = "UPDATE Users SET username=%s, role=%s WHERE id=%s"
        cursor.execute(query, (new_username, new_role, account[0]))
        cnx.commit()
        messagebox.showinfo("Info", "User updated successfully." if language=="English" else "Tài khoản đã được cập nhật thành công.")
        return True
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return False


def delete_user_operation(cursor, cnx, account, language):
    trans = translations[language]
    if account[2].lower() == "superuser":
        cursor.execute("SELECT COUNT(*) FROM Users WHERE LOWER(role) = 'superuser'")
        count = cursor.fetchone()[0]
        if count <= 1:
            messagebox.showerror("Error", "Cannot delete the only superuser account." if language=="English" else "Không thể xoá tài khoản superuser duy nhất.")
            return False
    confirm = messagebox.askyesno("Confirm",
                                  "Are you sure you want to delete this account?" if language=="English" else "Bạn có chắc muốn xoá tài khoản này?")
    if confirm:
        try:
            query = "DELETE FROM Users WHERE id=%s"
            cursor.execute(query, (account[0],))
            cnx.commit()
            messagebox.showinfo("Info", "User deleted successfully." if language=="English" else "Tài khoản đã được xoá thành công.")
            return True
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False
    return False
