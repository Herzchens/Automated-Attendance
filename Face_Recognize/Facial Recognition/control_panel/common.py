# control_panel/common.py
import customtkinter as ctk
from tkinter import messagebox

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
