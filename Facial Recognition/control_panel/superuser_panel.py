# superuser_panel.py
import customtkinter as ctk
from tkinter import messagebox, simpledialog
import sys, datetime
from DatabaseHooking import remove_student, update_student
from .common import translations, CustomTable
from .PanelOperations import (
    add_student_ui,
    edit_student_ui,
    remove_student_ui,
    set_cutoff_time,
    export_students_list,
    calculate_attendance_status
)

class SuperUserControlPanel(ctk.CTk):
    def __init__(self, user_info, cnx, cursor, language):
        super().__init__()
        self.user_info = user_info
        self.cnx = cnx
        self.cursor = cursor
        self.language = language
        self.trans = translations[self.language]
        self.current_mode = "Light"
        self.title(self.trans["control_title"] + " - SuperUser")
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

        # Tabview với 2 tab: Quản lý tài khoản và Quản lý học sinh
        self.tabview = ctk.CTkTabview(self, width=1000, height=500)
        self.tabview.pack(pady=10, padx=40, fill="both", expand=True)
        self.tabview.add(self.trans["user_accounts_tab"])
        self.tabview.add(self.trans["students_tab"])

        # Tab quản lý tài khoản (ví dụ: hiển thị danh sách tài khoản)
        accounts_frame = self.tabview.tab(self.trans["user_accounts_tab"])
        self.accounts_table = CustomTable(accounts_frame, columns=["ID", "Username", "Role"], corner_radius=8)
        self.accounts_table.pack(fill="both", expand=True)
        self.accounts_toolbar = ctk.CTkFrame(accounts_frame)
        self.accounts_toolbar.pack(pady=5)
        # Các nút thao tác tài khoản (edit, delete) nếu cần

        # Tab quản lý học sinh
        students_frame = self.tabview.tab(self.trans["students_tab"])
        self.student_toolbar = ctk.CTkFrame(students_frame)
        self.student_toolbar.pack(pady=5, fill="x")
        self.button_add_student = ctk.CTkButton(
            self.student_toolbar, text=self.trans["add_student"],
            command=lambda: add_student_ui(self, self.cnx, self.cursor, self.language, on_success_callback=self.load_students_data)
        )
        self.button_add_student.grid(row=0, column=0, padx=10, pady=5)
        self.button_edit_student = ctk.CTkButton(
            self.student_toolbar, text=self.trans["edit_student"],
            command=self.edit_student
        )
        self.button_edit_student.grid(row=0, column=1, padx=10, pady=5)
        self.button_delete_student = ctk.CTkButton(
            self.student_toolbar, text=self.trans["delete_student"],
            command=self.delete_student
        )
        self.button_delete_student.grid(row=0, column=2, padx=10, pady=5)
        self.button_export = ctk.CTkButton(
            self.student_toolbar, text=self.trans["export"],
            command=lambda: export_students_list(self.cursor, self.language)
        )
        self.button_export.grid(row=0, column=3, padx=10, pady=5)
        self.button_cutoff = ctk.CTkButton(
            self.student_toolbar, text=self.trans["set_cutoff"],
            command=lambda: set_cutoff_time(self.language)
        )
        self.button_cutoff.grid(row=0, column=4, padx=10, pady=5)

        self.search_frame = ctk.CTkFrame(self)
        self.search_frame.pack(pady=10)
        self.search_entry = ctk.CTkEntry(self.search_frame, placeholder_text=self.trans["search"])
        self.search_entry.pack(side="left", padx=10)
        self.search_button = ctk.CTkButton(self.search_frame, text=self.trans["search"], command=self.search_student)
        self.search_button.pack(side="left", padx=10)

        self.frame_buttons = ctk.CTkFrame(self)
        self.frame_buttons.pack(pady=10)
        self.button_logout = ctk.CTkButton(
            self.frame_buttons, text=self.trans["logout"], width=150,
            command=self.logout
        )
        self.button_logout.grid(row=0, column=0, padx=20, pady=10)
        self.button_quit = ctk.CTkButton(
            self.frame_buttons, text=self.trans["quit"], width=150,
            command=self.quit_app
        )
        self.button_quit.grid(row=0, column=1, padx=20, pady=10)

    def create_theme_toggle(self):
        btn_text = self.trans["toggle_light"] if self.current_mode == "Dark" else self.trans["toggle_dark"]
        self.toggle_button = ctk.CTkButton(self, text=btn_text, width=40, height=40, corner_radius=8, command=self.toggle_theme)
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
        self.load_accounts_data()
        self.load_students_data()

    def load_accounts_data(self):
        query = "SELECT id, username, role FROM Users ORDER BY id"
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            self.accounts_raw_data = rows
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching account data:\n{e}")
            return
        self.accounts_table.clear_rows()
        if not rows:
            self.accounts_table.pack_forget()
            self.accounts_watermark = ctk.CTkLabel(self.tabview.tab(self.trans["user_accounts_tab"]),
                                                   text=self.trans["no_data"], font=("Arial", 48), fg_color="transparent")
            self.accounts_watermark.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "accounts_watermark"):
                self.accounts_watermark.destroy()
            self.accounts_table.pack(fill="both", expand=True)
            for row in rows:
                self.accounts_table.add_row((row[0], row[1], row[2]))

    def load_students_data(self):
        query = "SELECT id, HoVaTen, DiemDanhStatus, ThoiGianDiemDanh FROM Students ORDER BY id"
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            self.students_raw_data = rows
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching student data:\n{e}")
            return
        # Cập nhật bảng hiển thị
        # (Giả sử bạn có một bảng hiển thị trong tab hoặc ngoài tab, tương tự như admin panel)
        if hasattr(self, "students_table"):
            self.students_table.clear_rows()
            if not rows:
                self.students_table.pack_forget()
                self.students_watermark = ctk.CTkLabel(self.tabview.tab(self.trans["students_tab"]),
                                                       text=self.trans["no_data"], font=("Arial", 48), fg_color="transparent")
                self.students_watermark.place(relx=0.5, rely=0.5, anchor="center")
            else:
                if hasattr(self, "students_watermark"):
                    self.students_watermark.destroy()
                self.students_table.pack(fill="both", expand=True)
                for idx, row in enumerate(rows, start=1):
                    if row[3] is not None and isinstance(row[3], datetime.datetime):
                        attendance = calculate_attendance_status(row[3], self.language)
                    else:
                        attendance = '✖'
                    self.students_table.add_row((idx, row[1], attendance))
        else:
            # Nếu không có bảng riêng, bạn có thể hiển thị lại theo cách khác
            pass

    def get_selected_student(self):
        idx = self.students_table.selected_row_index
        if idx is None or idx < 1 or idx > len(self.students_raw_data):
            return None
        return self.students_raw_data[idx - 1]

    def edit_student(self):
        student = self.get_selected_student()
        if not student:
            messagebox.showerror("Error", "Please select a student to edit." if self.language=="English"
                                                  else "Vui lòng chọn học sinh để chỉnh sửa.")
            return
        edit_student_ui(self, self.cnx, self.cursor, self.language, student, on_success_callback=self.load_students_data)

    def delete_student(self):
        student = self.get_selected_student()
        if not student:
            messagebox.showerror("Error", "Please select a student to delete." if self.language=="English"
                                                  else "Vui lòng chọn học sinh để xoá.")
            return
        remove_student_ui(self, self.cnx, self.cursor, self.language, student, on_success_callback=self.load_students_data)

    def search_student(self):
        search_term = self.search_entry.get().strip().lower()
        if not search_term:
            self.load_students_data()
            return
        query = "SELECT id, HoVaTen, DiemDanhStatus, ThoiGianDiemDanh FROM Students WHERE LOWER(HoVaTen) LIKE %s ORDER BY id"
        try:
            self.cursor.execute(query, (f"%{search_term}%",))
            rows = self.cursor.fetchall()
            self.students_raw_data = rows
        except Exception as e:
            messagebox.showerror("Error", f"Error searching data:\n{e}")
            return
        if hasattr(self, "students_table"):
            self.students_table.clear_rows()
            if not rows:
                self.students_table.pack_forget()
                self.students_watermark = ctk.CTkLabel(self.tabview.tab(self.trans["students_tab"]),
                                                       text=self.trans["no_data"], font=("Arial", 48), fg_color="transparent")
                self.students_watermark.place(relx=0.5, rely=0.5, anchor="center")
            else:
                if hasattr(self, "students_watermark"):
                    self.students_watermark.destroy()
                self.students_table.pack(fill="both", expand=True)
                for idx, row in enumerate(rows, start=1):
                    if row[3] is not None and isinstance(row[3], datetime.datetime):
                        attendance = calculate_attendance_status(row[3], self.language)
                    else:
                        attendance = '✖'
                    self.students_table.add_row((idx, row[1], attendance))

    def logout(self):
        self.destroy()

    def quit_app(self):
        self.destroy()
        sys.exit(0)
