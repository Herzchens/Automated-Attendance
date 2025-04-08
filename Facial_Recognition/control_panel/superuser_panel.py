import customtkinter as ctk
from tkinter import messagebox
from translator import translations
from .components import (
    CustomTable,
    add_student_ui,
    edit_student_ui,
    remove_student_ui,
    add_students_batch_ui,
    edit_user_operation,
    delete_user_operation,
    CutoffTimeWindowGMT
)
from DatabaseHooking import (
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

        self.create_tabs()
        self.create_theme_toggle()

        self.load_students_data()
        self.load_accounts_data()

    def create_tabs(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True)

        self.tab_students = self.tabview.add(self.trans["students_tab"])
        self.tab_users = self.tabview.add(self.trans["user_accounts_tab"])

        self.create_students_widgets(self.tab_students)
        self.create_users_widgets(self.tab_users)

    def create_students_widgets(self, parent):
        greeting = f"{self.trans['welcome']} {self.user_info[1]} ({self.user_info[2]})"
        self.label_greeting = ctk.CTkLabel(parent, text=greeting, font=("Arial", 24))
        self.label_greeting.pack(pady=20)

        self.student_toolbar = ctk.CTkFrame(parent)
        self.student_toolbar.pack(pady=5, fill="x")

        self.button_add_student = ctk.CTkButton(
            self.student_toolbar, text=self.trans["add_student"],
            command=lambda: add_student_ui(self, self.cnx, self.cursor, self.language,
                                           on_success_callback=self.load_students_data)
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

        self.button_export_students = ctk.CTkButton(
            self.student_toolbar, text=self.trans["export"],
            command=lambda: export_students_list(self.cursor, self.language)
        )
        self.button_export_students.grid(row=0, column=3, padx=10, pady=5)

        self.button_cutoff = ctk.CTkButton(
            self.student_toolbar, text=self.trans["set_cutoff"],
            command=lambda: CutoffTimeWindowGMT(self, self.cnx, self.cursor, self.language)
        )
        self.button_cutoff.grid(row=0, column=4, padx=10, pady=5)

        self.button_batch_add = ctk.CTkButton(
            self.student_toolbar, text=self.trans["batch_add"],
            command=lambda: add_students_batch_ui(self, self.cnx, self.cursor, self.language,
                                                  on_success_callback=self.load_students_data)
        )
        self.button_batch_add.grid(row=0, column=5, padx=10, pady=5)

        self.students_table = CustomTable(
            parent,
            columns=[self.trans["col_index"], self.trans["col_name"], "Lớp", self.trans["col_attendance"]],
            corner_radius=8
        )
        self.students_table.pack(fill="both", expand=True)

        self.search_frame = ctk.CTkFrame(parent)
        self.search_frame.pack(pady=10)
        self.search_entry = ctk.CTkEntry(self.search_frame, placeholder_text=self.trans["search"])
        self.search_entry.pack(side="left", padx=10)
        self.search_button = ctk.CTkButton(self.search_frame, text=self.trans["search"], command=self.search_student)
        self.search_button.pack(side="left", padx=10)

        self.frame_buttons_bottom = ctk.CTkFrame(parent)
        self.frame_buttons_bottom.pack(pady=10)
        self.button_logout = ctk.CTkButton(
            self.frame_buttons_bottom,
            text=self.trans["logout"],
            width=150,
            command=self.logout
        )
        self.button_logout.grid(row=0, column=0, padx=20, pady=10)
        self.button_quit = ctk.CTkButton(
            self.frame_buttons_bottom,
            text=self.trans["quit"],
            width=150,
            command=self.quit_app
        )
        self.button_quit.grid(row=0, column=1, padx=20, pady=10)

    def create_users_widgets(self, parent):
        label_accounts = ctk.CTkLabel(parent, text=self.trans["user_accounts_tab"], font=("Arial", 24))
        label_accounts.pack(pady=20)

        self.accounts_toolbar = ctk.CTkFrame(parent)
        self.accounts_toolbar.pack(pady=5, fill="x")

        self.button_add_user = ctk.CTkButton(
            self.accounts_toolbar, text=self.trans["add_user"],
            command=self.add_user
        )
        self.button_add_user.grid(row=0, column=0, padx=10, pady=5)

        self.button_edit_user = ctk.CTkButton(
            self.accounts_toolbar, text=self.trans["edit_user"],
            command=self.edit_user
        )
        self.button_edit_user.grid(row=0, column=1, padx=10, pady=5)

        self.button_delete_user = ctk.CTkButton(
            self.accounts_toolbar, text=self.trans["delete_user"],
            command=self.delete_user
        )
        self.button_delete_user.grid(row=0, column=2, padx=10, pady=5)

        self.users_table = CustomTable(
            parent,
            columns=[self.trans["col_index"], self.trans["username"], "Role"],
            corner_radius=8
        )
        self.users_table.pack(fill="both", expand=True)

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

    # =============== Phần quản lý học sinh ===============
    def load_students_data(self):
        query = "SELECT id, HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh FROM Students ORDER BY id"
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            self.students_raw_data = rows
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching student data:\n{e}")
            return

        self.students_table.clear_rows()
        if not rows:
            self.students_table.pack_forget()
            self.students_watermark = ctk.CTkLabel(
                self.tab_students,
                text=self.trans["no_data"],
                font=("Arial", 48),
                fg_color="transparent"
            )
            self.students_watermark.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "students_watermark"):
                self.students_watermark.destroy()
            self.students_table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                if row[4] is not None:
                    attendance = calculate_attendance_status(row[4], self.language)
                else:
                    attendance = '✖'
                self.students_table.add_row((idx, row[1], row[2], attendance))

    def edit_student(self):
        if self.students_table.selected_row_index is None:
            messagebox.showerror("Error", "Vui lòng chọn học sinh cần chỉnh sửa.")
            return
        index = self.students_table.selected_row_index - 1
        try:
            student = self.students_raw_data[index]
        except IndexError:
            messagebox.showerror("Error", "Lỗi chọn học sinh.")
            return
        try:
            edit_student_ui(self, self.cnx, self.cursor, self.language, student,
                            on_success_callback=self.load_students_data)
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi chỉnh sửa học sinh:\n{e}")

    def delete_student(self):
        if self.students_table.selected_row_index is None:
            messagebox.showerror("Error", "Vui lòng chọn học sinh cần xoá.")
            return
        index = self.students_table.selected_row_index - 1
        try:
            student = self.students_raw_data[index]
        except IndexError:
            messagebox.showerror("Error", "Lỗi chọn học sinh.")
            return
        confirm = messagebox.askyesno("Confirm", "Bạn có chắc muốn xoá học sinh này?")
        if confirm:
            try:
                remove_student_ui(self, self.cnx, self.cursor, self.language, student,
                                  on_success_callback=self.load_students_data)
            except Exception as e:
                messagebox.showerror("Error", f"Lỗi xoá học sinh:\n{e}")

    def search_student(self):
        keyword = self.search_entry.get().strip().lower()
        if not keyword:
            self.load_students_data()
            return

        query = "SELECT id, HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh FROM Students WHERE LOWER(HoVaTen) LIKE %s ORDER BY id"
        try:
            self.cursor.execute(query, (f"%{keyword}%",))
            rows = self.cursor.fetchall()
            self.students_raw_data = rows
        except Exception as e:
            messagebox.showerror("Error", f"Error searching for student:\n{e}")
            return

        self.students_table.clear_rows()
        if not rows:
            self.students_table.pack_forget()
            self.students_watermark = ctk.CTkLabel(
                self.table_frame,
                text=self.trans["no_data"],
                font=("Arial", 48),
                fg_color="transparent"
            )
            self.students_watermark.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "students_watermark"):
                self.students_watermark.destroy()
            self.students_table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                if row[4] is not None:
                    attendance = calculate_attendance_status(row[4], self.language)
                else:
                    attendance = '✖'
                self.students_table.add_row((idx, row[1], row[2], attendance))

    def load_accounts_data(self):
        query = "SELECT id, username, role FROM Users ORDER BY id"
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            self.accounts_raw_data = rows
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching account data:\n{e}")
            return

        self.users_table.clear_rows()
        if not rows:
            self.users_table.pack_forget()
            self.users_watermark = ctk.CTkLabel(
                self.tab_users,
                text=self.trans["no_data"],
                font=("Arial", 48),
                fg_color="transparent"
            )
            self.users_watermark.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "users_watermark"):
                self.users_watermark.destroy()
            self.users_table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                self.users_table.add_row((idx, row[1], row[2]))

    def add_user(self):
        import tkinter.simpledialog as sd
        trans = self.trans

        new_username = sd.askstring(
            trans["add_user"],
            "Username:" if self.language == "English" else "Nhập tên tài khoản:"
        )
        if not new_username:
            return

        new_password = sd.askstring(
            trans["add_user"],
            "Password:" if self.language == "English" else "Nhập mật khẩu:",
            show="*"
        )
        if not new_password:
            return

        new_role = sd.askstring(
            trans["add_user"],
            "Role (superuser/admin/moderator/user):" if self.language == "English"
            else "Quyền (superuser/admin/moderator/user):",
            initialvalue="user"
        )
        if not new_role:
            return

        try:
            sql = "INSERT INTO Users (username, password, role) VALUES (%s, %s, %s)"
            self.cursor.execute(sql, (new_username, new_password, new_role))
            self.cnx.commit()
            messagebox.showinfo("Info", "User added successfully." if self.language == "English" else "Thêm tài khoản thành công.")
            self.load_accounts_data()
        except Exception as e:
            messagebox.showerror("Error", f"Error adding user:\n{e}")

    def edit_user(self):
        if self.users_table.selected_row_index is None:
            return
        index = self.users_table.selected_row_index - 1
        try:
            account = self.accounts_raw_data[index]
        except IndexError:
            return
        if edit_user_operation(self.cursor, self.cnx, account, self.language):
            self.load_accounts_data()

    def delete_user(self):
        if self.users_table.selected_row_index is None:
            return
        index = self.users_table.selected_row_index - 1
        try:
            account = self.accounts_raw_data[index]
        except IndexError:
            return
        if delete_user_operation(self.cursor, self.cnx, account, self.language):
            self.load_accounts_data()
    def logout(self):
        self.destroy()

    def quit_app(self):
        self.destroy()

if __name__ == "__main__":
    user_info = (1, "SuperUser", "superuser")
    cnx, cursor = None, None
    language = "Tiếng Việt"
    app = SuperUserControlPanel(user_info, cnx, cursor, language)
    app.mainloop()
