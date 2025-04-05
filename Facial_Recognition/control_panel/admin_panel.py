import customtkinter as ctk
from tkinter import messagebox
import sys
from translator import translations
from DatabaseHooking import export_students_list, calculate_attendance_status
from .components import (
    CustomTable,
    add_student_ui,
    edit_student_ui,
    remove_student_ui,
    add_students_batch_ui,
    CutoffTimeWindowGMT
)

def open_cutoff_ui(self):
    CutoffTimeWindowGMT(self, self.cnx, self.cursor, self.language)

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

        # Search Bar
        self.search_frame = ctk.CTkFrame(self)
        self.search_frame.pack(pady=10)
        self.search_entry = ctk.CTkEntry(self.search_frame, placeholder_text=self.trans["search"])
        self.search_entry.pack(side="left", padx=10)
        self.search_button = ctk.CTkButton(self.search_frame, text=self.trans["search"], command=self.search_student)
        self.search_button.pack(side="left", padx=10)

        # Tools Bar
        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.pack(pady=10, padx=40, fill="x")
        self.button_add = ctk.CTkButton(
            self.frame_controls,
            text=self.trans["add_student"],
            command=lambda: add_student_ui(self, self.cnx, self.cursor, self.language, on_success_callback=self.fetch_data)
        )
        self.button_add.grid(row=0, column=0, padx=10, pady=10)
        self.button_delete = ctk.CTkButton(
            self.frame_controls,
            text=self.trans["delete_student"],
            command=self.delete_student
        )
        self.button_delete.grid(row=0, column=1, padx=10, pady=10)
        self.button_edit = ctk.CTkButton(
            self.frame_controls,
            text=self.trans["edit_student"],
            command=self.edit_student
        )
        self.button_edit.grid(row=0, column=2, padx=10, pady=10)
        self.button_batch_add = ctk.CTkButton(
            self.frame_controls,
            text=self.trans.get("batch_add", "Thêm hàng loạt"),
            command=lambda: add_students_batch_ui(self, self.cnx, self.cursor, self.language, on_success_callback=self.fetch_data)
        )
        self.button_batch_add.grid(row=0, column=3, padx=10, pady=10)
        self.button_cutoff = ctk.CTkButton(
            self.frame_controls,
            text=self.trans["set_cutoff"],
            command=lambda: open_cutoff_ui(self)
        )
        self.button_cutoff.grid(row=0, column=4, padx=10, pady=10)

        # Table of Students
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(pady=10, padx=40, fill="both", expand=True)
        columns = [self.trans["col_index"], self.trans["col_name"], "Lớp", self.trans["col_attendance"]]
        self.custom_table = CustomTable(self.table_frame, columns=columns, corner_radius=8)
        self.custom_table.pack(fill="both", expand=True)

        # Under Buttons
        self.frame_buttons_bottom = ctk.CTkFrame(self)
        self.frame_buttons_bottom.pack(pady=10)
        self.button_export = ctk.CTkButton(
            self.frame_buttons_bottom,
            text=self.trans["export"],
            width=150,
            command=lambda: export_students_list(self.cursor, self.language)
        )
        self.button_export.grid(row=0, column=0, padx=20, pady=10)
        self.button_logout = ctk.CTkButton(
            self.frame_buttons_bottom,
            text=self.trans["logout"],
            width=150,
            command=self.logout
        )
        self.button_logout.grid(row=0, column=1, padx=20, pady=10)
        self.button_quit = ctk.CTkButton(
            self.frame_buttons_bottom,
            text=self.trans["quit"],
            width=150,
            command=self.quit_app
        )
        self.button_quit.grid(row=0, column=2, padx=20, pady=10)

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
        query = "SELECT id, HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh FROM Students ORDER BY HoVaTen ASC"
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching data:\n{e}")
            return

        self.students_raw_data = rows
        self.custom_table.clear_rows()

        if not rows:
            self.custom_table.pack_forget()
            self.watermark_label = ctk.CTkLabel(
                self.table_frame,
                text=self.trans["no_data"],
                font=("Arial", 48),
                fg_color="transparent"
            )
            self.watermark_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "watermark_label") and self.watermark_label.winfo_exists():
                self.watermark_label.destroy()
            self.custom_table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                if row[4] is not None:
                    attendance = calculate_attendance_status(row[4], self.language)
                else:
                    attendance = '✖'
                self.custom_table.add_row((idx, row[1], row[2], attendance))

    def search_student(self):
        keyword = self.search_entry.get().strip().lower()
        if not keyword:
            self.fetch_data()
            return

        query = "SELECT id, HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh FROM facial_recognition.Students WHERE LOWER(HoVaTen) LIKE %s ORDER BY HoVaTen ASC"
        try:
            self.cursor.execute(query, (f"%{keyword}%",))
            rows = self.cursor.fetchall()
            self.students_raw_data = rows
        except Exception as e:
            messagebox.showerror("Error", f"Error searching for student:\n{e}")
            return

        self.custom_table.clear_rows()
        if not rows:
            self.custom_table.pack_forget()
            self.watermark_label = ctk.CTkLabel(
                self.table_frame,
                text=self.trans["no_data"],
                font=("Arial", 48),
                fg_color="transparent"
            )
            self.watermark_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "watermark_label") and self.watermark_label.winfo_exists():
                self.watermark_label.destroy()
            self.custom_table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                if row[4] is not None:
                    attendance = calculate_attendance_status(row[4], self.language)
                else:
                    attendance = '✖'
                self.custom_table.add_row((idx, row[1], row[2], attendance))

    def delete_student(self):
        if self.custom_table.selected_row_index is None:
            messagebox.showerror("Error", "Vui lòng chọn học sinh cần xoá.")
            return
        index = self.custom_table.selected_row_index - 1
        try:
            student = self.students_raw_data[index]
        except IndexError:
            messagebox.showerror("Error", "Lỗi chọn học sinh.")
            return
        confirm = messagebox.askyesno("Confirm", "Bạn có chắc muốn xoá học sinh này?")
        if confirm:
            try:
                remove_student_ui(self, self.cnx, self.cursor, self.language, student,
                                  on_success_callback=self.fetch_data)
            except Exception as e:
                messagebox.showerror("Error", f"Lỗi xoá học sinh:\n{e}")

    def edit_student(self):
        if self.custom_table.selected_row_index is None:
            messagebox.showerror("Error", "Vui lòng chọn học sinh cần chỉnh sửa.")
            return
        index = self.custom_table.selected_row_index - 1
        try:
            student = self.students_raw_data[index]
        except IndexError:
            messagebox.showerror("Error", "Lỗi chọn học sinh.")
            return
        try:
            edit_student_ui(self, self.cnx, self.cursor, self.language, student,
                            on_success_callback=self.fetch_data)
        except Exception as e:
            messagebox.showerror("Error", f"Lỗi chỉnh sửa học sinh:\n{e}")

    def logout(self):
        self.destroy()

    def quit_app(self):
        self.destroy()
        sys.exit(0)

if __name__ == "__main__":
    user_info = (1, "Admin", "admin")
    cnx, cursor = None, None
    language = "Tiếng Việt"
    app = AdminControlPanel(user_info, cnx, cursor, language)
    app.mainloop()
