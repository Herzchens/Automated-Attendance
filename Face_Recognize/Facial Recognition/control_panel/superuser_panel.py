# control_panel/superuser_panel.py
import customtkinter as ctk
from tkinter import messagebox
import sys
from .common import translations, CustomTable

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
        self.tabview = ctk.CTkTabview(self, width=1000, height=500)
        self.tabview.pack(pady=10, padx=40, fill="both", expand=True)
        self.tabview.add(self.trans["user_accounts_tab"])
        self.tabview.add(self.trans["students_tab"])
        accounts_frame = self.tabview.tab(self.trans["user_accounts_tab"])
        self.accounts_table = CustomTable(accounts_frame, columns=["ID", "Username", "Role"], corner_radius=8)
        self.accounts_table.pack(fill="both", expand=True)
        self.accounts_toolbar = ctk.CTkFrame(accounts_frame)
        self.accounts_toolbar.pack(pady=5)
        self.button_edit_account = ctk.CTkButton(self.accounts_toolbar, text=self.trans["edit_user"], command=self.edit_user)
        self.button_edit_account.grid(row=0, column=0, padx=10, pady=5)
        self.button_delete_account = ctk.CTkButton(self.accounts_toolbar, text=self.trans["delete_user"], command=self.delete_user)
        self.button_delete_account.grid(row=0, column=1, padx=10, pady=5)
        students_frame = self.tabview.tab(self.trans["students_tab"])
        columns = [self.trans["col_index"], self.trans["col_name"], self.trans["col_attendance"]]
        self.students_table = CustomTable(students_frame, columns=columns, corner_radius=8)
        self.students_table.pack(fill="both", expand=True)
        self.search_frame = ctk.CTkFrame(self)
        self.search_frame.pack(pady=10)
        self.search_entry = ctk.CTkEntry(self.search_frame, placeholder_text=self.trans["search"])
        self.search_entry.pack(side="left", padx=10)
        self.search_button = ctk.CTkButton(self.search_frame, text=self.trans["search"], command=self.search_student)
        self.search_button.pack(side="left", padx=10)
        self.frame_buttons = ctk.CTkFrame(self)
        self.frame_buttons.pack(pady=10)
        self.button_logout = ctk.CTkButton(self.frame_buttons, text=self.trans["logout"], width=150, command=self.logout)
        self.button_logout.grid(row=0, column=0, padx=20, pady=10)
        self.button_quit = ctk.CTkButton(self.frame_buttons, text=self.trans["quit"], width=150, command=self.quit_app)
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
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching account data:\n{e}")
            return
        self.accounts_table.clear_rows()
        if not rows:
            self.accounts_table.pack_forget()
            self.accounts_watermark = ctk.CTkLabel(self.tabview.tab(self.trans["user_accounts_tab"]), text=self.trans["no_data"], font=("Arial", 48), fg_color="transparent")
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
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching student data:\n{e}")
            return
        self.students_table.clear_rows()
        if not rows:
            self.students_table.pack_forget()
            self.students_watermark = ctk.CTkLabel(self.tabview.tab(self.trans["students_tab"]), text=self.trans["no_data"], font=("Arial", 48), fg_color="transparent")
            self.students_watermark.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "students_watermark"):
                self.students_watermark.destroy()
            self.students_table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                attendance = row[3] if row[3] is not None else row[2]
                self.students_table.add_row((idx, row[1], attendance))

    def search_student(self):
        search_term = self.search_entry.get().strip().lower()
        if not search_term:
            return
        self.students_table.clear_rows()
        query = "SELECT id, HoVaTen, DiemDanhStatus, ThoiGianDiemDanh FROM Students WHERE LOWER(HoVaTen) LIKE %s ORDER BY id"
        self.cursor.execute(query, (f"%{search_term}%",))
        rows = self.cursor.fetchall()
        if not rows:
            self.students_table.pack_forget()
            self.students_watermark = ctk.CTkLabel(self.tabview.tab(self.trans["students_tab"]), text=self.trans["no_data"], font=("Arial", 48), fg_color="transparent")
            self.students_watermark.place(relx=0.5, rely=0.5, anchor="center")
        else:
            if hasattr(self, "students_watermark"):
                self.students_watermark.destroy()
            self.students_table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                attendance = row[3] if row[3] is not None else row[2]
                self.students_table.add_row((idx, row[1], attendance))

    def export_data(self):
        messagebox.showinfo("Info", "Export List clicked!")

    def logout(self):
        self.destroy()
        from . import open_user_login_window
        open_user_login_window(self.cnx, self.cursor, self.language)

    def quit_app(self):
        self.destroy()
        sys.exit(0)

    def get_selected_account(self):
        idx = self.accounts_table.selected_row_index
        if not idx:
            return None
        return self.accounts_table.rows_data[idx - 1]

    def edit_user(self):
        account = self.get_selected_account()
        if not account:
            messagebox.showerror("Error", "Please select an account to edit." if self.language=="English" else "Vui lòng chọn tài khoản để chỉnh sửa.")
            return
        messagebox.showinfo("Info", f"Edit account {account[1]} clicked." if self.language=="English" else f"Chỉnh sửa tài khoản {account[1]} được chọn.")

    def delete_user(self):
        account = self.get_selected_account()
        if not account:
            messagebox.showerror("Error", "Please select an account to delete." if self.language=="English" else "Vui lòng chọn tài khoản để xoá.")
            return
        if account[2].lower() == "superuser":
            self.cursor.execute("SELECT COUNT(*) FROM Users WHERE LOWER(role) = 'superuser'")
            count = self.cursor.fetchone()[0]
            if count <= 1:
                messagebox.showerror("Error", "Cannot delete the only superuser account." if self.language=="English" else "Không thể xoá tài khoản superuser duy nhất.")
                return
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to delete this account?" if self.language=="English" else "Bạn có chắc muốn xoá tài khoản này?")
        if confirm:
            messagebox.showinfo("Info", f"Account {account[1]} deleted." if self.language=="English" else f"Tài khoản {account[1]} đã được xoá.")
            self.load_accounts_data()
