# control_panel/user_panel.py
import customtkinter as ctk
from tkinter import messagebox
import sys
from .common import translations, CustomTable

class UserControlPanel(ctk.CTk):
    def __init__(self, user_info, cnx, cursor, language):
        super().__init__()
        self.user_info = user_info
        self.cnx = cnx
        self.cursor = cursor
        self.language = language
        self.trans = translations[self.language]
        self.current_mode = "Light"
        self.title(self.trans["control_title"] + " - User")
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
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(pady=10, padx=40, fill="both", expand=True)
        columns = [self.trans["col_index"], self.trans["col_name"], self.trans["col_attendance"]]
        self.custom_table = CustomTable(self.table_frame, columns=columns, corner_radius=8)
        self.custom_table.pack(fill="both", expand=True)
        self.frame_buttons = ctk.CTkFrame(self)
        self.frame_buttons.pack(pady=10)
        self.button_export = ctk.CTkButton(self.frame_buttons, text=self.trans["export"], width=150, command=self.export_data)
        self.button_export.grid(row=0, column=0, padx=20, pady=10)
        self.button_logout = ctk.CTkButton(self.frame_buttons, text=self.trans["logout"], width=150, command=self.logout)
        self.button_logout.grid(row=0, column=1, padx=20, pady=10)
        self.button_quit = ctk.CTkButton(self.frame_buttons, text=self.trans["quit"], width=150, command=self.quit_app)
        self.button_quit.grid(row=0, column=2, padx=20, pady=10)

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
        assigned_class = "12S"
        query = "SELECT id, HoVaTen, DiemDanhStatus, ThoiGianDiemDanh FROM Students WHERE Lop = %s ORDER BY id"
        try:
            self.cursor.execute(query, (assigned_class,))
            rows = self.cursor.fetchall()
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching data:\n{e}")
            return
        self.custom_table.clear_rows()
        if not rows:
            self.custom_table.pack_forget()
            self.watermark_label = ctk.CTkLabel(self.table_frame, text=self.trans["no_data"], font=("Arial", 48), fg_color="transparent")
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
