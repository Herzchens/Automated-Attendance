# control_panel/moderator_panel.py
import customtkinter as ctk
from tkinter import messagebox
import sys
from .common import translations, CustomTable

class ModeratorControlPanel(ctk.CTk):
    def __init__(self, user_info, cnx, cursor, language):
        super().__init__()
        self.user_info = user_info
        self.cnx = cnx
        self.cursor = cursor
        self.language = language
        self.trans = translations[self.language]
        self.current_mode = "Light"
        self.title(self.trans["control_title"] + " - Moderator")
        self.geometry("1200x800")
        try:
            self.state("zoomed")
        except Exception:
            pass
        self.resizable(True, True)
        self.create_widgets()
        self.create_theme_toggle()
        self.load_tab_data()

    def create_widgets(self):
        greeting = f"{self.trans['welcome']} {self.user_info[1]} ({self.user_info[2]})"
        self.label_greeting = ctk.CTkLabel(self, text=greeting, font=("Arial", 24))
        self.label_greeting.pack(pady=20)
        self.tabview = ctk.CTkTabview(self, width=1000, height=500)
        self.tabview.pack(pady=10, padx=40, fill="both", expand=True)
        query = "SELECT DISTINCT Lop FROM Students ORDER BY Lop"
        try:
            self.cursor.execute(query)
            classes = [r[0] for r in self.cursor.fetchall()]
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching classes:\n{e}")
            classes = []
        self.classes = classes
        if not classes:
            no_data_label = ctk.CTkLabel(self, text=self.trans["no_data"], font=("Arial", 48))
            no_data_label.pack(expand=True)
        else:
            for lop in self.classes:
                self.tabview.add(lop)
                tab_frame = self.tabview.tab(lop)
                custom_table = CustomTable(tab_frame, columns=[self.trans["col_index"], self.trans["col_name"], self.trans["col_attendance"]], corner_radius=8)
                custom_table.pack(fill="both", expand=True)
                tab_frame.custom_table = custom_table
        self.search_frame = ctk.CTkFrame(self)
        self.search_frame.pack(pady=10)
        self.search_entry = ctk.CTkEntry(self.search_frame, placeholder_text=self.trans["search"])
        self.search_entry.pack(side="left", padx=10)
        self.search_button = ctk.CTkButton(self.search_frame, text=self.trans["search"], command=self.search_student)
        self.search_button.pack(side="left", padx=10)
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

    def load_tab_data(self):
        for lop in self.classes:
            tab_frame = self.tabview.tab(lop)
            table = tab_frame.custom_table
            table.clear_rows()
            query = "SELECT id, HoVaTen, DiemDanhStatus, ThoiGianDiemDanh FROM Students WHERE Lop=%s ORDER BY id"
            try:
                self.cursor.execute(query, (lop,))
                rows = self.cursor.fetchall()
            except Exception as e:
                messagebox.showerror("Error", f"Error fetching data for class {lop}:\n{e}")
                continue
            if not rows:
                table.pack_forget()
                watermark = ctk.CTkLabel(tab_frame, text=self.trans["no_data"], font=("Arial", 48), fg_color="transparent")
                watermark.place(relx=0.5, rely=0.5, anchor="center")
            else:
                table.pack(fill="both", expand=True)
                for idx, row in enumerate(rows, start=1):
                    attendance = row[3] if row[3] is not None else row[2]
                    table.add_row((idx, row[1], attendance))

    def search_student(self):
        search_term = self.search_entry.get().strip().lower()
        if not search_term:
            return
        current_tab = self.tabview.get()
        tab_frame = self.tabview.tab(current_tab)
        table = tab_frame.custom_table
        table.clear_rows()
        query = "SELECT id, HoVaTen, DiemDanhStatus, ThoiGianDiemDanh FROM Students WHERE Lop=%s AND LOWER(HoVaTen) LIKE %s ORDER BY id"
        self.cursor.execute(query, (current_tab, f"%{search_term}%"))
        rows = self.cursor.fetchall()
        if not rows:
            table.pack_forget()
            watermark = ctk.CTkLabel(tab_frame, text=self.trans["no_data"], font=("Arial", 48), fg_color="transparent")
            watermark.place(relx=0.5, rely=0.5, anchor="center")
        else:
            table.pack(fill="both", expand=True)
            for idx, row in enumerate(rows, start=1):
                attendance = row[3] if row[3] is not None else row[2]
                table.add_row((idx, row[1], attendance))

    def export_data(self):
        messagebox.showinfo("Info", "Export List clicked!")

    def logout(self):
        self.destroy()
        from . import open_user_login_window
        open_user_login_window(self.cnx, self.cursor, self.language)

    def quit_app(self):
        self.destroy()
        sys.exit(0)
