# control_panel/__init__.py
from .admin_panel import AdminControlPanel
from .moderator_panel import ModeratorControlPanel
from .user_panel import UserControlPanel
from .superuser_panel import SuperUserControlPanel
from .common import translations, CustomTable
from tkinter import messagebox
import sys

def open_user_login_window(cnx, cursor, language):
    from GUI import UserLoginWindow  # Giả sử bạn có file GUI.py chứa UserLoginWindow
    win = UserLoginWindow(cnx, cursor, language)
    try:
        win.state("zoomed")
    except Exception:
        pass
    win.mainloop()

def open_control_panel(user_info, cnx, cursor, language):
    role = user_info[2].lower()
    if role == "superuser":
        panel = SuperUserControlPanel(user_info, cnx, cursor, language)
    elif role == "admin":
        panel = AdminControlPanel(user_info, cnx, cursor, language)
    elif role == "moderator":
        panel = ModeratorControlPanel(user_info, cnx, cursor, language)
    else:
        panel = UserControlPanel(user_info, cnx, cursor, language)
    try:
        panel.state("zoomed")
    except Exception:
        pass
    panel.mainloop()
