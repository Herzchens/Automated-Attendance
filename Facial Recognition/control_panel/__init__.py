from .admin_panel import AdminControlPanel
from .moderator_panel import ModeratorControlPanel
from .user_panel import UserControlPanel
from .superuser_panel import SuperUserControlPanel
from .components import CustomTable, add_student_ui, edit_student_ui, remove_student_ui, add_students_batch_ui, edit_user_operation, delete_user_operation

def open_user_login_window(cnx, cursor, language):
    from GUI import UserLoginWindow
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
