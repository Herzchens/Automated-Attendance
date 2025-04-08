import traceback

import mysql.connector
from mysql.connector import errorcode
import os, time, random, datetime
from tkinter import messagebox, filedialog

try:
    from openpyxl import Workbook
except ImportError:
    messagebox.showwarning("Warning", "Thư viện 'openpyxl' chưa được cài đặt. Hãy cài bằng lệnh: pip install openpyxl")

CUTOFF_TIME = None

def connect_db(user, password, host, database="Facial_Recognition"):
    try:
        cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
        cursor = cnx.cursor(buffered=True)
        return cnx, cursor
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_BAD_DB_ERROR:
            try:
                cnx = mysql.connector.connect(user=user, password=password, host=host)
                cursor = cnx.cursor(buffered=True)
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
                cnx.database = database
                return cnx, cursor
            except mysql.connector.Error as err2:
                print(f"Failed creating database: {err2}")
                return None, None
        else:
            print(f"Error connecting to database: {err}")
            return None, None

def create_tables(cursor):
    create_students_table = """
    CREATE TABLE IF NOT EXISTS Students (
        id INT AUTO_INCREMENT PRIMARY KEY,
        UID VARCHAR(50) NOT NULL,
        HoVaTen VARCHAR(255) NOT NULL,
        NgaySinh DATE NOT NULL,
        Lop VARCHAR(50) NOT NULL,
        Gender ENUM('Nam', 'Nữ') NOT NULL,
        DiemDanhStatus VARCHAR(10) DEFAULT '❌',
        ThoiGianDiemDanh DATETIME NULL,
        ImagePath VARCHAR(255) NOT NULL
    )
    """
    cursor.execute(create_students_table)

    create_users_table = """
    CREATE TABLE IF NOT EXISTS Users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL,
        role ENUM('superuser', 'admin', 'moderator', 'user') NOT NULL DEFAULT 'user'
    )
    """
    cursor.execute(create_users_table)

def create_default_users(cursor, cnx):
    cursor.execute("SELECT COUNT(*) FROM Users")
    count = cursor.fetchone()[0]
    if count == 0:
        default_users = [
            ("superuser", "superpass", "superuser"),
            ("admin", "adminpass", "admin"),
            ("moderator", "modpass", "moderator"),
            ("user", "userpass", "user")
        ]
        sql = "INSERT INTO Users (username, password, role) VALUES (%s, %s, %s)"
        for user in default_users:
            cursor.execute(sql, user)
        cnx.commit()

def add_student(cursor, cnx, UID, HoVaTen, NgaySinh, Lop, Gender, ImagePath, DiemDanhStatus='❌', ThoiGianDiemDanh=None):
    if Gender not in ["Nam", "Nữ"]:
        raise ValueError("Giới tính phải là 'Nam' hoặc 'Nữ'")

    sql = """
    INSERT INTO Students (UID, HoVaTen, NgaySinh, Lop, Gender, DiemDanhStatus, ThoiGianDiemDanh, ImagePath)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (UID, HoVaTen, NgaySinh, Lop, Gender, DiemDanhStatus, ThoiGianDiemDanh, ImagePath))
    cnx.commit()


def update_student(cursor, cnx, student_id, UID=None, HoVaTen=None, NgaySinh=None, Lop=None, Gender=None, ImagePath=None):
    updates = []
    params = []
    if UID is not None:
        updates.append("UID=%s")
        params.append(UID)
    if HoVaTen is not None:
        updates.append("HoVaTen=%s")
        params.append(HoVaTen)
    if NgaySinh is not None:
        updates.append("NgaySinh=%s")
        params.append(NgaySinh)
    if Lop is not None:
        updates.append("Lop=%s")
        params.append(Lop)
    if Gender is not None:
        updates.append("Gender=%s")
        params.append(Gender)
    if ImagePath is not None:
        updates.append("ImagePath=%s")
        params.append(ImagePath)
    if updates:
        sql = "UPDATE Students SET " + ", ".join(updates) + " WHERE id=%s"
        params.append(student_id)
        cursor.execute(sql, tuple(params))
        cnx.commit()

def remove_student(cursor, cnx, student_id):
    sql = "DELETE FROM Students WHERE id=%s"
    cursor.execute(sql, (student_id,))
    cnx.commit()

def get_all_students(cursor):
    cursor.execute("""
        SELECT 
            id, 
            HoVaTen, 
            Lop, 
            ImagePath, 
            DiemDanhStatus, 
            ThoiGianDiemDanh
        FROM Students
    """)
    return cursor.fetchall()

def get_students_for_ui(cursor):
    cursor.execute("""
        SELECT UID, HoVaTen, Lop, Gender, NgaySinh, DiemDanhStatus, ThoiGianDiemDanh
        FROM Students
    """)
    return cursor.fetchall()

def get_students_by_class(cursor, class_name):
    sql = """
        SELECT id, UID, HoVaTen, NgaySinh, Lop, Gender, DiemDanhStatus, ThoiGianDiemDanh, ImagePath
        FROM Students
        WHERE Lop=%s
    """
    cursor.execute(sql, (class_name,))
    return cursor.fetchall()

def update_attendance(cursor, cnx, student_id, status, time):
    if status == '❌':
        time = None
    sql = "UPDATE Students SET DiemDanhStatus=%s, ThoiGianDiemDanh=%s WHERE id=%s"
    cursor.execute(sql, (status, time, int(student_id)))
    cnx.commit()

def export_students_list(cursor, language, save_to_file=True):
    try:
        query = """
            SELECT UID, HoVaTen, Lop, Gender, NgaySinh, DiemDanhStatus, ThoiGianDiemDanh
            FROM Students
            ORDER BY id
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        if not rows:
            if save_to_file:
                messagebox.showinfo("Info", "No data to export." if language=="English" else "Không có dữ liệu để xuất.")
            else:
                raise Exception("Không có dữ liệu để xuất.")
            return None

        wb = Workbook()
        ws = wb.active
        ws.title = "DanhSachHocSinh"
        header = ["STT", "UID", "Họ Và Tên", "Lớp", "Giới Tính", "Ngày Sinh", "Trạng Thái", "Thời Gian"]
        ws.append(header)
        for index, row in enumerate(rows):
            fixed_row = [str(index + 1)]
            for cell in row:
                if cell is None:
                    fixed_row.append("")
                elif isinstance(cell, datetime.datetime):
                    fixed_row.append(cell.strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    fixed_row.append(str(cell))
            ws.append(fixed_row)

        if save_to_file:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                title="Save as" if language=="English" else "Lưu dưới dạng"
            )
            if file_path:
                wb.save(file_path)
                wb.close()
                messagebox.showinfo("Info", "Exported successfully." if language=="English" else "Xuất dữ liệu thành công.")
        else:
            return wb

    except Exception as e:
        error_message = traceback.format_exc()
        if save_to_file:
            messagebox.showerror("Error", f"{str(e)}\n\n{error_message}")
        else:
            raise e

def update_cutoff_time(cnx, cursor, gmt, cutoff):
    value = f"{gmt} {cutoff}"
    create_config_table = """
    CREATE TABLE IF NOT EXISTS Config (
        config_key VARCHAR(50) PRIMARY KEY,
        config_value VARCHAR(255) NOT NULL
    )
    """
    cursor.execute(create_config_table)
    query = """
    INSERT INTO Config (config_key, config_value)
    VALUES ('cutoff_time', %s)
    ON DUPLICATE KEY UPDATE config_value = %s
    """
    cursor.execute(query, (value, value))
    cnx.commit()

def set_cutoff_time(language, cnx, cursor):
    global CUTOFF_TIME
    from tkinter import simpledialog, messagebox
    import datetime

    prompt_gmt = "Enter GMT offset (e.g., 7 or -5):" if language=="English" else "Nhập múi giờ (ví dụ: 7 hoặc -5):"
    gmt_str = simpledialog.askstring("Set Cutoff", prompt_gmt)
    if not gmt_str:
        return None
    try:
        gmt_int = int(gmt_str)
        gmt_display = f"GMT{gmt_int:+d}"
    except ValueError:
        messagebox.showerror("Error", "Invalid GMT offset." if language=="English" else "Múi giờ không hợp lệ.")
        return None

    prompt_cutoff = "Enter cutoff time (HH:MM):" if language=="English" else "Nhập thời gian hạn chót (HH:MM):"
    cutoff_str = simpledialog.askstring("Set Cutoff", prompt_cutoff)
    if not cutoff_str:
        return None
    try:
        cutoff_time = datetime.datetime.strptime(cutoff_str, "%H:%M").time()
    except ValueError:
        messagebox.showerror("Error", "Invalid time format. Please use HH:MM." if language=="English" else "Định dạng thời gian không hợp lệ. Vui lòng nhập theo định dạng HH:MM.")
        return None

    try:
        update_cutoff_time(cnx, cursor, gmt_display, cutoff_str)
        CUTOFF_TIME = cutoff_time
        msg = f"Cutoff time set to {gmt_display} {cutoff_str}" if language=="English" else f"Thời gian hạn chót được đặt là {gmt_display} {cutoff_str}"
        messagebox.showinfo("Info", msg)
        return cutoff_time
    except Exception as e:
        messagebox.showerror("Error", f"Error setting cutoff time: {e}")
        return None

def calculate_attendance_status(attendance_time, language):
    if attendance_time is None:
        return "✖"
    try:
        if CUTOFF_TIME is not None:
            if attendance_time.time() > CUTOFF_TIME:
                return "Late" if language=="English" else "Muộn"
            else:
                return attendance_time.strftime("%H:%M:%S")
        else:
            return attendance_time.strftime("%H:%M:%S")
    except Exception:
        return "Error"

def add_students_batch(cursor, cnx, language, folder):
    if not folder:
        return 0
    added_count = 0
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder, file_name)
            cursor.execute("SELECT COUNT(*) FROM Students WHERE ImagePath=%s", (image_path,))
            exists = cursor.fetchone()[0]
            if exists:
                continue
            base_name = os.path.splitext(file_name)[0]
            parts = base_name.split('_')
            if len(parts) < 2:
                messagebox.showerror("Error", f"Tên file {file_name} không hợp lệ. Định dạng: Họ_Tên_Lớp")
                continue
            class_name = parts[-1]
            student_name = " ".join(parts[:-1])
            ngay_sinh = datetime.datetime.now().strftime("%Y-%m-%d")
            uid = f"{int(time.time())}{random.randint(100, 999)}"
            try:
                add_student(cursor, cnx, uid, student_name, ngay_sinh, class_name, image_path)
                added_count += 1
            except Exception as e:
                messagebox.showerror("Error", f"Lỗi thêm học sinh {student_name}: {e}")
    messagebox.showinfo("Info",
                        f"Đã thêm {added_count} học sinh." if language=="Tiếng Việt" else f"Added {added_count} students.")
    return added_count
def add_user(cursor, cnx, username, password, role="user"):
    sql = "INSERT INTO Users (username, password, role) VALUES (%s, %s, %s)"
    cursor.execute(sql, (username, password, role))
    cnx.commit()

def verify_user(cursor, username, password):
    sql = "SELECT id, username, role FROM Users WHERE username=%s AND password=%s"
    cursor.execute(sql, (username, password))
    return cursor.fetchone()

def get_all_users(cursor):
    cursor.execute("SELECT id, username, role FROM Users")
    return cursor.fetchall()

def update_user(cursor, cnx, user_id, username, password, role):
    sql = "UPDATE Users SET username=%s, password=%s, role=%s WHERE id=%s"
    cursor.execute(sql, (username, password, role, user_id))
    cnx.commit()

def remove_user(cursor, cnx, user_id):
    sql = "DELETE FROM Users WHERE id=%s"
    cursor.execute(sql, (user_id,))
    cnx.commit()
