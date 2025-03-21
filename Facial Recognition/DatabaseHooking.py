import mysql.connector
from mysql.connector import Error, errorcode
import os, time, random, datetime
from tkinter import messagebox, simpledialog, filedialog

# Thêm import cho openpyxl
try:
    from openpyxl import Workbook
except ImportError:
    messagebox.showwarning("Warning", "Thư viện 'openpyxl' chưa được cài đặt. Hãy cài bằng lệnh: pip install openpyxl")

# Biến global dùng cho cutoff time
CUTOFF_TIME = None

############################
# KẾT NỐI VÀ TẠO DATABASE
############################
def connect_db(user, password, host, database="Facial_Recognition"):
    """
    Kết nối tới MySQL và sử dụng database được chỉ định.
    Nếu database không tồn tại, tự động tạo mới.
    """
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
    """
    4 tài khoản mặc định nếu bảng Users rỗng:
      - superuser: username: superuser, password: superpass, role: superuser
      - admin: username: admin, password: adminpass, role: admin
      - moderator: username: moderator, password: modpass, role: moderator
      - user: username: user, password: userpass, role: user
    """
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

############################
# THAO TÁC VỚI BẢNG STUDENTS
############################

def add_student(cursor, cnx, UID, HoVaTen, NgaySinh, Lop, ImagePath,
                DiemDanhStatus='❌', ThoiGianDiemDanh=None):
    """
    Thêm học sinh vào bảng Students.
    Mặc định cột Gender là 'Nam' (do ENUM('Nam','Nữ') NOT NULL).
    """
    sql = """
    INSERT INTO Students (UID, HoVaTen, NgaySinh, Lop, Gender, DiemDanhStatus, ThoiGianDiemDanh, ImagePath)
    VALUES (%s, %s, %s, %s, 'Nam', %s, %s, %s)
    """
    cursor.execute(sql, (UID, HoVaTen, NgaySinh, Lop, DiemDanhStatus, ThoiGianDiemDanh, ImagePath))
    cnx.commit()

def update_student(cursor, cnx, student_id, UID=None, HoVaTen=None, NgaySinh=None,
                   Lop=None, Gender=None, ImagePath=None):
    """
    Cập nhật thông tin học sinh theo student_id.
    Chỉ cập nhật các trường khác None.
    """
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
    """
    Xoá học sinh khỏi bảng Students theo student_id.
    """
    sql = "DELETE FROM Students WHERE id=%s"
    cursor.execute(sql, (student_id,))
    cnx.commit()

def get_all_students(cursor):
    """
    Lấy đúng 6 cột: (id, HoVaTen, Lop, ImagePath, DiemDanhStatus, ThoiGianDiemDanh).
    """
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

def export_students_list(cursor, language):
    """
    Xuất danh sách học sinh ra file Excel (.xlsx) thay vì CSV.
    Yêu cầu cài đặt openpyxl (pip install openpyxl).
    """
    try:
        query = """
            SELECT id, HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh
            FROM Students
            ORDER BY id
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        if not rows:
            messagebox.showinfo("Info", "No data to export." if language=="English" else "Không có dữ liệu để xuất.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save as" if language=="English" else "Lưu dưới dạng"
        )
        if file_path:
            wb = Workbook()
            ws = wb.active
            ws.append(["ID", "HoVaTen", "Lop", "DiemDanhStatus", "ThoiGianDiemDanh"])
            for row in rows:
                ws.append(list(row))
            wb.save(file_path)
            messagebox.showinfo("Info", "Exported successfully." if language=="English" else "Xuất dữ liệu thành công.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def update_cutoff_time(cnx, cursor, gmt, cutoff):
    """
    Cập nhật hạn chót vào bảng Config với khóa 'cutoff_time'.
    Lưu giá trị dưới dạng chuỗi "GMT+X HH:MM".
    Nếu bảng Config chưa tồn tại, tạo bảng đó.
    """
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
    """
    Yêu cầu nhập múi giờ (GMT offset) và thời gian hạn chót (HH:MM) từ người dùng,
    sau đó lưu vào bảng Config bằng cách gọi update_cutoff_time.
    Cập nhật biến global CUTOFF_TIME nếu thành công.
    """
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
    """
    Dựa vào thời gian điểm danh và CUTOFF_TIME, trả về:
      - "Late"/"Muộn" nếu điểm danh muộn.
      - Thời gian (HH:MM:SS) nếu đúng giờ.
      - "✖" nếu chưa điểm danh.
    """
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
    """
    Thêm học sinh hàng loạt từ thư mục chứa ảnh.
    Mỗi ảnh có tên file định dạng: Họ_Tên_Lớp (vd: Nguyen_Van_A_12A).
    Nếu đường dẫn ảnh đã tồn tại trong bảng Students thì không thêm nữa.
    Gender bị loại bỏ, mặc định là 'Nam'.
    """
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

############################
# THAO TÁC VỚI BẢNG USERS
############################
def add_user(cursor, cnx, username, password, role="user"):
    """
    Thêm người dùng mới vào bảng Users.
    """
    sql = "INSERT INTO Users (username, password, role) VALUES (%s, %s, %s)"
    cursor.execute(sql, (username, password, role))
    cnx.commit()

def verify_user(cursor, username, password):
    """
    Kiểm tra đăng nhập, trả về (id, username, role) nếu đúng.
    """
    sql = "SELECT id, username, role FROM Users WHERE username=%s AND password=%s"
    cursor.execute(sql, (username, password))
    return cursor.fetchone()

def get_all_users(cursor):
    """
    Lấy danh sách tất cả người dùng.
    """
    cursor.execute("SELECT id, username, role FROM Users")
    return cursor.fetchall()
