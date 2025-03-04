import mysql.connector
from mysql.connector import Error

def connect_db(user, password, host, database="Facial_Recognition"):
    try:
        cnx = mysql.connector.connect(user=user, password=password, host=host)
        # Sử dụng buffered cursor
        cursor = cnx.cursor(buffered=True)
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        cnx.database = database
        return cnx, cursor
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None, None

def create_tables(cursor):
    # Tạo bảng Students
    create_students_table = """
    CREATE TABLE IF NOT EXISTS Students (
        id INT AUTO_INCREMENT PRIMARY KEY,
        HoVaTen VARCHAR(255) NOT NULL,
        Lop VARCHAR(50) NOT NULL,
        ImagePath VARCHAR(255) NOT NULL,
        DiemDanhStatus VARCHAR(10) DEFAULT '✖',
        ThoiGianDiemDanh DATETIME NULL
    )
    """
    cursor.execute(create_students_table)

    # Tạo bảng Users
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

# ---------- Các thao tác với bảng Students ----------

def add_student(cursor, cnx, HoVaTen, Lop, ImagePath):
    """
    Thêm học sinh vào bảng Students.
    """
    sql = "INSERT INTO Students (HoVaTen, Lop, ImagePath) VALUES (%s, %s, %s)"
    cursor.execute(sql, (HoVaTen, Lop, ImagePath))
    cnx.commit()

def get_all_students(cursor):
    """
    Lấy danh sách tất cả học sinh.
    """
    cursor.execute("SELECT id, HoVaTen, Lop, ImagePath, DiemDanhStatus, ThoiGianDiemDanh FROM Students")
    return cursor.fetchall()

def get_students_by_class(cursor, class_name):
    """
    Lấy danh sách học sinh theo lớp.
    """
    sql = "SELECT id, HoVaTen, Lop, ImagePath, DiemDanhStatus, ThoiGianDiemDanh FROM Students WHERE Lop=%s"
    cursor.execute(sql, (class_name,))
    return cursor.fetchall()

def update_attendance(cursor, cnx, student_id, status, time):
    """
    Cập nhật trạng thái điểm danh của học sinh.
    """
    sql = "UPDATE Students SET DiemDanhStatus=%s, ThoiGianDiemDanh=%s WHERE id=%s"
    cursor.execute(sql, (status, time, student_id))
    cnx.commit()

def remove_student(cursor, cnx, student_id):
    """
    Xoá học sinh khỏi bảng.
    """
    sql = "DELETE FROM Students WHERE id=%s"
    cursor.execute(sql, (student_id,))
    cnx.commit()

def update_student(cursor, cnx, student_id, HoVaTen=None, Lop=None, ImagePath=None):
    """
    Cập nhật thông tin học sinh.
    """
    updates = []
    params = []
    if HoVaTen:
        updates.append("HoVaTen=%s")
        params.append(HoVaTen)
    if Lop:
        updates.append("Lop=%s")
        params.append(Lop)
    if ImagePath:
        updates.append("ImagePath=%s")
        params.append(ImagePath)
    if updates:
        sql = "UPDATE Students SET " + ", ".join(updates) + " WHERE id=%s"
        params.append(student_id)
        cursor.execute(sql, tuple(params))
        cnx.commit()

# ---------- Các thao tác với bảng Users ----------

def add_user(cursor, cnx, username, password, role="user"):
    """
    Thêm người dùng mới vào bảng Users. 
    """
    sql = "INSERT INTO Users (username, password, role) VALUES (%s, %s, %s)"
    cursor.execute(sql, (username, password, role))
    cnx.commit()

def verify_user(cursor, username, password):
    """
    Kiểm tra đăng nhập, trả về thông tin người dùng nếu đúng.
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
