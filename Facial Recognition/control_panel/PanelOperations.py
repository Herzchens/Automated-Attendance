# PanelOperations.py
import csv
from datetime import datetime
import customtkinter as ctk
from tkinter import messagebox, simpledialog, filedialog
from DatabaseHooking import add_student, update_student, remove_student
from .common import AddStudentImageWindow

# Biến toàn cục để lưu thời gian hạn chót (kiểu time)
CUTOFF_TIME = None

def set_cutoff_time(language):
    """
    Hiển thị hộp thoại nhập thời gian hạn chót (ví dụ "07:00") và lưu vào biến toàn cục.
    """
    global CUTOFF_TIME
    prompt = "Enter cutoff time (HH:MM):" if language == "English" else "Nhập thời gian hạn chót (HH:MM):"
    cutoff_str = simpledialog.askstring("Set Cutoff", prompt)
    if cutoff_str:
        try:
            cutoff_time = datetime.strptime(cutoff_str, "%H:%M").time()
            CUTOFF_TIME = cutoff_time
            msg = f"Cutoff time set to {cutoff_str}" if language == "English" else f"Thời gian hạn chót được đặt là {cutoff_str}"
            messagebox.showinfo("Info", msg)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid time format: {e}")
    return CUTOFF_TIME

def export_students_list(cursor, language):
    """
    Xuất danh sách học sinh ra file CSV.
    """
    try:
        query = "SELECT id, HoVaTen, Lop, DiemDanhStatus, ThoiGianDiemDanh FROM Students ORDER BY id"
        cursor.execute(query)
        rows = cursor.fetchall()
        if not rows:
            messagebox.showinfo("Info", "No data to export." if language == "English" else "Không có dữ liệu để xuất.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save as" if language == "English" else "Lưu dưới dạng"
        )
        if file_path:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "HoVaTen", "Lop", "DiemDanhStatus", "ThoiGianDiemDanh"])
                writer.writerows(rows)
            messagebox.showinfo("Info", "Exported successfully." if language == "English" else "Xuất dữ liệu thành công.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def calculate_attendance_status(attendance_time, language):
    """
    Dựa vào thời gian điểm danh và CUTOFF_TIME:
      - Nếu điểm danh sau hạn chót, trả về "Late" (hoặc "Muộn").
      - Nếu trước hạn chót, trả về thời gian (định dạng HH:MM:SS).
      - Nếu không có thời gian, trả về "✖".
    """
    if attendance_time is None:
        return "✖"
    try:
        if CUTOFF_TIME is not None:
            if attendance_time.time() > CUTOFF_TIME:
                return "Late" if language == "English" else "Muộn"
            else:
                return attendance_time.strftime("%H:%M:%S")
        else:
            return attendance_time.strftime("%H:%M:%S")
    except Exception:
        return "Error"

def add_student_ui(parent, cnx, cursor, language, on_success_callback=None):
    """
    Mở cửa sổ thêm học sinh từ ảnh.
    Nếu đã có cửa sổ AddStudentImageWindow trong common, ta chỉ gọi nó.
    """
    AddStudentImageWindow(parent, cnx, cursor, language, on_success_callback=on_success_callback)

def edit_student_ui(parent, cnx, cursor, language, student, on_success_callback=None):
    """
    Hiển thị hộp thoại cho phép chỉnh sửa thông tin học sinh.
    student: tuple chứa thông tin học sinh (id, HoVaTen, Lop, ...)
    """
    new_name = simpledialog.askstring(
        "Edit Student",
        "Enter new name:" if language == "English" else "Nhập tên mới:",
        initialvalue=student[1]
    )
    if new_name is None:
        return
    new_class = simpledialog.askstring(
        "Edit Student",
        "Enter new class:" if language == "English" else "Nhập lớp mới:",
        initialvalue=student[2]
    )
    if new_class is None:
        return
    try:
        update_student(cursor, cnx, student[0], HoVaTen=new_name, Lop=new_class)
        messagebox.showinfo(
            "Info",
            "Student updated successfully." if language == "English" else "Học sinh đã được cập nhật thành công."
        )
        if on_success_callback:
            on_success_callback()
    except Exception as e:
        messagebox.showerror("Error", f"Error updating student:\n{e}")

def remove_student_ui(parent, cnx, cursor, language, student, on_success_callback=None):
    """
    Hiển thị hộp thoại xác nhận và thực hiện xoá học sinh.
    """
    confirm = messagebox.askyesno(
        "Confirm",
        "Are you sure you want to delete this student?" if language == "English" else "Bạn có chắc muốn xoá học sinh này?"
    )
    if confirm:
        try:
            remove_student(cursor, cnx, student[0])
            messagebox.showinfo(
                "Info",
                "Student deleted successfully." if language == "English" else "Học sinh đã được xoá thành công."
            )
            if on_success_callback:
                on_success_callback()
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting student:\n{e}")

def search_student_ui(cursor, language, search_term, query_template, params):
    """
    Hàm tìm kiếm chung:
      - cursor: con trỏ database.
      - search_term: từ khóa tìm kiếm.
      - query_template: câu truy vấn có placeholder.
      - params: tuple chứa các tham số cho câu truy vấn.
    Trả về danh sách các hàng kết quả tìm kiếm.
    """
    if not search_term:
        return None
    try:
        cursor.execute(query_template, params)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        messagebox.showerror("Error", f"Error searching data:\n{e}")
        return None
