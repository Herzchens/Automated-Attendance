from flask import Flask, request, jsonify, send_from_directory
import subprocess
import json, os, base64
from DatabaseHooking import (
    connect_db, create_tables, create_default_users, verify_user,
    add_student, update_student, remove_student, add_students_batch,
    update_cutoff_time, add_user, update_user, remove_user, get_all_users,
    export_students_list, calculate_attendance_status, get_all_students, get_students_for_ui
)

app = Flask(__name__, static_folder='.', static_url_path='')

def load_config():
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8-sig") as f:
            config = json.load(f)
        return config
    return {}

@app.route('/')
def index():
    return send_from_directory('.', 'GUI.html')

@app.route('/panel/<path:filename>')
def serve_panel_static(filename):
    panel_folder = os.path.join(os.getcwd(), "panel")
    return send_from_directory(panel_folder, filename)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    language = data.get('language', 'Tiếng Việt')
    remember = data.get('remember', False)

    if not username or not password:
        return jsonify(success=False, message="❌ Vui lòng nhập tên đăng nhập và mật khẩu.")

    config = load_config()
    print("Loaded config:", config)
    db_host = config.get("db_host", "localhost")

    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception as e:
            print("Error decoding db_username:", e)
            db_username = "root"
    else:
        db_username = "root"

    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception as e:
            print("Error decoding db_password:", e)
            db_password = ""
    else:
        db_password = ""

    print(f"Using DB host: {db_host}")
    print(f"Using DB user: {db_username}")
    print(f"Using DB password: '{db_password}'")

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="❌ Không thể kết nối CSDL.")

    create_tables(cursor)
    create_default_users(cursor, cnx)

    user_info = verify_user(cursor, username, password)
    print("User info:", user_info)
    if user_info is None:
        return jsonify(success=False, message="❌ Tên đăng nhập hoặc mật khẩu không hợp lệ.")

    try:
        role = user_info[2]
    except Exception as e:
        print("Error extracting role from user_info:", e)
        role = "undefined"

    role_lower = role.lower() if role else "undefined"

    if role_lower == "superuser":
        redirect_url = "/panel/superuser_panel.html"
    elif role_lower == "admin":
        redirect_url = "/panel/admin_panel.html"
    elif role_lower == "moderator":
        redirect_url = "/panel/moderator_panel.html"
    else:
        redirect_url = "/panel/user_panel.html"

    return jsonify(success=True, role=role, redirect=redirect_url)

@app.route('/switch_app', methods=['POST'])
def switch_app():
    try:
        subprocess.Popen(['python', 'GUI.py'])
        return jsonify(success=True, redirect='/')
    except Exception as e:
        return jsonify(success=False, message=str(e))

# ==================== Endpoint cho Học sinh ====================

@app.route('/api/add_student', methods=['POST'])
def api_add_student():
    data = request.get_json()
    required_keys = ['UID', 'HoVaTen', 'NgaySinh', 'Lop', 'ImagePath']
    if not all(key in data for key in required_keys):
        return jsonify(success=False, message="Thiếu thông tin học sinh."), 400

    config = load_config()
    db_host = config.get("db_host", "localhost")
    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception:
            db_username = "root"
    else:
        db_username = "root"
    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception:
            db_password = ""
    else:
        db_password = ""

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="Không thể kết nối CSDL"), 500

    try:
        add_student(cursor, cnx,
                    data['UID'], data['HoVaTen'], data['NgaySinh'],
                    data['Lop'], data['ImagePath'])
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/edit_student', methods=['POST'])
def api_edit_student():
    data = request.get_json()
    student_id = data.get('id')
    if not student_id:
        return jsonify(success=False, message="Thiếu id học sinh."), 400

    update_fields = {}
    for field in ['UID', 'HoVaTen', 'NgaySinh', 'Lop', 'Gender', 'ImagePath']:
        if field in data:
            update_fields[field] = data[field]

    if not update_fields:
        return jsonify(success=False, message="Không có dữ liệu cập nhật."), 400

    config = load_config()
    db_host = config.get("db_host", "localhost")
    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception:
            db_username = "root"
    else:
        db_username = "root"
    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception:
            db_password = ""
    else:
        db_password = ""

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="Không thể kết nối CSDL"), 500

    try:
        update_student(cursor, cnx, student_id, **update_fields)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/delete_student', methods=['POST'])
def api_delete_student():
    data = request.get_json()
    student_id = data.get('id')
    if not student_id:
        return jsonify(success=False, message="Thiếu id học sinh."), 400

    config = load_config()
    db_host = config.get("db_host", "localhost")
    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception:
            db_username = "root"
    else:
        db_username = "root"
    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception:
            db_password = ""
    else:
        db_password = ""

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="Không thể kết nối CSDL"), 500

    try:
        remove_student(cursor, cnx, student_id)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/batch_add_students', methods=['POST'])
def api_batch_add_students():
    data = request.get_json()
    if not isinstance(data, list):
        return jsonify(success=False, message="Dữ liệu không hợp lệ, mong đợi một danh sách."), 400

    config = load_config()
    db_host = config.get("db_host", "localhost")
    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception:
            db_username = "root"
    else:
        db_username = "root"
    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception:
            db_password = ""
    else:
        db_password = ""

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="Không thể kết nối CSDL"), 500

    try:
        added_count = add_students_batch(cursor, cnx, language="Tiếng Việt", folder=data)
        return jsonify(success=True, added_count=added_count)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/set_cutoff', methods=['POST'])
def api_set_cutoff():
    data = request.get_json()
    gmt = data.get('gmt')
    cutoff = data.get('cutoff')
    if not gmt or not cutoff:
        return jsonify(success=False, message="Thiếu thông tin hạn chót."), 400

    config = load_config()
    db_host = config.get("db_host", "localhost")
    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception:
            db_username = "root"
    else:
        db_username = "root"
    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception:
            db_password = ""
    else:
        db_password = ""

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="Không thể kết nối CSDL"), 500

    try:
        update_cutoff_time(cnx, cursor, gmt, cutoff)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/students', methods=['GET'])
def api_get_students():
    config = load_config()
    db_host = config.get("db_host", "localhost")
    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception:
            db_username = "root"
    else:
        db_username = "root"

    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception:
            db_password = ""
    else:
        db_password = ""

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="Không thể kết nối CSDL"), 500

    try:
        students = get_students_for_ui(cursor)
        student_list = []

        for index, s in enumerate(students, start=1):
            student_list.append({
                "STT": index,
                "UID": s[0],
                "HoVaTen": s[1],
                "Lop": s[2],
                "Gender": s[3],
                "NgaySinh": s[4],
                "DiemDanhStatus": s[5],
                "ThoiGianDiemDanh": s[6]
            })

        return jsonify(success=True, students=student_list)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

# ==================== Endpoint cho Tài khoản ====================

@app.route('/api/add_user', methods=['POST'])
def api_add_user():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = data.get('role', 'user').strip()
    if not username or not password:
        return jsonify(success=False, message="Thiếu thông tin tài khoản."), 400

    config = load_config()
    db_host = config.get("db_host", "localhost")
    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception:
            db_username = "root"
    else:
        db_username = "root"
    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception:
            db_password = ""
    else:
        db_password = ""

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="Không thể kết nối CSDL"), 500

    try:
        add_user(cursor, cnx, username, password, role)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/edit_user', methods=['POST'])
def api_edit_user():
    data = request.get_json()
    user_id = data.get('id')
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    role = data.get('role', '').strip()
    if not user_id or not username or not password or not role:
        return jsonify(success=False, message="Thiếu thông tin tài khoản."), 400

    config = load_config()
    db_host = config.get("db_host", "localhost")
    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception:
            db_username = "root"
    else:
        db_username = "root"
    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception:
            db_password = ""
    else:
        db_password = ""

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="Không thể kết nối CSDL"), 500

    try:
        update_user(cursor, cnx, user_id, username, password, role)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

@app.route('/api/delete_user', methods=['POST'])
def api_delete_user():
    data = request.get_json()
    user_id = data.get('id')
    if not user_id:
        return jsonify(success=False, message="Thiếu id tài khoản."), 400

    config = load_config()
    db_host = config.get("db_host", "localhost")
    enc_db_username = config.get("db_username", "")
    enc_db_password = "SGVyemNoZW4="

    if enc_db_username:
        try:
            db_username = base64.b64decode(enc_db_username.encode()).decode()
        except Exception:
            db_username = "root"
    else:
        db_username = "root"
    if enc_db_password:
        try:
            db_password = base64.b64decode(enc_db_password.encode()).decode()
        except Exception:
            db_password = ""
    else:
        db_password = ""

    cnx, cursor = connect_db(db_username, db_password, db_host, database="Facial_Recognition")
    if cnx is None:
        return jsonify(success=False, message="Không thể kết nối CSDL"), 500

    try:
        remove_user(cursor, cnx, user_id)
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
