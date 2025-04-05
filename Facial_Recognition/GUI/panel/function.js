const PanelFunctions = {
  parseJSON: function(response) {
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.indexOf('application/json') !== -1) {
      return response.json();
    } else {
      return response.text().then(text => {
        try {
          return JSON.parse(text);
        } catch (e) {
          throw new Error("Invalid JSON: " + text);
        }
      });
    }
  },
  showNotification: function(message, type = "success") {
    alert("[" + type.toUpperCase() + "] " + message);
  },
  fetchStudents: function(callback) {
    fetch('/api/students')
      .then(this.parseJSON)
      .then(data => {
        if (data.success) {
          callback(data.students);
        } else {
          this.showNotification("Lỗi: " + data.message, "error");
          console.error("Error fetching students:", data.message);
        }
      })
      .catch(error => {
        this.showNotification("Lỗi: " + error.message, "error");
        console.error("Error fetching students:", error);
      });
  },
  addStudent: function(studentData, callback) {
    fetch('/api/add_student', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(studentData)
    })
      .then(this.parseJSON)
      .then(data => {
        if (data.success) {
          this.showNotification("Thêm học sinh thành công", "success");
          callback && callback();
        } else {
          this.showNotification("Lỗi: " + data.message, "error");
        }
      })
      .catch(error => {
        this.showNotification("Lỗi: " + error.message, "error");
        console.error("Error adding student:", error);
      });
  },
  editStudent: function(studentId, updatedData, callback) {
    fetch('/api/edit_student', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: studentId, ...updatedData })
    })
      .then(this.parseJSON)
      .then(data => {
        if (data.success) {
          this.showNotification("Chỉnh sửa học sinh thành công", "success");
          callback && callback();
        } else {
          this.showNotification("Lỗi: " + data.message, "error");
        }
      })
      .catch(error => {
        this.showNotification("Lỗi: " + error.message, "error");
        console.error("Error editing student:", error);
      });
  },
  deleteStudent: function(studentId, callback) {
    fetch('/api/delete_student', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: studentId })
    })
      .then(this.parseJSON)
      .then(data => {
        if (data.success) {
          this.showNotification("Xoá học sinh thành công", "success");
          callback && callback();
        } else {
          this.showNotification("Lỗi: " + data.message, "error");
        }
      })
      .catch(error => {
        this.showNotification("Lỗi: " + error.message, "error");
        console.error("Error deleting student:", error);
      });
  },
  batchAddStudents: function(batchData, callback) {
    fetch('/api/batch_add_students', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(batchData)
    })
      .then(this.parseJSON)
      .then(data => {
        if (data.success) {
          this.showNotification("Thêm hàng loạt học sinh thành công", "success");
          callback && callback();
        } else {
          this.showNotification("Lỗi: " + data.message, "error");
        }
      })
      .catch(error => {
        this.showNotification("Lỗi: " + error.message, "error");
        console.error("Error in batch adding students:", error);
      });
  },
  setCutoff: function(gmt, cutoff, callback) {
    fetch('/api/set_cutoff', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gmt, cutoff })
    })
      .then(this.parseJSON)
      .then(data => {
        if (data.success) {
          this.showNotification("Cài đặt hạn chót thành công", "success");
          callback && callback(data);
        } else {
          this.showNotification("Lỗi: " + data.message, "error");
        }
      })
      .catch(error => {
        this.showNotification("Lỗi: " + error.message, "error");
        console.error("Error setting cutoff:", error);
      });
  },
  addUser: function(username, password, role, callback) {
    fetch('/api/add_user', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password, role })
    })
      .then(this.parseJSON)
      .then(data => {
        if (data.success) {
          this.showNotification("Thêm tài khoản thành công", "success");
          callback && callback();
        } else {
          this.showNotification("Lỗi: " + data.message, "error");
        }
      })
      .catch(error => {
        this.showNotification("Lỗi: " + error.message, "error");
        console.error("Error adding user:", error);
      });
  },
  editUser: function(userId, username, password, role, callback) {
    fetch('/api/edit_user', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: userId, username, password, role })
    })
      .then(this.parseJSON)
      .then(data => {
        if (data.success) {
          this.showNotification("Chỉnh sửa tài khoản thành công", "success");
          callback && callback();
        } else {
          this.showNotification("Lỗi: " + data.message, "error");
        }
      })
      .catch(error => {
        this.showNotification("Lỗi: " + error.message, "error");
        console.error("Error editing user:", error);
      });
  },
  deleteUser: function(userId, callback) {
    fetch('/api/delete_user', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: userId })
    })
      .then(this.parseJSON)
      .then(data => {
        if (data.success) {
          this.showNotification("Xoá tài khoản thành công", "success");
          callback && callback();
        } else {
          this.showNotification("Lỗi: " + data.message, "error");
        }
      })
      .catch(error => {
        this.showNotification("Lỗi: " + error.message, "error");
        console.error("Error deleting user:", error);
      });
  }
};
window.PanelFunctions = PanelFunctions;
