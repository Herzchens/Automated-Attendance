document.addEventListener('DOMContentLoaded', () => {
  let studentsData = [];
  let currentSort = "az";

  function sortStudents(array, sortParam) {
    let sorted = [...array];
    switch (sortParam) {
      case "az":
        sorted.sort((a, b) => a.HoVaTen.localeCompare(b.HoVaTen));
        break;
      case "za":
        sorted.sort((a, b) => b.HoVaTen.localeCompare(a.HoVaTen));
        break;
      case "uid":
        sorted.sort((a, b) => a.UID.localeCompare(b.UID));
        break;
      case "gender":
        sorted.sort((a, b) => a.Gender.localeCompare(b.Gender));
        break;
      case "birthday":
        sorted.sort((a, b) => new Date(a.NgaySinh) - new Date(b.NgaySinh));
        break;
      case "attendance":
        sorted.sort((a, b) => a.DiemDanhStatus.localeCompare(b.DiemDanhStatus));
        break;
      case "time":
        sorted.sort((a, b) => (a.ThoiGianDiemDanh || "").localeCompare(b.ThoiGianDiemDanh || ""));
        break;
      default:
        break;
    }
    return sorted;
  }

  const tablinks = document.querySelectorAll('.tablink');
  const tabcontents = document.querySelectorAll('.tabcontent');
  tablinks.forEach(btn => {
    btn.addEventListener('click', () => {
      tablinks.forEach(b => b.classList.remove('active'));
      tabcontents.forEach(tc => tc.classList.remove('active'));
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');
    });
  });

  let selectedStudent = null;
  let selectedUser = null;

  function renderStudents(students) {
    const tbody = document.querySelector("#students-table tbody");
    tbody.innerHTML = "";
    selectedStudent = null;
    if (!students || students.length === 0) {
      document.getElementById("no-data").style.display = "block";
    } else {
      document.getElementById("no-data").style.display = "none";
      students.forEach((student, index) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${index + 1}</td>
          <td>${student.UID}</td>
          <td>${student.HoVaTen}</td>
          <td>${student.Lop}</td>
          <td>${student.Gender}</td>
          <td>${student.NgaySinh}</td>
          <td>${student.DiemDanhStatus}</td>
          <td>${student.ThoiGianDiemDanh || ""}</td>`;
        tr.addEventListener("click", () => {
          document.querySelectorAll("#students-table tbody tr").forEach(row => row.classList.remove("selected-row"));
          tr.classList.add("selected-row");
          selectedStudent = student;
        });
        tbody.appendChild(tr);
      });
    }
  }

  function renderUsers(users) {
    const tbody = document.querySelector("#users-table tbody");
    tbody.innerHTML = "";
    selectedUser = null;
    document.getElementById("user-detail").style.display = "none";
    if (!users || users.length === 0) {
      document.getElementById("no-users").style.display = "block";
    } else {
      document.getElementById("no-users").style.display = "none";
      users.forEach((user, index) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${index + 1}</td>
          <td>${user.username}</td>
          <td>${user.role}</td>`;
        tr.addEventListener("click", () => {
          document.querySelectorAll("#users-table tbody tr").forEach(row => row.classList.remove("selected-row"));
          tr.classList.add("selected-row");
          selectedUser = user;
          document.getElementById("detail-username").textContent = "Tên: " + user.username;
          document.getElementById("detail-role").textContent = "Quyền: " + user.role;
          document.getElementById("user-detail").style.display = "block";
        });
        tbody.appendChild(tr);
      });
    }
  }

  const PanelFunctions = {
    fetchStudents: function(callback) {
      fetch('/api/students')
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            studentsData = data.students;
            callback(sortStudents(studentsData, currentSort));
          }
          else showNotification("error", "Lỗi khi lấy danh sách học sinh: " + data.message);
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    },
    addStudent: function(studentData, callback) {
      fetch('/api/add_student', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(studentData)
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification("success", "Thêm học sinh thành công!");
            callback();
          } else showNotification("error", "Lỗi khi thêm học sinh: " + data.message);
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    },
    editStudent: function(studentId, updateData, callback) {
      updateData.id = studentId;
      fetch('/api/edit_student', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updateData)
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            const student = data.student;
            document.getElementById('edit-HoVaTen').value = student.HoVaTen;
            document.getElementById('edit-Lop').value = student.Lop;
            document.getElementById('edit-Gender').value = student.Gender;
            document.getElementById('edit-NgaySinh').value = student.NgaySinh;
            document.getElementById('edit-ImagePath').value = student.ImagePath;
            let preview = document.getElementById('edit-image-preview');
            if (student.ImagePath) {
              preview.src = student.ImagePath;
              preview.style.display = "block";
            } else {
              preview.style.display = "none";
            }
            showNotification("success", "Chỉnh sửa học sinh thành công!");
            callback();
          } else {
            showNotification("error", "Lỗi khi chỉnh sửa học sinh: " + data.message);
          }
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    },
    deleteStudent: function(UID, callback) {
      fetch('/api/delete_student', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ UID: UID })
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification("success", "Xoá học sinh thành công!");
            callback();
          } else showNotification("error", "Lỗi khi xoá học sinh: " + data.message);
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    },
    batchAddStudents: function(folderPath, callback) {
      fetch('/api/batch_add_students', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify([folderPath])
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification("success", "Đã thêm " + data.added_count + " học sinh.");
            callback();
          } else showNotification("error", "Lỗi khi thêm hàng loạt học sinh: " + data.message);
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    },
    setCutoff: function(gmt, cutoff, callback) {
      fetch('/api/set_cutoff', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gmt: gmt, cutoff: cutoff })
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification("success", "Đặt hạn chót thành công!");
            callback(data);
          } else showNotification("error", "Lỗi khi đặt thời hạn: " + data.message);
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    },
    exportStudents: function() {
      fetch('/api/export_students_excel')
        .then(response => response.blob())
        .then(blob => {
          const url = window.URL.createObjectURL(blob);
          const filename = new Date().toLocaleDateString("vi-VN").replace(/\//g, "-") + ".xlsx";
          const a = document.createElement('a');
          a.href = url;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          a.remove();
        })
        .catch(err => showNotification("error", "Lỗi khi xuất danh sách: " + err));
    },
    addUser: function(username, password, role, callback) {
      fetch('/api/add_user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, role })
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification("success", "Thêm người dùng thành công!");
            callback();
          } else showNotification("error", "Lỗi khi thêm tài khoản: " + data.message);
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    },
    editUser: function(userId, username, password, role, callback) {
      fetch('/api/edit_user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: userId, username, password, role })
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification("success", "Chỉnh sửa người dùng thành công!");
            callback();
          } else showNotification("error", "Lỗi khi chỉnh sửa tài khoản: " + data.message);
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    },
    deleteUser: function(userId, callback) {
      fetch('/api/delete_user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: userId })
      })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification("success", "Xoá người dùng thành công!");
            callback();
          } else showNotification("error", "Lỗi khi xoá tài khoản: " + data.message);
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    },
    fetchUsers: function(callback) {
      fetch('/api/users')
        .then(response => response.json())
        .then(data => {
          if (data.success) callback(data.users);
          else showNotification("error", "Lỗi khi lấy danh sách người dùng: " + data.message);
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    }
  };

  document.getElementById("export-students").addEventListener("click", () => {
    PanelFunctions.exportStudents();
  });

  document.getElementById("search-button").addEventListener("click", () => {
    const query = document.getElementById("search-input").value.trim();
    if (query) {
      fetch(`/api/search_student?query=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            studentsData = data.students;
            renderStudents(sortStudents(studentsData, currentSort));
          } else {
            showNotification("error", "Lỗi khi tìm kiếm: " + data.message);
          }
        })
        .catch(err => showNotification("error", "Lỗi kết nối: " + err));
    } else {
      PanelFunctions.fetchStudents((students) => {
        renderStudents(students);
      });
    }
  });

  const folderInput = document.getElementById("batch-folder-input");
  document.getElementById("batch-add").addEventListener("click", () => {
    document.getElementById("batch-student-count").querySelector("span").textContent = "0";
    openModal("batch-add-modal");
  });
  document.getElementById("select-folder").addEventListener("click", () => {
    folderInput.click();
  });
  folderInput.addEventListener("change", (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      document.getElementById("batch-student-count").querySelector("span").textContent = files.length;
    }
  });
  document.getElementById("save-batch-add").addEventListener("click", () => {
    const files = folderInput.files;
    if (files.length > 0) {
      const folderName = files[0].webkitRelativePath.split("/")[0];
      PanelFunctions.batchAddStudents(folderName, () => {
        PanelFunctions.fetchStudents((students) => {
          renderStudents(sortStudents(students, currentSort));
        });
        closeModal("batch-add-modal");
      });
    } else {
      alert("Vui lòng chọn thư mục.");
    }
  });
  document.getElementById("cancel-batch-add").addEventListener("click", () => closeModal("batch-add-modal"));

  document.getElementById("set-cutoff").addEventListener("click", () => {
    document.getElementById("cutoff-gmt").value = "";
    document.getElementById("cutoff-time").value = "";
    openModal("set-cutoff-modal");
  });
  document.getElementById("save-cutoff").addEventListener("click", () => {
    const gmt = document.getElementById("cutoff-gmt").value;
    const cutoff = document.getElementById("cutoff-time").value;
    if (gmt && cutoff) {
      PanelFunctions.setCutoff(gmt, cutoff, data => {
        alert("Đã cài đặt hạn chót: " + JSON.stringify(data));
        closeModal("set-cutoff-modal");
      });
    } else {
      alert("Vui lòng nhập đầy đủ thông tin.");
    }
  });
  document.getElementById("cancel-cutoff").addEventListener("click", () => closeModal("set-cutoff-modal"));

  document.getElementById("add-student").addEventListener("click", () => {
    document.getElementById("add-image").value = "";
    document.getElementById("add-HoVaTen").value = "";
    document.getElementById("add-Lop").value = "";
    document.getElementById("add-Gender").value = "";
    document.getElementById("add-NgaySinh").value = "";
    openModal("add-student-modal");
  });
  document.getElementById("add-image").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) autofillFromFilename(file.name, "add-HoVaTen", "add-Lop");
  });
  document.getElementById("save-add-student").addEventListener("click", () => {
    const fileInput = document.getElementById("add-image");
    const imagePath = fileInput.files[0] ? fileInput.files[0].name : "";
    const HoVaTen = document.getElementById("add-HoVaTen").value;
    const Lop = document.getElementById("add-Lop").value;
    const Gender = document.getElementById("add-Gender").value;
    const NgaySinh = document.getElementById("add-NgaySinh").value;
    const UID = Date.now().toString() + Math.floor(Math.random() * 900 + 100).toString();
    if (imagePath && HoVaTen && Lop && Gender && NgaySinh) {
      PanelFunctions.addStudent({ UID, HoVaTen, NgaySinh, Lop, Gender, ImagePath: imagePath }, () => {
        PanelFunctions.fetchStudents((students) => {
          renderStudents(sortStudents(students, currentSort));
        });
        closeModal("add-student-modal");
      });
    } else {
      alert("Vui lòng nhập đầy đủ thông tin.");
    }
  });
  document.getElementById("cancel-add-student").addEventListener("click", () => closeModal("add-student-modal"));

  document.getElementById("edit-student").addEventListener("click", () => {
    if (!selectedStudent) {
      alert("Vui lòng chọn học sinh cần chỉnh sửa.");
      return;
    }
    document.getElementById("edit-HoVaTen").value = selectedStudent.HoVaTen;
    document.getElementById("edit-Lop").value = selectedStudent.Lop;
    document.getElementById("edit-Gender").value = selectedStudent.Gender;
    document.getElementById("edit-NgaySinh").value = selectedStudent.NgaySinh;
    document.getElementById("edit-ImagePath").value = selectedStudent.ImagePath;
    document.getElementById("edit-image").value = "";
    openModal("edit-student-modal");
  });
  document.getElementById("edit-image").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) autofillFromFilename(file.name, "edit-HoVaTen", "edit-Lop");
  });
  document.getElementById("save-edit-student").addEventListener("click", () => {
    if (!selectedStudent) return;
    const fileInput = document.getElementById("edit-image");
    const imagePath = fileInput.files[0] ? fileInput.files[0].name : document.getElementById("edit-ImagePath").value;
    const HoVaTen = document.getElementById("edit-HoVaTen").value;
    const Lop = document.getElementById("edit-Lop").value;
    const Gender = document.getElementById("edit-Gender").value;
    const NgaySinh = document.getElementById("edit-NgaySinh").value;
    if (imagePath && HoVaTen && Lop && Gender && NgaySinh) {
      PanelFunctions.editStudent(selectedStudent.id, { HoVaTen, Lop, Gender, NgaySinh, ImagePath: imagePath }, () => {
        PanelFunctions.fetchStudents((students) => {
          renderStudents(sortStudents(students, currentSort));
        });
        closeModal("edit-student-modal");
      });
    } else {
      alert("Vui lòng nhập đầy đủ thông tin.");
    }
  });
  document.getElementById("cancel-edit-student").addEventListener("click", () => closeModal("edit-student-modal"));

  document.getElementById("delete-student").addEventListener("click", () => {
    if (!selectedStudent) {
      alert("Vui lòng chọn học sinh cần xoá.");
      return;
    }
    if (confirm("Bạn có chắc muốn xoá học sinh có UID " + selectedStudent.UID + "?")) {
      PanelFunctions.deleteStudent(selectedStudent.UID, () => {
        PanelFunctions.fetchStudents((students) => {
          renderStudents(sortStudents(students, currentSort));
        });
      });
    }
  });

  function fetchUsers() {
    PanelFunctions.fetchUsers(renderUsers);
  }
  fetchUsers();

  document.getElementById("add-user-btn").addEventListener("click", () => {
    document.getElementById("add-username").value = "";
    document.getElementById("add-password").value = "";
    document.getElementById("add-role").value = "user";
    openModal("add-user-modal");
  });
  document.getElementById("save-add-user").addEventListener("click", () => {
    const username = document.getElementById("add-username").value;
    const password = document.getElementById("add-password").value;
    const role = document.getElementById("add-role").value;
    if (username && password && role) {
      PanelFunctions.addUser(username, password, role, () => {
        fetchUsers();
        closeModal("add-user-modal");
      });
    } else {
      alert("Vui lòng nhập đầy đủ thông tin.");
    }
  });
  document.getElementById("cancel-add-user").addEventListener("click", () => closeModal("add-user-modal"));

  document.getElementById("edit-user-btn").addEventListener("click", () => {
    if (!selectedUser) {
      alert("Vui lòng chọn người dùng cần chỉnh sửa.");
      return;
    }
    document.getElementById("edit-username").value = selectedUser.username;
    document.getElementById("edit-password").value = "";
    document.getElementById("edit-role").value = selectedUser.role;
    openModal("edit-user-modal");
  });
  document.getElementById("save-edit-user").addEventListener("click", () => {
    if (!selectedUser) return;
    const username = document.getElementById("edit-username").value;
    const password = document.getElementById("edit-password").value;
    const role = document.getElementById("edit-role").value;
    if (username && password && role) {
      PanelFunctions.editUser(selectedUser.id, username, password, role, () => {
        fetchUsers();
        closeModal("edit-user-modal");
      });
    } else {
      alert("Vui lòng nhập đầy đủ thông tin.");
    }
  });
  document.getElementById("cancel-edit-user").addEventListener("click", () => closeModal("edit-user-modal"));

  document.getElementById("delete-user-btn").addEventListener("click", () => {
    if (!selectedUser) {
      alert("Vui lòng chọn người dùng cần xoá.");
      return;
    }
    document.getElementById("delete-username-display").textContent = selectedUser.username;
    openModal("delete-user-modal");
  });
  document.getElementById("confirm-delete-user").addEventListener("click", () => {
    if (!selectedUser) return;
    PanelFunctions.deleteUser(selectedUser.id, () => {
      fetchUsers();
      closeModal("delete-user-modal");
    });
  });
  document.getElementById("cancel-delete-user").addEventListener("click", () => closeModal("delete-user-modal"));

  document.getElementById("logout").addEventListener("click", () => window.location.href = "/");
  document.getElementById("quit").addEventListener("click", () => {
    if (confirm("Bạn có chắc muốn thoát?")) window.close();
  });

  const sortBtn = document.getElementById("sort-students");
  const sortDropdown = document.getElementById("sort-dropdown");

  sortBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (sortDropdown.style.display === "block") {
      sortDropdown.style.display = "none";
    } else {
      sortDropdown.style.display = "block";
    }
  });

  const sortOptions = sortDropdown.querySelectorAll("li");
  sortOptions.forEach(option => {
    option.addEventListener("click", () => {
      const sortValue = option.getAttribute("data-sort");
      currentSort = sortValue;
      sortBtn.textContent = "Sắp xếp theo: " + option.textContent;
      sortDropdown.style.display = "none";
      if (studentsData && studentsData.length > 0) {
        renderStudents(sortStudents(studentsData, currentSort));
      }
    });
  });

  document.addEventListener("click", (e) => {
    if (!sortDropdown.contains(e.target) && e.target !== sortBtn) {
      sortDropdown.style.display = "none";
    }
  });

  PanelFunctions.fetchStudents((students) => {
    renderStudents(students);
  });
});
