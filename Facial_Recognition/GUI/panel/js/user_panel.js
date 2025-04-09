document.addEventListener('DOMContentLoaded', () => {
  let studentsData = [];
  let currentSort = "az";

function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = "block";
  }
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = "none";
  }
}

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
