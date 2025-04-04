function parseJSON(response) {
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
}
function showNotification(message, type = "success") {
  let borderColor, svgContent;
  if (type === "success") {
    borderColor = "#4CAF50";
    svgContent = '<svg width="24" height="24" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12" style="fill:none;stroke:#4CAF50;stroke-width:2"/></svg>';
  } else if (type === "error") {
    borderColor = "#F44336";
    svgContent = '<svg width="24" height="24" viewBox="0 0 24 24"><line x1="4" y1="4" x2="20" y2="20" style="stroke:#F44336;stroke-width:2"/><line x1="20" y1="4" x2="4" y2="20" style="stroke:#F44336;stroke-width:2"/></svg>';
  } else if (type === "info") {
    borderColor = "#2196F3";
    svgContent = '<svg width="24" height="24" viewBox="0 0 24 24"><text x="50%" y="50%" text-anchor="middle" fill="#2196F3" font-size="18" font-family="Arial" dy=".35em">i</text></svg>';
  }

  let notification = document.createElement('div');
  notification.className = 'notification';

  let icon = document.createElement('div');
  icon.className = 'notification-icon';
  if (type === "error") {
    icon.classList.add('error');
  } else if (type === "info") {
    icon.classList.add('info');
  }

  let messageElem = document.createElement('div');
  messageElem.className = 'notification-message';
  messageElem.textContent = message;

  notification.appendChild(icon);
  notification.appendChild(messageElem);
  document.body.appendChild(notification);

  setTimeout(() => {
    notification.classList.add('show');
    icon.classList.add('animate');
    setTimeout(() => {
      icon.style.borderTopColor = borderColor;
      icon.innerHTML = svgContent;
      icon.classList.remove('animate');
    }, 1000);
  }, 10);

  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => {
      notification.remove();
    }, 500);
  }, 3000);
}
function fetchStudents(callback) {
  fetch('/api/students')
    .then(parseJSON)
    .then(data => {
      if (data.success) {
        callback(data.students);
      } else {
        showNotification("Lỗi: " + data.message, "error");
      }
    })
    .catch(error => {
      console.error("Error fetching students:", error);
      showNotification("Lỗi: " + error.message, "error");
    });
}
function searchStudents(students, keyword) {
  if (!keyword) return students;
  keyword = keyword.trim().toLowerCase();
  return students.filter(student => student.name.toLowerCase().includes(keyword));
}
document.getElementById('loginForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    try {
      const response = await fetch('/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
      });
      const data = await response.json();

      if (data.success) {
        showNotification("Đăng nhập thành công", "success");
        setTimeout(() => {
          window.location.href = data.redirect;
        }, 2000);
      } else {
        showNotification(data.message || "Đăng nhập thất bại", "error");
      }
    } catch (error) {
      console.error("Error during login:", error);
      showNotification("Lỗi khi đăng nhập: " + error.message, "error");
    }
});
document.getElementById('register') && document.getElementById('register').addEventListener('click', function() {
    showNotification("Liên hệ Quản Trị Viên để được đăng ký.", "info");
});
document.getElementById('forgetPassword') && document.getElementById('forgetPassword').addEventListener('click', function() {
    showNotification("Liên hệ Quản Trị Viên để lấy lại mật khẩu.", "info");
});
