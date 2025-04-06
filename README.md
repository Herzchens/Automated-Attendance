# 🎓 Automated-Attendance 📸

A Python-based **Automatic Attendance System** using Facial Recognition 🧠

---

## 📚 Table of Contents
- 📌 [Introduction](#-introduction)
- ✨ [Features](#-features)
- 🚀 [Getting Started](#-getting-started)
- ⚙️ [Usage](#-usage)
- 🧰 [Technologies Used](#-technologies-used)
- 📦 [Installation](#-installation)
- 🤝 [Contributing](#-contributing)
- 🪪 [License](#-license)
- ⚠️ [Notes](#-notes)

---

## 📌 Introduction
Welcome to **Automated-Attendance**, an automatic attendance system powered by **facial recognition**!  
This system can **detect and recognize faces in real-time** from a webcam feed and is ideal for classrooms, offices, and other attendance-tracking use cases.

---

## ✨ Features
- 🧠 **Facial Recognition-Based Attendance**  
- 🎥 **Real-time detection from webcam**  
- 📸 Support for **images/videos** (⚠️ *Planned feature*)  
- 🎯 **High accuracy** with pretrained models (Haar Cascade, Dlib, or CNN)  
- 🔗 Easy to **integrate with existing systems**  
- 📊 **MySQL integration** for storing attendance data  
- 🖥️ Optional **HTML-based GUI** with Flask  

---

## 🚀 Getting Started

### 📦 Installation
> ⚙️ _You can use Docker later. For now, install dependencies manually:_

```bash
pip install mysql-connector-python opencv-python torch torchvision torchaudio numpy pillow customtkinter scikit-learn facenet-pytorch tk pyinstaller openpyxl flask
```

---

## ⚙️ Usage

### 🧑‍💻 Run the Real-time Attendance GUI

#### 🪟 Basic Python GUI:
```bash
python GUI.py
```

#### 🌐 HTML (Web-based) GUI:
```bash
python GUI/backend.py
```

### 🛠️ Configuration:
Edit the `config.json` file to change settings like camera index, attendance thresholds, database credentials, etc.

---

## 🧰 Technologies Used
| 🛠 Technology      | 💡 Use Case |
|------------------|-------------|
| 🐍 **Python 3.9+**    | Core language for backend logic and image processing |
| 📷 **OpenCV**         | Image processing & face detection |
| 🧠 **Dlib**           | Advanced facial recognition |
| 🔬 **FaceNet (facenet-pytorch)** | Face embedding and comparison |
| 🔗 **MySQL**          | Storing attendance data |
| 🌐 **Flask**          | Backend API server for HTML GUI |
| 🎨 **HTML/CSS**       | Web interface layout and styling |
| ⚙️ **JavaScript**     | Frontend interactivity and webcam integration |
| 🖼️ **Pillow, Numpy**  | Additional image processing |
| 🧱 **CustomTkinter**  | Modern desktop GUI |
| 📦 **PyInstaller**    | Packaging the application into executable |

> Facial recognition is powered by the awesome [face_recognition](https://github.com/ageitgey/face_recognition) library 🙌

---

## 🤝 Contributing

We ❤️ contributions!  
Feel free to:
- 🛠 Fork the repo
- 🐞 Report issues
- 📬 Submit pull requests

> For big changes, open an issue first to discuss your ideas.

🔗 **Repository:** [Herzchens/Automated-Attendance](https://github.com/Herzchens/Automated-Attendance)  
💬 **Contact Me on Discord:** [itztli_herzchen](https://discord.com/users/984085171408080897)

---

## 🪪 License

This project is licensed under the **GNU General Public License v3.0**.  
See the `LICENSE` file for full details.

---

## ⚠️ Notes

- 🚧 **This project is still under development** – expect bugs and missing features.
- 📷 **A compatible webcam is required** for real-time attendance.
- 💡 Want Docker support? Coming soon!

---

> ⭐️ If you like this project, give it a star on GitHub to support the development!

