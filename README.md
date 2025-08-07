
# 🧠 LLM Draw.io Assistant

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A full-stack application combining a frontend interface and a FastAPI backend to enable LLM-powered interactions with Draw.io diagrams.

---

## 🗂️ Project Structure

```
root/
│
├── frontend/        # Frontend (e.g., Vite/React/Vue)
└── backend/         # FastAPI-based backend
```

---

## 🚀 Getting Started

### ⚙️ Prerequisites

- [Node.js](https://nodejs.org/) (v18 or later)
- [Python 3.10](https://www.python.org/downloads/release/python-3100/)
- Docker or a strong local GPU (optional)
- `.rar` extractor (if running backend locally)

---

## 📦 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

> The frontend will be running at: [http://localhost:5173](http://localhost:5173)

---

## 🧠 Backend Setup

You have two options for running the backend:

---

### 🔁 Option 1: Inside the Innkube Container

```bash
cd llm-drawio-backend/

apt update
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install python3.10 python3.10-venv python3.10-dev -y

python3.10 -m venv venv310
source venv310/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

> The backend will be available at: [http://localhost:8000](http://localhost:8000)

---

### 💻 Option 2: Locally with Files

1. Download the necessary resources archive from this link:  
   📦 [Download `.rar` file](https://drive.google.com/file/d/1juTLR8cXnwdFNHNTXVizJju0SCBS5kMW/view?usp=sharing) 

2. Extract it directly into the `backend/` directory.

3. Run the backend locally:

```bash
cd backend
python3.10 -m venv venv310
source venv310/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🌐 Accessing the App

Once both servers are running:

🔗 Open [http://localhost:5173](http://localhost:5173) in your browser.

There is a file called "testing.txt", you can find some prompts there to try out the chatbot

---

## ❓ Need Help?

Feel free to reach out:

- **Souhail Karam** – 📧 [souhailkaram.studies@gmail.com](mailto:souhailkaram.studies@gmail.com)
- **Chaimaa El Argoubi** – 📧 [chaimae.elargoubi.studies@gmail.com](mailto:chaimae.elargoubi.studies@gmail.com)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---


