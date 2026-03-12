# Brain Tumor Detection System (MRI)

This project is a **Brain Tumor Detection System** that predicts whether a brain tumor is present in an MRI image using a deep learning model based on **VGG16**.

The system provides a simple **web interface** where users can upload an MRI image and get the prediction result.

---

## Project Technologies

Frontend:

* HTML
* CSS

Backend:

* Python
* FastAPI
* TensorFlow
* NumPy
* Pillow

Model:

* VGG16 CNN Model (`brain_tumor_vgg16.h5`)

---

## Project Structure

```
brain-tumor-detection-vgg16
│
├── frontend
│   ├── index.html
│   └── style.css
│
├── backend
│   ├── main.py
│   ├── brain_tumor_vgg16.h5
│   └── requirements.txt
│
└── README.md
```

---

## Frontend Setup

The frontend is built using **HTML and CSS**.

1. Open the `frontend` folder.
2. Open `index.html` in your browser.

This page allows users to upload an MRI image.

---

## Backend Setup

Follow these steps to run the backend.

### 1. Go to Backend Folder

```
cd backend
```

### 2. Create Virtual Environment

```
python -m venv venv
```

### 3. Activate Virtual Environment

Windows:

```
venv\Scripts\activate
```

Linux / Mac:

```
source venv/bin/activate
```

---

### 4. Install Required Libraries

```
pip install fastapi uvicorn tensorflow numpy pillow
```

Or install from requirements file:

```
pip install -r requirements.txt
```

---

### 5. Run the Backend Server

```
uvicorn main:app --reload
```

The backend server will start at:

```
http://127.0.0.1:8000
```

---

## Model

The trained model file used in this project:

```
brain_tumor_vgg16.h5
```

This model predicts whether the MRI image contains a tumor or not.

---

## Features

* Upload MRI Image
* Brai
