# NeuroScan AI: Alzheimer's MRI Analysis & Assistant

> An AI-powered web application for Alzheimer's disease classification from MRI scans, with an integrated medical chatbot, scan history, and PDF report generation.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-black?style=flat-square&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue?style=flat-square&logo=sqlite)

---

##  Overview

NeuroScan AI is an end-to-end deep learning web application that:

- Classifies Alzheimer's disease severity from MRI scans using a fine-tuned **ResNet18** model
- Provides an AI-powered **medical chatbot** (Llama 3.3 70B via Groq) for detailed Alzheimer's Q&A
- Saves **scan history** and **chat history** per user account
- Generates **professional PDF reports** for each MRI analysis
- Features a **dark medical-grade UI** built with Tailwind-inspired CSS

> Note: **Disclaimer:** This project is for **educational and research purposes only**. It does not constitute a medical diagnosis. Always consult a qualified neurologist.

---

##  Features

| Feature | Description |
|---|---|
| MRI Classification | ResNet18 classifies into 4 Alzheimer's stages |
| Medical Chatbot | Context-aware assistant using Llama 3.3 70B via Groq API |
| User Authentication | Register/login with hashed passwords |
| Dashboard | Stats, scan history, chat history per user |
| PDF Reports | Downloadable clinical-style reports per scan |
| Chat History | Persistent per-session saved conversations |
| Data Management | Delete scans and chat sessions |
| Responsive UI | Dark theme, professional medical design |

---

## Classification Classes

| Class | Description | GDS Stage |
|---|---|---|
| `NonDemented` | No signs of dementia | GDS 1–2 |
| `VeryMildDemented` | Very early cognitive changes | GDS 3 |
| `MildDemented` | Mild cognitive decline | GDS 4 |
| `ModerateDemented` | Moderate cognitive decline | GDS 5–6 |

---

## Project Structure

```
AlzeigmersChatBot/
├── chatbot/                        # Flask web application
│   ├── app.py                      # Main Flask app, routes, auth, DB
│   ├── inference.py                # Model loading, prediction, transform
│   ├── generate_reports.py         # PDF report generation (ReportLab)
│   ├── .env                        # API keys (not committed)
│   ├── instance/
│   │   └── neuroscan.db            # SQLite database (auto-created)
│   ├── static/
│   │   ├── uploads/                # Temporary MRI uploads
│   │   └── reports/                # Generated PDF reports
│   └── templates/
│       ├── index.html              # Main scan + chat interface
│       ├── login.html              # Login / Register page
│       ├── dashboard.html          # User dashboard
│       ├── history.html            # Full scan history
│       ├── reports.html            # PDF reports page
│       ├── chat_history_list.html  # All chat sessions
│       └── chat_history.html       # Single chat session view
│
├── models/
│   ├── resnet_model.py             # ResNet18 architecture
│   └── resnet2d_alzheimers_best.pth # Trained model weights
│
├── utils/
│   └── data_preprocessing.py      # DataLoaders, transforms (RGB + ImageNet norm)
│
├── data/
│   ├── train/                      # Original training data
│   ├── train_aug/                  # Augmented + balanced training data
│   └── test/                       # Test data
│
├── plots/
│   ├── training_curves.png         # Loss & accuracy curves
│   └── confusion_matrix.png        # Confusion matrix
│
└── main.py                         # Training script
```

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Alzheimers_ChatBot.git
cd neuroscan-ai
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file inside the `chatbot/` folder:

```env
GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Add Trained Model Weights

Place your trained model file at:
```
models/resnet2d_alzheimers_best.pth
```

To train from scratch, see [Training](#-training) below.

### 6. Run the App

```bash
cd chatbot
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Requirements

```txt
flask>=3.1
flask-sqlalchemy
torch>=2.0
torchvision
pillow
python-dotenv
groq
werkzeug
reportlab
scikit-learn
matplotlib
seaborn
```

Install all at once:
```bash
pip install flask flask-sqlalchemy torch torchvision pillow python-dotenv groq werkzeug reportlab scikit-learn matplotlib seaborn
```

---

## Training

### Dataset

This project uses the [Alzheimer MRI Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) from Kaggle — 10,240 MRI images across 4 classes, pre-split into train/test.

> Note: **Known Limitation:** The Kaggle dataset has documented data leakage — the same patient's scans may appear in both train and test sets. Validation accuracy may appear inflated. For clinical use, re-split the data by patient ID.

### Model Architecture

- **Base:** ResNet18 (pretrained on ImageNet)
- **Input:** 224×224 RGB (grayscale MRI replicated to 3 channels)
- **Normalization:** ImageNet mean/std `[0.485, 0.456, 0.406]`
- **Classifier:** `Dropout(0.3)` → `Linear(512, 4)`
- **Full fine-tuning:** No layers frozen

### Run Training

```bash
python main.py
```

**Hyperparameters (main.py):**

| Parameter | Value |
|---|---|
| Batch Size | 32 |
| Image Size | 224 × 224 |
| Epochs | 20 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Early Stopping | Patience = 5 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |

Training outputs:
- Best model → `models/resnet2d_alzheimers_best.pth`
- Loss & accuracy curves → `plots/training_curves.png`
- Confusion matrix → `plots/confusion_matrix.png`

---

## Database Schema

The app uses **SQLite** via Flask-SQLAlchemy. The database is auto-created at `chatbot/instance/neuroscan.db` on first run.

```
User
 ├── id, name, email, password (hashed), created_at
 ├── → Scan (one-to-many)
 └── → ChatSession (one-to-many)

Scan
 ├── id, user_id, prediction, confidence, all_probs (JSON), warning, created_at
 └── → ChatSession (one-to-many)

ChatSession
 ├── id, user_id, scan_id, scan_stage, created_at
 └── → ChatMessage (one-to-many)

ChatMessage
 └── id, chat_session_id, role (user/assistant), content, created_at
```

---

## Chatbot

The assistant uses **Llama 3.3 70B** via the [Groq API](https://groq.com):

- Responds in structured markdown (`##` headings, bullet points, bold text)
- Context-aware — knows the MRI prediction stage and confidence
- Conversation history saved to database (last 10 messages as context)
- Free tier: **14,400 requests/day**

---

## PDF Reports

Reports are generated with **ReportLab** and include:

- Prediction classification + confidence score
- GDS (Global Deterioration Scale) staging
- Probability distribution table for all 4 classes
- Stage-specific symptoms and clinical recommendations
- Colour-coded prognosis
- About Alzheimer's disease section
- Medical disclaimer

---

## API Routes

| Method | Route | Description |
|---|---|---|
| `GET/POST` | `/` | Main scan + chat page |
| `GET/POST` | `/login` | Login page |
| `POST` | `/register` | Register new user |
| `GET` | `/logout` | Logout |
| `GET` | `/dashboard` | User dashboard |
| `GET` | `/history` | Full scan history |
| `GET` | `/reports` | PDF reports list |
| `GET` | `/chat_history` | Chat sessions list |
| `GET` | `/chat_history/<id>` | Single chat session view |
| `POST` | `/chat` | Send chat message (JSON) |
| `POST` | `/clear_chat` | Clear current chat session |
| `GET` | `/download_report` | Download report for current scan |
| `GET` | `/download_report/<id>` | Download report for any past scan |
| `POST` | `/delete_scan/<id>` | Delete a scan and linked chats |
| `POST` | `/delete_chat/<id>` | Delete a chat session |

---

## Security Notes

- Passwords hashed using **Werkzeug's `generate_password_hash`** (PBKDF2-SHA256)
- All routes protected with `@login_required` decorator
- User data scoped by `user_id` — no cross-user data access
- API keys stored in `.env` — never committed to version control

---

## Model Performance

> Results on the Kaggle Alzheimer MRI Dataset (note: dataset has train/test data leakage by patient)

| Metric | Value |
|---|---|
| Architecture | ResNet18 (fine-tuned) |
| Input | 224×224 RGB |
| Best Val Accuracy | ~65–75% (varies by run) |
| Classes | 4 |

Realistic accuracy expectations:
- ResNet18 fine-tuned: **65–75%**
- ResNet50 with tuning: **75–82%**
- Best published on this dataset: **85–90%**

---

## Deployment

### Render (Free)

1. Push to GitHub
2. Create new **Web Service** on [render.com](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn chatbot.app:app`
5. Add environment variables: `GROQ_API_KEY`, `FLASK_SECRET_KEY`

### Hugging Face Spaces

1. Create a new Space with **Docker** SDK
2. Upload all files
3. Add secrets for API keys in Space settings

---

## Roadmap

- [ ] RGB training pipeline (currently grayscale → 3 channels)
- [ ] Grad-CAM heatmap visualization
- [ ] MMSE score input for multi-modal assessment
- [ ] Longitudinal scan tracking (trend: improving/stable/worsening)
- [ ] ResNet50 / EfficientNet upgrade
- [ ] Patient risk factor form
- [ ] Admin dashboard

---

## Acknowledgements

- [Kaggle Alzheimer MRI Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) by Tourist55
- [Groq](https://groq.com) for ultra-fast LLM inference
- [Meta AI](https://ai.meta.com) for the Llama 3.3 model
- [PyTorch](https://pytorch.org) and [torchvision](https://pytorch.org/vision)
- [ReportLab](https://www.reportlab.com) for PDF generation

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Medical Disclaimer

> NeuroScan AI is an **educational tool** built to demonstrate AI applications in medical imaging. It is **not a medical device** and must **not be used for clinical diagnosis**. All results should be interpreted by qualified medical professionals. The developers accept no liability for clinical decisions made based on this tool's output.

---

*Built with  using PyTorch, Flask, and Groq · NeuroScan AI*
