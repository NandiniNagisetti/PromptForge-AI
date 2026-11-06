
# PromptForge AI 🚀
### Intelligent Prompt Engineering & LLM Evaluation Platform

PromptForge AI is a full-stack application designed to experiment with prompt engineering strategies and evaluate how changes in prompts influence Large Language Model (LLM) responses.

The platform connects a React frontend, Node.js backend, and Python-based AI service to provide an interactive environment for prompt experimentation and response comparison.

## ✨ Key Features

- **Prompt Experimentation:** Test prompts using different prompting strategies.
- **LLM Response Comparison:** Generate and compare responses from an AI model.
- **Prompt Sensitivity Analysis:** Explore how prompt modifications affect model outputs.
- **Custom Prompt Testing:** Experiment with user-defined prompts.
- **Response Evaluation:** Compare generated responses using similarity-based analysis.
- **Performance Analysis:** Examine response generation time and latency.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Node.js, Express.js |
| AI Service | Python, FastAPI |
| Language Model | Ollama, Qwen2.5:3b |
| Version Control | Git, GitHub |

## 🏗️ Project Architecture

```text
PromptForge-AI/
│
├── frontend/       # React application
│
├── backend/        # Node.js and Express server
│
├── ai_service/     # Python AI service
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ How It Works

1. The user enters a prompt through the frontend.
2. The backend handles application requests.
3. The Python AI service communicates with the locally running language model.
4. The model generates responses based on the selected prompting strategy.
5. The application displays the generated responses for comparison and evaluation.

## 💻 Local Setup

### Prerequisites

- Node.js and npm
- Python 3.10+
- Ollama
- Git

### 1. Clone the repository

```bash
git clone https://github.com/NandiniNagisetti/PromptForge-AI.git
cd PromptForge-AI
```

### 2. Set up the AI model

Install Ollama and download the model:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running before using the AI features.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
```

Start the frontend using the development command configured in its package.json.

### 5. Start the backend services

Install the backend dependencies and start the Node.js and Python services using their respective project configurations.

## 🎯 Project Objective

The objective of PromptForge AI is to explore prompt engineering, understand LLM response behavior, and build an interactive platform for evaluating prompt effectiveness.

## 👩‍💻 Developer

**Nandini Nagisetti**

- GitHub: [NandiniNagisetti](https://github.com/NandiniNagisetti)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/nandini-nandu-89b705372/)

---

*PromptForge AI — Experiment. Evaluate. Improve.*
