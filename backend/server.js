
const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = 5000;
const PYTHON_SERVICE = 'http://127.0.0.1:8002';

// Middleware
app.use(cors());
app.use(express.json());

// Welcome route
app.get('/', (req, res) => {
    res.send('Welcome to PromptForge AI Backend!');
});

// Health check route
app.get('/api/health', (req, res) => {
    res.json({
        status: 'success',
        message: 'PromptForge AI Backend is running!'
    });
});

// Generate AI response through Python FastAPI
app.post('/api/generate', async (req, res) => {
    try {
        const response = await fetch(
            `${PYTHON_SERVICE}/api/generate`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(req.body)
            }
        );

        const data = await response.json();
        res.status(response.status).json(data);

    } catch (error) {
        console.error('AI Service Error:', error.message);

        res.status(500).json({
            status: 'error',
            message: 'Could not connect to Python AI service.'
        });
    }
});

// Evaluate prompt experiment responses through Python FastAPI
app.post('/api/evaluate', async (req, res) => {
    try {
        const response = await fetch(
            `${PYTHON_SERVICE}/api/evaluate`,
            {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(req.body)
            }
        );

        const data = await response.json();
        res.status(response.status).json(data);

    } catch (error) {
        console.error('Evaluation Service Error:', error.message);

        res.status(500).json({
            status: 'error',
            message: 'Could not connect to Python evaluation service.'
        });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`PromptForge AI Backend running on http://localhost:${PORT}`);
});
