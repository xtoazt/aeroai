# Oumi ML Platform - Vercel Deployment

This is a Vercel-optimized version of the Oumi ML platform that provides web-based access to key ML operations through REST APIs.

## 🚀 What's Included

### Core Features
- **Inference API**: Run inference on various models (OpenAI, Anthropic, Gemini, Together AI)
- **Evaluation API**: Evaluate models on standard benchmarks
- **Judgment API**: Use LLM judges to evaluate response quality
- **Web Interface**: Modern React frontend for easy interaction

### Key Differences from Full Oumi
- **API-based**: Uses external APIs instead of local model inference
- **Lightweight**: Removed heavy dependencies (PyTorch, Transformers, etc.)
- **Serverless**: Optimized for Vercel's serverless environment
- **Simplified**: Focuses on core inference and evaluation capabilities

## 📁 Project Structure

```
├── api/
│   ├── main.py              # FastAPI application
│   └── oumi_lite.py         # Lightweight Oumi implementation
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # React application
│   │   ├── main.tsx         # Entry point
│   │   └── index.css        # Styles
│   ├── package.json         # Frontend dependencies
│   └── vite.config.ts       # Vite configuration
├── vercel.json              # Vercel deployment config
├── requirements.txt         # Python dependencies
└── README_VERCEL.md         # This file
```

## 🛠️ Setup and Deployment

### Prerequisites
- Vercel account
- API keys for the services you want to use:
  - OpenAI API key
  - Anthropic API key (optional)
  - Google Gemini API key (optional)
  - Together AI API key (optional)

### 1. Deploy to Vercel

#### Option A: Deploy via Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variables
vercel env add OPENAI_API_KEY
vercel env add ANTHROPIC_API_KEY  # Optional
vercel env add GEMINI_API_KEY     # Optional
vercel env add TOGETHER_API_KEY   # Optional
```

#### Option B: Deploy via GitHub
1. Push this code to a GitHub repository
2. Connect the repository to Vercel
3. Set environment variables in Vercel dashboard
4. Deploy

### 2. Environment Variables

Set these in your Vercel project settings:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional
GEMINI_API_KEY=your_gemini_api_key        # Optional
TOGETHER_API_KEY=your_together_api_key    # Optional
```

### 3. Local Development

#### Backend (API)
```bash
cd api
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🔧 API Endpoints

### Inference
```bash
POST /api/inference
{
  "model_name": "gpt-4o-mini",
  "prompt": "Explain machine learning",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.95,
  "engine": "OPENAI"
}
```

### Evaluation
```bash
POST /api/evaluate
{
  "model_name": "gpt-4o-mini",
  "tasks": ["arc_challenge", "hellaswag"],
  "num_samples": 10
}
```

### Judgment
```bash
POST /api/judge
{
  "judge_model": "gpt-4o-mini",
  "prompt_template": "Rate this response: {response}",
  "dataset": [
    {"question": "What is AI?", "answer": "AI is..."}
  ]
}
```

### Available Models
```bash
GET /api/models
```

### Available Tasks
```bash
GET /api/tasks
```

## 🎯 Usage Examples

### Using the Web Interface
1. Open your deployed Vercel URL
2. Select the "Inference" tab
3. Choose a model and enter your prompt
4. Click "Run Inference"

### Using the API Directly
```python
import requests

# Run inference
response = requests.post("https://your-app.vercel.app/api/inference", json={
    "model_name": "gpt-4o-mini",
    "prompt": "What is machine learning?",
    "engine": "OPENAI"
})

print(response.json())
```

## ⚠️ Limitations

### Vercel Constraints
- **Function timeout**: 30 seconds max (can be extended with Pro plan)
- **Memory limit**: 1GB (can be extended with Pro plan)
- **Package size**: 50MB limit for serverless functions
- **Cold starts**: May experience delays on first request

### Feature Limitations
- **No local training**: Training requires significant compute resources
- **Simplified evaluation**: Uses basic question-answer format
- **API dependencies**: Requires external API keys and internet connectivity
- **No model hosting**: Cannot host large models locally

## 🔄 Migration from Full Oumi

If you're migrating from the full Oumi platform:

1. **Training**: Use the full Oumi CLI for training, then use this API for inference
2. **Evaluation**: This provides simplified evaluation; use full Oumi for comprehensive benchmarks
3. **Model hosting**: Use external APIs or deploy models separately
4. **Configuration**: API-based configuration instead of YAML files

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are in requirements.txt
2. **API key errors**: Verify environment variables are set correctly
3. **Timeout errors**: Reduce max_tokens or use faster models
4. **Memory errors**: Use smaller models or reduce batch sizes

### Debug Mode
Set `DEBUG=true` in environment variables for detailed error messages.

## 📚 Resources

- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Oumi Documentation](https://oumi.ai/docs)

## 🤝 Contributing

This is a simplified version of Oumi. For full features and contributions:
- Visit the main [Oumi repository](https://github.com/oumi-ai/oumi)
- Check the [documentation](https://oumi.ai/docs)
- Join the [community](https://discord.gg/oumi)

## 📄 License

Same license as the main Oumi project (Apache 2.0).
