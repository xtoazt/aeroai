import React, { useState, useEffect } from 'react'
import { Brain, Zap, BarChart3, Gavel, Upload, CheckCircle, AlertCircle } from 'lucide-react'
import axios from 'axios'

const API_BASE = '/api'

interface Model {
  name: string
  provider: string
  description: string
}

interface Task {
  name: string
  category: string
  description: string
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('inference')
  const [models, setModels] = useState<Model[]>([])
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  // Inference state
  const [inferenceForm, setInferenceForm] = useState({
    model: 'gpt-4o-mini',
    prompt: 'Explain machine learning in simple terms.',
    maxTokens: 512,
    temperature: 0.7,
    topP: 0.95,
    engine: 'OPENAI'
  })

  // Evaluation state
  const [evaluationForm, setEvaluationForm] = useState({
    model: 'gpt-4o-mini',
    tasks: ['arc_challenge', 'hellaswag'],
    numSamples: 10
  })

  // Judge state
  const [judgeForm, setJudgeForm] = useState({
    judgeModel: 'gpt-4o-mini',
    promptTemplate: 'Rate the quality of this response on a scale of 1-10: {response}',
    dataset: [
      { question: 'What is AI?', answer: 'Artificial Intelligence is a field of computer science.' },
      { question: 'How does ML work?', answer: 'Machine learning uses algorithms to learn patterns.' }
    ]
  })

  useEffect(() => {
    loadModels()
    loadTasks()
  }, [])

  const loadModels = async () => {
    try {
      const response = await axios.get(`${API_BASE}/models`)
      const modelData = response.data
      const modelList: Model[] = []
      
      Object.entries(modelData).forEach(([provider, modelNames]: [string, any]) => {
        modelNames.forEach((name: string) => {
          modelList.push({
            name,
            provider: provider.toUpperCase(),
            description: `${provider} model`
          })
        })
      })
      
      setModels(modelList)
    } catch (err) {
      console.error('Failed to load models:', err)
    }
  }

  const loadTasks = async () => {
    try {
      const response = await axios.get(`${API_BASE}/tasks`)
      const taskData = response.data
      const taskList: Task[] = []
      
      Object.entries(taskData).forEach(([category, taskNames]: [string, any]) => {
        taskNames.forEach((name: string) => {
          taskList.push({
            name,
            category,
            description: `${category} evaluation task`
          })
        })
      })
      
      setTasks(taskList)
    } catch (err) {
      console.error('Failed to load tasks:', err)
    }
  }

  const handleInference = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_BASE}/inference`, inferenceForm)
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Inference failed')
    } finally {
      setLoading(false)
    }
  }

  const handleEvaluation = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_BASE}/evaluate`, evaluationForm)
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Evaluation failed')
    } finally {
      setLoading(false)
    }
  }

  const handleJudgment = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_BASE}/judge`, judgeForm)
      setResult(response.data)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Judgment failed')
    } finally {
      setLoading(false)
    }
  }

  const renderInferenceTab = () => (
    <div className="tab-content active">
      <div className="form-group">
        <label>Model</label>
        <select
          value={inferenceForm.model}
          onChange={(e) => setInferenceForm({ ...inferenceForm, model: e.target.value })}
        >
          {models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.name} ({model.provider})
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Prompt</label>
        <textarea
          rows={4}
          value={inferenceForm.prompt}
          onChange={(e) => setInferenceForm({ ...inferenceForm, prompt: e.target.value })}
          placeholder="Enter your prompt here..."
        />
      </div>

      <div className="grid">
        <div className="form-group">
          <label>Max Tokens</label>
          <input
            type="number"
            value={inferenceForm.maxTokens}
            onChange={(e) => setInferenceForm({ ...inferenceForm, maxTokens: parseInt(e.target.value) })}
          />
        </div>

        <div className="form-group">
          <label>Temperature</label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="2"
            value={inferenceForm.temperature}
            onChange={(e) => setInferenceForm({ ...inferenceForm, temperature: parseFloat(e.target.value) })}
          />
        </div>

        <div className="form-group">
          <label>Top P</label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="1"
            value={inferenceForm.topP}
            onChange={(e) => setInferenceForm({ ...inferenceForm, topP: parseFloat(e.target.value) })}
          />
        </div>

        <div className="form-group">
          <label>Engine</label>
          <select
            value={inferenceForm.engine}
            onChange={(e) => setInferenceForm({ ...inferenceForm, engine: e.target.value })}
          >
            <option value="OPENAI">OpenAI</option>
            <option value="ANTHROPIC">Anthropic</option>
            <option value="GEMINI">Google Gemini</option>
            <option value="TOGETHER">Together AI</option>
          </select>
        </div>
      </div>

      <button className="btn" onClick={handleInference} disabled={loading}>
        {loading ? 'Running Inference...' : 'Run Inference'}
      </button>
    </div>
  )

  const renderEvaluationTab = () => (
    <div className="tab-content active">
      <div className="form-group">
        <label>Model</label>
        <select
          value={evaluationForm.model}
          onChange={(e) => setEvaluationForm({ ...evaluationForm, model: e.target.value })}
        >
          {models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.name} ({model.provider})
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Evaluation Tasks</label>
        <div className="grid">
          {tasks.map((task) => (
            <div
              key={task.name}
              className={`model-card ${evaluationForm.tasks.includes(task.name) ? 'selected' : ''}`}
              onClick={() => {
                const newTasks = evaluationForm.tasks.includes(task.name)
                  ? evaluationForm.tasks.filter(t => t !== task.name)
                  : [...evaluationForm.tasks, task.name]
                setEvaluationForm({ ...evaluationForm, tasks: newTasks })
              }}
            >
              <h3>{task.name}</h3>
              <p>{task.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="form-group">
        <label>Number of Samples</label>
        <input
          type="number"
          value={evaluationForm.numSamples}
          onChange={(e) => setEvaluationForm({ ...evaluationForm, numSamples: parseInt(e.target.value) })}
        />
      </div>

      <button className="btn" onClick={handleEvaluation} disabled={loading}>
        {loading ? 'Running Evaluation...' : 'Run Evaluation'}
      </button>
    </div>
  )

  const renderJudgeTab = () => (
    <div className="tab-content active">
      <div className="form-group">
        <label>Judge Model</label>
        <select
          value={judgeForm.judgeModel}
          onChange={(e) => setJudgeForm({ ...judgeForm, judgeModel: e.target.value })}
        >
          {models.map((model) => (
            <option key={model.name} value={model.name}>
              {model.name} ({model.provider})
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Prompt Template</label>
        <textarea
          rows={3}
          value={judgeForm.promptTemplate}
          onChange={(e) => setJudgeForm({ ...judgeForm, promptTemplate: e.target.value })}
          placeholder="Use {response} placeholder for the response to judge"
        />
      </div>

      <div className="form-group">
        <label>Dataset (JSON format)</label>
        <textarea
          rows={6}
          value={JSON.stringify(judgeForm.dataset, null, 2)}
          onChange={(e) => {
            try {
              const parsed = JSON.parse(e.target.value)
              setJudgeForm({ ...judgeForm, dataset: parsed })
            } catch (err) {
              // Invalid JSON, don't update
            }
          }}
        />
      </div>

      <button className="btn" onClick={handleJudgment} disabled={loading}>
        {loading ? 'Running Judgment...' : 'Run Judgment'}
      </button>
    </div>
  )

  return (
    <div className="container">
      <div className="header">
        <h1>🧠 Oumi ML Platform</h1>
        <p>Train, evaluate, and deploy foundation models with ease</p>
      </div>

      <div className="card">
        <div className="tabs">
          <div
            className={`tab ${activeTab === 'inference' ? 'active' : ''}`}
            onClick={() => setActiveTab('inference')}
          >
            <Zap size={20} style={{ marginRight: '8px' }} />
            Inference
          </div>
          <div
            className={`tab ${activeTab === 'evaluation' ? 'active' : ''}`}
            onClick={() => setActiveTab('evaluation')}
          >
            <BarChart3 size={20} style={{ marginRight: '8px' }} />
            Evaluation
          </div>
          <div
            className={`tab ${activeTab === 'judge' ? 'active' : ''}`}
            onClick={() => setActiveTab('judge')}
          >
            <Gavel size={20} style={{ marginRight: '8px' }} />
            Judge
          </div>
        </div>

        {activeTab === 'inference' && renderInferenceTab()}
        {activeTab === 'evaluation' && renderEvaluationTab()}
        {activeTab === 'judge' && renderJudgeTab()}

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
          </div>
        )}

        {error && (
          <div className={`result error`}>
            <AlertCircle size={20} style={{ marginRight: '8px' }} />
            {error}
          </div>
        )}

        {result && !loading && (
          <div className={`result success`}>
            <CheckCircle size={20} style={{ marginRight: '8px' }} />
            {JSON.stringify(result, null, 2)}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
