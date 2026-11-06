
import { useState } from 'react'
import './App.css'

const strategies = [
  'Original',
  'Instruction Added',
  'Direct Instruction',
  'Detailed Instruction',
  'Concise Instruction',
  'Typos Introduced',
]

type ComparisonResult = {
  strategy: string
  response: string
  duration: number
}

function buildStrategyPrompt(originalPrompt: string, strategy: string) {
  switch (strategy) {
    case 'Instruction Added':
      return `${originalPrompt}\n\nPlease provide a clear, accurate and easy-to-understand answer with relevant examples.`

    case 'Direct Instruction':
      return `Follow this instruction directly and answer the request: ${originalPrompt}`

    case 'Detailed Instruction':
      return `${originalPrompt}\n\nProvide a detailed, well-organized explanation. Include important points, examples and a simple conclusion.`

    case 'Concise Instruction':
      return `${originalPrompt}\n\nGive a short, concise and direct answer. Focus only on the most important points.`

    case 'Typos Introduced':
      return originalPrompt.replace(/\b([A-Za-z])([A-Za-z])([A-Za-z]+)\b/, '$2$1$3')

    default:
      return originalPrompt
  }
}

function App() {
  const [activeTab, setActiveTab] = useState('Dashboard')
  const [prompt, setPrompt] = useState('')
  const [model, setModel] = useState('qwen2.5:3b')
  const [strategy, setStrategy] = useState('Original')

  const [status, setStatus] = useState('')
  const [backendMessage, setBackendMessage] = useState('Not connected yet')
  const [checkingBackend, setCheckingBackend] = useState(false)

  const [generatedResponse, setGeneratedResponse] = useState('')
  const [generating, setGenerating] = useState(false)

  const [comparisonResults, setComparisonResults] = useState<ComparisonResult[]>([])
  const [comparing, setComparing] = useState(false)

  const menuItems = [
    'Dashboard',
    'Prompt Experiments',
    'Model Evaluation',
    'Analytics',
    'Experiment History',
  ]

  async function checkBackend() {
    setCheckingBackend(true)
    setBackendMessage('Connecting to backend...')

    try {
      const response = await fetch('http://localhost:5000/api/health')

      if (!response.ok) {
        throw new Error('Backend request failed')
      }

      const data = await response.json()
      setBackendMessage(data.message)
    } catch {
      setBackendMessage(
        'Connection failed. Please check whether Express is running.'
      )
    } finally {
      setCheckingBackend(false)
    }
  }

  async function generateAIResponse(promptText: string) {
    const response = await fetch('http://localhost:5000/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: promptText,
        model: model,
        temperature: 0.7,
        max_tokens: 512,
      }),
    })

    const data = await response.json()

    if (!response.ok || data.status !== 'success') {
      throw new Error(data.message || 'AI generation failed.')
    }

    return data.generated_response || 'The model returned an empty response.'
  }

  async function runExperiment() {
    if (!prompt.trim()) {
      setStatus('Please enter a prompt before running an experiment.')
      setGeneratedResponse('')
      return
    }

    setGenerating(true)
    setGeneratedResponse('')
    setComparisonResults([])
    setStatus('Generating AI response. Please wait...')

    try {
      const strategyPrompt = buildStrategyPrompt(prompt, strategy)
      const result = await generateAIResponse(strategyPrompt)

      setGeneratedResponse(result)
      setStatus(
        `Experiment completed successfully! Model: ${model} | Strategy: ${strategy}`
      )
    } catch (error) {
      setStatus(
        error instanceof Error
          ? error.message
          : 'Something went wrong while generating the response.'
      )
    } finally {
      setGenerating(false)
    }
  }

  async function runComparison() {
    if (!prompt.trim()) {
      setStatus('Please enter a prompt before comparing strategies.')
      return
    }

    setComparing(true)
    setComparisonResults([])
    setGeneratedResponse('')
    setStatus('Running all six prompt strategies. Please wait...')

    const results: ComparisonResult[] = []

    try {
      for (const currentStrategy of strategies) {
        setStatus(`Running strategy: ${currentStrategy}...`)

        const strategyPrompt = buildStrategyPrompt(prompt, currentStrategy)

        const startTime = Date.now()
        const result = await generateAIResponse(strategyPrompt)
        const duration = (Date.now() - startTime) / 1000

        results.push({
          strategy: currentStrategy,
          response: result,
          duration,
        })

        // Show each completed result immediately
        setComparisonResults([...results])
      }

      setStatus(
        `Comparison completed successfully! ${results.length} strategies tested using ${model}.`
      )
    } catch (error) {
      setStatus(
        error instanceof Error
          ? `Comparison stopped: ${error.message}`
          : 'Something went wrong during comparison.'
      )
    } finally {
      setComparing(false)
    }
  }

  return (
    <div className="app-container">

      {/* Sidebar */}
      <aside className="sidebar">
        <h2 className="logo">PromptForge AI</h2>

        <p className="sidebar-subtitle">
          LLM Evaluation Platform
        </p>

        <nav>
          {menuItems.map((item) => (
            <button
              key={item}
              className={`nav-item ${activeTab === item ? 'active' : ''}`}
              onClick={() => {
                setActiveTab(item)
                setStatus('')
              }}
            >
              {item}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          Powered by AI
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">

        <header className="topbar">
          <div>
            <h1>{activeTab}</h1>

            <p>
              Intelligent Prompt Engineering and LLM Evaluation
            </p>
          </div>
        </header>

        {/* Dashboard */}
        {activeTab === 'Dashboard' && (
          <section className="dashboard">

            <h2>Welcome to PromptForge AI</h2>

            <p>
              Explore prompt engineering, experiment with language
              models, and evaluate model responses.
            </p>

            <div className="response-panel">
              <h3>Backend Connection</h3>

              <button
                className="run-button"
                onClick={checkBackend}
                disabled={checkingBackend}
              >
                {checkingBackend
                  ? 'Checking...'
                  : 'Test Backend Connection'}
              </button>

              <p>{backendMessage}</p>
            </div>

            <div className="stats-grid">

              <div className="stat-card">
                <h3>50</h3>
                <p>Benchmark Questions</p>
              </div>

              <div className="stat-card">
                <h3>6</h3>
                <p>Prompt Variants</p>
              </div>

              <div className="stat-card">
                <h3>Qwen 2.5</h3>
                <p>Selected Model</p>
              </div>

              <div className="stat-card">
                <h3>Ready</h3>
                <p>Frontend Status</p>
              </div>

            </div>

            <div className="module-grid">

              {menuItems.slice(1).map((item) => (
                <button
                  className="module-card"
                  key={item}
                  onClick={() => setActiveTab(item)}
                >
                  <h3>{item}</h3>
                  <p>Open {item} module</p>
                </button>
              ))}

            </div>
          </section>
        )}

        {/* Prompt Experiments */}
        {activeTab === 'Prompt Experiments' && (
          <section className="experiment-section">

            <h2>Run a Prompt Experiment</h2>

            <p>
              Enter a prompt, choose a language model and select
              a prompt strategy.
            </p>

            <label htmlFor="prompt">
              Your Prompt
            </label>

            <textarea
              id="prompt"
              className="prompt-input"
              placeholder="Example: Explain artificial intelligence in simple words..."
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={7}
              disabled={generating || comparing}
            />

            <div className="experiment-options">

              <div>
                <label htmlFor="model">
                  Select Model
                </label>

                <select
                  id="model"
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  disabled={generating || comparing}
                >
                  <option value="qwen2.5:3b">
                    Qwen 2.5:3b
                  </option>

                  <option value="llama3.2:1b">
                    Llama 3.2:1b
                  </option>
                </select>
              </div>

              <div>
                <label htmlFor="strategy">
                  Prompt Strategy
                </label>

                <select
                  id="strategy"
                  value={strategy}
                  onChange={(event) => setStrategy(event.target.value)}
                  disabled={generating || comparing}
                >
                  {strategies.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </div>

            </div>

            {/* Experiment Buttons */}
            <div style={{
              display: 'flex',
              gap: '12px',
              flexWrap: 'wrap',
              marginTop: '20px'
            }}>
              <button
                className="run-button"
                onClick={runExperiment}
                disabled={generating || comparing}
              >
                {generating ? 'Generating...' : 'Run Single Experiment'}
              </button>

              <button
                className="run-button"
                onClick={runComparison}
                disabled={generating || comparing}
              >
                {comparing ? 'Comparing Strategies...' : 'Compare All 6 Strategies'}
              </button>
            </div>

            {/* Experiment Status */}
            {status && (
              <div className="response-panel">
                <h3>Experiment Status</h3>
                <p>{status}</p>
              </div>
            )}

            {/* Single AI Response */}
            {!comparing && generatedResponse && (
              <div className="response-panel">
                <h3>Model Response - {strategy}</h3>

                <p style={{ whiteSpace: 'pre-wrap' }}>
                  {generatedResponse}
                </p>
              </div>
            )}

            {/* Comparison Results */}
            {comparisonResults.length > 0 && (
              <section style={{ marginTop: '30px' }}>
                <h2>Prompt Strategy Comparison Results</h2>

                <p>
                  Completed {comparisonResults.length} of 6 strategies.
                </p>

                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
                  gap: '18px',
                  marginTop: '20px'
                }}>
                  {comparisonResults.map((item) => (
                    <div
                      className="response-panel"
                      key={item.strategy}
                      style={{ minWidth: 0 }}
                    >
                      <h3>{item.strategy}</h3>

                      <p>
                        <strong>Generation time:</strong>{' '}
                        {item.duration.toFixed(2)} seconds
                      </p>

                      <p style={{
                        whiteSpace: 'pre-wrap',
                        overflowWrap: 'anywhere'
                      }}>
                        {item.response}
                      </p>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Initial Response Placeholder */}
            {!generatedResponse && comparisonResults.length === 0 && !generating && (
              <div className="response-panel">
                <h3>Model Response</h3>
                <p>Your AI-generated response will appear here.</p>
              </div>
            )}

          </section>
        )}

        {/* Other Modules */}
        {activeTab !== 'Dashboard' &&
          activeTab !== 'Prompt Experiments' && (
            <section className="placeholder-section">

              <h2>{activeTab}</h2>

              <p>
                This module will be implemented in the next
                development stage.
              </p>

            </section>
          )}

      </main>
    </div>
  )
}

export default App
