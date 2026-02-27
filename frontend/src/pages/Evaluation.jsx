import { useState, useEffect } from 'react'
import axios from 'axios'
import './Evaluation.css'

const API = 'http://localhost:8000'
const HISTORY_KEY = 'ragpy_eval_history'

function ScoreBar({ label, value, color }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div className="score-bar-item">
      <div className="score-bar-header">
        <span className="score-bar-label">{label}</span>
        <span className="score-bar-value" style={{ color }}>{pct}%</span>
      </div>
      <div className="score-bar-track">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}

function ScoreInterpret({ score }) {
  if (score > 0.7) return <span className="interp excellent">✓ Excellent</span>
  if (score > 0.4) return <span className="interp good">◎ Good</span>
  if (score > 0.2) return <span className="interp fair">⚠ Fair</span>
  return <span className="interp poor">✕ Poor</span>
}

function ScoreBadge({ score }) {
  const pct = Math.round((score || 0) * 100)
  const color = score > 0.7 ? 'var(--success)' : score > 0.4 ? 'var(--info)' : score > 0.2 ? 'var(--warning)' : 'var(--error)'
  return <span className="score-badge" style={{ color, borderColor: color }}>{pct}%</span>
}

export default function Evaluation() {
  const [documents, setDocuments] = useState([])
  const [question, setQuestion] = useState('')
  const [docId, setDocId] = useState('')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [step, setStep] = useState(0)
  const [history, setHistory] = useState([])
  const [selectedHistory, setSelectedHistory] = useState(null)
  const [activeTab, setActiveTab] = useState('run') // 'run' | 'history'

  const steps = [
    'Searching relevant chunks...',
    'Generating reference answer...',
    'Generating RAG answer...',
    'Computing NLP scores...',
    'Running LLM Judge...',
  ]

  useEffect(() => {
    axios.get(`${API}/api/documents`)
      .then(r => setDocuments(r.data.filter(d => d.status === 'Ready')))
      .catch(() => {})

    // Load history from localStorage
    try {
      const saved = localStorage.getItem(HISTORY_KEY)
      if (saved) setHistory(JSON.parse(saved))
    } catch (e) {
      console.error('Failed to load history:', e)
    }
  }, [])

  const saveToHistory = (res, q, docName) => {
    const entry = {
      id: Date.now(),
      question: q,
      document: docName || 'All Documents',
      overall_score: res.overall_score,
      bleu_score: res.bleu_score,
      gleu_score: res.gleu_score,
      f1_score: res.f1_score,
      llm_judge_score: res.llm_judge_score,
      llm_judge_explanation: res.llm_judge_explanation,
      auto_generated_reference: res.auto_generated_reference,
      generated_answer: res.generated_answer,
      source_documents: res.source_documents,
      timestamp: new Date().toISOString(),
    }
    const updated = [entry, ...history].slice(0, 50) // keep last 50
    setHistory(updated)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
    return entry
  }

  const clearHistory = () => {
    if (!confirm('Clear all evaluation history?')) return
    setHistory([])
    setSelectedHistory(null)
    localStorage.removeItem(HISTORY_KEY)
  }

  const deleteHistoryItem = (id, e) => {
    e.stopPropagation()
    const updated = history.filter(h => h.id !== id)
    setHistory(updated)
    if (selectedHistory?.id === id) setSelectedHistory(null)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
  }

  const runEvaluation = async () => {
    if (!question.trim()) return
    setLoading(true)
    setResult(null)
    setError('')
    setStep(0)

    const interval = setInterval(() => {
      setStep(s => (s < steps.length - 1 ? s + 1 : s))
    }, 2500)

    try {
      const res = await axios.post(`${API}/api/evaluation`, {
        question: question.trim(),
        document_id: docId || null,
        top_k: topK
      })
      setResult(res.data)
      const docName = documents.find(d => d.id === docId)?.file_name
      saveToHistory(res.data, question.trim(), docName)
    } catch (e) {
      setError(e.response?.data?.detail || 'Evaluation failed.')
    } finally {
      clearInterval(interval)
      setLoading(false)
    }
  }

  const displayResult = selectedHistory || result
  const overall = displayResult?.overall_score || 0

  return (
    <div className="eval-page">
      <div className="eval-header">
        <div>
          <h1>Evaluation</h1>
          <p>Test your RAG pipeline quality with BLEU, GLEU, F1 and LLM Judge</p>
        </div>
      </div>

      <div className="eval-layout">
        {/* Left panel */}
        <div className="eval-left">
          {/* Tabs */}
          <div className="eval-tabs">
            <button
              className={`eval-tab ${activeTab === 'run' ? 'active' : ''}`}
              onClick={() => { setActiveTab('run'); setSelectedHistory(null) }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              Run
            </button>
            <button
              className={`eval-tab ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
              </svg>
              History
              {history.length > 0 && <span className="tab-badge">{history.length}</span>}
            </button>
          </div>

          {/* Run tab */}
          {activeTab === 'run' && (
            <div className="eval-form">
              <div className="form-group">
                <label>Question</label>
                <textarea
                  className="eval-textarea"
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  placeholder="Ask a question answerable from your documents..."
                  rows={3}
                  disabled={loading}
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Document (optional)</label>
                  <select
                    className="eval-select"
                    value={docId}
                    onChange={e => setDocId(e.target.value)}
                    disabled={loading}
                  >
                    <option value="">All Documents</option>
                    {documents.map(d => (
                      <option key={d.id} value={d.id}>{d.file_name}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group form-group-sm">
                  <label>Top K</label>
                  <input
                    type="number"
                    className="eval-input"
                    value={topK}
                    onChange={e => setTopK(parseInt(e.target.value) || 5)}
                    min={1} max={20}
                    disabled={loading}
                  />
                </div>
              </div>

              <button
                className="btn-primary eval-btn"
                onClick={runEvaluation}
                disabled={loading || !question.trim()}
              >
                {loading ? (
                  <><span className="spinner" />Evaluating...</>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                    </svg>
                    Run Evaluation
                  </>
                )}
              </button>

              {loading && (
                <div className="steps-progress">
                  {steps.map((s, i) => (
                    <div key={i} className={`step-item ${i < step ? 'done' : i === step ? 'active' : 'pending'}`}>
                      <span className="step-dot">
                        {i < step ? '✓' : i === step ? <span className="step-spinner" /> : '○'}
                      </span>
                      <span>{s}</span>
                    </div>
                  ))}
                </div>
              )}

              {error && (
                <div className="eval-error">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  {error}
                </div>
              )}
            </div>
          )}

          {/* History tab */}
          {activeTab === 'history' && (
            <div className="history-panel">
              {history.length === 0 ? (
                <div className="history-empty">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                  </svg>
                  <p>No evaluations yet</p>
                  <span>Run an evaluation to see history here</span>
                </div>
              ) : (
                <>
                  <div className="history-toolbar">
                    <span className="history-count">{history.length} evaluations</span>
                    <button className="btn-danger" onClick={clearHistory}>Clear All</button>
                  </div>
                  <div className="history-list">
                    {history.map(h => (
                      <div
                        key={h.id}
                        className={`history-item ${selectedHistory?.id === h.id ? 'selected' : ''}`}
                        onClick={() => { setSelectedHistory(h); setResult(null) }}
                      >
                        <div className="history-item-top">
                          <ScoreBadge score={h.overall_score} />
                          <span className="history-date">
                            {new Date(h.timestamp).toLocaleDateString()} {new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                          <button
                            className="history-delete"
                            onClick={(e) => deleteHistoryItem(h.id, e)}
                            title="Delete"
                          >✕</button>
                        </div>
                        <div className="history-question">{h.question}</div>
                        <div className="history-doc">
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                          </svg>
                          {h.document}
                        </div>
                        <div className="history-scores">
                          <span>BLEU {Math.round(h.bleu_score * 100)}%</span>
                          <span>GLEU {Math.round(h.gleu_score * 100)}%</span>
                          <span>F1 {Math.round(h.f1_score * 100)}%</span>
                          <span>Judge {Math.round(h.llm_judge_score * 100)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Right panel — results */}
        <div className="eval-right">
          {!displayResult ? (
            <div className="result-empty">
              <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>
              <p>Run an evaluation or select from history to see results</p>
            </div>
          ) : (
            <div className="eval-results">
              {/* Question shown in result */}
              <div className="result-question">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
                {displayResult.question}
              </div>

              {/* Overall score */}
              <div className="overall-card">
                <div className="overall-left">
                  <div className="overall-score" style={{
                    color: overall > 0.7 ? 'var(--success)' : overall > 0.4 ? 'var(--info)' : overall > 0.2 ? 'var(--warning)' : 'var(--error)'
                  }}>
                    {Math.round(overall * 100)}%
                  </div>
                  <div className="overall-label">Overall Score</div>
                  <ScoreInterpret score={overall} />
                </div>
                <div className="overall-right">
                  <ScoreBar label="BLEU" value={displayResult.bleu_score} color="#10a37f" />
                  <ScoreBar label="GLEU" value={displayResult.gleu_score} color="#1a7fa8" />
                  <ScoreBar label="F1" value={displayResult.f1_score} color="#7c6fcd" />
                  <ScoreBar label="LLM Judge" value={displayResult.llm_judge_score} color="#f5a623" />
                </div>
              </div>

              {/* LLM Judge explanation */}
              {displayResult.llm_judge_explanation && (
                <div className="judge-card">
                  <div className="judge-title">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                    LLM Judge Explanation
                  </div>
                  <p className="judge-text">{displayResult.llm_judge_explanation}</p>
                </div>
              )}

              {/* Answer comparison */}
              <div className="answers-grid">
                <div className="answer-card">
                  <div className="answer-title reference">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
                    </svg>
                    Reference Answer
                  </div>
                  <p className="answer-text">{displayResult.auto_generated_reference}</p>
                </div>
                <div className="answer-card">
                  <div className="answer-title rag">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/>
                    </svg>
                    RAG Answer
                  </div>
                  <p className="answer-text">{displayResult.generated_answer}</p>
                </div>
              </div>

              {/* Sources */}
              {displayResult.source_documents?.length > 0 && (
                <div className="sources-card">
                  <div className="sources-card-title">Source Documents</div>
                  <div className="source-docs">
                    {displayResult.source_documents.map((s, i) => (
                      <span key={i} className="source-doc-chip">{s}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}