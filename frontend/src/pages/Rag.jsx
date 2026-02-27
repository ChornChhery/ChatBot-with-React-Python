import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './Rag.css'

const API = 'http://localhost:8000'
const FILTER_KEY = 'ragpy_chat_doc_filter'

const STRATEGIES = [
  { value: 0, label: 'Fixed Size', desc: 'Fast, 500-char chunks with overlap. Best for most documents.' },
  { value: 1, label: 'Content Aware', desc: 'Respects paragraphs & headings. Best for markdown/structured docs.' },
  { value: 2, label: 'Semantic', desc: 'Groups by topic coherence. Best for research papers & dense content.' },
]

function StatusBadge({ status }) {
  const map = {
    Ready: { cls: 'badge-ready', icon: '✓', label: 'Ready' },
    Processing: { cls: 'badge-processing', icon: '⟳', label: 'Processing' },
    Uploading: { cls: 'badge-uploading', icon: '↑', label: 'Uploading' },
    Failed: { cls: 'badge-failed', icon: '✕', label: 'Failed' },
  }
  const s = map[status] || map.Failed
  return <span className={`badge ${s.cls}`}>{s.icon} {s.label}</span>
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function Rag() {
  const [documents, setDocuments] = useState([])
  const [strategy, setStrategy] = useState(0)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [cacheStats, setCacheStats] = useState(null)
  const [error, setError] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [selectedDoc, setSelectedDoc] = useState('')
  const [savedDoc, setSavedDoc] = useState('')
  const [saveSuccess, setSaveSuccess] = useState(false)
  const fileRef = useRef(null)
  const pollRef = useRef(null)

  const fetchDocs = () => {
    axios.get(`${API}/api/documents`)
      .then(r => setDocuments(r.data))
      .catch(() => setError('Cannot connect to backend. Make sure it is running.'))
  }

  const fetchCache = () => {
    axios.get(`${API}/api/documents/cache-stats`)
      .then(r => setCacheStats(r.data))
      .catch(() => {})
  }

  const startPolling = () => {
    if (pollRef.current) return
    pollRef.current = setInterval(() => { fetchDocs(); fetchCache() }, 3000)
  }

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  useEffect(() => {
    fetchDocs()
    fetchCache()
    startPolling()

    const saved = localStorage.getItem(FILTER_KEY) || ''
    setSelectedDoc(saved)
    setSavedDoc(saved)

    // FIX: Pause polling when tab is hidden, resume when visible again.
    // Prevents unnecessary API calls while user is on another tab.
    const handleVisibilityChange = () => {
      if (document.hidden) {
        stopPolling()
      } else {
        fetchDocs()
        fetchCache()
        startPolling()
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      stopPolling()
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [])

  const handleFileSelect = (file) => {
    if (!file) return
    setSelectedFile(file)
    setError('')
    if (fileRef.current) fileRef.current.value = ''
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFileSelect(e.dataTransfer.files[0])
  }

  const cancelSelect = () => { setSelectedFile(null); setError('') }

  const confirmUpload = async () => {
    if (!selectedFile) return
    setUploading(true)
    setError('')
    const form = new FormData()
    form.append('file', selectedFile)
    try {
      await axios.post(`${API}/api/documents/upload?strategy=${strategy}`, form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setSelectedFile(null)
      fetchDocs()
    } catch (e) {
      setError(e.response?.data?.detail || 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  const saveFilter = () => {
    localStorage.setItem(FILTER_KEY, selectedDoc)
    setSavedDoc(selectedDoc)
    setSaveSuccess(true)
    setTimeout(() => setSaveSuccess(false), 2000)
  }

  const deleteDoc = async (id) => {
    if (!window.confirm('Delete this document and all its chunks?')) return
    if (savedDoc === id) {
      localStorage.setItem(FILTER_KEY, '')
      setSavedDoc('')
      setSelectedDoc('')
    }
    await axios.delete(`${API}/api/documents/${id}`)
    fetchDocs()
    fetchCache()
  }

  const readyCount = documents.filter(d => d.status === 'Ready').length
  const processingCount = documents.filter(d => d.status === 'Processing' || d.status === 'Uploading').length
  const hasUnsavedChange = selectedDoc !== savedDoc
  const savedDocName = documents.find(d => d.id === savedDoc)?.file_name

  return (
    <div className="rag-page">
      <div className="rag-header">
        <div>
          <h1>Documents</h1>
          <p>Upload and manage your knowledge base</p>
        </div>
        <div className="rag-stats">
          <div className="stat-pill">
            <span className="stat-num">{readyCount}</span>
            <span className="stat-label">Ready</span>
          </div>
          {processingCount > 0 && (
            <div className="stat-pill stat-processing">
              <span className="stat-num">{processingCount}</span>
              <span className="stat-label">Processing</span>
            </div>
          )}
          {cacheStats && (
            <div className="stat-pill">
              <span className="stat-num">{cacheStats.totalChunks}</span>
              <span className="stat-label">Chunks in RAM</span>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="error-banner">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          {error}
        </div>
      )}

      <div className="rag-content">

        {/* Upload section */}
        <div className="upload-section">
          <h2 className="section-title">Upload Document</h2>
          <div className="strategy-grid">
            {STRATEGIES.map(s => (
              <button
                key={s.value}
                className={`strategy-card ${strategy === s.value ? 'active' : ''}`}
                onClick={() => setStrategy(s.value)}
                disabled={!!selectedFile || uploading}
              >
                <div className="strategy-label">{s.label}</div>
                <div className="strategy-desc">{s.desc}</div>
              </button>
            ))}
          </div>

          {!selectedFile && !uploading && (
            <div
              className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
              onClick={() => fileRef.current.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".pdf,.txt,.md"
                style={{ display: 'none' }}
                onChange={e => handleFileSelect(e.target.files[0])}
              />
              <div className="drop-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              </div>
              <p className="drop-main">Drop a file here or <span className="drop-link">browse</span></p>
              <p className="drop-sub">Supports PDF, TXT, MD</p>
            </div>
          )}

          {selectedFile && !uploading && (
            <div className="confirm-panel">
              <div className="confirm-file-info">
                <div className="confirm-file-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                  </svg>
                </div>
                <div className="confirm-file-details">
                  <div className="confirm-file-name">{selectedFile.name}</div>
                  <div className="confirm-file-meta">
                    {formatFileSize(selectedFile.size)}
                    <span className="confirm-dot">·</span>
                    {STRATEGIES[strategy].label} chunking
                  </div>
                </div>
              </div>
              <div className="confirm-actions">
                <button className="btn-secondary" onClick={cancelSelect}>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                  Cancel
                </button>
                <button className="btn-primary" onClick={confirmUpload}>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  Upload & Embed
                </button>
              </div>
            </div>
          )}

          {uploading && (
            <div className="drop-zone uploading">
              <div className="upload-loading">
                <span className="spinner-lg" />
                <span>Uploading & starting embedding...</span>
              </div>
            </div>
          )}
        </div>

        {/* Chat Document Filter */}
        <div className="doc-filter-section">
          <h2 className="section-title">Chat Document Filter</h2>
          <p className="section-desc">
            Choose which document the Chat page will use when answering questions.
            Select <strong>All Documents</strong> to let the AI search across everything,
            or pick a specific document to limit responses to that source only.
          </p>

          <div className="doc-filter-row">
            <select
              className="doc-filter-select"
              value={selectedDoc}
              onChange={e => setSelectedDoc(e.target.value)}
            >
              <option value="">All Documents</option>
              {documents.filter(d => d.status === 'Ready').map(d => (
                <option key={d.id} value={d.id}>{d.file_name}</option>
              ))}
            </select>

            <button
              className={`btn-save ${saveSuccess ? 'btn-save-success' : ''}`}
              onClick={saveFilter}
              disabled={!hasUnsavedChange && !saveSuccess}
            >
              {saveSuccess ? (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polyline points="20 6 9 17 4 12"/>
                  </svg>
                  Saved!
                </>
              ) : (
                <>
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>
                    <polyline points="17 21 17 13 7 13 7 21"/>
                    <polyline points="7 3 7 8 15 8"/>
                  </svg>
                  Save
                </>
              )}
            </button>
          </div>

          <div className="doc-filter-status">
            <span className="filter-status-dot" />
            {savedDoc && savedDocName
              ? <>Chat is using: <strong>{savedDocName}</strong></>
              : <>Chat is using: <strong>All Documents</strong></>
            }
            {hasUnsavedChange && (
              <span className="filter-unsaved"> · unsaved change</span>
            )}
          </div>
        </div>

        {/* Documents list */}
        <div className="docs-section">
          <h2 className="section-title">
            Documents
            <span className="section-count">{documents.length}</span>
          </h2>

          {documents.length === 0 ? (
            <div className="docs-empty">
              <p>No documents yet. Upload one to get started.</p>
            </div>
          ) : (
            <div className="docs-list">
              {documents.map(doc => (
                <div key={doc.id} className={`doc-card ${savedDoc === doc.id ? 'doc-card-active' : ''}`}>
                  <div className="doc-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                      <polyline points="14 2 14 8 20 8"/>
                    </svg>
                  </div>
                  <div className="doc-info">
                    <div className="doc-name">
                      {doc.file_name}
                      {savedDoc === doc.id && <span className="doc-active-tag">Chat source</span>}
                    </div>
                    <div className="doc-meta">
                      <StatusBadge status={doc.status} />
                      {doc.chunk_count > 0 && (
                        <span className="doc-chunks">{doc.chunk_count} chunks</span>
                      )}
                      <span className="doc-date">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  <button className="btn-danger" onClick={() => deleteDoc(doc.id)}>
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/>
                    </svg>
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Cache stats */}
        {cacheStats && (
          <div className="cache-card">
            <h2 className="section-title">Cache Stats</h2>
            <div className="cache-grid">
              <div className="cache-stat">
                <span className="cache-num">{cacheStats.totalChunks}</span>
                <span className="cache-label">Total Chunks</span>
              </div>
              <div className="cache-stat">
                <span className="cache-num">{cacheStats.totalDocuments}</span>
                <span className="cache-label">Documents</span>
              </div>
              <div className="cache-stat">
                <span className="cache-num">{cacheStats.estimatedMemoryMb} MB</span>
                <span className="cache-label">RAM Used</span>
              </div>
              <div className="cache-stat">
                <span className={`cache-num ${cacheStats.isLoaded ? 'text-accent' : 'text-warning'}`}>
                  {cacheStats.isLoaded ? '✓ Loaded' : '○ Empty'}
                </span>
                <span className="cache-label">Status</span>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}