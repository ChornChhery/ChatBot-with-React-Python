import { useState, useRef, useEffect, useCallback } from 'react'
import axios from 'axios'
import './Chat.css'

const API = 'http://localhost:8000'
const WS = 'ws://localhost:8000'
const FILTER_KEY = 'ragpy_chat_doc_filter'

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [docFilterName, setDocFilterName] = useState('')
  const wsRef = useRef(null)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  // FIX: Read docFilter fresh from localStorage each time sendMessage is called
  // instead of capturing it once at mount — so filter changes on Rag page take effect
  // without needing a full page reload
  const getDocFilter = () => localStorage.getItem(FILTER_KEY) || ''

  useEffect(() => {
    const filter = getDocFilter()
    if (filter) {
      axios.get(`${API}/api/documents`)
        .then(r => {
          const doc = r.data.find(d => d.id === filter)
          setDocFilterName(doc?.file_name || '')
        })
        .catch(() => {})
    }
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [input])

  // FIX: Close WebSocket on component unmount to prevent dangling connections
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const sendMessage = useCallback(() => {
    const question = input.trim()
    if (!question || isStreaming) return

    const docFilter = getDocFilter()

    const userMsg = { role: 'user', content: question, id: Date.now() }
    const assistantMsg = { role: 'assistant', content: '', id: Date.now() + 1, streaming: true }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setInput('')
    setIsStreaming(true)

    // FIX: Filter out empty-content messages from interrupted streams before
    // sending history to backend — empty assistant messages confuse the LLM
    const history = messages
      .filter(m => m.content && m.content.trim())
      .map(m => ({ role: m.role, content: m.content }))

    const ws = new WebSocket(`${WS}/api/chat/ws`)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify({
        question,
        history,
        document_id: docFilter || undefined
      }))
    }

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (!data.isFinal) {
        setMessages(prev => prev.map(m =>
          m.id === assistantMsg.id
            ? { ...m, content: m.content + data.token }
            : m
        ))
      } else {
        setMessages(prev => prev.map(m =>
          m.id === assistantMsg.id
            ? {
                ...m,
                streaming: false,
                sources: data.sources || [],
                // FIX: Only show sources when backend confirms RAG was actually used
                // (score above threshold). If ragUsed is false, LLM answered from
                // its own knowledge and there are no meaningful sources to show.
                ragUsed: data.ragUsed || false
              }
            : m
        ))
        setIsStreaming(false)
        ws.close()
      }
    }

    ws.onerror = () => {
      setMessages(prev => prev.map(m =>
        m.id === assistantMsg.id
          ? { ...m, content: 'Connection error. Make sure the backend is running.', streaming: false, error: true }
          : m
      ))
      setIsStreaming(false)
    }
  }, [input, isStreaming, messages])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => setMessages([])

  const docFilter = getDocFilter()

  return (
    <div className="chat-page">
      <div className="chat-header">
        <div className="chat-header-left">
          <h1>Chat</h1>
          <p>Ask questions about your documents</p>
        </div>
        <div className="chat-header-right">
          {messages.length > 0 && (
            <button className="btn-secondary" onClick={clearChat}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/>
              </svg>
              Clear
            </button>
          )}
        </div>
      </div>

      <div className="chat-body">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="empty-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            </div>
            <h3>Start a conversation</h3>
            <p>Upload documents on the Documents page, then ask anything about them.</p>
            <div className="empty-hints">
              {['What is this document about?', 'Summarize the key points', 'What are the main topics?'].map(hint => (
                <button key={hint} className="hint-chip" onClick={() => setInput(hint)}>
                  {hint}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages">
            {messages.map(msg => (
              <div key={msg.id} className={`message-row ${msg.role === 'user' ? 'row-user' : 'row-assistant'}`}>
                {msg.role === 'assistant' && (
                  <div className="avatar avatar-ai">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/>
                    </svg>
                  </div>
                )}

                <div className={`bubble ${msg.role === 'user' ? 'bubble-user' : 'bubble-ai'} ${msg.error ? 'bubble-error' : ''}`}>
                  <div className="bubble-text">
                    {msg.content || (msg.streaming && <span className="typing-cursor">▋</span>)}
                    {msg.streaming && msg.content && <span className="typing-cursor">▋</span>}
                  </div>

                  {/* FIX: Only show sources when ragUsed=true (backend found relevant chunks
                      above the score threshold). Never show when LLM answered on its own. */}
                  {msg.role === 'assistant' && !msg.streaming && msg.ragUsed && msg.sources?.length > 0 && (
                    <div className="bubble-sources">
                      <div className="sources-label">
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                          <polyline points="14 2 14 8 20 8"/>
                        </svg>
                        Sources
                      </div>
                      <div className="sources-chips">
                        {msg.sources.map((s, i) => (
                          <div key={i} className="source-item">
                            <span className="source-score">{(s.score * 100).toFixed(0)}%</span>
                            <span className="source-preview">{s.content}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {msg.role === 'user' && (
                  <div className="avatar avatar-user">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/>
                    </svg>
                  </div>
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="chat-input-area">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question... (Enter to send, Shift+Enter for newline)"
            rows={1}
            disabled={isStreaming}
          />
          <button
            className="send-btn"
            onClick={sendMessage}
            disabled={!input.trim() || isStreaming}
          >
            {isStreaming ? (
              <span className="spinner" />
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            )}
          </button>
        </div>
        <p className="input-hint">
          {docFilter && docFilterName
            ? <>Searching in: <strong>{docFilterName}</strong></>
            : 'Searching across all documents'
          }
        </p>
      </div>
    </div>
  )
}