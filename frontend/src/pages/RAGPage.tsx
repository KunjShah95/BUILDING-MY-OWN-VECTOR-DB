import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, Link, FileText, Trash2, Send, Loader2,
  CheckCircle2, XCircle, Bot, User, RefreshCw, ChevronDown, ChevronUp,
} from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────

interface Document {
  doc_id: string
  source: string
  chunk_count: number
  ingested_at: string
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  model?: string
}

interface Source {
  doc_id: string
  chunk_index: number
  score: number
  text: string
}

type IngestMode = 'url' | 'text' | 'file'
type Status = 'idle' | 'loading' | 'success' | 'error'

// ── API helpers ───────────────────────────────────────────────────────────────

// Routes match api/routers/rag.py: /collections/{collectionId}/...
const col = (id: string) => `/collections/${encodeURIComponent(id)}`

async function ingestURL(url: string, collectionId: string) {
  const r = await fetch(`${col(collectionId)}/ingest/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  })
  return r.json()
}

async function ingestText(text: string, sourceLabel: string, collectionId: string) {
  const r = await fetch(`${col(collectionId)}/ingest/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, source: sourceLabel }),
  })
  return r.json()
}

async function ingestFile(file: File, collectionId: string) {
  const form = new FormData()
  form.append('file', file)
  const r = await fetch(`${col(collectionId)}/ingest/file`, { method: 'POST', body: form })
  return r.json()
}

async function listDocs(collectionId: string): Promise<Document[]> {
  const r = await fetch(`${col(collectionId)}/documents`)
  const d = await r.json()
  return d.documents ?? []
}

async function deleteDoc(docId: string, collectionId: string) {
  await fetch(`${col(collectionId)}/documents?doc_id=${encodeURIComponent(docId)}`, {
    method: 'DELETE',
  })
}

async function* streamQuery(query: string, collectionId: string, model: string) {
  const r = await fetch(`${col(collectionId)}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, model }),
  })
  if (!r.body) return
  const reader = r.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const payload = line.slice(6).trim()
        if (payload === '[DONE]') return
        try {
          const obj = JSON.parse(payload)
          yield obj
        } catch { /* skip malformed */ }
      }
    }
  }
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusBadge({ status, msg }: { status: Status; msg?: string }) {
  if (status === 'idle') return null
  const map: Record<Exclude<Status, 'idle'>, { icon: JSX.Element; cls: string }> = {
    loading: { icon: <Loader2 className="w-4 h-4 animate-spin" />, cls: 'text-blue-400' },
    success: { icon: <CheckCircle2 className="w-4 h-4" />, cls: 'text-green-400' },
    error:   { icon: <XCircle className="w-4 h-4" />,   cls: 'text-red-400'   },
  }
  const { icon, cls } = map[status as Exclude<Status, 'idle'>]
  return (
    <span className={`flex items-center gap-1 text-sm ${cls}`}>
      {icon} {msg}
    </span>
  )
}

function SourceCard({ src, expanded, onToggle }: { src: Source; expanded: boolean; onToggle: () => void }) {
  return (
    <div className="border border-white/10 rounded-lg overflow-hidden text-xs">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2 bg-white/5 hover:bg-white/10 transition-colors"
      >
        <span className="text-white/60 font-mono truncate">{src.doc_id} · chunk {src.chunk_index}</span>
        <span className="flex items-center gap-1 text-white/40">
          {(src.score * 100).toFixed(0)}%
          {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        </span>
      </button>
      {expanded && (
        <div className="px-3 py-2 bg-black/20 text-white/70 leading-relaxed whitespace-pre-wrap">
          {src.text}
        </div>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function RAGPage() {
  const [collectionId, setCollectionId] = useState('rag')
  const [model, setModel] = useState('groq:llama3-8b-8192')
  const [ingestMode, setIngestMode] = useState<IngestMode>('url')
  const [urlInput, setUrlInput] = useState('')
  const [textInput, setTextInput] = useState('')
  const [textLabel, setTextLabel] = useState('')
  const [fileRef, setFileRef] = useState<File | null>(null)
  const [ingestStatus, setIngestStatus] = useState<Status>('idle')
  const [ingestMsg, setIngestMsg] = useState('')
  const [docs, setDocs] = useState<Document[]>([])
  const [docsLoading, setDocsLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({})
  const bottomRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const refreshDocs = useCallback(async () => {
    setDocsLoading(true)
    try { setDocs(await listDocs(collectionId)) } catch { /* ignore */ }
    setDocsLoading(false)
  }, [collectionId])

  useEffect(() => { refreshDocs() }, [refreshDocs])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleIngest() {
    setIngestStatus('loading')
    setIngestMsg('Ingesting…')
    try {
      let res: { success: boolean; message?: string; chunks?: number }
      if (ingestMode === 'url') {
        res = await ingestURL(urlInput.trim(), collectionId)
        if (res.success) setUrlInput('')
      } else if (ingestMode === 'text') {
        res = await ingestText(textInput, textLabel || 'manual', collectionId)
        if (res.success) { setTextInput(''); setTextLabel('') }
      } else {
        if (!fileRef) { setIngestStatus('error'); setIngestMsg('No file selected'); return }
        res = await ingestFile(fileRef, collectionId)
        if (res.success) { setFileRef(null); if (fileInputRef.current) fileInputRef.current.value = '' }
      }
      if (res.success) {
        setIngestStatus('success')
        setIngestMsg(`Ingested ${res.chunks ?? '?'} chunks`)
        refreshDocs()
      } else {
        setIngestStatus('error')
        setIngestMsg(res.message ?? 'Failed')
      }
    } catch (e: unknown) {
      setIngestStatus('error')
      setIngestMsg(e instanceof Error ? e.message : 'Network error')
    }
    setTimeout(() => setIngestStatus('idle'), 4000)
  }

  async function handleSend() {
    const q = input.trim()
    if (!q || streaming) return
    setInput('')
    const userMsg: Message = { role: 'user', content: q }
    setMessages(prev => [...prev, userMsg])
    setStreaming(true)
    const assistantMsg: Message = { role: 'assistant', content: '', model }
    setMessages(prev => [...prev, assistantMsg])
    try {
      for await (const chunk of streamQuery(q, collectionId, model)) {
        // Backend sends {token: "..."} or {type:"delta",text:"..."} or {type:"sources",sources:[...]}
        const text = chunk.token ?? (chunk.type === 'delta' ? chunk.text : null)
        if (text) {
          setMessages(prev => {
            const msgs = [...prev]
            msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: msgs[msgs.length - 1].content + text }
            return msgs
          })
        } else if (chunk.type === 'sources') {
          setMessages(prev => {
            const msgs = [...prev]
            msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], sources: chunk.sources }
            return msgs
          })
        }
      }
    } catch (e: unknown) {
      setMessages(prev => {
        const msgs = [...prev]
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: `[Error: ${e instanceof Error ? e.message : 'stream failed'}]` }
        return msgs
      })
    }
    setStreaming(false)
  }

  const modeTab = (m: IngestMode, label: string) => (
    <button
      onClick={() => setIngestMode(m)}
      className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${
        ingestMode === m ? 'bg-white/15 text-white' : 'text-white/50 hover:text-white/80'
      }`}
    >
      {label}
    </button>
  )

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white flex flex-col">
      {/* Header */}
      <div className="border-b border-white/10 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Bot className="w-6 h-6 text-violet-400" />
          <span className="font-semibold text-lg">RAG Chat</span>
        </div>
        <div className="flex items-center gap-3">
          <input
            value={collectionId}
            onChange={e => setCollectionId(e.target.value)}
            placeholder="collection"
            className="bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm w-32 focus:outline-none focus:border-violet-500"
          />
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-violet-500"
          >
            <optgroup label="🆓 Groq (free, fast)">
              <option value="groq:llama3-8b-8192">groq · llama3-8b</option>
              <option value="groq:llama3-70b-8192">groq · llama3-70b</option>
              <option value="groq:mixtral-8x7b-32768">groq · mixtral-8x7b</option>
              <option value="groq:gemma2-9b-it">groq · gemma2-9b</option>
            </optgroup>
            <optgroup label="🆓 Google Gemini (free)">
              <option value="gemini:gemini-1.5-flash">gemini-1.5-flash</option>
              <option value="gemini:gemini-1.5-flash-8b">gemini-1.5-flash-8b</option>
              <option value="gemini:gemini-2.0-flash-exp">gemini-2.0-flash-exp</option>
            </optgroup>
            <optgroup label="🆓 OpenRouter (free models)">
              <option value="openrouter:mistralai/mistral-7b-instruct:free">openrouter · mistral-7b (free)</option>
              <option value="openrouter:meta-llama/llama-3.2-3b-instruct:free">openrouter · llama-3.2-3b (free)</option>
              <option value="openrouter:google/gemma-2-9b-it:free">openrouter · gemma2-9b (free)</option>
              <option value="openrouter:qwen/qwen-2-7b-instruct:free">openrouter · qwen2-7b (free)</option>
            </optgroup>
            <optgroup label="🆓 Mistral AI (free tier)">
              <option value="mistral:mistral-small-latest">mistral-small</option>
              <option value="mistral:open-mistral-7b">open-mistral-7b</option>
              <option value="mistral:open-mixtral-8x7b">open-mixtral-8x7b</option>
            </optgroup>
            <optgroup label="🆓 NVIDIA NIM (free credits)">
              <option value="nvidia:meta/llama3-8b-instruct">nvidia · llama3-8b</option>
              <option value="nvidia:meta/llama3-70b-instruct">nvidia · llama3-70b</option>
              <option value="nvidia:mistralai/mistral-7b-instruct-v0.3">nvidia · mistral-7b</option>
            </optgroup>
            <optgroup label="🆓 Together AI (free credits)">
              <option value="together:meta-llama/Llama-3-8b-chat-hf">together · llama3-8b</option>
              <option value="together:mistralai/Mistral-7B-Instruct-v0.3">together · mistral-7b</option>
            </optgroup>
            <optgroup label="🆓 DeepSeek (cheap)">
              <option value="deepseek:deepseek-chat">deepseek-chat</option>
              <option value="deepseek:deepseek-coder">deepseek-coder</option>
            </optgroup>
            <optgroup label="🆓 HuggingFace Inference">
              <option value="hf:microsoft/Phi-3-mini-4k-instruct">hf · phi-3-mini</option>
              <option value="hf:HuggingFaceH4/zephyr-7b-beta">hf · zephyr-7b</option>
            </optgroup>
            <optgroup label="🏠 Ollama (local, unlimited)">
              <option value="ollama:llama3">ollama · llama3</option>
              <option value="ollama:mistral">ollama · mistral</option>
              <option value="ollama:phi3">ollama · phi3</option>
              <option value="ollama:gemma2">ollama · gemma2</option>
            </optgroup>
            <optgroup label="💳 Anthropic (paid)">
              <option value="claude-haiku-4-5-20251001">claude-haiku-4-5</option>
              <option value="claude-sonnet-4-6">claude-sonnet-4-6</option>
            </optgroup>
            <optgroup label="💳 OpenAI (paid)">
              <option value="gpt-4o-mini">gpt-4o-mini</option>
              <option value="gpt-4o">gpt-4o</option>
            </optgroup>
          </select>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar — ingest + docs */}
        <aside className="w-80 border-r border-white/10 flex flex-col overflow-hidden shrink-0">
          {/* Ingest panel */}
          <div className="p-4 border-b border-white/10">
            <div className="flex items-center gap-1 mb-3">
              {modeTab('url', 'URL')}
              {modeTab('text', 'Text')}
              {modeTab('file', 'File')}
            </div>

            {ingestMode === 'url' && (
              <div className="space-y-2">
                <input
                  value={urlInput}
                  onChange={e => setUrlInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleIngest()}
                  placeholder="https://example.com/article"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-violet-500"
                />
                <button
                  onClick={handleIngest}
                  disabled={!urlInput.trim() || ingestStatus === 'loading'}
                  className="w-full flex items-center justify-center gap-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 rounded-lg py-2 text-sm font-medium transition-colors"
                >
                  <Link className="w-4 h-4" /> Ingest URL
                </button>
              </div>
            )}

            {ingestMode === 'text' && (
              <div className="space-y-2">
                <input
                  value={textLabel}
                  onChange={e => setTextLabel(e.target.value)}
                  placeholder="Source label (optional)"
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-violet-500"
                />
                <textarea
                  value={textInput}
                  onChange={e => setTextInput(e.target.value)}
                  placeholder="Paste text to index…"
                  rows={5}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-violet-500 resize-none"
                />
                <button
                  onClick={handleIngest}
                  disabled={!textInput.trim() || ingestStatus === 'loading'}
                  className="w-full flex items-center justify-center gap-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 rounded-lg py-2 text-sm font-medium transition-colors"
                >
                  <FileText className="w-4 h-4" /> Ingest Text
                </button>
              </div>
            )}

            {ingestMode === 'file' && (
              <div className="space-y-2">
                <label
                  className="flex flex-col items-center justify-center gap-2 border-2 border-dashed border-white/20 rounded-lg py-6 cursor-pointer hover:border-violet-500 transition-colors"
                  onDragOver={e => e.preventDefault()}
                  onDrop={e => {
                    e.preventDefault()
                    const f = e.dataTransfer.files[0]
                    if (f) setFileRef(f)
                  }}
                >
                  <Upload className="w-6 h-6 text-white/40" />
                  <span className="text-sm text-white/50">
                    {fileRef ? fileRef.name : 'Drop file or click — TXT, DOCX, CSV, PDF'}
                  </span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".txt,.md,.docx,.csv,.pdf"
                    className="hidden"
                    onChange={e => setFileRef(e.target.files?.[0] ?? null)}
                  />
                </label>
                <button
                  onClick={handleIngest}
                  disabled={!fileRef || ingestStatus === 'loading'}
                  className="w-full flex items-center justify-center gap-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 rounded-lg py-2 text-sm font-medium transition-colors"
                >
                  <Upload className="w-4 h-4" /> Upload & Index
                </button>
              </div>
            )}

            <div className="mt-2 min-h-[20px]">
              <StatusBadge status={ingestStatus} msg={ingestMsg} />
            </div>
          </div>

          {/* Document list */}
          <div className="flex-1 overflow-y-auto p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-white/50 uppercase tracking-wide">Documents ({docs.length})</span>
              <button onClick={refreshDocs} className="text-white/40 hover:text-white transition-colors">
                <RefreshCw className={`w-3.5 h-3.5 ${docsLoading ? 'animate-spin' : ''}`} />
              </button>
            </div>
            <AnimatePresence>
              {docs.map(doc => (
                <motion.div
                  key={doc.doc_id}
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, height: 0 }}
                  className="flex items-start justify-between gap-2 py-2 border-b border-white/5 last:border-0"
                >
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-white/80 truncate font-mono">{doc.doc_id}</p>
                    <p className="text-xs text-white/40 truncate">{doc.source}</p>
                    <p className="text-xs text-white/30">{doc.chunk_count} chunks</p>
                  </div>
                  <button
                    onClick={async () => {
                      await deleteDoc(doc.doc_id, collectionId)
                      refreshDocs()
                    }}
                    className="text-white/30 hover:text-red-400 transition-colors mt-1 shrink-0"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </motion.div>
              ))}
            </AnimatePresence>
            {docs.length === 0 && !docsLoading && (
              <p className="text-center text-white/30 text-sm py-8">No documents yet</p>
            )}
          </div>
        </aside>

        {/* Chat area */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-white/30">
                <Bot className="w-12 h-12 mb-4 text-violet-500/40" />
                <p className="text-lg">Ask anything about your documents</p>
                <p className="text-sm mt-1">Ingest URLs, text, or files on the left, then chat here.</p>
              </div>
            )}
            <AnimatePresence initial={false}>
              {messages.map((msg, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                    msg.role === 'user' ? 'bg-violet-600' : 'bg-white/10'
                  }`}>
                    {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                  </div>
                  <div className={`max-w-[75%] space-y-2 ${msg.role === 'user' ? 'items-end flex flex-col' : ''}`}>
                    <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                      msg.role === 'user'
                        ? 'bg-violet-600 text-white rounded-tr-sm'
                        : 'bg-white/8 text-white/90 rounded-tl-sm'
                    }`}>
                      {msg.content || <Loader2 className="w-4 h-4 animate-spin text-white/40" />}
                    </div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="space-y-1 w-full">
                        <p className="text-xs text-white/30 px-1">Sources</p>
                        {msg.sources.map((src, si) => {
                          const key = `${i}-${si}`
                          return (
                            <SourceCard
                              key={key}
                              src={src}
                              expanded={!!expandedSources[key]}
                              onToggle={() => setExpandedSources(p => ({ ...p, [key]: !p[key] }))}
                            />
                          )
                        })}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={bottomRef} />
          </div>

          {/* Input bar */}
          <div className="border-t border-white/10 px-6 py-4">
            <div className="flex gap-3 items-end">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSend()
                  }
                }}
                placeholder="Ask about your documents… (Enter to send, Shift+Enter for newline)"
                rows={1}
                className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-violet-500 resize-none leading-relaxed"
                style={{ minHeight: '48px', maxHeight: '160px' }}
                onInput={e => {
                  const t = e.currentTarget
                  t.style.height = 'auto'
                  t.style.height = `${Math.min(t.scrollHeight, 160)}px`
                }}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || streaming}
                className="w-12 h-12 flex items-center justify-center bg-violet-600 hover:bg-violet-500 disabled:opacity-40 rounded-xl transition-colors shrink-0"
              >
                {streaming ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
