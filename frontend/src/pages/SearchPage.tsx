import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search, Globe, ArrowLeft, Loader2, Sparkles,
  ExternalLink, Zap, AlignLeft, Download, BarChart3,
  CheckCircle2, XCircle, ChevronLeft, ChevronRight,
} from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────

interface ResultMeta { url: string; title: string; snippet: string; source?: string }
interface SearchResult { vector_id: string; score: number; metadata: ResultMeta }
interface SearchResponse {
  success: boolean; query: string; expanded_query: string; cached?: boolean
  page: number; page_size: number; total_fetched: number; has_next: boolean
  results: SearchResult[]
}
interface CrawlStats { tracked: number; total_changes: number; avg_interval_sec: number }
interface IndexStats {
  collection_id: string; freshness: CrawlStats
  cache_entries: number; cache_ttl_seconds: number; recent_queries: number
}
interface JobStatus {
  status: 'running' | 'done' | 'failed'; pages: number; ingested: number
  last_url: string | null; error?: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const HINTS = [
  'HNSW index tuning', 'vector similarity search',
  'BM25 vs dense retrieval', 'RAG pipeline setup', 'product quantization',
]

function domain(url: string): string {
  try { return new URL(url).hostname.replace('www.', '') } catch { return url }
}
function favicon(url: string): string {
  try { return `https://www.google.com/s2/favicons?domain=${new URL(url).hostname}&sz=16` }
  catch { return '' }
}

// Bold query-matching terms in text.
function Highlight({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return <>{text}</>
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2)
  if (!terms.length) return <>{text}</>
  const pattern = new RegExp(`(${terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi')
  const parts = text.split(pattern)
  return (
    <>
      {parts.map((part, i) =>
        terms.some(t => t.toLowerCase() === part.toLowerCase())
          ? <mark key={i} className="bg-cyan-500/20 text-cyan-300 rounded px-0.5 not-italic">{part}</mark>
          : <span key={i}>{part}</span>
      )}
    </>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function AnswerCard({ results, query }: { results: SearchResult[]; query: string }) {
  const top = results.slice(0, 3).filter(r => r.metadata.snippet)
  if (!top.length) return null
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border border-cyan-500/25 bg-gradient-to-b from-cyan-950/30 to-transparent p-5 mb-8 relative overflow-hidden"
    >
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/40 to-transparent" />
      <div className="flex items-center gap-2 text-cyan-400 text-xs font-medium mb-3">
        <Sparkles className="size-3.5" /> AI Answer
      </div>
      <p className="text-sm text-gray-200 leading-relaxed">
        {top.map((r, i) => (
          <span key={r.vector_id}>
            <Highlight text={r.metadata.snippet.trim()} query={query} />
            <sup
              className="ml-0.5 text-cyan-400 hover:text-cyan-300 cursor-pointer"
              onClick={() => window.open(r.metadata.url, '_blank')}
            >[{i + 1}]</sup>
            {i < top.length - 1 ? ' ' : ''}
          </span>
        ))}
      </p>
      <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-white/5">
        {top.map((r, i) => (
          <a key={r.vector_id} href={r.metadata.url} target="_blank" rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-2.5 py-1 bg-white/[0.04] border border-white/8 rounded-full text-xs text-gray-400 hover:text-cyan-400 hover:border-cyan-500/30 transition-colors"
          >
            <span className="w-4 h-4 rounded-full bg-cyan-500/20 text-[9px] flex items-center justify-center font-bold text-cyan-300 flex-shrink-0">{i + 1}</span>
            {domain(r.metadata.url)}
          </a>
        ))}
      </div>
    </motion.div>
  )
}

function ResultCard({ result, idx, query, onClickResult }: {
  result: SearchResult; idx: number; query: string
  onClickResult?: (url: string, pos: number) => void
}) {
  const meta = result.metadata
  const fav = favicon(meta.url)
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay: idx * 0.04 }} className="mb-6 group"
    >
      <div className="flex items-center gap-1.5 mb-1">
        {fav
          ? <img src={fav} alt="" className="size-3.5 rounded-sm opacity-70" onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
          : <Globe className="size-3.5 text-gray-600" />}
        <span className="text-xs text-gray-500">{domain(meta.url)}</span>
        <span className="text-gray-700 text-xs">›</span>
        <span className="text-xs text-gray-600 truncate max-w-xs">{meta.url}</span>
      </div>
      <a
        href={meta.url} target="_blank" rel="noopener noreferrer"
        className="flex items-start gap-1 mb-1 w-fit"
        onClick={() => onClickResult?.(meta.url, idx)}
      >
        <h3 className="text-[15px] font-medium text-cyan-400 group-hover:underline leading-tight">
          <Highlight text={meta.title || domain(meta.url)} query={query} />
        </h3>
        <ExternalLink className="size-3 text-gray-600 group-hover:text-cyan-500 mt-[3px] flex-shrink-0 transition-colors" />
      </a>
      <p className="text-sm text-gray-400 leading-relaxed line-clamp-3">
        <Highlight text={meta.snippet} query={query} />
      </p>
      <span className="text-[10px] text-gray-700 mt-1 inline-block">relevance {result.score.toFixed(4)}</span>
    </motion.div>
  )
}

function CrawlPanel() {
  const [seeds, setSeeds] = useState('')
  const [collectionId, setCollectionId] = useState('web')
  const [maxPages, setMaxPages] = useState(50)
  const [maxDepth, setMaxDepth] = useState(2)
  const [sameDomain, setSameDomain] = useState(true)
  const [crawling, setCrawling] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<JobStatus | null>(null)
  const [stats, setStats] = useState<IndexStats | null>(null)
  const [loadingStats, setLoadingStats] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchStats = async () => {
    setLoadingStats(true)
    try {
      const r = await fetch(`/api/web/index/stats?collection_id=${collectionId}`)
      if (r.ok) setStats(await r.json())
    } catch { /* offline */ }
    setLoadingStats(false)
  }

  useEffect(() => { fetchStats() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Poll job status while running.
  useEffect(() => {
    if (!jobId) return
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`/api/web/crawl/status/${jobId}`)
        if (r.ok) {
          const data: JobStatus = await r.json()
          setJob(data)
          if (data.status !== 'running') {
            clearInterval(pollRef.current!)
            fetchStats()
          }
        }
      } catch { /* ignore */ }
    }, 1500)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [jobId]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleCrawl = async (e: React.FormEvent) => {
    e.preventDefault()
    const seedList = seeds.split('\n').map(s => s.trim()).filter(Boolean)
    if (!seedList.length) return
    setCrawling(true)
    setJob(null)
    try {
      const r = await fetch('/api/web/crawl', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          seeds: seedList, collection_id: collectionId,
          max_pages: maxPages, max_depth: maxDepth,
          same_domain_only: sameDomain, run_in_background: true,
        }),
      })
      const data = await r.json()
      if (data.job_id) setJobId(data.job_id)
    } catch {
      setJob({ status: 'failed', pages: 0, ingested: 0, last_url: null, error: 'API unreachable' })
    }
    setCrawling(false)
  }

  const jobDone = job?.status === 'done'
  const jobFailed = job?.status === 'failed'

  return (
    <div className="max-w-2xl mx-auto px-4 pt-8 pb-16">
      <h2 className="text-lg font-semibold mb-1">Crawl & Index</h2>
      <p className="text-xs text-gray-500 mb-6">Feed URLs into the search index. Respects robots.txt and rate limits.</p>

      {/* Index stats */}
      <div className="rounded-xl border border-white/8 bg-white/[0.02] p-4 mb-6">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-medium text-gray-400 flex items-center gap-1.5"><BarChart3 className="size-3.5" /> Index Stats</span>
          <button onClick={fetchStats} disabled={loadingStats} className="text-[10px] text-gray-600 hover:text-gray-400 transition-colors">
            {loadingStats ? <Loader2 className="size-3 animate-spin inline" /> : 'refresh'}
          </button>
        </div>
        {stats ? (
          <div className="grid grid-cols-4 gap-3">
            {([
              ['Tracked URLs', stats.freshness.tracked],
              ['Changes', stats.freshness.total_changes],
              ['Cache entries', stats.cache_entries],
              ['Recent queries', stats.recent_queries],
            ] as [string, number][]).map(([label, val]) => (
              <div key={label} className="text-center">
                <div className="text-xl font-bold text-cyan-400">{val}</div>
                <div className="text-[10px] text-gray-600 mt-0.5">{label}</div>
              </div>
            ))}
          </div>
        ) : <p className="text-xs text-gray-700">No stats — API may be offline.</p>}
      </div>

      {/* Job progress */}
      <AnimatePresence>
        {(job || crawling) && (
          <motion.div
            initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className={`rounded-xl border p-4 mb-5 ${
              jobFailed ? 'border-red-500/20 bg-red-500/5'
              : jobDone ? 'border-emerald-500/20 bg-emerald-500/5'
              : 'border-cyan-500/20 bg-cyan-500/5'
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              {jobFailed ? <XCircle className="size-4 text-red-400" />
                : jobDone ? <CheckCircle2 className="size-4 text-emerald-400" />
                : <Loader2 className="size-4 text-cyan-400 animate-spin" />}
              <span className="text-sm font-medium">
                {jobFailed ? 'Crawl failed' : jobDone ? 'Crawl complete' : 'Crawling…'}
              </span>
              {jobId && <span className="text-[10px] text-gray-600 ml-auto font-mono">{jobId}</span>}
            </div>
            {job && (
              <div className="text-xs text-gray-500 space-y-1">
                <div className="flex gap-4">
                  <span>Pages visited: <b className="text-gray-300">{job.pages}</b></span>
                  <span>Ingested: <b className="text-gray-300">{job.ingested}</b></span>
                </div>
                {job.last_url && <div className="truncate text-gray-600">{job.last_url}</div>}
                {job.error && <div className="text-red-400">{job.error}</div>}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Crawl form */}
      <form onSubmit={handleCrawl} className="space-y-4">
        <div>
          <label className="text-xs text-gray-500 mb-1.5 block">Seed URLs <span className="text-gray-700">(one per line)</span></label>
          <textarea
            value={seeds} onChange={e => setSeeds(e.target.value)} rows={4}
            placeholder={'https://example.com\nhttps://docs.example.com'}
            className="w-full bg-white/[0.03] border border-white/10 rounded-xl px-3 py-2.5 text-sm font-mono focus:outline-none focus:border-cyan-500/40 transition-all resize-none placeholder-gray-700"
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-gray-500 mb-1.5 block">Collection ID</label>
            <input value={collectionId} onChange={e => setCollectionId(e.target.value)}
              className="w-full bg-white/[0.03] border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-cyan-500/40 transition-all" />
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1.5 block">Max pages</label>
            <input type="number" min={1} max={5000} value={maxPages} onChange={e => setMaxPages(Number(e.target.value))}
              className="w-full bg-white/[0.03] border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-cyan-500/40 transition-all" />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-gray-500 mb-1.5 block">Max depth</label>
            <input type="number" min={0} max={10} value={maxDepth} onChange={e => setMaxDepth(Number(e.target.value))}
              className="w-full bg-white/[0.03] border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-cyan-500/40 transition-all" />
          </div>
          <div className="flex items-end pb-2">
            <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-400">
              <input type="checkbox" checked={sameDomain} onChange={e => setSameDomain(e.target.checked)} className="accent-cyan-500" />
              Same domain only
            </label>
          </div>
        </div>
        <button type="submit" disabled={crawling || !seeds.trim()}
          className="w-full flex items-center justify-center gap-2 py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-xl text-sm font-medium transition-colors"
        >
          {crawling ? <Loader2 className="size-4 animate-spin" /> : <Download className="size-4" />}
          {crawling ? 'Starting…' : 'Start Crawl'}
        </button>
      </form>
    </div>
  )
}

function Skeleton() {
  return (
    <div className="max-w-3xl mx-auto px-4 pt-8 pb-16">
      <div className="flex items-center gap-2 text-gray-500 text-sm mb-6">
        <Loader2 className="size-4 animate-spin text-cyan-400" /> Searching...
      </div>
      <div className="rounded-2xl border border-cyan-500/15 bg-cyan-950/20 p-5 mb-6 animate-pulse">
        {[1/4, 1, 5/6, 3/4].map((w, i) => <div key={i} className={`h-3 bg-white/5 rounded mb-2`} style={{ width: `${w * 100}%` }} />)}
      </div>
      {[1, 2, 3].map(i => (
        <div key={i} className="mb-6 animate-pulse">
          <div className="h-2.5 bg-white/5 rounded w-1/3 mb-2" />
          <div className="h-4 bg-white/8 rounded w-2/3 mb-2" />
          <div className="h-2.5 bg-white/5 rounded w-full mb-1" />
          <div className="h-2.5 bg-white/5 rounded w-4/5" />
        </div>
      ))}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function SearchPage() {
  const [mode, setMode] = useState<'search' | 'crawl'>('search')
  const [query, setQuery] = useState('')
  const [submittedQuery, setSubmittedQuery] = useState('')
  const [activeTab, setActiveTab] = useState<'all' | 'neural' | 'keyword'>('all')
  const [page, setPage] = useState(1)
  const [results, setResults] = useState<SearchResult[]>([])
  const [hasNext, setHasNext] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [cached, setCached] = useState(false)
  const [collectionId, setCollectionId] = useState('web')
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSugg, setShowSugg] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const topInputRef = useRef<HTMLInputElement>(null)
  const suggTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const hasResults = Boolean(submittedQuery && !loading)

  const clickResult = useCallback(async (url: string, position: number) => {
    try {
      await fetch('/api/web/click', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: submittedQuery, result_url: url, position }),
      })
    } catch { /* fire-and-forget */ }
  }, [submittedQuery])

  useEffect(() => { inputRef.current?.focus() }, [])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement !== inputRef.current && document.activeElement !== topInputRef.current) {
        e.preventDefault()
        ;(hasResults ? topInputRef : inputRef).current?.focus()
      }
      if (e.key === 'Escape') setShowSugg(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [hasResults])

  const fetchSuggestions = useCallback((q: string) => {
    if (suggTimerRef.current) clearTimeout(suggTimerRef.current)
    if (q.length < 2) { setSuggestions([]); return }
    suggTimerRef.current = setTimeout(async () => {
      try {
        const r = await fetch(`/api/web/suggest?q=${encodeURIComponent(q)}&limit=5`)
        if (r.ok) {
          const data = await r.json()
          setSuggestions(data.suggestions || [])
          setShowSugg(true)
        }
      } catch { /* offline */ }
    }, 200)
  }, [])

  const doSearch = async (q: string, tab = activeTab, pg = 1) => {
    if (!q.trim()) return
    setLoading(true)
    setError('')
    setResults([])
    setSubmittedQuery(q)
    setPage(pg)
    setShowSugg(false)
    const start = Date.now()
    const expand = tab !== 'keyword'
    const rerank = tab !== 'keyword'
    try {
      const res = await fetch(
        `/api/web/search?q=${encodeURIComponent(q)}&k=10&page=${pg}&collection_id=${encodeURIComponent(collectionId)}&expand=${expand}&rerank=${rerank}`
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data: SearchResponse = await res.json()
      if (!data.success) throw new Error('Search returned failure')
      setResults(data.results)
      setHasNext(data.has_next)
      setElapsed(Date.now() - start)
      setCached(Boolean(data.cached))
    } catch {
      setError('Search unavailable — ensure the API server is running at localhost:8000.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => { e.preventDefault(); doSearch(query) }
  const handleHint = (hint: string) => { setQuery(hint); doSearch(hint) }
  const handleTab = (tab: typeof activeTab) => { setActiveTab(tab); if (submittedQuery) doSearch(submittedQuery, tab, 1) }
  const handlePrev = () => { const p = Math.max(1, page - 1); doSearch(submittedQuery, activeTab, p) }
  const handleNext = () => doSearch(submittedQuery, activeTab, page + 1)

  const SearchInput = ({ ref: ref_, compact }: { ref_?: React.RefObject<HTMLInputElement | null>; compact?: boolean }) => (
    <div className="relative flex-1">
      <Search className={`absolute left-3 top-1/2 -translate-y-1/2 text-gray-500 ${compact ? 'size-4' : 'size-5 left-4'}`} />
      <input
        ref={ref_}
        type="text" value={query}
        onChange={e => { setQuery(e.target.value); fetchSuggestions(e.target.value) }}
        onFocus={() => suggestions.length && setShowSugg(true)}
        onBlur={() => setTimeout(() => setShowSugg(false), 150)}
        className={`w-full bg-white/[0.04] border border-white/10 hover:border-white/20 focus:border-cyan-500/50 rounded-full focus:outline-none transition-all placeholder-gray-700 ${
          compact ? 'pl-9 pr-4 py-2 text-sm' : 'pl-12 pr-[130px] py-3.5 text-base'
        }`}
        placeholder={compact ? 'Search the web…' : 'Search anything…'}
      />
      {/* Autocomplete dropdown */}
      <AnimatePresence>
        {showSugg && suggestions.length > 0 && (
          <motion.ul
            initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            className="absolute top-full left-0 right-0 mt-1 bg-zinc-900 border border-white/10 rounded-xl overflow-hidden z-50 shadow-2xl"
          >
            {suggestions.map(s => (
              <li key={s}>
                <button
                  className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:bg-white/5 flex items-center gap-2"
                  onMouseDown={() => { setQuery(s); doSearch(s) }}
                >
                  <Search className="size-3 text-gray-600 flex-shrink-0" /> {s}
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )

  return (
    <div className="min-h-screen bg-black text-white">

      {/* ── Results header ── */}
      {submittedQuery && (
        <header className="sticky top-0 z-50 border-b border-white/5 bg-black/90 backdrop-blur-md px-4 py-3">
          <div className="max-w-3xl mx-auto flex items-center gap-3">
            <a href="/" className="text-xs text-gray-500 hover:text-white flex items-center gap-1 flex-shrink-0">
              <ArrowLeft className="size-3.5" /> VectorDB
            </a>
            <form onSubmit={handleSubmit} className="flex-1 flex gap-2">
              <SearchInput ref_={topInputRef} compact />
              <button type="submit" className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 rounded-full text-sm font-medium transition-colors flex-shrink-0">
                Search
              </button>
            </form>
          </div>
          <div className="max-w-3xl mx-auto flex items-center gap-1 mt-2 pl-[72px]">
            {([['all', 'All', AlignLeft], ['neural', 'Neural', Sparkles], ['keyword', 'Keyword', Zap]] as const).map(([id, label, Icon]) => (
              <button key={id} onClick={() => handleTab(id)}
                className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  activeTab === id ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                <Icon className="size-3" />{label}
              </button>
            ))}
            <div className="ml-auto flex items-center gap-1.5">
              <span className="text-[10px] text-gray-600">collection</span>
              <select
                value={collectionId}
                onChange={e => { setCollectionId(e.target.value); if (submittedQuery) doSearch(submittedQuery, activeTab, 1) }}
                className="text-xs bg-white/5 border border-white/10 rounded-lg px-2 py-0.5 text-gray-400 focus:outline-none focus:border-cyan-500/30 cursor-pointer"
              >
                {['web', 'docs', 'news', 'research'].map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>
        </header>
      )}

      {/* ── Crawl mode ── */}
      {mode === 'crawl' && !submittedQuery && <CrawlPanel />}

      {/* ── Landing state ── */}
      <AnimatePresence>
        {mode === 'search' && !submittedQuery && (
          <motion.div key="landing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, y: -20 }}
            className="min-h-screen flex flex-col items-center justify-center px-4"
          >
            <motion.div initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.1 }}
              className="mb-8 text-center"
            >
              <a href="/" className="inline-flex items-center gap-2 text-gray-600 text-xs hover:text-gray-400 transition-colors mb-6">
                <ArrowLeft className="size-3" /> Back to home
              </a>
              <div className="flex items-center justify-center gap-2.5 mb-2">
                <span className="size-3 rounded-full bg-cyan-400 shadow-[0_0_12px_#22d3ee]" />
                <span className="text-3xl font-light tracking-tight">VectorDB</span>
                <span className="text-3xl font-bold text-cyan-400">Search</span>
              </div>
              <p className="text-gray-600 text-sm">Hybrid neural + keyword · self-hosted · zero external APIs</p>
            </motion.div>

            <motion.form initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}
              onSubmit={handleSubmit} className="w-full max-w-lg"
            >
              <div className="flex gap-2">
                <SearchInput ref_={inputRef} />
                <button type="submit" disabled={!query.trim()}
                  className="px-5 py-3.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed rounded-full text-sm font-medium transition-colors flex-shrink-0"
                >
                  Search
                </button>
              </div>
              <div className="flex items-center justify-between mt-2 px-1">
                <p className="text-[11px] text-gray-700">
                  Press <kbd className="px-1 py-0.5 bg-white/5 rounded text-gray-600">/</kbd> to focus
                </p>
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-gray-600">collection</span>
                  <select
                    value={collectionId} onChange={e => setCollectionId(e.target.value)}
                    className="text-xs bg-white/5 border border-white/10 rounded-lg px-2 py-0.5 text-gray-400 focus:outline-none focus:border-cyan-500/30 cursor-pointer"
                  >
                    {['web', 'docs', 'news', 'research'].map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              </div>
            </motion.form>

            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }}
              className="flex flex-wrap gap-2 mt-6 justify-center max-w-lg"
            >
              {HINTS.map(hint => (
                <button key={hint} onClick={() => handleHint(hint)}
                  className="px-3.5 py-1.5 text-xs text-gray-500 bg-white/[0.03] border border-white/8 rounded-full hover:border-cyan-500/30 hover:text-cyan-400 hover:bg-cyan-500/5 transition-colors"
                >
                  {hint}
                </button>
              ))}
            </motion.div>

            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
              className="flex items-center gap-1 mt-8 bg-white/[0.03] border border-white/8 rounded-full p-0.5"
            >
              {([['search', 'Search', Search], ['crawl', 'Crawl', Download]] as const).map(([id, label, Icon]) => (
                <button key={id} onClick={() => setMode(id)}
                  className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-medium transition-colors ${
                    mode === id ? 'bg-cyan-600 text-white' : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  <Icon className="size-3" />{label}
                </button>
              ))}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {loading && <Skeleton />}

      {/* ── Results ── */}
      {hasResults && (
        <div className="max-w-3xl mx-auto px-4 pt-6 pb-16">
          <p className="text-[11px] text-gray-700 mb-5 flex items-center gap-1.5">
            <span>{results.length} results</span>
            <span>·</span>
            <span>{(elapsed / 1000).toFixed(2)}s</span>
            {cached && <><span>·</span><span className="text-cyan-700 flex items-center gap-0.5"><Zap className="size-2.5" /> cached</span></>}
            <span>·</span>
            <span className="text-gray-600">page {page}</span>
            <span>·</span>
            <span className="italic text-gray-600">{submittedQuery}</span>
          </p>

          {error && (
            <div className="text-red-400 text-sm mb-6 p-3 bg-red-500/10 rounded-xl border border-red-500/20">{error}</div>
          )}

          {!error && results.length === 0 && (
            <div className="text-gray-600 text-sm py-8 text-center">
              No results for <em>"{submittedQuery}"</em>.<br />
              <span className="text-xs mt-2 block">Switch to <b>Crawl</b> mode to seed the index first.</span>
            </div>
          )}

          {results.length > 0 && (
            <>
              <AnswerCard results={results} query={submittedQuery} />
              {results.map((r, idx) => (
                <ResultCard key={r.vector_id} result={r} idx={idx} query={submittedQuery} onClickResult={clickResult} />
              ))}

              {/* Pagination */}
              <div className="flex items-center justify-between pt-6 border-t border-white/5 mt-4">
                <button onClick={handlePrev} disabled={page <= 1}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-full text-sm text-gray-400 hover:text-white hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeft className="size-4" /> Previous
                </button>
                <span className="text-xs text-gray-600">Page {page}</span>
                <button onClick={handleNext} disabled={!hasNext}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-full text-sm text-gray-400 hover:text-white hover:bg-white/5 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  Next <ChevronRight className="size-4" />
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
