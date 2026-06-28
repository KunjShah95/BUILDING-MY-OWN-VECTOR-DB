import { useEffect, useRef, useState } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import { GitForkIcon, BarChart3, Brain, Search, Globe, Layers, Shield, Zap, Cpu, TestTube, ExternalLink, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import AnimatedHeading from '@/components/Hero/AnimatedHeading'
import CssFadeIn from '@/components/Hero/FadeIn'

const NAV_ITEMS = [
  { label: 'Features', href: '#features' },
  { label: 'SDKs', href: '#sdks' },
  { label: 'Benchmarks', href: '#benchmarks' },
]

const FEATURES = [
  { icon: Zap, title: '9 ANN Algorithms', desc: 'HNSW, IVF+PQ, Vamana/DiskANN, Int8, LSH, KD-Tree, VP-Tree, BM25, Hybrid RRF', color: '#06b6d4' },
  { icon: Brain, title: 'Agentic Memory', desc: 'pgvector-backed persistent memory with CRUD, semantic search, LLM chat, and SSE streaming', color: '#8b5cf6' },
  { icon: Search, title: 'SPLADE + ColBERT', desc: 'Learned sparse retrieval via transformers and late-interaction MaxSim scoring', color: '#10b981' },
  { icon: Globe, title: 'Polyglot SDKs', desc: 'Python, TypeScript, Go, Java, Rust, .NET — type-safe clients for every language', color: '#eab308' },
  { icon: BarChart3, title: 'Self-Tuning Indexes', desc: 'AI-recommended parameters and adaptive per-query index routing', color: '#f97316' },
  { icon: Shield, title: 'Enterprise Compliance', desc: 'SOC2/GDPR reports, retention policies, query budgets, AES-256 encryption', color: '#ec4899' },
  { icon: Layers, title: 'Real-Time Streaming', desc: 'SSE subscriptions, webhooks, lock-free HNSW writes, materialized views', color: '#22d3ee' },
  { icon: Cpu, title: 'Ecosystem Integrations', desc: 'Haystack, Semantic Kernel, LangChain, LlamaIndex, Arrow Flight, MCP Server', color: '#a855f7' },
  { icon: TestTube, title: '660+ Tests', desc: 'Full coverage: API, indexes, services, durability, ML models, infrastructure', color: '#facc15' },
]
const SDKS = [
  { icon: '🐍', name: 'Python', stable: true },
  { icon: '🔷', name: 'TypeScript', stable: true },
  { icon: '🐹', name: 'Go', stable: true },
  { icon: '☕', name: 'Java', stable: true },
  { icon: '🦀', name: 'Rust', stable: true },
  { icon: '💧', name: '.NET', stable: true },
]
const BENCHMARKS = [
  { index: 'HNSW', params: '(M=16, ef=200)', recall: '0.981', latency: '0.129 s', p99: '0.789 s', build: '45.2 s', highlight: true },
  { index: 'IVF', params: '(nlist=100, nprobe=10)', recall: '0.940', latency: '0.350 s', p99: '1.200 s', build: '1.8 s', highlight: false },
  { index: 'Brute Force', params: '', recall: '1.000', latency: '4.200 s', p99: '9.100 s', build: '—', highlight: false },
  { index: 'Vamana/DiskANN', params: '(L=75, R=50)', recall: '0.970', latency: '0.150 s', p99: '0.850 s', build: '38.0 s', highlight: false },
]

function useCountUp(ref: React.RefObject<HTMLElement | null>, target: number, suffix = '') {
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return
      let current = 0
      const increment = Math.max(1, Math.floor(target / 40))
      const timer = setInterval(() => {
        current += increment
        if (current >= target) { current = target; clearInterval(timer) }
        el.textContent = current + suffix
      }, 30)
      observer.disconnect()
    }, { threshold: 0.5 })
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref, target, suffix])
}

function ScrollFadeIn({ children, className = '', delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.6, delay, ease: [0.175, 0.885, 0.32, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

function StatItem({ target, label, suffix = '' }: { target: number; label: string; suffix?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  useCountUp(ref, target, suffix)
  return (
    <div className="text-center">
      <div ref={ref} className="text-5xl font-extrabold bg-gradient-to-br from-white to-cyan-400 bg-clip-text text-transparent">0</div>
      <div className="text-sm text-gray-500 mt-2">{label}</div>
    </div>
  )
}

function DashboardReveal() {
  const sectionRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({ target: sectionRef, offset: ['start end', 'end start'] })
  const y = useTransform(scrollYProgress, [0, 1], [80, -120])
  const opacity = useTransform(scrollYProgress, [0, 0.25, 0.8, 1], [0, 1, 1, 0.6])

  return (
    <section ref={sectionRef} className="relative px-6 py-24 overflow-hidden">
      <div className="absolute inset-x-0 top-1/3 mx-auto size-[500px] rounded-full bg-cyan-500/10 blur-[140px] pointer-events-none" />
      <motion.div style={{ y, opacity }} className="relative z-10 mx-auto max-w-5xl">
        <div className="liquid-glass rounded-2xl overflow-hidden shadow-2xl shadow-cyan-500/10" style={{ mixBlendMode: 'luminosity' }}>
          <div className="flex items-center gap-2 px-4 py-3 border-b border-white/5">
            <span className="size-3 rounded-full bg-red-400/60" /><span className="size-3 rounded-full bg-amber-400/60" /><span className="size-3 rounded-full bg-emerald-400/60" />
            <span className="ml-3 text-xs text-gray-500">vectordb · dashboard</span>
          </div>
          <div className="grid grid-cols-4 gap-3 p-5">
            {[['Vectors', '4.2M'], ['QPS', '12.8K'], ['Recall@10', '0.981'], ['p99', '0.79s']].map(([k, v]) => (
              <div key={k} className="rounded-xl bg-white/[0.03] border border-white/5 p-4">
                <div className="text-[10px] uppercase tracking-wider text-gray-500">{k}</div>
                <div className="text-xl font-bold text-cyan-400 mt-1">{v}</div>
              </div>
            ))}
            <div className="col-span-4 h-40 rounded-xl bg-white/[0.02] border border-white/5 flex items-end gap-2 p-4">
              {[40, 65, 50, 80, 55, 90, 70, 95, 60, 85, 75, 100].map((h, i) => (
                <div key={i} className="flex-1 rounded-t bg-gradient-to-t from-cyan-500/40 to-purple-500/40" style={{ height: `${h}%` }} />
              ))}
            </div>
          </div>
        </div>
      </motion.div>
    </section>
  )
}

export default function App() {
  return (
    <div className="min-h-screen bg-black text-white overflow-x-hidden">
      {/* ───── HERO ───── */}
      <section className="relative min-h-screen bg-black text-white">
        <video autoPlay loop muted playsInline className="absolute inset-0 w-full h-full object-cover">
          <source src="https://d8j0ntlcm91z4.cloudfront.net/user_38xzZboKViGWJOttwIXH07lWA1P/hf_20260423_084718_72a17915-4964-4059-afcd-22d59399b72e.mp4" type="video/mp4" />
        </video>
        <div className="relative z-10 min-h-screen">
          <div className="absolute top-0 left-0 right-0 px-6 md:px-12 lg:px-16 pt-6">
            <div className="liquid-glass rounded-xl flex items-center justify-between px-4 py-2">
              <a href="/" className="flex items-center gap-2 text-lg font-semibold tracking-tight">
                <span className="size-2 rounded-full bg-cyan-400" />
                VectorDB
              </a>
              <div className="hidden md:flex gap-8 text-sm">
                {NAV_ITEMS.map(item => (
                  <a key={item.label} href={item.href} className="hover:text-gray-300 transition-colors">{item.label}</a>
                ))}
                <a href="/search" className="hover:text-gray-300 transition-colors flex items-center gap-1.5">
                  <Search className="size-3.5" /> Search
                </a>
                <a href="/rag" className="hover:text-gray-300 transition-colors flex items-center gap-1.5">
                  <Brain className="size-3.5" /> RAG
                </a>
                <a href="https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB" target="_blank" rel="noopener noreferrer" className="hover:text-gray-300 transition-colors flex items-center gap-1.5">
                  <GitForkIcon className="size-3.5" /> GitHub
                </a>
              </div>
              <a href="/dashboard"><button className="bg-white text-black px-5 py-1.5 rounded-lg text-sm font-medium hover:bg-gray-100 transition-colors">Dashboard</button></a>
            </div>
          </div>
          <div className="min-h-screen px-6 md:px-12 lg:px-16 flex flex-col items-center justify-center text-center">
            <div className="w-full max-w-4xl flex flex-col items-center">
              <AnimatedHeading
                text="Build intelligent&#10;search at scale."
                className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-light mb-6"
                delay={200}
                charDelay={30}
              />
              <CssFadeIn delay={800} duration={1000}>
                <p className="text-base md:text-lg text-gray-400 max-w-2xl mb-8 leading-relaxed">
                  A production-grade vector database with 9+ ANN algorithms, agentic memory, multi-vector search, and enterprise compliance — built for AI workloads.
                </p>
              </CssFadeIn>
              <CssFadeIn delay={1200} duration={1000}>
                <div className="flex flex-wrap gap-4 justify-center">
                  <a href="/dashboard"><button className="bg-white text-black px-8 py-3 rounded-lg font-medium hover:bg-gray-100 transition-colors">Launch Dashboard</button></a>
                  <a href="https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB" target="_blank" rel="noopener noreferrer">
                    <button className="liquid-glass border border-white/20 text-white px-8 py-3 rounded-lg font-medium hover:bg-white hover:text-black transition-colors"><GitForkIcon className="size-4 inline mr-1.5" />GitHub</button>
                  </a>
                </div>
              </CssFadeIn>
            </div>
          </div>
          <div className="absolute bottom-0 left-0 right-0 px-6 md:px-12 lg:px-16 pb-12 lg:pb-16 flex justify-center">
            <CssFadeIn delay={1400} duration={1000}>
              <div className="liquid-glass border border-white/20 px-6 py-3 rounded-xl">
                <p className="text-sm md:text-base text-gray-300">HNSW · IVF · Vamana · BM25 · Hybrid RRF</p>
              </div>
            </CssFadeIn>
          </div>
        </div>
      </section>

      {/* ───── DASHBOARD REVEAL ───── */}
      <DashboardReveal />

      {/* ───── FEATURES ───── */}
      <section id="features" className="px-6 py-24 max-w-6xl mx-auto">
        <ScrollFadeIn>
          <Badge variant="outline" className="border-cyan-500/20 text-cyan-400 bg-cyan-500/5 mb-4">Capabilities</Badge>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">Everything you need for vector search</h2>
          <p className="text-gray-400 max-w-xl mb-12">9 ANN algorithms, multi-modal embeddings, real-time streaming, enterprise compliance, and SDKs for every major language.</p>
        </ScrollFadeIn>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((f, i) => (
            <ScrollFadeIn key={f.title} delay={i * 0.05}>
              <Card className="group border-white/5 bg-white/[0.02] hover:bg-white/[0.04] hover:border-cyan-500/20 transition-all duration-500 cursor-default overflow-hidden relative">
                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <CardContent className="p-6">
                  <div className="size-11 rounded-xl flex items-center justify-center mb-4 text-lg" style={{ background: `${f.color}15`, color: f.color }}>
                    <f.icon className="size-5" />
                  </div>
                  <h3 className="font-semibold mb-1.5">{f.title}</h3>
                  <p className="text-sm text-gray-400 leading-relaxed">{f.desc}</p>
                </CardContent>
              </Card>
            </ScrollFadeIn>
          ))}
        </div>
      </section>

      {/* ───── SDKs ───── */}
      <section id="sdks" className="px-6 py-24 max-w-6xl mx-auto">
        <ScrollFadeIn>
          <Badge variant="outline" className="border-purple-500/20 text-purple-400 bg-purple-500/5 mb-4">SDKs & Clients</Badge>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">Ship in any language</h2>
          <p className="text-gray-400 max-w-xl mb-12">Idiomatic, type-safe clients for every major ecosystem.</p>
        </ScrollFadeIn>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
          {SDKS.map((s, i) => (
            <ScrollFadeIn key={s.name} delay={i * 0.08}>
              <div className="p-5 rounded-xl bg-white/[0.02] border border-white/5 hover:border-cyan-500/20 hover:bg-white/[0.04] transition-all text-center cursor-default group">
                <div className="text-3xl mb-2">{s.icon}</div>
                <div className="text-sm font-semibold">{s.name}</div>
                <div className="text-[10px] text-emerald-400 mt-1 opacity-0 group-hover:opacity-100 transition-opacity">✓ Stable</div>
              </div>
            </ScrollFadeIn>
          ))}
        </div>
      </section>

      {/* ───── STATS ───── */}
      <section className="px-6 py-24 max-w-4xl mx-auto">
        <ScrollFadeIn>
          <Badge variant="outline" className="border-amber-500/20 text-amber-400 bg-amber-500/5 mb-4">By the Numbers</Badge>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3 text-center">Built for production</h2>
        </ScrollFadeIn>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-8 mt-12">
          <StatItem target={9} label="ANN Algorithms" />
          <StatItem target={11} label="SDKs & Integrations" />
          <StatItem target={660} label="Tests" suffix="+" />
          <StatItem target={80} label="REST Endpoints" suffix="+" />
        </div>
      </section>

      {/* ───── BENCHMARKS ───── */}
      <section id="benchmarks" className="px-6 py-24 max-w-5xl mx-auto">
        <ScrollFadeIn>
          <Badge variant="outline" className="border-cyan-500/20 text-cyan-400 bg-cyan-500/5 mb-4">Performance</Badge>
          <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-3">Benchmark snapshot</h2>
          <p className="text-gray-400 max-w-xl mb-12">10K vectors, 128-dim — real numbers from the built-in benchmark suite.</p>
        </ScrollFadeIn>
        <ScrollFadeIn>
          <div className="overflow-x-auto rounded-2xl border border-white/5">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5 bg-white/[0.02]">
                  <th className="text-left p-4 font-semibold text-gray-500 text-xs uppercase tracking-wider">Index</th>
                  <th className="text-left p-4 font-semibold text-gray-500 text-xs uppercase tracking-wider">Recall@10</th>
                  <th className="text-left p-4 font-semibold text-gray-500 text-xs uppercase tracking-wider">Avg Latency</th>
                  <th className="text-left p-4 font-semibold text-gray-500 text-xs uppercase tracking-wider">p99 Latency</th>
                  <th className="text-left p-4 font-semibold text-gray-500 text-xs uppercase tracking-wider">Build Time</th>
                </tr>
              </thead>
              <tbody>
                {BENCHMARKS.map(b => (
                  <tr key={b.index} className="border-b border-white/[0.02] last:border-0 hover:bg-cyan-500/5 transition-colors">
                    <td className="p-4">
                      <span className="font-semibold">{b.index}</span>
                      {b.params && <span className="text-gray-500 ml-1 text-xs">{b.params}</span>}
                    </td>
                    <td className={`p-4 font-mono ${b.highlight ? 'text-cyan-400' : ''}`}>{b.recall}</td>
                    <td className="p-4 font-mono text-gray-400">{b.latency}</td>
                    <td className="p-4 font-mono text-gray-400">{b.p99}</td>
                    <td className="p-4 font-mono text-gray-400">{b.build}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ScrollFadeIn>
      </section>

      {/* ───── CTA ───── */}
      <section className="px-6 py-24 text-center">
        <ScrollFadeIn>
          <div className="liquid-glass rounded-3xl max-w-2xl mx-auto p-12 border border-white/5">
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight mb-4">Ready to build?</h2>
            <p className="text-gray-400 mb-8 max-w-md mx-auto">Get started with the dashboard or explore the codebase on GitHub.</p>
            <div className="flex flex-wrap gap-4 justify-center">
              <a href="/dashboard"><button className="bg-white text-black px-8 py-3 rounded-lg font-medium hover:bg-gray-100 transition-colors">Launch Dashboard</button></a>
              <a href="https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB" target="_blank" rel="noopener noreferrer">
                <button className="liquid-glass border border-white/20 text-white px-8 py-3 rounded-lg font-medium hover:bg-white hover:text-black transition-colors"><GitForkIcon className="size-4 inline mr-1.5" />GitHub</button>
              </a>
            </div>
          </div>
        </ScrollFadeIn>
      </section>

      {/* ───── FOOTER ───── */}
      <footer className="border-t border-white/5 px-6 py-10 text-center text-sm text-gray-500">
        <div className="flex justify-center gap-6 mb-4 flex-wrap">
          {['Features', 'SDKs', 'Benchmarks', 'GitHub', 'Issues'].map(item => (
            <a key={item} href={item === 'GitHub' ? 'https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB' : item === 'Issues' ? 'https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/issues' : `#${item.toLowerCase()}`}
              className="hover:text-white transition-colors">{item}</a>
          ))}
        </div>
        <p>Built with Rust, Python, and TypeScript. Open source under MIT License.</p>
      </footer>
    </div>
  )
}
