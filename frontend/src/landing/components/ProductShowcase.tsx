import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";
import { BentoGridShowcase } from "../../components/ui/bento-product-features";
import { Cpu, Database, Search, ShieldCheck, Terminal, Layers } from "lucide-react";

{/* ── Linear-Style Isometric Line-Art Animated SVGs ── */}

// FIG 0.1: Layered Isometric Chunk Stack
const IsometricStackSvg = () => (
  <div className="relative flex h-48 w-full items-center justify-center overflow-hidden py-4">
    <svg width="220" height="180" viewBox="0 0 220 180" fill="none" className="text-zinc-500">
      {/* Top Layer */}
      <motion.g
        initial={{ y: -10, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <path d="M110 20 L170 50 L110 80 L50 50 Z" stroke="currentColor" strokeWidth="1.2" strokeDasharray="3 3" fill="rgba(99,102,241,0.06)" />
        <path d="M110 20 L170 50 L110 80 L50 50 Z" stroke="#6366f1" strokeWidth="1.2" />
        <circle cx="110" cy="50" r="4" fill="#818cf8" className="animate-pulse" />
      </motion.g>

      {/* Middle Layer */}
      <motion.g
        initial={{ y: 0, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.2 }}
      >
        <path d="M110 50 L170 80 L110 110 L50 80 Z" stroke="currentColor" strokeWidth="1" fill="rgba(255,255,255,0.02)" />
        <path d="M50 80 L50 90 L110 120 L170 90 L170 80" stroke="currentColor" strokeWidth="1" />
      </motion.g>

      {/* Bottom Layer */}
      <motion.g
        initial={{ y: 10, opacity: 0 }}
        whileInView={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.4 }}
      >
        <path d="M110 80 L170 110 L110 140 L50 110 Z" stroke="currentColor" strokeWidth="1" fill="rgba(255,255,255,0.02)" />
        <path d="M50 110 L50 130 L110 160 L170 130 L170 110" stroke="currentColor" strokeWidth="1" />
        <path d="M110 140 L110 160" stroke="#6366f1" strokeWidth="1.5" />
      </motion.g>

      {/* Vertical Axis Guide Lines */}
      <line x1="110" y1="20" x2="110" y2="160" stroke="rgba(99,102,241,0.3)" strokeWidth="1" strokeDasharray="2 2" />
    </svg>
  </div>
);

// FIG 0.2: Isometric Cubes (Router Nodes)
const IsometricCubesSvg = () => (
  <div className="relative flex h-36 w-full items-center justify-center overflow-hidden">
    <svg width="200" height="130" viewBox="0 0 200 130" fill="none" className="text-zinc-600">
      {/* Cube 1 */}
      <g transform="translate(40, 20)">
        <path d="M30 0 L60 15 L30 30 L0 15 Z" stroke="currentColor" strokeWidth="1" fill="rgba(255,255,255,0.03)" />
        <path d="M0 15 L0 45 L30 60 L30 30 Z" stroke="currentColor" strokeWidth="1" fill="rgba(0,0,0,0.4)" />
        <path d="M60 15 L60 45 L30 60 L30 30 Z" stroke="currentColor" strokeWidth="1" />
      </g>
      {/* Cube 2 (Highlighted) */}
      <g transform="translate(100, 35)">
        <path d="M30 0 L60 15 L30 30 L0 15 Z" stroke="#818cf8" strokeWidth="1.2" fill="rgba(99,102,241,0.15)" />
        <path d="M0 15 L0 45 L30 60 L30 30 Z" stroke="#6366f1" strokeWidth="1.2" fill="rgba(99,102,241,0.05)" />
        <path d="M60 15 L60 45 L30 60 L30 30 Z" stroke="#6366f1" strokeWidth="1.2" />
        <circle cx="30" cy="15" r="3" fill="#818cf8" className="animate-ping" />
      </g>
    </svg>
  </div>
);

// FIG 0.3: Stepped Pyramid (Performance)
const IsometricPyramidSvg = () => (
  <div className="relative flex h-36 w-full items-center justify-center overflow-hidden">
    <svg width="180" height="130" viewBox="0 0 180 130" fill="none" className="text-zinc-600">
      <path d="M90 10 L150 40 L90 70 L30 40 Z" stroke="#6366f1" strokeWidth="1.2" fill="rgba(99,102,241,0.08)" />
      <path d="M90 30 L135 52.5 L90 75 L45 52.5 Z" stroke="currentColor" strokeWidth="1" />
      <path d="M90 50 L120 65 L90 80 L60 65 Z" stroke="currentColor" strokeWidth="1" />
      <path d="M30 40 L30 90 L90 120 L150 90 L150 40" stroke="currentColor" strokeWidth="1" />
      <path d="M90 70 L90 120" stroke="#6366f1" strokeWidth="1.2" />
    </svg>
  </div>
);

export default function ProductShowcase() {
  return (
    <section id="system-features" className="border-t border-white/[0.05] bg-[#09090b] py-28">
      <div className="mx-auto max-w-6xl px-6">
        {/* Header */}
        <div className="mb-16">
          <p className="mb-3 font-mono text-[11px] font-medium uppercase tracking-[0.12em] text-indigo-400">
            Engine Specs
          </p>
          <h2
            className="text-3xl font-medium leading-tight tracking-[-0.03em] text-white md:text-5xl"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            Purpose-built RAG Architecture
          </h2>
          <p className="mt-4 max-w-md text-[15px] leading-relaxed text-zinc-500">
            Shaped by the principles of production AI retrieval: tool routing, strict citations, and low latency.
          </p>
        </div>

        {/* Bento Grid Showcase */}
        <BentoGridShowcase
          integration={
            <Card className="flex h-full flex-col justify-between p-2">
              <CardHeader>
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">FIG 0.1</span>
                  <Badge variant="outline" className="border-indigo-500/30 text-indigo-400">Hierarchical Chunking</Badge>
                </div>
                <CardTitle className="text-xl">Parent-Child Vector Store</CardTitle>
                <CardDescription>
                  Documents are decomposed into granular sub-chunks for similarity search, mapping back to full parent paragraphs for coherent LLM context generation.
                </CardDescription>
              </CardHeader>

              <CardContent className="mt-auto">
                <IsometricStackSvg />
              </CardContent>
            </Card>
          }

          trackers={
            <Card className="h-full">
              <CardContent className="flex h-full flex-col justify-between p-6">
                <div>
                  <div className="mb-3 flex items-center justify-between">
                    <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">FIG 0.2</span>
                    <Layers className="h-4 w-4 text-indigo-400" />
                  </div>
                  <CardTitle className="text-base font-medium">Hybrid BM25 + Vector</CardTitle>
                  <CardDescription className="mt-1 text-xs">Reciprocal Rank Fusion blending exact matching and semantic vector search.</CardDescription>
                </div>
                <IsometricCubesSvg />
              </CardContent>
            </Card>
          }

          statistic={
            <Card className="relative h-full w-full overflow-hidden">
              <div
                className="absolute inset-0 opacity-15"
                style={{
                  backgroundImage: "radial-gradient(#ffffff 1px, transparent 1px)",
                  backgroundSize: "16px 16px",
                }}
              />
              <CardContent className="relative z-10 flex h-full flex-col justify-between p-6">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">FIG 0.3</span>
                  <Badge variant="outline" className="border-emerald-500/30 text-emerald-400">RAGAS Benchmark</Badge>
                </div>
                <div className="my-auto text-center">
                  <span className="text-6xl font-bold tracking-tight text-white md:text-7xl">99%</span>
                  <p className="mt-2 text-xs font-medium text-zinc-400">Groundedness Score</p>
                </div>
              </CardContent>
            </Card>
          }

          focus={
            <Card className="h-full group cursor-default">
              <CardContent className="flex h-full flex-col justify-between p-6">
                <div className="flex items-start justify-between">
                  <div>
                    <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">FIG 0.4</span>
                    <CardTitle className="mt-1 text-base font-medium">Router Latency</CardTitle>
                    <CardDescription className="text-xs">Single-pass function call</CardDescription>
                  </div>
                  <Badge variant="outline" className="border-indigo-500/30 text-indigo-400 group-hover:bg-indigo-500/10 transition-colors duration-500">
                    &lt;150ms
                  </Badge>
                </div>
                <div className="mt-4 flex justify-center h-24">
                   <div className="relative w-full h-full flex items-end gap-1 opacity-60 group-hover:opacity-100 transition-opacity duration-500">
                      {[30, 45, 20, 60, 35, 80, 40, 25, 55, 30].map((h, i) => (
                        <div 
                           key={i} 
                           className="flex-1 bg-indigo-500/40 rounded-t-sm transition-all duration-500 ease-out origin-bottom group-hover:bg-indigo-400 group-hover:scale-y-[1.2] group-hover:shadow-[0_0_10px_rgba(99,102,241,0.5)]" 
                           style={{ height: `${h}%`, transitionDelay: `${i * 30}ms` }} 
                        />
                      ))}
                   </div>
                </div>
              </CardContent>
            </Card>
          }

          productivity={
            <Card className="h-full">
              <CardContent className="flex h-full flex-col justify-between p-6">
                <div>
                  <span className="font-mono text-[10px] uppercase tracking-widest text-zinc-500">FIG 0.5</span>
                  <CardTitle className="mt-1 text-base font-medium">Postgres + pgvector</CardTitle>
                  <CardDescription className="mt-1 text-xs">
                    Single relational database for document metadata and high-dimensional vector embeddings.
                  </CardDescription>
                </div>
                <div className="mt-4 flex items-center gap-2 font-mono text-[11px] text-zinc-400">
                  <Database className="h-4 w-4 text-indigo-400" />
                  <span>pgvector SQL extension</span>
                </div>
              </CardContent>
            </Card>
          }
        />
      </div>
    </section>
  );
}
