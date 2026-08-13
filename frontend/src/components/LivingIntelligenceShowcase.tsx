import React, { useMemo, useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BrainCircuit } from "lucide-react";

const ribbons = [
  // Main bright crossing wave (Violet)
  {
    id: 1,
    d: "M -100, 550 C 500, 800, 1500, 200, 2200, 100",
    color: "#8B5CF6",
    width: 6,
    duration: 50,
    dash: "1500 1500",
    primary: true,
  },
  // Secondary crossing wave (Blue/Cyan)
  {
    id: 2,
    d: "M -100, 850 C 600, 300, 1500, 600, 2200, 350",
    color: "#38BDF8",
    width: 5,
    duration: 40,
    dash: "1000 2000",
    primary: true,
  },
  // Top swooping down (Deep Violet)
  {
    id: 3,
    d: "M -100, 700 C 600, 900, 1500, 700, 2200, 800",
    color: "#7C3AED",
    width: 8,
    duration: 64,
    dash: "2000 1000",
    primary: false,
  },
  // Wide ambient glow mimicking Ribbon 1
  {
    id: 4,
    d: "M -100, 600 C 550, 850, 1450, 150, 2200, 50",
    color: "#6366F1",
    width: 14,
    duration: 70,
    dash: "2500 1500",
    primary: false,
  },
  // Fast accent strand crossing high to low
  {
    id: 5,
    d: "M -100, 750 C 500, 150, 1400, 950, 2200, 900",
    color: "#4F46E5",
    width: 3,
    duration: 36,
    dash: "1200 1800",
    primary: false,
  },
  // Mid-screen core filler
  {
    id: 6,
    d: "M -100, 750 C 550, 450, 1450, 250, 2200, 450",
    color: "#A855F7",
    width: 10,
    duration: 70,
    dash: "2500 1500",
    primary: false,
  },
  // Faint accent line
  {
    id: 7,
    d: "M 300, 900 C 900, 100, 1600, 500, 2200, 200",
    color: "#EC4899",
    width: 2,
    duration: 44,
    dash: "1000 2000",
    primary: true,
  },
  // High sweeping curve (Light Purple) fanning to top-right
  {
    id: 8,
    d: "M -100, 800 C 500, 900, 1500, 0, 2400, -100",
    color: "#A855F7",
    width: 6,
    duration: 56,
    dash: "1500 1500",
    primary: false,
  },
  // Deep crossing curve hitting the center-right
  {
    id: 9,
    d: "M 400, 100 C 700, 200, 1600, 600, 2400, 500",
    color: "#6366F1",
    width: 8,
    duration: 66,
    dash: "2000 1000",
    primary: true,
  },
  // Upper-right accent (Sky Blue)
  {
    id: 10,
    d: "M -100, 600 C 600, 500, 1600, 100, 2400, 150",
    color: "#38BDF8",
    width: 4,
    duration: 42,
    dash: "1000 2000",
    primary: false,
  },
  // --- New additions mimicking the reference veil & parallel grid lines ---
  // Massive trailing veil background swooping into right side
  {
    id: 11,
    d: "M 200, 1200 C 800, 900, 1500, 400, 2500, 0",
    color: "#7C3AED", 
    width: 25,
    duration: 90,
    dash: "3000 2000",
    primary: false,
  },
  // Sharp parallel grid line 1
  {
    id: 12,
    d: "M 200, 1150 C 780, 850, 1480, 380, 2480, -20",
    color: "#C084FC",
    width: 1.5,
    duration: 45,
    dash: "1500 1000",
    primary: true,
  },
  // Sharp parallel grid line 2
  {
    id: 13,
    d: "M 250, 1200 C 820, 900, 1520, 420, 2520, 20",
    color: "#C084FC",
    width: 1,
    duration: 48,
    dash: "1200 1500",
    primary: false,
  },
  // Sharp parallel grid line 3 (Pinkish tint)
  {
    id: 14,
    d: "M 150, 1100 C 750, 800, 1450, 350, 2450, -50",
    color: "#E879F9",
    width: 2,
    duration: 52,
    dash: "2000 1000",
    primary: true,
  },
  // Crossing swooping arc that creates the geometric "V" intersection in the reference
  {
    id: 15,
    d: "M 2200, 1100 C 1800, 800, 1700, 400, 2100, -100",
    color: "#8B5CF6", 
    width: 12,
    duration: 75,
    dash: "2500 1500",
    primary: false,
  },
  // --- New additions: Top-center swooping down to mid-right (Avoiding ASCII text) ---
  {
    id: 16,
    d: "M 400, 150 C 700, 300, 1200, 600, 2400, 500",
    color: "#7C3AED",
    width: 8,
    duration: 60,
    dash: "2000 1000",
    primary: false,
  },
  {
    id: 17,
    d: "M 450, 200 C 750, 350, 1250, 650, 2450, 550",
    color: "#C084FC",
    width: 2,
    duration: 48,
    dash: "1500 1000",
    primary: true,
  },
  {
    id: 18,
    d: "M 500, 250 C 900, 400, 1400, 500, 2200, 400",
    color: "#38BDF8",
    width: 3,
    duration: 54,
    dash: "1800 1200",
    primary: false,
  },
  // --- New additions: Sweeping high above the ASCII Text (Adjusted for visible viewBox slice) ---
  {
    id: 19,
    d: "M -100, 330 C 800, 200, 1600, 350, 2500, 200",
    color: "#38BDF8",
    width: 3,
    duration: 50,
    dash: "1500 1000",
    primary: false,
  },
  {
    id: 20,
    d: "M -100, 350 C 900, 250, 1800, 400, 2600, 250",
    color: "#A855F7",
    width: 6,
    duration: 60,
    dash: "2000 1500",
    primary: true,
  }
];

const GlassHUDCard = () => {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPhase((prev) => (prev + 1) % 3);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div 
      animate={{ y: [0, -10, 0] }}
      transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
      whileHover={{ rotateX: 5, rotateY: -5, scale: 1.02 }}
      style={{ perspective: 1000 }}
      className="absolute bottom-32 left-10 w-80 bg-[#050a19]/50 backdrop-blur-2xl border border-indigo-500/20 rounded-xl p-5 shadow-[0_0_30px_rgba(99,102,241,0.15)] z-20 pointer-events-auto cursor-default"
    >
      <AnimatePresence mode="wait">
        {phase === 0 && (
          <motion.div
            key="phase0"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col gap-2"
          >
            <div className="flex items-center gap-2 text-[10px] font-semibold text-purple-300 tracking-wider">
              <div className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
              DOCUMENT INGESTION
            </div>
            <div className="text-white text-sm font-medium mt-1">Q3_Report.pdf</div>
            <div className="text-zinc-400 text-xs">Indexing &middot; 35 pages</div>
          </motion.div>
        )}
        {phase === 1 && (
          <motion.div
            key="phase1"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col gap-2"
          >
            <div className="flex items-center gap-2 text-[10px] font-semibold text-blue-300 tracking-wider">
              <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
              SEMANTIC ANALYSIS
            </div>
            <div className="text-zinc-300 text-sm font-medium mt-1">Market &middot; Risk &middot; Q3 &middot; Growth</div>
            <div className="text-zinc-400 text-xs">Extracting context...</div>
          </motion.div>
        )}
        {phase === 2 && (
          <motion.div
            key="phase2"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.5 }}
            className="flex flex-col gap-2"
          >
            <div className="flex items-center gap-2 text-[10px] font-semibold text-emerald-300 tracking-wider">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              ANSWER GENERATED
            </div>
            <div className="text-white text-sm font-medium mt-1">94% confidence</div>
            <div className="text-zinc-400 text-xs italic">"Key risks in Q3 are..."</div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

const AsciiHeadline = () => (
  <motion.div 
    animate={{ opacity: [1, 0.85, 1, 0.9, 1] }}
    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
    className="absolute top-[25%] left-[10%] pointer-events-none select-none z-20"
  >
    <div className="relative text-indigo-200/90 font-mono whitespace-pre drop-shadow-[0_0_20px_rgba(99,102,241,0.8)]" style={{ fontSize: '11px', lineHeight: '1.2', letterSpacing: '0.15em' }}>
{`
██████╗  ██████╗  ██████╗██╗   ██╗███╗   ███╗██╗███╗   ██╗██████╗ 
██╔══██╗██╔═══██╗██╔════╝██║   ██║████╗ ████║██║████╗  ██║██╔══██╗
██║  ██║██║   ██║██║     ██║   ██║██╔████╔██║██║██╔██╗ ██║██║  ██║
██║  ██║██║   ██║██║     ██║   ██║██║╚██╔╝██║██║██║╚██╗██║██║  ██║
██████╔╝╚██████╔╝╚██████╗╚██████╔╝██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ 
`}
    </div>
    <div className="mt-4 flex flex-col gap-1.5 ml-2">
      <div className="text-[11px] font-mono text-indigo-300/80 tracking-[0.3em] uppercase">
        System Initialization Sequence
      </div>
      <div className="text-[10px] font-mono text-indigo-400/60 tracking-[0.2em] uppercase">
        V.9.4.1 // SECURE CONNECTION ESTABLISHED
      </div>
    </div>
  </motion.div>
);

export function LivingIntelligenceShowcase() {
  
  return (
    <div className="relative w-full h-full min-h-[600px] bg-transparent overflow-hidden flex items-center justify-center pointer-events-none">
      
      {/* Ascii Headline in the background */}
      <AsciiHeadline />

      {/* Glass HUD Card mapped to the 15s data particle loop */}
      <GlassHUDCard />

      {/* Bottom Branding (Untouched) */}
      <div className="absolute bottom-8 left-10 flex flex-col gap-1 z-30 pointer-events-none">
        <h2 className="text-lg font-medium text-zinc-200 tracking-tight">Intelligence from your documents.</h2>
        <p className="text-xs text-zinc-500 font-light tracking-wide">
          Neural search <span className="text-zinc-700 mx-1">&middot;</span> Semantic reasoning <span className="text-zinc-700 mx-1">&middot;</span> Agentic retrieval
        </p>
      </div>

    </div>
  );
}

export function AsciiHalftoneBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let time = 0;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", resize);
    resize();

    const chars = " .:-=+*#%@";
    
    let lastTime = 0;
    const fps = 15;
    const interval = 1000 / fps;
    
    const draw = (currentTime: number) => {
      animationFrameId = requestAnimationFrame(draw);
      
      const delta = currentTime - lastTime;
      if (delta < interval) return;
      lastTime = currentTime - (delta % interval);

      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.font = "18px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      const time = currentTime * 0.0005;
      const cols = Math.floor(canvas.width / 18);
      const rows = Math.floor(canvas.height / 18);

      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const nx = x / cols;
          const ny = y / rows;
          const wave1 = Math.sin(nx * 4 + time);
          const wave2 = Math.cos(ny * 3 + time * 0.8);
          const wave3 = Math.sin((nx + ny) * 6 - time * 1.2);
          
          let intensity = (wave1 + wave2 + wave3 + 3) / 6;
          const cx = nx - 0.5;
          const cy = ny - 0.5;
          const dist = Math.sqrt(cx * cx + cy * cy);
          intensity *= Math.max(0, 1 - dist * 1.2);

          if (intensity > 0.05) {
            const charIndex = Math.min(chars.length - 1, Math.floor(intensity * chars.length));
            const char = chars[charIndex];
            const r = 99 + intensity * 40;
            const g = 102 + intensity * 20;
            const b = 241 + intensity * 14;
            
            ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${intensity * 0.8})`;
            ctx.fillText(char, x * 18 + 9, y * 18 + 9);
          }
        }
      }
    };

    animationFrameId = requestAnimationFrame(draw);

    return () => {
      window.removeEventListener("resize", resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <motion.div 
      className="w-full h-full"
    >
      <canvas
        ref={canvasRef}
        className="w-full h-full pointer-events-none"
        style={{ imageRendering: "pixelated" }}
      />
    </motion.div>
  );
}

export function FadedBackgroundRibbons() {
  return (
    <svg 
      className="absolute inset-0 w-full h-full"
      viewBox="0 0 1000 1000" 
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        <radialGradient id="veil-fade" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="white" stopOpacity="1" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </radialGradient>
      </defs>
      
      <g opacity="0.8">
        {ribbons.map(ribbon => (
          <g key={`bg-ribbon-${ribbon.id}`}>
            <path
              d={ribbon.d}
              fill="none"
              stroke={ribbon.color}
              strokeWidth={ribbon.width * 1.5}
              strokeLinecap="round"
              className="opacity-20"
            />
            <path
              d={ribbon.d}
              fill="none"
              stroke={ribbon.color}
              strokeWidth={ribbon.width * 0.8}
              strokeLinecap="round"
              className="opacity-40"
            />
          </g>
        ))}
      </g>
    </svg>
  );
}
