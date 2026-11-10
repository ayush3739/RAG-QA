import { motion } from 'framer-motion';
import { Database, Code2, Server, Zap, Layers, Boxes, TerminalSquare, Cpu } from 'lucide-react';

const technologies = [
  { name: 'React', icon: Code2 },
  { name: 'Tailwind CSS', icon: Zap },
  { name: 'FastAPI', icon: TerminalSquare },
  { name: 'PostgreSQL', icon: Database },
  { name: 'pgvector', icon: Layers },
  { name: 'Python', icon: Server },
  { name: 'Framer Motion', icon: Boxes },
  { name: 'LLM Routing', icon: Cpu },
];

export default function LogoMarquee() {
  // Duplicate the array to create a seamless infinite loop
  const duplicatedTech = [...technologies, ...technologies, ...technologies];

  return (
    <section className="py-16 border-y border-border bg-background overflow-hidden">
      <div className="max-w-7xl mx-auto px-6 mb-8 text-center">
        <p className="text-xs font-medium text-muted-foreground">
          Powered by industry-standard open source
        </p>
      </div>
      
      <div className="relative flex w-full overflow-hidden">
        {/* Gradient Masks for smooth fade on edges */}
        <div className="absolute left-0 top-0 bottom-0 w-32 bg-gradient-to-r from-background to-transparent z-10 pointer-events-none" />
        <div className="absolute right-0 top-0 bottom-0 w-32 bg-gradient-to-l from-background to-transparent z-10 pointer-events-none" />

        <motion.div
          className="flex items-center gap-16 whitespace-nowrap"
          animate={{ x: ["0%", "-50%"] }}
          transition={{
            repeat: Infinity,
            ease: "linear",
            duration: 30, // Adjust speed here
          }}
        >
          {duplicatedTech.map((tech, index) => {
            const Icon = tech.icon;
            return (
              <div 
                key={index} 
                className="flex items-center gap-3 text-muted-foreground hover:text-foreground transition-colors duration-150"
              >
                <Icon className="w-6 h-6" aria-hidden="true" />
                <span className="text-sm font-medium">{tech.name}</span>
              </div>
            );
          })}
        </motion.div>
      </div>
    </section>
  );
}
