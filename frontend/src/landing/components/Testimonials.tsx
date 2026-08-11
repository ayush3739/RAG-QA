import { motion } from 'framer-motion';

const testimonials = [
  {
    content: "The agentic routing is absurdly fast. We dropped DocuMind into our stack and saw a 40% reduction in hallucination rates immediately.",
    author: "Sarah Chen",
    role: "Staff Engineer",
    company: "DataScale",
    image: "https://i.pravatar.cc/150?u=sarah"
  },
  {
    content: "We evaluated a lot of RAG frameworks, but nothing matched the sub-chunk citation mapping out of the box like this does. Beautiful architecture.",
    author: "Marcus Johnson",
    role: "CTO",
    company: "NexusHealth",
    image: "https://i.pravatar.cc/150?u=marcus"
  },
  {
    content: "Having pgvector and document metadata in a single Postgres database makes scaling infinitely easier. No more vector DB synchronization nightmares.",
    author: "Elena Rodriguez",
    role: "Lead Architect",
    company: "FinTech Solutions",
    image: "https://i.pravatar.cc/150?u=elena"
  },
  {
    content: "The API is incredibly clean. We were able to pipe the streaming JSON into our React frontend in under an hour. Highly recommend.",
    author: "David Kim",
    role: "Frontend Developer",
    company: "Streamline",
    image: "https://i.pravatar.cc/150?u=david"
  },
  {
    content: "Finally, a RAG system that understands that retrieving the context is only half the battle. The routing layer is where the magic happens.",
    author: "Alex Morgan",
    role: "AI Researcher",
    company: "Cognitive Labs",
    image: "https://i.pravatar.cc/150?u=alex"
  },
  {
    content: "DocuMind's latency is unmatched. Hitting sub-150ms on complex queries with reciprocal rank fusion is seriously impressive engineering.",
    author: "James Wilson",
    role: "Backend Engineer",
    company: "Velocity",
    image: "https://i.pravatar.cc/150?u=james"
  }
];

export default function Testimonials() {
  return (
    <section className="py-28 bg-[#050508] relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-6 relative z-10">
        <div className="text-center mb-20">
          <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4" style={{ fontFamily: "'DM Sans', sans-serif" }}>
            Loved by <span className="text-indigo-400">engineers</span>
          </h2>
          <p className="text-zinc-400 max-w-xl mx-auto">
            See what technical teams are saying about building with our architecture.
          </p>
        </div>

        <div className="columns-1 md:columns-2 lg:columns-3 gap-6 space-y-6">
          {testimonials.map((testimonial, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
              className="break-inside-avoid bg-[#0c0c0e] border border-white/[0.05] rounded-2xl p-6 shadow-xl hover:border-indigo-500/30 transition-colors duration-300"
            >
              <div className="flex items-center gap-1 mb-4 text-indigo-400">
                {[...Array(5)].map((_, i) => (
                  <svg key={i} className="w-4 h-4 fill-current" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                ))}
              </div>
              <p className="text-zinc-300 text-[15px] leading-relaxed mb-6">
                "{testimonial.content}"
              </p>
              <div className="flex items-center gap-4">
                <img 
                  src={testimonial.image} 
                  alt={testimonial.author} 
                  className="w-10 h-10 rounded-full bg-zinc-800 object-cover"
                />
                <div>
                  <h4 className="text-sm font-medium text-white">{testimonial.author}</h4>
                  <p className="text-xs text-zinc-500">{testimonial.role} @ {testimonial.company}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
