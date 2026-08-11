import { Check, X } from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../../components/ui/card';
import { Button } from '../../components/ui/button';

const tiers = [
  {
    name: 'Community',
    description: 'Perfect for testing out the architecture and small personal projects.',
    price: 'Free',
    features: [
      { text: 'Up to 50 documents', included: true },
      { text: 'Basic RAG architecture', included: true },
      { text: 'Standard vector search', included: true },
      { text: 'Agentic routing', included: false },
      { text: 'Sub-chunk citation mapping', included: false },
    ],
    buttonText: 'Get Started',
    popular: false,
  },
  {
    name: 'Pro',
    description: 'For teams building production-grade RAG applications.',
    price: 'Coming Soon',
    period: '',
    features: [
      { text: 'Up to 5,000 documents', included: true },
      { text: 'Agentic query routing', included: true },
      { text: 'Hybrid search (BM25 + Vector)', included: true },
      { text: 'Sub-chunk citation mapping', included: true },
      { text: 'Priority support', included: false },
    ],
    buttonText: '',
    popular: true,
  },
];

export default function PricingSection() {
  return (
    <section id="pricing" className="py-28 bg-[#09090b] relative border-t border-white/[0.05]">
      {/* Background Glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[300px] bg-indigo-500/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="max-w-7xl mx-auto px-6 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4" style={{ fontFamily: "'DM Sans', sans-serif" }}>
            Simple, transparent <span className="text-indigo-400">pricing</span>
          </h2>
          <p className="text-zinc-400 max-w-xl mx-auto">
            Choose the plan that best fits your scale. All plans include full access to our open-source core.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 max-w-4xl mx-auto gap-8">
          {tiers.map((tier) => (
            <Card 
              key={tier.name} 
              className={`relative flex flex-col h-full bg-[#0c0c0e] border ${
                tier.popular 
                  ? 'border-indigo-500 shadow-[0_0_30px_rgba(99,102,241,0.15)]' 
                  : 'border-white/[0.08]'
              }`}
            >
              {tier.popular && (
                <div className="absolute -top-3 inset-x-0 flex justify-center">
                  <span className="bg-indigo-500 text-white text-[10px] font-bold tracking-widest uppercase px-3 py-1 rounded-full">
                    Most Popular
                  </span>
                </div>
              )}
              
              <CardHeader className="p-8 pb-4">
                <CardTitle className="text-xl text-white mb-2">{tier.name}</CardTitle>
                <CardDescription className="text-zinc-400 h-10">{tier.description}</CardDescription>
              </CardHeader>
              
              <CardContent className="p-8 pt-0 flex-grow">
                <div className="mb-8 mt-4 flex items-end">
                  <span className="text-4xl font-bold text-white tracking-tight">{tier.price}</span>
                  {tier.period && <span className="text-zinc-500 mb-1 ml-1">{tier.period}</span>}
                </div>
                
                <ul className="space-y-4">
                  {tier.features.map((feature, i) => (
                    <li key={i} className="flex items-start text-sm">
                      {feature.included ? (
                        <Check className="w-5 h-5 text-indigo-400 mr-3 shrink-0" />
                      ) : (
                        <X className="w-5 h-5 text-zinc-700 mr-3 shrink-0" />
                      )}
                      <span className={feature.included ? 'text-zinc-300' : 'text-zinc-600'}>
                        {feature.text}
                      </span>
                    </li>
                  ))}
                </ul>
              </CardContent>
              
              {tier.buttonText && (
                <CardFooter className="p-8 pt-0 mt-auto">
                  <Button 
                    className={`w-full ${
                      tier.popular 
                        ? 'bg-indigo-500 hover:bg-indigo-600 text-white' 
                        : 'bg-white/5 hover:bg-white/10 text-white border border-white/10'
                    }`}
                    variant={tier.popular ? 'default' : 'outline'}
                  >
                    {tier.buttonText}
                  </Button>
                </CardFooter>
              )}
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
