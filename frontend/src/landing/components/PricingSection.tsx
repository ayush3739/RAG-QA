import { Check, X } from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { useStore } from '../../store/useStore';
import { navigate } from '../../lib/navigation';

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
  const setAuthMode = useStore((s) => s.setAuthMode);
  const isAuthenticated = useStore((s) => s.isAuthenticated);

  const handleAction = () => {
    if (isAuthenticated) {
      navigate('/dashboard');
    } else {
      setAuthMode('register');
      navigate('/signup');
    }
  };
  return (
    <section id="pricing" className="py-28 bg-background relative border-t border-border">
      <div className="max-w-7xl mx-auto px-6 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-foreground mb-4" style={{ fontFamily: "'DM Sans', sans-serif" }}>
            Simple, transparent <span className="text-primary">pricing</span>
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Choose the plan that best fits your scale. All plans include full access to our open-source core.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 max-w-4xl mx-auto gap-8">
          {tiers.map((tier) => (
            <Card 
              key={tier.name} 
              className={`relative flex flex-col h-full bg-surface border transition-all duration-300 ease-out hover:-translate-y-1.5 hover:shadow-2xl hover:shadow-primary/10 dark:hover:shadow-black/80 ${
                tier.popular 
                  ? 'border-primary hover:border-primary' 
                  : 'border-border hover:border-primary/60'
              }`}
            >
              {tier.popular && (
                <div className="absolute -top-3 inset-x-0 flex justify-center">
                  <span className="bg-primary text-primary-foreground text-[11px] font-semibold px-3 py-0.5 rounded-full shadow-sm">
                    Popular
                  </span>
                </div>
              )}
              
              <CardHeader className="p-8 pb-4">
                <CardTitle className="text-xl text-foreground mb-2">{tier.name}</CardTitle>
                <CardDescription className="text-muted-foreground h-10">{tier.description}</CardDescription>
              </CardHeader>
              
              <CardContent className="p-8 pt-0 flex-grow">
                <div className="mb-8 mt-4 flex items-end">
                  <span className="text-4xl font-bold text-foreground tracking-tight">{tier.price}</span>
                  {tier.period && <span className="text-muted-foreground mb-1 ml-1">{tier.period}</span>}
                </div>
                
                <ul className="space-y-4">
                  {tier.features.map((feature, i) => (
                    <li key={i} className="flex items-start text-sm">
                      {feature.included ? (
                        <Check className="w-5 h-5 text-primary mr-3 shrink-0" aria-hidden="true" />
                      ) : (
                        <X className="w-5 h-5 text-muted-foreground mr-3 shrink-0" aria-hidden="true" />
                      )}
                      <span className={feature.included ? 'text-foreground' : 'text-muted-foreground'}>
                        {feature.text}
                      </span>
                    </li>
                  ))}
                </ul>
              </CardContent>
              
              {tier.buttonText && (
                <CardFooter className="p-8 pt-0 mt-auto">
                  <Button 
                    onClick={handleAction}
                    className={`w-full cursor-pointer transition-colors duration-150 active:scale-[0.98] disabled:opacity-50 motion-reduce:transform-none focus-visible:ring-2 focus-visible:ring-primary ${
                      tier.popular 
                        ? 'bg-primary hover:bg-primary/90 text-primary-foreground' 
                        : 'bg-surface-container hover:bg-surface-container-high text-foreground border border-border'
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
