import { SourceDocument } from "./types";

export const INITIAL_DOCUMENTS: SourceDocument[] = [
  {
    id: "doc-1",
    name: "fiscal_projections_q3_revised.xlsx",
    type: "spreadsheet",
    content: `
MONOLITH PROJECT - REVISED Q3 FISCAL PROJECTIONS
Prepared by: Financial Operations Division
Status: APPROVED

Summary of Key Financial Elements:
- Overall Q3 Projected Revenue: $14.8M (up 18% from Q2)
- Operational Efficiency Gains: 24% overall increase in workflow speed when paired with the Modular RAG Framework.
- Savings due to Latency Reduction: Estimated at $120,000 in cloud compute costs due to embedding caching.
- Monolith Project Budget Allocation:
  * R&D and AI Infrastructure: $4.2M
  * API Integration and Hosting: $1.8M
  * Client Support and Deployment: $1.1M
- Projected ROI for the RAG Implementation: 312% over 12 months.
    `,
    addedAt: "Added 2h ago",
    size: "1.2 MB",
    active: true
  },
  {
    id: "doc-2",
    name: "monolith_project_blueprint.pdf",
    type: "pdf",
    content: `
MONOLITH SYSTEM ARCHITECTURE BLUEPRINT V4.2
Confidential - For Internal Use Only

Core Framework: Modular RAG (Retrieval-Augmented Generation) Pipeline
This system architecture acts as the backbone of Deep Forest Intelligence. It is designed to scale across multi-disciplinary document sets securely.

Primary Technical Specifications:
1. Embedding Caching: By caching high-density semantic embeddings at the edge, the pipeline reduces overall document retrieval time by 120ms per query session.
2. Hierarchical Re-ranking: Improves result relevance by introducing a dual-stage neural ranker. Stage 1 retrieves top 100 documents, Stage 2 refines the set to the top 5 highest-relevance items before feeding into the context window.
3. Vector Retrieval Model: Uses high-dimensional cosine similarity indexing.
    `,
    addedAt: "Added 4d ago",
    size: "4.8 MB",
    active: true
  },
  {
    id: "doc-3",
    name: "whitepaper_2024_v2.pdf",
    type: "pdf",
    content: `
WHITE PAPER: HIGH-PERFORMANCE SEMANTIC INTEGRATION IN ENTERPRISE SYSTEMS
Published: March 2024
Author: Research Division

Abstract:
Enterprise RAG (Retrieval-Augmented Generation) systems often suffer from latency overheads and context retrieval noise. This paper outlines methods to solve these issues.

Key Discoveries:
- Context Window Optimization: Feeding precise, re-ranked snippets rather than raw paragraphs increases accuracy by 34%.
- Semantic Cache Hit Ratio: Achieving a cache hit ratio of over 85% yields sub-50ms vector query times.
- Efficiency Vector: The application of multi-stage validation ensures that generated text adheres to the ground truth provided by source documents.
    `,
    addedAt: "Added 1w ago",
    size: "3.2 MB",
    active: false
  },
  {
    id: "doc-4",
    name: "internal_wiki_efficiency",
    type: "link",
    content: `
DEEP FOREST INTERNAL WIKI - WORKSPACE EFFICIENCY METRICS
Section: Devops & Knowledge Base

Operational Speedups:
Deploying the Modular RAG Framework across the principal research directories has successfully triggered a 24% increase in research discovery rates and operational execution.
- Embedding generation latency: reduced to 18ms.
- Average retrieval query latency: reduced by 120ms.
- User feedback indicates a 94% satisfaction score regarding the accuracy of context citation.
    `,
    addedAt: "Added 2w ago",
    size: "24 KB",
    active: false
  }
];
