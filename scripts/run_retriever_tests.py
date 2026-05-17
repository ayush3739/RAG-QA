import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.core.retriever import Retriever
import traceback

# instantiate retriever for collection 'test.pdf'
r = Retriever('test.pdf')
queries = [
    "How do I insert documents into MongoDB using Mongoose?",
    "How can I send emails with SendGrid in the project?",
    "What does Lesson 9: Querying Documents cover?",
    "How does automatic scrolling work in the chat application?",
]

for q in queries:
    print('\n=== QUERY ===')
    print(q)
    print('--- ANSWER ---')
    try:
        ans = r.answer(q, k=10)
        print(ans)
    except Exception:
        traceback.print_exc()
        print('Error during retrieval/generation for query:', q)
