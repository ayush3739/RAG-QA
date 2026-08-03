For testing your RAG system, don't just use simple questions. Use a mix of:

### Easy Retrieval Questions

These should be answered from a single chunk.

1. What is the first rule discussed in the document?
2. Who is the author of this ebook?
3. What is the 20-minute rule?
4. What does the author compare learning to code with?
5. According to the document, how many hours did the author spend at the hospital while working as a doctor?
6. What is Rule 3 called?
7. What does the author say about the "perfect programming language"?
8. What website is mentioned throughout the document?
9. What is Imposter Syndrome?
10. What percentage of people are said to experience Imposter Syndrome? 

---

### Multi-Chunk Questions

These require information from multiple sections.

1. Why does the author believe coding projects are more motivating than coding exercises?
2. How are the 20-minute rule and habit formation connected?
3. What advice does the author give for choosing a programming language?
4. Why should beginners avoid tutorials that jump from beginner to advanced concepts?
5. What qualities does the author think employers value most in programmers?
6. How does the document suggest handling programming problems you cannot solve?
7. Compare Rule 2 and Rule 4. How do they help beginners learn more effectively?
8. What are the common mistakes self-taught programmers make according to the document?

---

### Summarization Questions

Good for testing context synthesis.

1. Summarize the main ideas of this document.
2. Summarize the first four rules in less than 150 words.
3. What are the key lessons a beginner programmer should learn from this ebook?
4. Give a chapter-by-chapter summary of the document.
5. What recurring themes appear throughout the ebook?

---

### Questions That Should Return "Not Found"

A good RAG must refuse these.

1. What is the author's favorite programming language?
2. What year was this ebook published?
3. What salary do Google software engineers earn?
4. What are the latest features of Python 3.15?
5. What operating system does Angela Yu use?
6. What is the best laptop for programming according to the document?

Expected answer:

```text
I could not find this information in the provided document.
```

---

### Hard Questions

These are excellent RAG tests.

1. Why does the author believe internal motivation is important for learning programming?
2. How does the author use the tool analogy to explain programming languages?
3. Why does the author think memorizing code is less important than problem-solving ability?
4. What is the relationship between habit formation and coding consistency?
5. According to the document, what makes a tutorial suitable for beginners?
6. How does the concept of "ramping" relate to long-term programming growth?
7. Why does the author encourage using Google and Stack Overflow while learning?
8. What evidence does the author provide that professional programmers frequently use references?

---

### Stress-Test Questions

These expose retrieval weaknesses.

1. What coding method does the author recommend for maintaining motivation?
2. What should a beginner do when they encounter bugs they don't understand?
3. Why does the author mention Game of Thrones and ice cream?
4. What examples of student-built apps are given in the ebook?
5. How does the author recommend selecting a programming language for a project?
6. What does the author say about the value of information versus the value of thinking?
7. Explain the author's views on Stack Overflow.
8. What are all 12 rules mentioned in the document? 

If your RAG can answer most of the "Hard Questions" correctly and refuses the "Not Found" questions instead of hallucinating, then the retrieval and grounding are working well.
