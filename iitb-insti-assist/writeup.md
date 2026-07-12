
# IITB Insti-Assist: Project Write-up

### 1. Chosen Scope and Rationale
**Scope:** Academic Life
**Rationale:** Navigating the dense administrative guidelines, academic policies, graduation requirements, and probation rules at IIT Bombay is notoriously difficult for undergraduate students. The rule book is lengthy and heavily cross-referenced, making quick information retrieval a challenge. An AI assistant dedicated to academic regulations directly addresses a high-friction area of student life, providing immediate, accurate administrative guidance without requiring students to manually sift through dozens of PDF pages.

### 2. Data Sources
To ensure comprehensive academic coverage, the following 5 real source documents were ingested into the vector database:
1. **IIT Bombay Undergraduate Rules & Regulations** (Core primary source containing grading policies, credit structures, and disciplinary rules).
2. **Academic Calendar 2023-2024** (For dates regarding course registration, drops, and exam periods).
3. **B.Tech Curriculum Structure** (Department-specific credit requirements).
4. **Minor and Honors Program Guidelines** (Rules for opting into additional academic tracks).
5. **Hostel Allocation Rules for Academic Year** (Intersecting rules for students facing academic probation/year-drops).

### 3. Chunking Strategy
**Method:** `RecursiveCharacterTextSplitter`
**Parameters:** Chunk Size = 1000 characters, Chunk Overlap = 200 characters.
**Technical Reasoning:** 
Legal and academic texts are structured logically into paragraphs and sub-clauses. Standard token splitting might cut a rule directly in half. The recursive character splitter attempts to split on logical boundaries (paragraphs `\n\n`, then sentences `\n`, then spaces) before defaulting to character limits. A chunk size of 1000 ensures that a complete policy or rule (along with its immediate sub-points) is captured within a single vector. The 200-character overlap prevents vital context from being lost if a long, continuous rule spills over into the next chunk, ensuring the LLM receives unbroken conceptual information.

### 4. Limitations & Future Enhancements
**Current Limitations & Edge Cases:**
* **Tabular Data:** The current PyPDF extraction struggles heavily with tables (e.g., credit requirement tables or grading rubrics). When chunked, the tabular formatting is destroyed, causing the LLM to misinterpret rows and columns.
* **Cross-Referencing:** If a rule states "Refer to Section 4.2," the vector retrieval might not pull Section 4.2 unless the semantic search explicitly links the current query to it.
* **Lack of Memory:** Because conversational memory was scoped out, users cannot ask follow-up questions like "What does that mean?" without restating the entire context of their previous query.

**Future Enhancements:**
If given more time, I would implement:
1. **Advanced PDF Parsing (e.g., Unstructured.io):** To accurately extract and preserve Markdown representations of tables and hierarchical headings.
2. **Conversational Memory:** Utilizing a `ConversationBufferMemory` setup to allow for natural, multi-turn follow-up questions.
3. **Metadata Filtering:** Allowing users to select which specific document they want to query (e.g., filtering strictly by the "Academic Calendar" rather than the entire database) to improve accuracy.
