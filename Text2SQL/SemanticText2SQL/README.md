# 📚 SemanticText2SQL: Advanced Natural Language Database Querying

A **revolutionary AI-powered system** that combines vector embeddings, fuzzy matching, and traditional SQL to answer **any question** about your database in natural language.

🎥 **[Watch the Full Tutorial on YouTube](https://youtu.be/OZ4BUW4TmsI)** - Complete walkthrough and demonstration

---

## 🌟 Why This System is Truly Advanced

### The Problem with Traditional Database Access

Traditional database systems have fundamental limitations:
- ❌ **Requires SQL expertise** - Users must know complex syntax
- ❌ **No typo tolerance** - "Stephan King" returns 0 results instead of "Stephen King"
- ❌ **Can't understand concepts** - Searching for "dystopia" won't find books about totalitarianism
- ❌ **Brittle and inflexible** - One spelling mistake breaks everything

### The Solution: A Three-Pillar Approach

This system combines **three complementary technologies** to create truly intelligent database querying:

#### 1️⃣ **Traditional SQL Filters** → Precision & Speed
```sql
WHERE publication_date > '2010-01-01' AND retail_price < 20
```
- ✅ Fast, indexed lookups
- ✅ Exact numeric/date comparisons
- ✅ Complex boolean logic
- **Best for**: Structured data (prices, dates, IDs, flags)

#### 2️⃣ **Levenshtein Fuzzy Matching** → Typo Tolerance
```sql
WHERE levenshtein(LOWER(author_name), LOWER('Orrwell')) <= 2
```
- ✅ Handles spelling errors (1-3 character differences)
- ✅ Works with any VARCHAR field
- ✅ Ranks by similarity (closest matches first)
- **Best for**: Names, titles, categories when user might misspell

#### 3️⃣ **Vector Embeddings** → Semantic Understanding
```sql
WHERE book_description_embed <-> embedding("dystopian themes") < 0.5
```
- ✅ Understands **meaning**, not just keywords
- ✅ Finds conceptually similar content
- ✅ Works across languages and synonyms
- **Best for**: Thematic searches, similarity, recommendations

### Why This Combination Changes Everything

Most systems use **only ONE approach**. This system uses **ALL THREE INTELLIGENTLY**:

| System Type | Handles Typos? | Understands Concepts? | Exact Filters? | Real Example |
|-------------|----------------|----------------------|----------------|--------------|
| **Traditional SQL-only** | ❌ No | ❌ No | ✅ Yes | "Find books WHERE author = 'Orwell'" (breaks if misspelled) |
| **Embeddings-only** | ⚠️ Sometimes | ✅ Yes | ❌ No | Can't filter "books under $20 published in 2020" |
| **Fuzzy-only** | ✅ Yes | ❌ No | ⚠️ Limited | Can't find "books about dystopia" (concept) |
| **🚀 THIS SYSTEM** | ✅ Yes | ✅ Yes | ✅ Yes | **Handles ANY question combining all three!** |

### Real-World Example: The Power of Combination

```
User Question (in Italian):
"Trova libri distopici simili a 1984 di autori con nome che finisce in 
'well', pubblicati dopo 2000 da editori inglesi, prezzo tra $12-$18, 
con recensioni che parlano di libertà"

System Processing:
┌─────────────────────────────────────────────────────────────┐
│ 🌍 TRANSLATION (Multi-language support)                     │
│ "Trova" → "Find", "distopici" → "dystopian", etc.           │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 🧠 SEMANTIC UNDERSTANDING                                   │
│ "similar to 1984" → embedding("totalitarian surveillance")  │
│ "about freedom" → embedding("liberty freedom rights")       │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 🔍 FUZZY MATCHING                                           │
│ "nome che finisce in 'well'" → LIKE '%well'                 │
│ (Tolerates typos in author names)                           │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 📊 SQL FILTERS                                              │
│ publication_date > '2000-01-01'                             │
│ retail_price BETWEEN 12 AND 18                              │
│ publisher.country = 'UK'                                    │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
                    SINGLE OPTIMIZED QUERY
                  (Combines all strategies)
                            ▼
                  Natural Language Answer:
"I found 2 dystopian books similar to '1984' by authors whose 
names end in 'well': [results with details]..."
```

**One question. One query. Perfect results.** Even with typos, in another language, with complex criteria.

### Universal Question Answering

This system can answer **virtually ANY question** about your database:

| Question Type | Example | What Makes It Hard | How This System Handles It |
|--------------|---------|-------------------|---------------------------|
| **Simple exact** | "Books published in 2020" | None | ✅ SQL filter |
| **With typos** | "Books by Stephan King" | Spelling error | ✅ Levenshtein distance |
| **Conceptual** | "Books about artificial intelligence" | No exact keyword "AI" in data | ✅ Vector embedding similarity |
| **Similarity** | "Books like Harry Potter" | Subjective "like" | ✅ Embedding of HP themes |
| **Fuzzy + Filters** | "Books by Tolkein under $15" | Typo + price filter | ✅ Levenshtein + SQL |
| **Semantic + Filters** | "Dystopian books after 2010" | Concept + date | ✅ Embeddings + SQL |
| **All combined** | "Fantasy by Pratchet, UK publishers, $12-$18, with magic in reviews" | Everything at once | ✅ **All three techniques!** |
| **Ultra-complex** | "Compare books similar to both '1984' AND 'Brave New World', authors with 'well'/'ley' in name, literary publishers, $12-$18, reviews about social commentary, last 30 years" | Multiple embeddings, fuzzy patterns, filters, aggregations | ✅ **Handles perfectly!** |

### What Makes This System Truly Revolutionary

#### 🎯 **Intelligent Strategy Selection**
The LLM **automatically chooses** the right technique for each part of your question:
- Recognizes when fuzzy matching is needed (typos)
- Detects semantic intent (concepts, themes, similarity)
- Applies exact filters where appropriate (numbers, dates)
- **Combines them seamlessly** in a single query

#### 🔄 **Self-Healing with Retry**
If a query fails (wrong syntax, missing GROUP BY, etc.):
1. System captures the error
2. Feeds it back to the LLM with full context
3. LLM analyzes ALL previous failures
4. Generates corrected query
5. **Up to 4 attempts** until success

**Real example:**
```
Attempt 1: Missing GROUP BY → PostgreSQL error
Attempt 2: Added GROUP BY with vector column → Validation blocks it
Attempt 3: Removed vector, fixed all issues → SUCCESS ✓
```

#### 🔒 **Production-Grade Security**
Every query is validated before execution:
- ✅ Only SELECT queries (read-only)
- ✅ No SQL injection possible
- ✅ No write/delete/drop operations
- ✅ Structural validation (no invalid syntax reaches DB)

#### 🌍 **True Multi-Language Support**
Ask in **any language**, get accurate results:
```
English:  "Find science fiction books"
Italian:  "Trova libri di fantascienza"
Spanish:  "Busca libros de ciencia ficción"
French:   "Trouve des livres de science-fiction"

→ All translate to: category = 'Science Fiction'
→ All return the same accurate results
```

### The Result: Answer ANY Question

With this system, users can ask questions like:

✅ "Show me books" → Simple list  
✅ "Books by Orwell" → Author filter  
✅ "Books by Orrwell" → Fuzzy match (typo)  
✅ "Books about dystopia" → Semantic search  
✅ "Books similar to 1984" → Embedding similarity  
✅ "Dystopian books by Orrwell under $20" → Fuzzy + Semantic + Filter  
✅ "Compare fantasy books similar to Harry Potter and Lord of the Rings, by British authors, published after 2000, under $25, with reviews mentioning magic" → **Everything combined**  

**No SQL knowledge needed. No exact spelling required. No question too complex.**

This is database interaction reimagined for the AI era. 🚀

---

## 🚀 Quick Start

### Prerequisites
- Docker
- Docker Compose
- Python 3.8+ (for AI agent)
- Ollama installed locally
- Ollama model: `qwen3:4b-instruct`

### Starting the Database

```bash
# Start the PostgreSQL container
docker-compose up -d

# Check if the container is running
docker-compose ps

# View logs
docker-compose logs -f
```

### Connecting to the Database

**Connection Details:**
- **Host:** localhost
- **Port:** 5432
- **Database:** offerte_ristorazione
- **Username:** bookadmin
- **Password:** bookpass123

**Using psql:**
```bash
docker exec -it books_database psql -U bookadmin -d offerte_ristorazione
```

**Using connection string:**
```
postgresql://bookadmin:bookpass123@localhost:5432/offerte_ristorazione
```

### Stopping the Database

```bash
# Stop the container
docker-compose down

# Stop and remove data volumes (⚠️ deletes all data)
docker-compose down -v
```

## 🐍 Python Environment Setup

### Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Local LLM via Ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3:4b-instruct
```

Download the model once:

```bash
ollama pull qwen3:4b-instruct
```

### Deactivating Virtual Environment

```bash
# Deactivate when done
deactivate
```

---

## 🤖 Text-to-SQL Agent System

This project includes an advanced **AI-powered Text-to-SQL Agent** that converts natural language questions into SQL queries, executes them, and generates natural language answers.

### System Overview

The Text-to-SQL system is a complete pipeline that:

1. **Accepts natural language questions** in multiple languages
2. **Generates optimized SQL queries** using LLM
3. **Validates queries** for security and correctness
4. **Executes queries** against the PostgreSQL database
5. **Generates natural language answers** from results

### Key Features

#### 🧠 **Semantic Search with Vector Embeddings**
- Uses **pgvector** extension for similarity searches
- Supports **OpenAI text-embedding-3-small** (1536 dimensions)
- Semantic understanding of concepts, themes, and content
- Handles queries like: *"Find books similar to '1984'"*

#### 🔍 **Fuzzy Matching with Levenshtein Distance**
- Tolerates **typos and spelling errors** in user queries
- Uses **fuzzystrmatch** PostgreSQL extension
- Applied to: titles, author names, publisher names, categories
- Handles queries like: *"Find books by George Orrwell"* (with typo)

#### 🔄 **Intelligent Retry Mechanism**
- Automatically retries failed queries up to **4 times**
- Provides **comprehensive error feedback** to the LLM
- Learns from previous failures to avoid repeating mistakes
- Distinguishes between security issues and fixable errors

#### 🔒 **SQL Security & Validation**
- Uses **sqlglot** for SQL parsing and validation
- **Blocks write operations** (INSERT, UPDATE, DELETE, DROP, etc.)
- Prevents **SQL injection** attacks
- Validates query structure before execution

#### 🌍 **Multi-Language Support**
- Accepts queries in **English, Italian, Spanish, French**, etc.
- Automatically **translates** search terms to English (database language)
- Example: *"Trova libri di fantascienza"* → searches for "science fiction"

#### 📊 **Complex Query Support**
- Multi-table JOINs
- Aggregations (COUNT, AVG, SUM, MIN, MAX)
- Date range filters
- Price range filters
- Combined fuzzy + semantic + filters

### System Architecture

![System Architecture Flow](agent_flow.png)

### Database Extensions

The system requires these PostgreSQL extensions:

1. **vector** - For semantic similarity search with embeddings
2. **fuzzystrmatch** - For Levenshtein distance (typo tolerance)
3. **uuid-ossp** - For UUID generation

All extensions are automatically installed via `init-db.sql`.

### How It Works

#### 1. Query Understanding

When you ask a question, the LLM analyzes:
- **Intent**: What is the user trying to find?
- **Strategy**: Should it use fuzzy matching, semantic search, or both?
- **Filters**: What conditions need to be applied?

Example:
```
User: "Find books by George Orrwell about dystopia"

Analysis:
- "George Orrwell" → Fuzzy match (typo in "Orwell")
- "about dystopia" → Semantic search (concept)
- Strategy: Combine both
```

#### 2. SQL Generation

The LLM generates SQL following strict rules:

**For Fuzzy Matching** (VARCHAR fields with typos):
```sql
WHERE levenshtein(LOWER(last_name), LOWER('Orrwell')) <= 2
ORDER BY levenshtein(LOWER(last_name), LOWER('Orrwell'))
```

**For Semantic Search** (TEXT fields with concepts):
```sql
WHERE book_description_embed <-> %s::vector < 0.5
ORDER BY book_description_embed <-> %s::vector
```
- Placeholder `%s::vector` will be replaced with actual embedding

**For Combined Queries**:
```sql
SELECT b.title, a.last_name,
       levenshtein(LOWER(a.last_name), LOWER('Orrwell')) AS name_distance,
       b.book_description_embed <-> %s::vector AS content_similarity
FROM books b
JOIN authors a ON b.author_id = a.author_id
WHERE levenshtein(LOWER(a.last_name), LOWER('Orrwell')) <= 2
  AND b.book_description_embed IS NOT NULL
ORDER BY name_distance, content_similarity
LIMIT 15;
```

#### 3. Query Validation

Before execution, every query is validated:

✅ **Security Checks**:
- Must be a SELECT statement
- No INSERT, UPDATE, DELETE, DROP, CREATE, etc.
- No multiple statements (SQL injection protection)
- No SELECT INTO operations

✅ **Structural Checks**:
- Parseable SQL syntax
- No vector columns in GROUP BY clause
- Proper use of aggregate functions

❌ **Rejected Examples**:
```sql
-- Rejected: Write operation
INSERT INTO books VALUES (...)

-- Rejected: Dangerous operation
DROP TABLE books;

-- Rejected: Vector in GROUP BY
SELECT title, AVG(rating)
FROM books
GROUP BY title, book_description_embed;  -- ❌ _embed cannot be in GROUP BY
```

#### 4. Embedding Generation

If the query needs semantic search:

1. **Extract search terms** from user question
2. **Generate embeddings** using OpenAI API
3. **Format as PostgreSQL vectors**: `[0.123, 0.456, ...]`
4. **Substitute into query** replacing `%s::vector` placeholders

Example:
```
User: "Books about dystopia"
→ Embedding text: "dystopian totalitarian surveillance oppression"
→ Vector: [0.123, 0.456, ..., 0.789]  (1536 dimensions)
→ Query: book_description_embed <-> '[0.123, 0.456, ..., 0.789]'::vector
```

#### 5. Query Execution

The system:
1. **Connects** to PostgreSQL database
2. **Substitutes parameters** (manual substitution for complex queries)
3. **Executes query**
4. **Fetches results** as list of dictionaries
5. **Handles errors** with detailed logging

#### 6. Retry Mechanism

If execution fails:

```
Attempt 1: Initial query generation
    ↓ [FAIL] → Error: "column X must appear in GROUP BY"
    
Attempt 2: Regenerate with error feedback
    ↓ [FAIL] → Error: "syntax error near Y"
    
Attempt 3: Regenerate with ALL previous errors
    ↓ [FAIL] → Error: "type mismatch"
    
Attempt 4: Final attempt with complete history
    ↓ [SUCCESS] ✓
```

The LLM receives:
- Original user request
- Complete history of all failed attempts
- Specific error messages for each failure
- Instructions to avoid repeating same mistakes

#### 7. Answer Generation

After successful execution:
- LLM receives the **original question** + **query results**
- Generates a **clear, conversational answer**
- Handles edge cases (no results, errors, etc.)

Example:
```
User: "How many books by Stephen King?"
Results: [{"count": 7}]
Answer: "There are 7 books by Stephen King in the database."
```

### Usage Examples

#### Basic Usage

```python
from text_to_sql_agent import AgentTextToSql

# Initialize agent
agent = AgentTextToSql()

# Process a question
result = agent.process_request_with_execution(
    "Find books similar to 1984"
)

if result['success']:
    print(f"Answer: {result['final_answer']}")
    print(f"Retrieved {result['query_results']['row_count']} books")
else:
    print(f"Error: {result['error']}")
```

#### Interactive Mode

```bash
# Run interactive mode
python main.py --interactive

# Then ask questions in natural language:
📝 Your question: Find books by Terry Pratchet about fantasy
📝 Your question: Quanti libri ci sono di fantascienza?
📝 Your question: Books similar to Harry Potter under $20
```

#### Example Queries

**Simple Search:**
```
"Find all books by George Orwell"
→ SQL: Uses fuzzy matching on author name
→ Result: Books by George Orwell
```

**Semantic Search:**
```
"Find books about artificial intelligence"
→ SQL: Uses vector embeddings for concept search
→ Result: Books discussing AI/ML themes
```

**Combined Search:**
```
"Find dystopian books by Margret Atwood under $20"
→ SQL: Fuzzy on name + semantic on content + price filter
→ Result: Atwood's dystopian novels under $20
```

**Complex Multi-Criteria:**
```
"Show highly rated fantasy books published after 2010 
 by UK publishers with reviews mentioning magic"
→ SQL: Multi-JOIN + semantic + filters + aggregations
→ Result: Filtered and ranked results
```

### Query Strategy Selection

The system intelligently chooses strategies:

| User Query Type | Strategy | Example |
|----------------|----------|---------|
| Exact data (names, dates, IDs) | **Traditional SQL** | "Books published in 2020" |
| Text with possible typos | **Levenshtein** | "Books by Stephan King" |
| Concepts and themes | **Semantic (embeddings)** | "Books about dystopia" |
| Similarity requests | **Semantic** | "Books similar to 1984" |
| Mixed requirements | **Combined** | "Dystopian books by Orwell" |

### Result Limits

The system enforces smart limits:

- **Semantic searches**: Default 15-20 results
- **General lists**: Default 50 results
- **Hard maximum**: 100 results (unless explicitly requested)
- **User-specified**: Honors explicit counts ("top 5 books")

### Configuration

#### Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here
```

#### Database Configuration

Default configuration (in `text_to_sql_agent.py`):

```python
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'books_db',
    'user': 'bookadmin',
    'password': 'bookpass123'
}
```

#### Model Configuration

```python
# Initialize with custom settings
agent = AgentTextToSql(
    model="gpt-4.1",             
    temperature=0.1,             # Default: 0.1 (low for consistency)
    db_config=custom_db_config   # Optional custom DB config
)
```

### Running the System

#### 1. Generate Embeddings (First Time)

Before using semantic search, generate embeddings for the database:

```bash
# Activate virtual environment
source venv/bin/activate

# Generate embeddings for all text fields
python gen_embeddings.py
```

This will:
- Scan all tables for `_embed` columns
- Generate embeddings for corresponding text fields
- Update the database with vector values
- Show progress and estimated cost

#### 2. Run Example Queries

```bash
# Run predefined examples
python main.py

# Examples include:
# - Traditional SQL (no embeddings)
# - Semantic search (with embeddings)
# - Combined queries
# - Aggregations
```

#### 3. Interactive Mode

```bash
# Start interactive session
python main.py --interactive

# Ask questions in natural language
# Type 'quit' or 'exit' to stop
```

### Advanced Features

#### Retry with Error Feedback

The system automatically handles query failures:

```python
# If a query fails, the system:
1. Captures the error message
2. Sends it back to LLM with the original request
3. LLM analyzes the error and generates a corrected query
4. Repeats up to 4 times total

# Example retry scenario:
Attempt 1: GROUP BY missing required columns → FAIL
Attempt 2: Added GROUP BY but included vector column → FAIL  
Attempt 3: Removed vector from GROUP BY, fixed syntax → SUCCESS
```

#### Multi-Language Translation

```python
# Italian query
"Trova libri di fantascienza pubblicati dopo il 2010"
# Translated internally to:
# "Find science fiction books published after 2010"
# SQL uses: category_name = 'Science Fiction'
```

#### Parameter Replication

When queries reuse the same embedding:

```sql
-- Query uses same embedding twice
((book_embed <-> %s::vector) + (book_embed <-> %s::vector)) / 2

-- System automatically:
1. Detects 2 placeholders
2. Generates 1 embedding
3. Replicates it for both placeholders
4. Substitutes correctly
```

### Testing

See `QUESTIONS.md` for 30 comprehensive test questions covering:
- Fuzzy matching with typos
- Semantic search for concepts
- Multi-table joins
- Aggregations and statistics
- Complex combined queries

### Architecture Files

| File | Purpose |
|------|---------|
| `text_to_sql_agent.py` | Main agent class with pipeline logic |
| `prompt.py` | All LLM prompts and templates |
| `utils.py` | Database schema extraction utilities |
| `gen_embeddings.py` | Batch embedding generation script |
| `main.py` | Example usage and interactive mode |
| `init-db.sql` | Database schema with sample data |
| `QUESTIONS.md` | 30 test questions for validation |

### Performance Considerations

**Embedding Generation:**
- Cost: ~$0.020 per 1M tokens (text-embedding-3-small)
- Speed: ~300-500ms per embedding
- Batch generation recommended for initial setup

**Query Execution:**
- Simple queries: <100ms
- Semantic searches: 500ms-2s (including embedding generation)
- Complex queries: 1-5s (multiple embeddings + aggregations)

**LLM API Calls:**
- Query generation: ~2-5s
- Answer generation: ~1-3s
- Retry regeneration: ~2-5s per attempt

### Error Handling

The system handles various error types:

**Security Errors** (immediate abort):
```
Error: Dangerous operation detected: INSERT
→ No retry, query rejected
```

**Fixable Errors** (retry with feedback):
```
Error: Vector columns cannot be in GROUP BY
→ Retry with feedback to LLM
→ LLM removes vector from GROUP BY
→ Success on retry
```

**Execution Errors** (retry with feedback):
```
Error: column "xyz" does not exist
→ Retry with feedback
→ LLM corrects column name
→ Success on retry
```

### Limitations

- **Maximum 100 results** by default (hard limit for performance)
- **Complex queries** may require multiple retry attempts
- **Embeddings required** for semantic search (run `gen_embeddings.py` first)
- **English database** - queries in other languages are translated
- **Read-only** - only SELECT queries allowed

### Best Practices

1. **Generate embeddings first** before using semantic search
2. **Be specific** in questions for better results
3. **Use natural language** - the system understands conversational queries
4. **Check retry info** if queries take multiple attempts
5. **Review generated SQL** to understand the strategy used

### Troubleshooting

**Issue: "Embeddings not found"**
```bash
# Solution: Generate embeddings
python gen_embeddings.py
```

**Issue: "Query validation failed"**
- Check if you're trying write operations (not allowed)
- Review the error message for specific issue

**Issue: "Multiple retries, still failing"**
- Query might be too complex
- Try simplifying the question
- Check `failed_attempts` in result for details

**Issue: "No results found"**
- Criteria might be too restrictive
- Try relaxing some filters
- Check if embeddings are generated

