#!/usr/bin/env python3
"""
Text-to-SQL Agent for local LLM usage.
Professional agent for converting natural language queries to SQL using Ollama models.
"""

import os
import re
import json
import logging
import requests
import psycopg2
import psycopg2.extensions
import sqlglot
import traceback
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from utils import generate_db_schema
from prompt import (
    create_text_to_sql_prompt, 
    create_final_answer_prompt, 
    create_sql_retry_prompt,
    create_sql_retry_prompt_for_phase,
    create_final_answer_user_message
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "qwen3:4b-instruct"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_BROAD_RESULT_THRESHOLD = 200


class AgentTextToSql:
    """
    Professional Text-to-SQL Agent that converts natural language queries to SQL.
    
    This agent uses a local Ollama model to understand user intent and generate
    accurate SQL queries based on the database schema.
    """
    
    # Default database configuration
    DEFAULT_DB_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'offerte_ristorazione',
        'user': 'bookadmin',
        'password': 'bookpass123'
    }

    GENERIC_QUERY_TERMS = {
        "macchina",
        "macchine",
        "prodotto",
        "prodotti",
        "articolo",
        "articoli",
        "elemento",
        "elementi",
        "capacità",
        "capacita",
        "capienza",
        "dimensione",
        "dimensioni",
        "tipo",
        "tipologia",
        "categoria",
        "potenza",
        "funzione",
        "funzioni",
        "tensione",
        "corrente",
        "materiale",
        "materiali",
        "colore",
        "sopra",
        "sotto",
        "vicino",
        "piu",
        "più",
        "meno",
        "grande",
        "grandi"
        "LT", 
        "litri",
        "cm",
        "centimetri",
        "mm",
        "millimetri",
        "kg",
        "chilogrammi",
        "g",
        "grammi",
        "m",
        "metri"
    }
    
    def __init__(self, db_config: Dict[str, Any] = None, model: str = DEFAULT_LOCAL_MODEL, temperature: float = 0.0):
        """
        Initialize the Text-to-SQL Agent.
        
        Args:
            db_config: Database configuration dictionary (optional, uses default if not provided)
            model: Ollama model to use (default: qwen3:4b-instruct)
            temperature: Model temperature for response generation (default: 0.0 for consistency)
        """
        self.model = os.getenv("LLM_MODEL", model)
        self.temperature = temperature
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.broad_result_threshold = int(os.getenv("BROAD_RESULT_THRESHOLD", str(DEFAULT_BROAD_RESULT_THRESHOLD)))
        self.database_schema = None
        if db_config:
            self.db_config = db_config
        else:
            # Allow environment overrides while keeping working local defaults.
            self.db_config = {
                'host': os.getenv('PGHOST', self.DEFAULT_DB_CONFIG['host']),
                'port': int(os.getenv('PGPORT', str(self.DEFAULT_DB_CONFIG['port']))),
                'database': os.getenv('PGDATABASE', self.DEFAULT_DB_CONFIG['database']),
                'user': os.getenv('PGUSER', self.DEFAULT_DB_CONFIG['user']),
                'password': os.getenv('PGPASSWORD', self.DEFAULT_DB_CONFIG['password'])
            }
        
        # Initialize local LLM settings
        self._initialize_llm_client()
        
        # Load database schema
        self._load_database_schema()
    
    def _initialize_llm_client(self) -> None:
        """Initialize local LLM settings (Ollama)."""
        logger.info(
            "Using local Ollama model '%s' at %s",
            self.model,
            self.ollama_base_url,
        )
    
    def _load_database_schema(self) -> None:
        """Generate database schema directly from database using utils."""
        try:
            # Connect to database
            connection = psycopg2.connect(**self.db_config)
            
            # Generate schema
            formatted_text, json_data = generate_db_schema(connection)
            self.database_schema = formatted_text
            
            # Close connection
            connection.close()
            
            logger.info("Database schema generated successfully")
            
        except Exception as e:
            logger.error(f"Error loading database schema: {str(e)}")
            raise
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for the Text-to-SQL agent.
        
        Returns:
            str: System prompt for LLM API
        """
        return create_text_to_sql_prompt(self.database_schema)

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        """Extract a JSON object from model output, handling code fences if present."""
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _call_local_llm(self, system_prompt: str, user_prompt: str, temperature: float, expect_json: bool) -> str:
        """Call Ollama chat endpoint and return model text."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1200,
            },
        }

        if expect_json:
            payload["format"] = "json"

        response = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "")
        if not content:
            raise ValueError("Empty response from local model")

        return content

    def _sanitize_user_request_for_sql(self, user_request: str) -> str:
        """
        Remove generic low-signal terms from user request before SQL generation.

        This avoids over-constraining queries with words like "macchine" that are
        too generic and often reduce recall.
        """
        if not user_request:
            return user_request

        tokens = user_request.split()
        cleaned_tokens: List[str] = []
        for token in tokens:
            normalized = re.sub(r"[^\w]", "", token, flags=re.UNICODE).lower()
            if normalized in self.GENERIC_QUERY_TERMS:
                continue
            cleaned_tokens.append(token)

        sanitized = " ".join(cleaned_tokens).strip()
        if sanitized and sanitized != user_request:
            logger.info("Sanitized user request for SQL generation: %s", sanitized)
            return sanitized

        return user_request

    def _enforce_specifiche_descrizione_pairing(self, sql_query: str) -> str:
        """
        Enforce a hard-coded rule for text filters:
        every `specifiche ILIKE '...` predicate must also include the same
        token on `descrizione` with an OR clause.

        This normalization is intentionally deterministic to avoid relying
        only on model compliance.
        """
        pattern = re.compile(r"(?<![\w.])specifiche\s+ILIKE\s+'((?:''|[^'])*)'", flags=re.IGNORECASE)

        def _replace(match: re.Match) -> str:
            literal = match.group(1)
            if re.search(rf"descrizione\s+ILIKE\s+'{re.escape(literal)}'", sql_query, flags=re.IGNORECASE):
                return match.group(0)
            return f"(specifiche ILIKE '{literal}' OR descrizione ILIKE '{literal}')"

        normalized = pattern.sub(_replace, sql_query)
        if normalized != sql_query:
            logger.info("Applied hard-coded specifiche/descrizione pairing normalization")
        return normalized

    def _build_double_letter_tolerant_regex(self, like_literal: str) -> str:
        """
        Build a PostgreSQL regex pattern tolerant to double-letter mismatches.

        Example:
            '%cappottina%' -> '.*c+a+p+o+t+i+n+a+.*'
        """
        core = like_literal.replace('%', '').strip()
        if not core:
            return ""

        # Collapse consecutive duplicates first (e.g. cappottina -> capotina),
        # then allow one-or-more per character in regex.
        collapsed_chars: List[str] = []
        prev_char: str | None = None
        for ch in core:
            if prev_char is not None and ch.lower() == prev_char.lower() and ch.isalpha() and prev_char.isalpha():
                continue
            collapsed_chars.append(ch)
            prev_char = ch

        parts: List[str] = []
        prev_kind: str | None = None
        for ch in collapsed_chars:
            if ch.isdigit():
                curr_kind = "digit"
            elif ch.isalpha():
                curr_kind = "alpha"
            elif ch.isspace():
                curr_kind = "space"
            else:
                curr_kind = "other"

            # Allow optional spacing between numeric and alphabetic boundaries:
            # this makes 35cm match 35 cm (and vice versa).
            if prev_kind in {"digit", "alpha"} and curr_kind in {"digit", "alpha"} and prev_kind != curr_kind:
                parts.append(r"\s*")

            if ch.isspace():
                parts.append(r"\s+")
            elif ch.isalpha():
                parts.append(re.escape(ch) + "+")
            else:
                parts.append(re.escape(ch))

            prev_kind = curr_kind

        return ".*" + "".join(parts) + ".*"

    def _build_morph_variants(self, core_token: str) -> List[str]:
        """
        Build lightweight Italian morphological variants for a single token.

        This is intentionally generic and rule-based (no word-specific dictionaries).
        """
        token = (core_token or "").lower().strip()
        if not token or not re.fullmatch(r"[a-zA-Z]+", token):
            return []

        variants = {token}

        # Agentive noun alternation (e.g., friggitore <-> friggitrice)
        if token.endswith("tore") and len(token) > 5:
            stem = token[:-4]
            variants.update({stem + "trice", stem + "tori", stem + "trici"})
        elif token.endswith("trice") and len(token) > 6:
            stem = token[:-5]
            variants.update({stem + "tore", stem + "tori", stem + "trici"})

        # Generic gender/number alternation for common endings.
        if token.endswith("o") and len(token) > 3:
            stem = token[:-1]
            variants.update({stem + "a", stem + "i", stem + "e"})
        elif token.endswith("a") and len(token) > 3:
            stem = token[:-1]
            variants.update({stem + "o", stem + "e", stem + "i"})
        elif token.endswith("i") and len(token) > 3:
            stem = token[:-1]
            variants.update({stem + "o", stem + "a", stem + "e"})
        elif token.endswith("e") and len(token) > 3:
            stem = token[:-1]
            variants.update({stem + "a", stem + "o", stem + "i"})

        return sorted(variants)

    def _enforce_double_letter_tolerant_matching(self, sql_query: str) -> str:
        """
        Enforce generic typo tolerance for doubled letters on all ILIKE predicates.

        For each `field ILIKE '...` clause, add an OR regex fallback that accepts
        one-or-more occurrences for letters, improving matches like single/double
        consonant variations.
        """
        pattern = re.compile(
            r"(?<![\w.])([A-Za-z_][\w.]*)\s+ILIKE\s+'((?:''|[^'])*)'",
            flags=re.IGNORECASE,
        )

        def _replace(match: re.Match) -> str:
            field = match.group(1)
            literal = match.group(2)
            core = literal.replace('%', '').strip()

            # Skip very short tokens to avoid excessive broadening.
            if len(core) < 4:
                return match.group(0)

            regex_pattern = self._build_double_letter_tolerant_regex(literal)
            if not regex_pattern:
                return match.group(0)

            regex_patterns = [regex_pattern]

            # Add generic morphology-aware alternatives for single-token alphabetic terms.
            if re.fullmatch(r"[A-Za-z]+", core):
                for variant in self._build_morph_variants(core):
                    if variant == core:
                        continue
                    variant_pattern = self._build_double_letter_tolerant_regex(variant)
                    if variant_pattern and variant_pattern not in regex_patterns:
                        regex_patterns.append(variant_pattern)

            sql_safe_regexes = [p.replace("'", "''") for p in regex_patterns]

            # Avoid duplicating if a regex fallback for this field/literal already exists.
            existing_regex = re.search(
                rf"{re.escape(field)}\s+~\*\s+'.*{re.escape(sql_safe_regexes[0])}.*'",
                sql_query,
                flags=re.IGNORECASE,
            )
            if existing_regex:
                return match.group(0)

            regex_clause = " OR ".join(f"{field} ~* '{rx}'" for rx in sql_safe_regexes)
            return f"({match.group(0)} OR {regex_clause})"

        normalized = pattern.sub(_replace, sql_query)
        if normalized != sql_query:
            logger.info("Applied generic double-letter tolerant matching normalization")
        return normalized

    def _enforce_numeric_literal_cross_field_matching(self, sql_query: str) -> str:
        """
        Enforce cross-field matching for literals containing digits.

        If a numeric literal is filtered only on specifiche/descrizione, expand it
        to nome OR descrizione OR specifiche. This improves recall for dimensions
        and capacities often stored in `nome` (e.g., 10LT, 35cm).
        """
        pattern = re.compile(
            r"(?<![\w.])(specifiche|descrizione)\s+ILIKE\s+'((?:''|[^'])*)'",
            flags=re.IGNORECASE,
        )

        def _replace(match: re.Match) -> str:
            literal = match.group(2)
            if not re.search(r"\d", literal):
                return match.group(0)

            # Avoid repeated expansion if already present for this literal.
            already_expanded = re.search(
                rf"nome\s+ILIKE\s+'{re.escape(literal)}'",
                sql_query,
                flags=re.IGNORECASE,
            )
            if already_expanded:
                return match.group(0)

            return (
                f"(nome ILIKE '{literal}' OR descrizione ILIKE '{literal}' "
                f"OR specifiche ILIKE '{literal}')"
            )

        normalized = pattern.sub(_replace, sql_query)
        if normalized != sql_query:
            logger.info("Applied numeric literal cross-field normalization")
        return normalized

    def _enforce_codice_in_list_select(self, sql_query: str, user_request: str) -> str:
        """
        Ensure `codice` is selected for list/order requests when applicable.

        This keeps final answer formatting consistent and avoids N/D when codice
        exists in DB but is not selected by the generated SQL.
        """
        if not self._is_list_or_order_request(user_request):
            return sql_query

        if not re.match(r"^\s*SELECT\b", sql_query, flags=re.IGNORECASE):
            return sql_query

        # Skip aggregate-style queries.
        if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql_query, flags=re.IGNORECASE):
            return sql_query

        # SELECT * already includes codice.
        if re.search(r"^\s*SELECT\s+\*\s+FROM\b", sql_query, flags=re.IGNORECASE):
            return sql_query

        from_match = re.search(r"\bFROM\b", sql_query, flags=re.IGNORECASE)
        if not from_match:
            return sql_query

        select_part = sql_query[:from_match.start()]
        if re.search(r"\bcodice\b", select_part, flags=re.IGNORECASE):
            return sql_query

        normalized = select_part.rstrip() + ", codice " + sql_query[from_match.start():]
        logger.info("Applied list-select normalization: added codice to SELECT fields")
        return normalized

    def generate_sql(self, user_request: str) -> Dict[str, Any]:
        """
        Generate SQL query from natural language request.
        
        Args:
            user_request: Natural language description of the desired query
            
        Returns:
            Dict with 'sql_query' and 'need_embedding' fields
        """
        try:
            logger.info(f"Processing user request: {user_request}")
            sql_user_request = self._sanitize_user_request_for_sql(user_request)
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Make API call to local model
            response_text = self._call_local_llm(
                system_prompt=system_prompt,
                user_prompt=sql_user_request,
                temperature=self.temperature,
                expect_json=True,
            )

            # Extract and parse JSON response
            result = self._extract_json_object(response_text)
            
            # Validate response structure
            required_fields = ["sql_query", "need_embedding", "embedding_params"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field '{field}' in LLM response")
            
            # Validate embedding_params structure
            if not isinstance(result["embedding_params"], list):
                raise ValueError("embedding_params must be a list")
            
            # If need_embedding is true, ensure embedding_params is not empty
            if result["need_embedding"] and not result["embedding_params"]:
                raise ValueError("need_embedding is true but embedding_params is empty")
            
            # If need_embedding is false, ensure embedding_params is empty
            if not result["need_embedding"] and result["embedding_params"]:
                raise ValueError("need_embedding is false but embedding_params is not empty")
            
            # Validate that embedding_params count matches placeholder count in SQL
            if result["need_embedding"]:
                placeholder_count = result["sql_query"].count('%s')
                params_count = len(result["embedding_params"])
                
                if placeholder_count != params_count:
                    logger.warning(
                        f"Placeholder mismatch: SQL has {placeholder_count} %s placeholders "
                        f"but embedding_params has {params_count} entries"
                    )
                    logger.warning("This may cause execution errors. The LLM should be retrained.")
            
            logger.info("SQL query generated successfully")
            logger.info(f"Generated SQL: {result['sql_query']}")
            logger.info(f"Needs Embedding: {result['need_embedding']}")
            if result['need_embedding']:
                logger.info(f"Embedding Parameters: {len(result['embedding_params'])} parameter(s)")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            raise
    
    def process_request(self, user_request: str) -> Dict[str, Any]:
        """
        Process a user request and return structured response.
        
        Args:
            user_request: Natural language description of the desired query
            
        Returns:
            Dict containing the generated SQL, need_embedding flag, embedding_params, and metadata
        """
        try:
            result = self.generate_sql(user_request)
            
            return {
                "success": True,
                "user_request": user_request,
                "sql_query": result["sql_query"],
                "need_embedding": result["need_embedding"],
                "embedding_params": result["embedding_params"],
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "success": False,
                "user_request": user_request,
                "error": str(e),
                "model_used": self.model,
                "need_embedding": None,
                "embedding_params": []
            }
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded database schema.
        
        Returns:
            Dict containing schema metadata
        """
        return {
            "schema_loaded": self.database_schema is not None,
            "schema_length": len(self.database_schema) if self.database_schema else 0,
            "model": self.model,
            "temperature": self.temperature
        }
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        raise NotImplementedError(
            "Embedding generation is disabled in local MVP mode. "
            "Set need_embedding=false and embedding_params=[] in prompts."
        )
    
    def _generate_embeddings_for_params(self, embedding_params: List[Dict[str, str]]) -> List[str]:
        """
        Generate embeddings for all parameters in the query.
        
        Args:
            embedding_params: List of embedding parameter dictionaries
            
        Returns:
            List of embedding vectors as PostgreSQL-formatted strings
        """
        embeddings = []
        
        for param in embedding_params:
            text_to_embed = param['text_to_embed']
            embedding = self._generate_embedding(text_to_embed)
            
            # Convert to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            embeddings.append(embedding_str)
        
        return embeddings
    
    def _validate_sql_query(self, sql_query: str) -> tuple[bool, Optional[str]]:
        """
        Validate SQL query for security and safety.
        
        Checks:
        1. Query must be parseable
        2. Only SELECT statements allowed (no INSERT, UPDATE, DELETE, DROP, etc.)
        3. No dangerous operations (CREATE, ALTER, TRUNCATE, etc.)
        4. No multiple statements (semicolon separation)
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check for multiple statements (basic protection against SQL injection)
            statements = sql_query.strip().split(';')
            # Remove empty strings from split
            statements = [s.strip() for s in statements if s.strip()]
            
            if len(statements) > 1:
                return False, "Multiple SQL statements detected. Only single SELECT queries are allowed."
            
            # Define dangerous keywords that should not appear
            dangerous_keywords = [
                'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
                'TRUNCATE', 'REPLACE', 'MERGE', 'GRANT', 'REVOKE',
                'EXEC', 'EXECUTE', 'CALL'
            ]
            
            # Check for dangerous keywords in uppercase query
            query_upper = sql_query.upper()
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    return False, f"Dangerous operation detected: {keyword}"
            
            # Additional check: look for INTO clause (SELECT INTO is a write operation)
            if 'INTO' in query_upper and 'INTO' not in query_upper[query_upper.find('FROM'):]:
                # INTO appears before FROM, which could be SELECT INTO
                return False, "SELECT INTO operations are not allowed"
            
            # Check if query starts with SELECT (case-insensitive)
            if not query_upper.strip().startswith('SELECT'):
                return False, "Only SELECT queries are allowed"
            
            # Check for vector columns in GROUP BY (this will cause errors in PostgreSQL)
            if 'GROUP BY' in query_upper:
                # Extract the GROUP BY clause
                group_by_start = query_upper.find('GROUP BY')
                group_by_clause = sql_query[group_by_start:].split('ORDER BY')[0].split('LIMIT')[0]
                
                # Check if any _embed fields are in GROUP BY
                if '_embed' in group_by_clause.lower():
                    return False, "Vector columns (fields ending with '_embed') cannot be used in GROUP BY clause. Use primary keys or scalar fields only."
            
            # Prepare query for parsing by handling pgvector operators
            # sqlglot doesn't understand PostgreSQL's <-> operator for vector distance
            # We'll temporarily replace it for parsing validation
            query_for_parsing = sql_query
            has_vector_ops = False
            
            # Replace pgvector operators with standard operators for parsing
            if '<->' in query_for_parsing:
                has_vector_ops = True
                # Replace vector distance operator with a dummy function call
                query_for_parsing = query_for_parsing.replace('<->', '+')
            
            # Parse the SQL query to check structure
            try:
                parsed = sqlglot.parse_one(query_for_parsing, read='postgres')
            except Exception as e:
                # If parsing fails, do a basic syntax check instead
                logger.warning(f"SQL parsing warning: {str(e)}")
                # Allow the query if it passed all other checks
                if query_upper.strip().startswith('SELECT'):
                    logger.info("SQL query validation passed (basic check)")
                    return True, None
                return False, f"SQL parsing error: {str(e)}"
            
            # Check if it's a SELECT statement
            if not isinstance(parsed, sqlglot.exp.Select):
                statement_type = type(parsed).__name__
                return False, f"Only SELECT queries are allowed. Detected: {statement_type}"
            
            logger.info("SQL query validation passed")
            return True, None
            
        except Exception as e:
            logger.error(f"SQL validation error: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def execute_sql(self, sql_query: str, need_embedding: bool = False, 
                    embedding_params: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Execute SQL query against the database with validation.
        
        Args:
            sql_query: SQL query to execute
            need_embedding: Whether the query needs embedding parameters
            embedding_params: List of embedding parameter dictionaries (required if need_embedding is True)
            
        Returns:
            Dict containing query results and metadata
        """
        try:
            # SECURITY: Validate SQL query before execution
            logger.info("Validating SQL query for security...")
            is_valid, error_message = self._validate_sql_query(sql_query)
            
            if not is_valid:
                logger.error(f"SQL validation failed: {error_message}")
                
                # Check if this is a security issue or a fixable error
                is_security_issue = any(keyword in error_message for keyword in [
                    'Dangerous operation', 'Multiple SQL statements', 'Only SELECT queries',
                    'SELECT INTO', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE'
                ])
                
                return {
                    "success": False,
                    "error": f"Query validation failed: {error_message}",
                    "results": [],
                    "column_names": [],
                    "row_count": 0,
                    "validation_failed": True,
                    "is_security_issue": is_security_issue  # Distinguish security from syntax errors
                }
            
            logger.info("Connecting to database for query execution...")
            connection = psycopg2.connect(**self.db_config)
            cursor = connection.cursor()
            
            # Generate embeddings if needed
            query_params = []
            if need_embedding:
                if not embedding_params:
                    raise ValueError("embedding_params required when need_embedding is True")
                
                logger.info(f"Generating {len(embedding_params)} embedding(s) for query...")
                embeddings_generated = self._generate_embeddings_for_params(embedding_params)
                
                # Count how many %s placeholders are in the query
                placeholder_count = sql_query.count('%s')
                
                # If there are more placeholders than embeddings, replicate embeddings in order
                # This handles cases where the LLM reuses the same embedding multiple times
                if placeholder_count > len(embeddings_generated):
                    logger.info(f"Replicating {len(embeddings_generated)} embeddings to match {placeholder_count} placeholders")
                    
                    # Replicate embeddings cyclically to match placeholder count
                    query_params = []
                    for i in range(placeholder_count):
                        embedding_index = i % len(embeddings_generated)
                        query_params.append(embeddings_generated[embedding_index])
                else:
                    query_params = embeddings_generated
            
            # Execute query
            placeholder_count = sql_query.count('%s')
            logger.info(f"Executing SQL query...")
            
            # Verify parameter count matches
            if query_params and len(query_params) != placeholder_count:
                error_msg = f"Parameter count mismatch: query expects {placeholder_count} but got {len(query_params)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if query_params:
                try:
                    # Manual parameter substitution to avoid psycopg2 issues with complex queries
                    query_with_params = sql_query
                    for param in query_params:
                        # Replace first occurrence of %s::vector with the embedding
                        query_with_params = query_with_params.replace('%s::vector', f"'{param}'::vector", 1)
                    
                    # Execute without parameters (they're already in the query)
                    cursor.execute(query_with_params)
                except psycopg2.Error as e:
                    # Catch ALL PostgreSQL-specific errors and provide better error message
                    logger.error(f"PostgreSQL execution error: {str(e)}")
                    cursor.close()
                    connection.close()
                    raise ValueError(f"PostgreSQL error: {str(e)}")
                except Exception as e:
                    # Catch any other errors
                    logger.error(f"Unexpected error during execution: {str(e)}")
                    cursor.close()
                    connection.close()
                    raise
            else:
                try:
                    cursor.execute(sql_query)
                except psycopg2.Error as e:
                    logger.error(f"PostgreSQL execution error: {str(e)}")
                    cursor.close()
                    connection.close()
                    raise ValueError(f"PostgreSQL error: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error during execution: {str(e)}")
                    cursor.close()
                    connection.close()
                    raise
            
            # Fetch results
            try:
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Convert results to list of dictionaries
                results_list = []
                for row in results:
                    row_dict = {}
                    for i, col_name in enumerate(column_names):
                        row_dict[col_name] = row[i]
                    results_list.append(row_dict)
                
                logger.info(f"Query executed successfully. Retrieved {len(results_list)} row(s)")
                
                # Close connection
                cursor.close()
                connection.close()
                
                return {
                    "success": True,
                    "results": results_list,
                    "column_names": column_names,
                    "row_count": len(results_list)
                }
                
            except psycopg2.ProgrammingError:
                # No results to fetch (e.g., INSERT, UPDATE, DELETE)
                connection.commit()
                affected_rows = cursor.rowcount
                
                cursor.close()
                connection.close()
                
                logger.info(f"Query executed successfully. {affected_rows} row(s) affected")
                
                return {
                    "success": True,
                    "results": [],
                    "column_names": [],
                    "row_count": 0,
                    "affected_rows": affected_rows
                }
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            if 'connection' in locals() and connection:
                connection.rollback()
                connection.close()
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "column_names": [],
                "row_count": 0
            }
    
    def generate_final_answer(self, user_request: str, query_results: Dict[str, Any], 
                              sql_query: str = None) -> str:
        """
        Generate a natural language answer based on the user request and query results.
        
        Args:
            user_request: Original user request
            query_results: Results from the SQL query execution
            sql_query: The SQL query that was executed (optional, for context)
            
        Returns:
            Natural language answer as a string
        """
        try:
            logger.info("Generating final natural language answer...")

            # For list/order requests, return a deterministic compact list without extra feature prose.
            if (
                query_results.get('success', False)
                and query_results.get('row_count', 0) > 0
                and self._is_list_or_order_request(user_request)
            ):
                return self._format_ordered_list_answer(user_request, query_results)
            
            # Prepare the results summary
            if not query_results.get('success', False):
                results_text = f"Error executing query: {query_results.get('error', 'Unknown error')}"
            elif query_results['row_count'] == 0:
                results_text = "No results found."
            else:
                # Format results as a readable text
                results_text = f"Found {query_results['row_count']} result(s):\n\n"
                for i, row in enumerate(query_results['results'][:20], 1):  # Limit to first 20 rows
                    results_text += f"Result {i}:\n"
                    for key, value in row.items():
                        # Skip embedding columns and similarity scores for readability
                        if not key.endswith('_embed') and key != 'similarity' and key != 'combined_similarity':
                            results_text += f"  - {key}: {value}\n"
                    results_text += "\n"
                
                if query_results['row_count'] > 20:
                    results_text += f"... and {query_results['row_count'] - 20} more results\n"
            
            # Create prompt for final answer generation
            system_prompt = create_final_answer_prompt()
            user_message = create_final_answer_user_message(user_request, results_text, sql_query)
            
            # Make API call to generate answer
            answer = self._call_local_llm(
                system_prompt=system_prompt,
                user_prompt=user_message,
                temperature=0.3,
                expect_json=False,
            ).strip()
            return answer
            
        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            return f"I apologize, but I encountered an error while generating the answer: {str(e)}"

    def _is_list_or_order_request(self, user_request: str) -> bool:
        """Detect requests that explicitly ask for list/elenco and/or sorted ordering."""
        text = (user_request or "").lower()
        markers = [
            "lista",
            "elenca",
            "elenco",
            "mostrami",
            "in ordine",
            "ordine crescente",
            "ordine decrescente",
            "ordered",
            "sorted",
            "ascending",
            "descending",
        ]
        return any(marker in text for marker in markers)

    def _format_ordered_list_answer(self, user_request: str, query_results: Dict[str, Any]) -> str:
        """Format query results as an ordered list with essential fields only."""
        rows = query_results.get('results', [])
        if not rows:
            return "Non ho trovato risultati."

        lines = [f"Ecco l'elenco in ordine richiesto ({len(rows)} risultati):"]
        for idx, row in enumerate(rows, 1):
            nome = row.get('nome', 'N/D')
            fornitore = row.get('fornitore', 'N/D')
            codice = row.get('codice', 'N/D')
            prezzo = row.get('prezzo')
            cliente = row.get('cliente')

            prezzo_str = "N/D" if prezzo is None else str(prezzo)

            item = f"{idx}. {nome} | fornitore: {fornitore} | codice: {codice} | prezzo: {prezzo_str}"
            if cliente:
                item += f" | cliente: {cliente}"
            lines.append(item)

        return "\n".join(lines)
    
    def _regenerate_sql_with_error_feedback(self, user_request: str, 
                                              attempt_history: List[Dict[str, str]], 
                                              attempt: int) -> Dict[str, Any]:
        """
        Regenerate SQL query by providing comprehensive error feedback to the LLM.
        
        Args:
            user_request: Original user request
            attempt_history: List of previous attempts with their SQL and errors
            attempt: Current attempt number
            
        Returns:
            Dict with regenerated SQL query
        """
        logger.info(f"Regenerating SQL query with error feedback from {len(attempt_history)} previous attempt(s)...")
        sql_user_request = self._sanitize_user_request_for_sql(user_request)
        
        # Create system prompt
        system_prompt = self._create_system_prompt()
        
        # Build comprehensive error history
        error_history_text = ""
        for i, prev_attempt in enumerate(attempt_history, 1):
            error_history_text += f"""
ATTEMPT {i}:
SQL Query: {prev_attempt['sql']}
Error: {prev_attempt['error']}
---"""
        
        # Create user message using a phase-specific retry prompt for deterministic broadening.
        if attempt >= 2:
            user_message = create_sql_retry_prompt_for_phase(sql_user_request, error_history_text, attempt)
        else:
            user_message = create_sql_retry_prompt(sql_user_request, error_history_text)
        
        try:
            # Make API call to local model
            response_text = self._call_local_llm(
                system_prompt=system_prompt,
                user_prompt=user_message,
                temperature=self.temperature,
                expect_json=True,
            )

            # Extract and parse JSON response
            result = self._extract_json_object(response_text)
            
            # Validate response structure
            required_fields = ["sql_query", "need_embedding", "embedding_params"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field '{field}' in LLM response")
            
            return result
            
        except Exception as e:
            logger.error(f"Error regenerating SQL: {str(e)}")
            raise
    
    def process_request_with_execution(self, user_request: str, max_retries: int = 4) -> Dict[str, Any]:
        """
        Complete pipeline with retry mechanism: Generate SQL, execute it, and generate final answer.
        
        If execution fails, the system will retry up to max_retries times, providing comprehensive
        error feedback including all previous attempts to the LLM for regeneration.
        
        Args:
            user_request: Natural language description of the desired query
            max_retries: Maximum number of retry attempts (default: 4)
            
        Returns:
            Dict containing all information including the final answer
        """
        attempt = 0
        last_error = None
        sql_result = None
        query_results = None
        attempt_history = []  # Track all previous attempts
        
        while attempt < max_retries:
            try:
                attempt += 1
                logger.info("=" * 80)
                logger.info(f"ATTEMPT {attempt}/{max_retries}")
                logger.info("=" * 80)
                
                # Step 1: Generate SQL query (or regenerate with error feedback)
                if attempt == 1:
                    logger.info("STEP 1: GENERATING SQL QUERY")
                    logger.info("=" * 80)
                    sql_result = self.generate_sql(user_request)
                else:
                    logger.info(f"STEP 1: REGENERATING SQL QUERY (Attempt {attempt})")
                    logger.info("=" * 80)
                    sql_result = self._regenerate_sql_with_error_feedback(
                        user_request=user_request,
                        attempt_history=attempt_history,
                        attempt=attempt
                    )
                
                # Step 2: Execute SQL query
                logger.info("=" * 80)
                logger.info("STEP 2: EXECUTING SQL QUERY")
                logger.info("=" * 80)
                sql_result['sql_query'] = self._enforce_codice_in_list_select(sql_result['sql_query'], user_request)
                sql_result['sql_query'] = self._enforce_numeric_literal_cross_field_matching(sql_result['sql_query'])
                sql_result['sql_query'] = self._enforce_double_letter_tolerant_matching(sql_result['sql_query'])
                sql_result['sql_query'] = self._enforce_specifiche_descrizione_pairing(sql_result['sql_query'])
                query_results = self.execute_sql(
                    sql_query=sql_result['sql_query'],
                    need_embedding=sql_result['need_embedding'],
                    embedding_params=sql_result['embedding_params']
                )
                
                # Check if execution was successful
                if not query_results.get('success', False):
                    last_error = query_results.get('error', 'Unknown execution error')
                    logger.warning(f"Attempt {attempt} failed: {last_error}")
                    
                    # Add this failed attempt to history
                    attempt_history.append({
                        'sql': sql_result['sql_query'],
                        'error': last_error
                    })
                    
                    # Check if this is a security issue (should abort) or fixable error (can retry)
                    if query_results.get('validation_failed', False):
                        is_security_issue = query_results.get('is_security_issue', True)
                        
                        if is_security_issue:
                            logger.error("SECURITY ISSUE detected - aborting retries")
                            break
                        else:
                            logger.warning("Validation failed but error is fixable - will retry with feedback")
                            # Continue to next attempt
                            continue
                    
                    # Continue to next attempt
                    continue

                # Stop fallback if broadening has already produced a very large candidate set.
                if query_results.get('row_count', 0) >= self.broad_result_threshold and attempt < max_retries:
                    logger.warning(
                        "Attempt %s produced %s rows (threshold=%s). "
                        "Stopping retries to avoid over-broad filtering.",
                        attempt,
                        query_results.get('row_count', 0),
                        self.broad_result_threshold,
                    )
                    query_results['is_broad_result'] = True
                    query_results['broad_result_threshold'] = self.broad_result_threshold
                    
                    # Continue to final answer generation without further retries.
                    
                
                
                # If query executed but returned no rows, retry with no-results feedback.
                if query_results.get('row_count', 0) == 0 and attempt < max_retries:
                    retry_phase = min(attempt + 1, max_retries)
                    no_results_error = (
                        "Query executed successfully but returned 0 rows. "
                        f"Apply hard-coded fallback phase for next attempt={retry_phase}. "
                        "Use progressively broader filtering: split long phrase constraints, reduce strict AND "
                        "chains, keep core product terms, and broaden ILIKE predicates across "
                        "nome/descrizione/specifiche."
                    )
                    logger.warning(
                        "Attempt %s returned 0 rows. Retrying with phase-based broader strategy (next attempt=%s).",
                        attempt,
                        retry_phase,
                    )
                    attempt_history.append({
                        'sql': sql_result['sql_query'],
                        'error': no_results_error
                    })
                    continue

                # Step 3: Generate final answer (only if execution succeeded)
                logger.info("=" * 80)
                logger.info("STEP 3: GENERATING FINAL ANSWER")
                logger.info("=" * 80)
                final_answer = self.generate_final_answer(
                    user_request=user_request,
                    query_results=query_results,
                    sql_query=sql_result['sql_query']
                )
                
                logger.info("=" * 80)
                logger.info(f"PIPELINE COMPLETED SUCCESSFULLY (Attempt {attempt})")
                logger.info("=" * 80)
                
                return {
                    "success": True,
                    "user_request": user_request,
                    "sql_query": sql_result['sql_query'],
                    "need_embedding": sql_result['need_embedding'],
                    "embedding_params": sql_result['embedding_params'],
                    "query_results": query_results,
                    "final_answer": final_answer,
                    "model_used": self.model,
                    "attempts": attempt,
                    "failed_attempts": attempt_history  # Include history for transparency
                }
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Attempt {attempt} error: {last_error}")
                
                # Add this failed attempt to history
                if sql_result:
                    attempt_history.append({
                        'sql': sql_result.get('sql_query', 'Query generation failed'),
                        'error': last_error
                    })
                
                # If this is not the last attempt, continue retrying
                if attempt < max_retries:
                    continue
                # If it's the last attempt, break and return error
                break
        
        # All attempts failed
        logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
        
        return {
            "success": False,
            "user_request": user_request,
            "error": f"Failed after {attempt} attempts. Last error: {last_error}",
            "model_used": self.model,
            "attempts": attempt,
            "last_sql_query": sql_result['sql_query'] if sql_result else None,
            "failed_attempts": attempt_history  # Include complete history for debugging
        }
