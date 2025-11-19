#!/usr/bin/env python3
"""
PostgreSQL Database Connector with pgvector for Medical Documents
Handles document storage, embedding generation, and semantic search

Features:
- Connection pooling for batch operations
- Embedding generation with multilingual-e5-large-instruct
- Semantic search with pgvector
- HIPAA-compliant audit logging
- Transaction handling with rollback on errors

Environment Variables Required:
- DB_HOST: PostgreSQL host (default: localhost)
- DB_PORT: PostgreSQL port (default: 5432)
- DB_NAME: Database name (default: medical_records)
- DB_USER: Database user
- DB_PASSWORD: Database password
- DB_POOL_SIZE: Connection pool size (default: 10)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
import json

import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor, execute_values
import numpy as np

# Embedding model
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Embedding generation disabled.")


@dataclass
class DocumentRecord:
    """Represents a clinical document record"""
    filename: str
    original_path: str
    document_type: str
    ocr_text: str
    ocr_confidence: float
    char_count: int
    word_count: int
    model_used: str
    prompt_type: str
    processing_time_seconds: float
    gpu_memory_used_mb: float
    image_dimensions: Dict
    file_size_bytes: int
    base_size: int
    image_size: int
    crop_mode: bool
    markdown_path: Optional[str]
    boxes_image_path: Optional[str]
    entities_json_path: Optional[str]
    medical_entities: Optional[Dict]
    entity_count: int
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    processed_by: Optional[int] = None
    uploaded_by: Optional[int] = None
    specialty: Optional[str] = None
    metadata: Optional[Dict] = None


class EmbeddingGenerator:
    """
    Generate embeddings using multilingual-e5-large-instruct

    Model: intfloat/multilingual-e5-large-instruct (1024 dimensions)
    Optimized for multilingual semantic search including Spanish
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        """
        Initialize embedding generator

        Args:
            model_name: Hugging Face model name
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Install with: pip install transformers")

        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.logger = logging.getLogger('EmbeddingGenerator')
        self.logger.info(f"Loading embedding model: {model_name}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"Model loaded on {self.device}")

    def generate_embedding(self, text: str, instruction: str = "Represent this medical document for retrieval:") -> np.ndarray:
        """
        Generate embedding for text

        Args:
            text: Input text
            instruction: Task instruction (for instruct models)

        Returns:
            numpy array of shape (1024,)
        """
        # Prepend instruction for better embeddings
        input_text = f"{instruction} {text}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()[0]

    def generate_batch_embeddings(
        self,
        texts: List[str],
        instruction: str = "Represent this medical document for retrieval:",
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batched for efficiency)

        Args:
            texts: List of texts
            instruction: Task instruction
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (len(texts), 1024)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Prepend instruction
            input_texts = [f"{instruction} {text}" for text in batch_texts]

            # Tokenize
            inputs = self.tokenizer(
                input_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)


class DatabaseConnector:
    """
    PostgreSQL connector for medical documents with pgvector support

    Features:
    - Connection pooling
    - Transaction management
    - Embedding storage and retrieval
    - Semantic search
    - HIPAA audit logging
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        pool_size: int = 10
    ):
        """
        Initialize database connector

        Args:
            host: PostgreSQL host (default: env DB_HOST or localhost)
            port: PostgreSQL port (default: env DB_PORT or 5432)
            database: Database name (default: env DB_NAME or medical_records)
            user: Database user (default: env DB_USER)
            password: Database password (default: env DB_PASSWORD)
            pool_size: Connection pool size (default: env DB_POOL_SIZE or 10)
        """
        self.logger = logging.getLogger('DatabaseConnector')

        # Get credentials from environment or parameters
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.port = port or int(os.getenv('DB_PORT', '5432'))
        self.database = database or os.getenv('DB_NAME', 'medical_records')
        self.user = user or os.getenv('DB_USER')
        self.password = password or os.getenv('DB_PASSWORD')

        if not self.user or not self.password:
            raise ValueError(
                "Database credentials not provided. "
                "Set DB_USER and DB_PASSWORD environment variables or pass as parameters."
            )

        self.pool_size = pool_size or int(os.getenv('DB_POOL_SIZE', '10'))

        # Initialize connection pool
        self.logger.info(f"Initializing connection pool to {self.host}:{self.port}/{self.database}")
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=self.pool_size,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.logger.info("Connection pool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise

        # Initialize embedding generator (lazy loading)
        self._embedding_generator = None

    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        """Lazy load embedding generator"""
        if self._embedding_generator is None:
            self._embedding_generator = EmbeddingGenerator()
        return self._embedding_generator

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections

        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users")
        """
        conn = self.connection_pool.getconn()
        try:
            yield conn
        finally:
            self.connection_pool.putconn(conn)

    @contextmanager
    def get_cursor(self, commit: bool = True, dict_cursor: bool = True):
        """
        Context manager for database cursors with automatic commit/rollback

        Args:
            commit: Auto-commit on success
            dict_cursor: Use RealDictCursor for dict results

        Usage:
            with db.get_cursor() as cursor:
                cursor.execute("SELECT * FROM users")
                results = cursor.fetchall()
        """
        conn = self.connection_pool.getconn()
        try:
            cursor_factory = RealDictCursor if dict_cursor else None
            cursor = conn.cursor(cursor_factory=cursor_factory)
            yield cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Transaction failed, rolled back: {e}")
            raise
        finally:
            cursor.close()
            self.connection_pool.putconn(conn)

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_cursor(commit=False) as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                self.logger.info("Database connection test successful")
                return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False

    def insert_document(
        self,
        document: DocumentRecord,
        generate_embedding: bool = True,
        user_id: Optional[int] = None
    ) -> int:
        """
        Insert clinical document into database

        Args:
            document: DocumentRecord to insert
            generate_embedding: Whether to generate embedding
            user_id: User ID for audit logging

        Returns:
            Document ID
        """
        start_time = datetime.now()

        try:
            # Generate embedding if requested
            embedding = None
            if generate_embedding and document.ocr_text:
                self.logger.debug(f"Generating embedding for {document.filename}")
                embedding_vector = self.embedding_generator.generate_embedding(document.ocr_text)
                embedding = embedding_vector.tolist()

            with self.get_cursor() as cursor:
                # Insert document
                cursor.execute("""
                    INSERT INTO clinical_documents (
                        filename, original_path, document_type, specialty,
                        ocr_text, ocr_confidence, char_count, word_count,
                        model_used, prompt_type, processing_time_seconds, gpu_memory_used_mb,
                        image_dimensions, file_size_bytes, base_size, image_size, crop_mode,
                        markdown_path, boxes_image_path, entities_json_path,
                        embedding, medical_entities, entity_count,
                        patient_id, encounter_id, processed_by, uploaded_by, metadata
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s, %s
                    ) RETURNING id, document_uuid
                """, (
                    document.filename, document.original_path, document.document_type, document.specialty,
                    document.ocr_text, document.ocr_confidence, document.char_count, document.word_count,
                    document.model_used, document.prompt_type, document.processing_time_seconds, document.gpu_memory_used_mb,
                    json.dumps(document.image_dimensions), document.file_size_bytes, document.base_size, document.image_size, document.crop_mode,
                    document.markdown_path, document.boxes_image_path, document.entities_json_path,
                    embedding, json.dumps(document.medical_entities) if document.medical_entities else None, document.entity_count,
                    document.patient_id, document.encounter_id, document.processed_by, document.uploaded_by,
                    json.dumps(document.metadata) if document.metadata else None
                ))

                result = cursor.fetchone()
                document_id = result['id']
                document_uuid = result['document_uuid']

                # Log audit event
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                self.log_audit_event(
                    cursor=cursor,
                    event_type='CREATE',
                    event_category='DOCUMENT_ACCESS',
                    user_id=user_id,
                    resource_type='clinical_document',
                    resource_id=document_id,
                    document_uuid=document_uuid,
                    action_description=f"Inserted document: {document.filename}",
                    duration_ms=int(duration_ms),
                    metadata={'document_type': document.document_type, 'entity_count': document.entity_count}
                )

                self.logger.info(f"✓ Inserted document {document.filename} (ID: {document_id}, UUID: {document_uuid})")
                return document_id

        except Exception as e:
            self.logger.error(f"Failed to insert document {document.filename}: {e}")
            raise

    def insert_documents_batch(
        self,
        documents: List[DocumentRecord],
        generate_embeddings: bool = True,
        user_id: Optional[int] = None,
        batch_size: int = 100
    ) -> List[int]:
        """
        Insert multiple documents in batch (more efficient)

        Args:
            documents: List of DocumentRecord objects
            generate_embeddings: Whether to generate embeddings
            user_id: User ID for audit logging
            batch_size: Batch size for embedding generation

        Returns:
            List of document IDs
        """
        document_ids = []

        # Generate embeddings in batch if requested
        embeddings = None
        if generate_embeddings:
            texts = [doc.ocr_text for doc in documents if doc.ocr_text]
            if texts:
                self.logger.info(f"Generating embeddings for {len(texts)} documents...")
                embedding_vectors = self.embedding_generator.generate_batch_embeddings(texts, batch_size=batch_size)
                embeddings = [vec.tolist() for vec in embedding_vectors]

        # Insert documents
        for i, document in enumerate(documents):
            embedding = embeddings[i] if embeddings and i < len(embeddings) else None

            # Temporarily override embedding in document
            original_embedding = document.medical_entities

            try:
                doc_id = self.insert_document(document, generate_embedding=False, user_id=user_id)

                # Update embedding separately if generated
                if embedding:
                    with self.get_cursor() as cursor:
                        cursor.execute(
                            "UPDATE clinical_documents SET embedding = %s WHERE id = %s",
                            (embedding, doc_id)
                        )

                document_ids.append(doc_id)

            except Exception as e:
                self.logger.error(f"Failed to insert document {document.filename}: {e}")
                continue

        return document_ids

    def search_similar_documents(
        self,
        query_text: str,
        limit: int = 10,
        threshold: float = 0.7,
        document_type: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Semantic search for similar documents using pgvector

        Args:
            query_text: Search query
            limit: Maximum results to return
            threshold: Minimum similarity threshold (0-1)
            document_type: Filter by document type
            user_id: User ID for audit logging

        Returns:
            List of similar documents with similarity scores
        """
        start_time = datetime.now()

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(
                query_text,
                instruction="Represent this query for retrieving medical documents:"
            )

            with self.get_cursor(commit=False) as cursor:
                # Build query
                query = """
                    SELECT
                        id,
                        document_uuid,
                        filename,
                        document_type,
                        ocr_text,
                        medical_entities,
                        processed_at,
                        patient_id,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM clinical_documents
                    WHERE
                        embedding IS NOT NULL
                        AND deleted = false
                        AND 1 - (embedding <=> %s::vector) > %s
                """
                params = [query_embedding.tolist(), query_embedding.tolist(), threshold]

                # Add document type filter if specified
                if document_type:
                    query += " AND document_type = %s"
                    params.append(document_type)

                query += " ORDER BY embedding <=> %s::vector LIMIT %s"
                params.extend([query_embedding.tolist(), limit])

                cursor.execute(query, params)
                results = cursor.fetchall()

                # Log audit event
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                self.log_audit_event(
                    cursor=cursor,
                    event_type='SEARCH',
                    event_category='DATABASE_QUERY',
                    user_id=user_id,
                    resource_type='clinical_document',
                    action_description=f"Semantic search: {query_text[:100]}",
                    records_affected=len(results),
                    duration_ms=int(duration_ms),
                    metadata={'threshold': threshold, 'limit': limit, 'document_type': document_type}
                )

                self.logger.info(f"✓ Found {len(results)} similar documents (threshold: {threshold})")
                return results

        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            raise

    def full_text_search(
        self,
        query: str,
        limit: int = 50,
        document_type: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Full-text search on OCR text (PostgreSQL FTS with Spanish)

        Args:
            query: Search query
            limit: Maximum results
            document_type: Filter by type
            user_id: User ID for audit

        Returns:
            List of matching documents
        """
        try:
            with self.get_cursor(commit=False) as cursor:
                query_sql = """
                    SELECT
                        id,
                        document_uuid,
                        filename,
                        document_type,
                        ocr_text,
                        medical_entities,
                        processed_at,
                        ts_rank(to_tsvector('spanish', ocr_text), query) as rank
                    FROM clinical_documents,
                         to_tsquery('spanish', %s) query
                    WHERE to_tsvector('spanish', ocr_text) @@ query
                        AND deleted = false
                """
                params = [query]

                if document_type:
                    query_sql += " AND document_type = %s"
                    params.append(document_type)

                query_sql += " ORDER BY rank DESC LIMIT %s"
                params.append(limit)

                cursor.execute(query_sql, params)
                results = cursor.fetchall()

                # Log audit event
                self.log_audit_event(
                    cursor=cursor,
                    event_type='SEARCH',
                    event_category='DATABASE_QUERY',
                    user_id=user_id,
                    resource_type='clinical_document',
                    action_description=f"Full-text search: {query}",
                    records_affected=len(results)
                )

                return results

        except Exception as e:
            self.logger.error(f"Full-text search failed: {e}")
            raise

    def log_audit_event(
        self,
        cursor,
        event_type: str,
        event_category: str,
        user_id: Optional[int] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[int] = None,
        document_uuid: Optional[str] = None,
        action_description: Optional[str] = None,
        records_affected: int = 0,
        duration_ms: int = 0,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log audit event for HIPAA compliance

        Args:
            cursor: Database cursor
            event_type: CREATE, READ, UPDATE, DELETE, SEARCH, EXPORT
            event_category: DOCUMENT_ACCESS, ENTITY_EXTRACTION, DATABASE_QUERY
            user_id: User ID
            resource_type: Resource type
            resource_id: Resource ID
            document_uuid: Document UUID
            action_description: Description of action
            records_affected: Number of records affected
            duration_ms: Duration in milliseconds
            success: Whether operation succeeded
            error_message: Error message if failed
            metadata: Additional metadata
        """
        try:
            cursor.execute("""
                INSERT INTO audit_logs (
                    event_type, event_category, user_id,
                    resource_type, resource_id, document_uuid,
                    action_description, records_affected, success,
                    error_message, duration_ms, metadata
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
            """, (
                event_type, event_category, user_id,
                resource_type, resource_id, document_uuid,
                action_description, records_affected, success,
                error_message, duration_ms,
                json.dumps(metadata) if metadata else None
            ))
        except Exception as e:
            # Don't raise - audit logging should not break operations
            self.logger.warning(f"Failed to log audit event: {e}")

    def get_document_by_id(self, document_id: int, user_id: Optional[int] = None) -> Optional[Dict]:
        """Get document by ID"""
        try:
            with self.get_cursor(commit=False) as cursor:
                cursor.execute("""
                    SELECT * FROM clinical_documents
                    WHERE id = %s AND deleted = false
                """, (document_id,))

                result = cursor.fetchone()

                if result:
                    self.log_audit_event(
                        cursor=cursor,
                        event_type='READ',
                        event_category='DOCUMENT_ACCESS',
                        user_id=user_id,
                        resource_type='clinical_document',
                        resource_id=document_id,
                        document_uuid=result['document_uuid'],
                        action_description=f"Retrieved document: {result['filename']}"
                    )

                return result

        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            raise

    def get_documents_by_patient(self, patient_id: str, user_id: Optional[int] = None) -> List[Dict]:
        """Get all documents for a patient"""
        try:
            with self.get_cursor(commit=False) as cursor:
                cursor.execute("""
                    SELECT * FROM clinical_documents
                    WHERE patient_id = %s AND deleted = false
                    ORDER BY processed_at DESC
                """, (patient_id,))

                results = cursor.fetchall()

                self.log_audit_event(
                    cursor=cursor,
                    event_type='READ',
                    event_category='DOCUMENT_ACCESS',
                    user_id=user_id,
                    resource_type='clinical_document',
                    action_description=f"Retrieved patient documents: {patient_id}",
                    records_affected=len(results)
                )

                return results

        except Exception as e:
            self.logger.error(f"Failed to get documents for patient {patient_id}: {e}")
            raise

    def close(self):
        """Close all connections in pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Connection pool closed")


# Convenience function
def get_database_connector(
    config_path: Optional[str] = None,
    **kwargs
) -> DatabaseConnector:
    """
    Get database connector with configuration

    Args:
        config_path: Path to config file (optional)
        **kwargs: Override parameters

    Returns:
        DatabaseConnector instance
    """
    # Load from config file if provided
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            db_config = config.get('database', {})
            kwargs = {**db_config, **kwargs}

    return DatabaseConnector(**kwargs)


if __name__ == '__main__':
    # Example usage
    import sys

    logging.basicConfig(level=logging.INFO)

    print("=== PostgreSQL + pgvector Connector Demo ===\n")

    # Test connection
    try:
        db = DatabaseConnector()

        if db.test_connection():
            print("✓ Database connection successful\n")

            # Example: Semantic search
            print("Example: Semantic search")
            query = "paciente con diabetes y presión alta"
            results = db.search_similar_documents(query, limit=5, threshold=0.5)

            print(f"\nFound {len(results)} similar documents for query: '{query}'")
            for doc in results:
                print(f"  - {doc['filename']} (similarity: {doc['similarity']:.2f})")

            db.close()
        else:
            print("✗ Database connection failed")
            sys.exit(1)

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
