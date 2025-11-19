-- Medical Records Database Schema v1.0
-- PostgreSQL 15+ with pgvector extension
-- HIPAA compliant schema for medical document storage and retrieval

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

-- Enable pgvector for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USERS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'viewer', -- admin, doctor, nurse, viewer, analyst
    department VARCHAR(100),
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

-- Index for faster user lookups
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- ============================================================================
-- CLINICAL DOCUMENTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS clinical_documents (
    id SERIAL PRIMARY KEY,

    -- Document identification
    document_uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    filename VARCHAR(500) NOT NULL,
    original_path TEXT,

    -- Document classification
    document_type VARCHAR(50), -- historia_clinica, laboratorio, receta, radiologia, etc.
    specialty VARCHAR(100), -- cardiologia, neurologia, etc.

    -- OCR results
    ocr_text TEXT,
    ocr_confidence FLOAT CHECK (ocr_confidence >= 0 AND ocr_confidence <= 1),
    char_count INTEGER,
    word_count INTEGER,

    -- Processing metadata
    model_used VARCHAR(100),
    prompt_type VARCHAR(50),
    processing_time_seconds FLOAT,
    gpu_memory_used_mb FLOAT,

    -- Image metadata
    image_dimensions JSONB, -- {width: 2480, height: 3508}
    file_size_bytes BIGINT,
    base_size INTEGER,
    image_size INTEGER,
    crop_mode BOOLEAN,

    -- Paths to generated files
    markdown_path TEXT,
    boxes_image_path TEXT,
    entities_json_path TEXT,

    -- Embeddings for semantic search (1024 dimensions for multilingual-e5-large)
    embedding vector(1024),

    -- Medical entities (stored as JSONB for flexibility)
    medical_entities JSONB,
    entity_count INTEGER DEFAULT 0,

    -- Timestamps
    processed_at TIMESTAMP DEFAULT NOW(),
    uploaded_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- User tracking
    processed_by INTEGER REFERENCES users(id),
    uploaded_by INTEGER REFERENCES users(id),

    -- Soft delete for HIPAA compliance (never actually delete data)
    deleted BOOLEAN DEFAULT false,
    deleted_at TIMESTAMP,
    deleted_by INTEGER REFERENCES users(id),

    -- Patient linking (anonymized patient ID)
    patient_id VARCHAR(100), -- Hash or anonymized ID
    encounter_id VARCHAR(100),

    -- Additional metadata
    metadata JSONB
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_clinical_docs_type ON clinical_documents(document_type);
CREATE INDEX IF NOT EXISTS idx_clinical_docs_patient ON clinical_documents(patient_id);
CREATE INDEX IF NOT EXISTS idx_clinical_docs_encounter ON clinical_documents(encounter_id);
CREATE INDEX IF NOT EXISTS idx_clinical_docs_processed_at ON clinical_documents(processed_at);
CREATE INDEX IF NOT EXISTS idx_clinical_docs_deleted ON clinical_documents(deleted);
CREATE INDEX IF NOT EXISTS idx_clinical_docs_uuid ON clinical_documents(document_uuid);

-- Full-text search index on OCR text (Spanish language)
CREATE INDEX IF NOT EXISTS idx_clinical_docs_ocr_text_fts
    ON clinical_documents
    USING GIN(to_tsvector('spanish', ocr_text));

-- GIN index for medical_entities JSONB queries
CREATE INDEX IF NOT EXISTS idx_clinical_docs_entities
    ON clinical_documents
    USING GIN(medical_entities);

-- pgvector index for similarity search (HNSW for better performance)
CREATE INDEX IF NOT EXISTS idx_clinical_docs_embedding
    ON clinical_documents
    USING hnsw(embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- MEDICAL ENTITIES TABLE (Normalized)
-- ============================================================================

CREATE TABLE IF NOT EXISTS medical_entities (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES clinical_documents(id) ON DELETE CASCADE,

    -- Entity type and value
    entity_type VARCHAR(50) NOT NULL, -- diagnostico, medicamento, signo_vital, etc.
    entity_value TEXT NOT NULL,

    -- Entity metadata
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    start_pos INTEGER,
    end_pos INTEGER,
    context_before TEXT,
    context_after TEXT,

    -- For medications
    medication_name VARCHAR(255),
    medication_dose VARCHAR(100),
    medication_frequency VARCHAR(100),
    medication_duration VARCHAR(100),
    medication_route VARCHAR(50),

    -- For vital signs
    vital_sign_name VARCHAR(100),
    vital_sign_value VARCHAR(50),
    vital_sign_unit VARCHAR(20),
    vital_sign_numeric FLOAT,

    -- For lab results
    lab_test_name VARCHAR(255),
    lab_test_value VARCHAR(50),
    lab_test_unit VARCHAR(50),
    lab_test_numeric FLOAT,
    lab_range_min FLOAT,
    lab_range_max FLOAT,

    -- For dates
    date_value DATE,
    date_type VARCHAR(50), -- ingreso, alta, cirugia, etc.

    -- Standardized codes (future: ICD-10, SNOMED-CT)
    icd10_code VARCHAR(20),
    snomed_code VARCHAR(50),

    -- Timestamps
    extracted_at TIMESTAMP DEFAULT NOW(),

    -- Additional metadata
    metadata JSONB
);

-- Indexes for entity queries
CREATE INDEX IF NOT EXISTS idx_entities_document ON medical_entities(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON medical_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_value ON medical_entities(entity_value);
CREATE INDEX IF NOT EXISTS idx_entities_med_name ON medical_entities(medication_name);
CREATE INDEX IF NOT EXISTS idx_entities_vital_name ON medical_entities(vital_sign_name);
CREATE INDEX IF NOT EXISTS idx_entities_lab_name ON medical_entities(lab_test_name);
CREATE INDEX IF NOT EXISTS idx_entities_date ON medical_entities(date_value);

-- Full-text search on entity values
CREATE INDEX IF NOT EXISTS idx_entities_value_fts
    ON medical_entities
    USING GIN(to_tsvector('spanish', entity_value));

-- ============================================================================
-- AUDIT LOGS TABLE (HIPAA Compliance)
-- ============================================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,

    -- Event identification
    event_uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- CREATE, READ, UPDATE, DELETE, SEARCH, EXPORT
    event_category VARCHAR(50), -- DOCUMENT_ACCESS, ENTITY_EXTRACTION, DATABASE_QUERY

    -- User information
    user_id INTEGER REFERENCES users(id),
    username VARCHAR(100),
    user_role VARCHAR(50),
    user_ip VARCHAR(45), -- IPv6 compatible

    -- Resource information
    resource_type VARCHAR(50), -- clinical_document, medical_entity, user
    resource_id INTEGER,
    document_uuid UUID,
    patient_id VARCHAR(100),

    -- Action details
    action_description TEXT,
    query_executed TEXT, -- SQL query (anonymized)
    filters_applied JSONB, -- Search filters used

    -- Results
    records_affected INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,

    -- Timing
    timestamp TIMESTAMP DEFAULT NOW(),
    duration_ms INTEGER,

    -- Session information
    session_id VARCHAR(100),
    application VARCHAR(50) DEFAULT 'deepseek-ocr-batch',

    -- Additional context
    metadata JSONB
);

-- Indexes for audit queries
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_username ON audit_logs(username);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_document ON audit_logs(document_uuid);
CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_success ON audit_logs(success);

-- ============================================================================
-- DOCUMENT VERSIONS TABLE (Track changes)
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_versions (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES clinical_documents(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,

    -- Snapshot of document state
    ocr_text TEXT,
    medical_entities JSONB,
    metadata JSONB,

    -- Change tracking
    changed_by INTEGER REFERENCES users(id),
    change_reason TEXT,
    changed_at TIMESTAMP DEFAULT NOW(),

    -- Checksum for integrity
    content_hash VARCHAR(64), -- SHA-256

    UNIQUE(document_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_versions_document ON document_versions(document_id);
CREATE INDEX IF NOT EXISTS idx_versions_changed_at ON document_versions(changed_at);

-- ============================================================================
-- SEARCH QUERIES TABLE (Track search patterns)
-- ============================================================================

CREATE TABLE IF NOT EXISTS search_queries (
    id SERIAL PRIMARY KEY,

    -- Query information
    query_text TEXT,
    query_type VARCHAR(50), -- full_text, semantic, entity_based, sql
    query_embedding vector(1024),

    -- Filters used
    filters JSONB,

    -- Results
    results_count INTEGER,
    execution_time_ms INTEGER,

    -- User tracking
    user_id INTEGER REFERENCES users(id),
    timestamp TIMESTAMP DEFAULT NOW(),

    -- Metadata
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_search_user ON search_queries(user_id);
CREATE INDEX IF NOT EXISTS idx_search_timestamp ON search_queries(timestamp);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for clinical_documents
CREATE TRIGGER update_clinical_documents_updated_at
    BEFORE UPDATE ON clinical_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for users
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function for semantic search
CREATE OR REPLACE FUNCTION search_similar_documents(
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    document_id integer,
    document_uuid uuid,
    filename varchar,
    document_type varchar,
    ocr_text text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cd.id,
        cd.document_uuid,
        cd.filename,
        cd.document_type,
        cd.ocr_text,
        1 - (cd.embedding <=> query_embedding) as similarity
    FROM clinical_documents cd
    WHERE
        cd.embedding IS NOT NULL
        AND cd.deleted = false
        AND 1 - (cd.embedding <=> query_embedding) > match_threshold
    ORDER BY cd.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function to log audit events
CREATE OR REPLACE FUNCTION log_audit_event(
    p_event_type varchar,
    p_user_id integer,
    p_resource_type varchar,
    p_resource_id integer,
    p_action_description text,
    p_metadata jsonb DEFAULT '{}'::jsonb
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    v_event_uuid uuid;
BEGIN
    INSERT INTO audit_logs (
        event_type,
        user_id,
        resource_type,
        resource_id,
        action_description,
        metadata
    ) VALUES (
        p_event_type,
        p_user_id,
        p_resource_type,
        p_resource_id,
        p_action_description,
        p_metadata
    ) RETURNING event_uuid INTO v_event_uuid;

    RETURN v_event_uuid;
END;
$$;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View for recent documents with entity counts
CREATE OR REPLACE VIEW recent_documents_summary AS
SELECT
    cd.id,
    cd.document_uuid,
    cd.filename,
    cd.document_type,
    cd.ocr_confidence,
    cd.char_count,
    cd.word_count,
    cd.entity_count,
    cd.processed_at,
    cd.patient_id,
    u.username as processed_by_username,
    COUNT(me.id) as extracted_entities_count
FROM clinical_documents cd
LEFT JOIN users u ON cd.processed_by = u.id
LEFT JOIN medical_entities me ON cd.id = me.document_id
WHERE cd.deleted = false
GROUP BY cd.id, u.username
ORDER BY cd.processed_at DESC;

-- View for audit summary by user
CREATE OR REPLACE VIEW audit_summary_by_user AS
SELECT
    u.username,
    u.full_name,
    u.role,
    COUNT(*) as total_events,
    COUNT(CASE WHEN al.event_type = 'READ' THEN 1 END) as read_events,
    COUNT(CASE WHEN al.event_type = 'CREATE' THEN 1 END) as create_events,
    COUNT(CASE WHEN al.event_type = 'UPDATE' THEN 1 END) as update_events,
    COUNT(CASE WHEN al.event_type = 'DELETE' THEN 1 END) as delete_events,
    COUNT(CASE WHEN al.success = false THEN 1 END) as failed_events,
    MIN(al.timestamp) as first_event,
    MAX(al.timestamp) as last_event
FROM users u
LEFT JOIN audit_logs al ON u.id = al.user_id
GROUP BY u.id, u.username, u.full_name, u.role
ORDER BY total_events DESC;

-- ============================================================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================================================

-- Insert default admin user (password should be hashed in production)
INSERT INTO users (username, email, full_name, role, department)
VALUES
    ('admin', 'admin@hospital.pe', 'Administrador del Sistema', 'admin', 'IT'),
    ('dr_garcia', 'garcia@hospital.pe', 'Dr. Carlos García', 'doctor', 'Cardiología'),
    ('nurse_lopez', 'lopez@hospital.pe', 'Enf. María López', 'nurse', 'Emergencias')
ON CONFLICT (username) DO NOTHING;

-- ============================================================================
-- GRANTS (Adjust for your security model)
-- ============================================================================

-- Create roles (example - adjust for your setup)
-- CREATE ROLE medical_app_user WITH LOGIN PASSWORD 'your_secure_password';
-- GRANT CONNECT ON DATABASE medical_records TO medical_app_user;
-- GRANT USAGE ON SCHEMA public TO medical_app_user;
-- GRANT SELECT, INSERT, UPDATE ON clinical_documents TO medical_app_user;
-- GRANT SELECT, INSERT ON medical_entities TO medical_app_user;
-- GRANT SELECT, INSERT ON audit_logs TO medical_app_user;

-- ============================================================================
-- INDEXES FOR COMMON QUERIES
-- ============================================================================

-- Composite index for date range + document type queries
CREATE INDEX IF NOT EXISTS idx_clinical_docs_type_date
    ON clinical_documents(document_type, processed_at);

-- Index for patient document retrieval
CREATE INDEX IF NOT EXISTS idx_clinical_docs_patient_date
    ON clinical_documents(patient_id, processed_at);

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE clinical_documents IS 'Main table for storing OCR-processed medical documents with pgvector embeddings';
COMMENT ON TABLE medical_entities IS 'Normalized medical entities extracted from documents (diagnoses, medications, vital signs, etc.)';
COMMENT ON TABLE audit_logs IS 'HIPAA-compliant audit trail for all document access and modifications';
COMMENT ON TABLE users IS 'System users (doctors, nurses, administrators, analysts)';
COMMENT ON COLUMN clinical_documents.embedding IS 'Document embedding (1024-dim) from multilingual-e5-large-instruct for semantic search';
COMMENT ON COLUMN clinical_documents.medical_entities IS 'Denormalized medical entities as JSONB for fast access';
COMMENT ON COLUMN audit_logs.event_uuid IS 'Unique identifier for each audit event (immutable)';

-- ============================================================================
-- VERIFICATION QUERIES
-- ============================================================================

-- Check if pgvector is installed
-- SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check table sizes
-- SELECT
--     schemaname,
--     tablename,
--     pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
-- FROM pg_tables
-- WHERE schemaname = 'public'
-- ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
