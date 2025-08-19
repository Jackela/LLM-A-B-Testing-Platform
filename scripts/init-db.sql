-- PostgreSQL initialization script for LLM A/B Testing Platform
-- This script creates the necessary databases for development and testing

-- Create development database
CREATE DATABASE llm_testing_dev;

-- Create test database (for CI/CD)
CREATE DATABASE test_db;

-- Create user for development (optional - uses postgres user by default)
-- CREATE USER llm_testing WITH PASSWORD 'password';
-- GRANT ALL PRIVILEGES ON DATABASE llm_testing_dev TO llm_testing;
-- GRANT ALL PRIVILEGES ON DATABASE test_db TO llm_testing;

-- Enable required extensions for the databases
\c llm_testing_dev;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

\c test_db;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Return to default database
\c postgres;