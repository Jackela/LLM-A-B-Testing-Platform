-- Performance Enhancement: Composite Indexes for Analytics and Heavy Queries
-- This script adds optimized indexes to improve query performance for analytics workloads

-- Performance analytics composite index
-- Improves queries that filter by test_id, model_config_id, and status
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_model_responses_analytics 
ON model_responses (test_id, model_config_id, status, request_time);

-- Evaluation results performance index
-- Optimizes analytics queries that need evaluation results with timing
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evaluation_results_performance 
ON evaluation_results (test_id, model_response_id, completed_at);

-- Test samples batch processing index
-- Speeds up sample counting and batch operations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_test_samples_batch_processing
ON test_samples (test_id, status, created_at);

-- User activity and audit index
-- Improves user activity queries and audit trail performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_activity_audit
ON tests (user_id, status, created_at DESC);

-- Time-series analytics index
-- Optimizes time-based analytics and reporting queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_analytics_time_series
ON model_responses (created_at DESC, test_id, status)
WHERE status IN ('completed', 'success');

-- Multi-model comparison index
-- Speeds up queries comparing multiple models in the same test
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_multi_model_comparison
ON model_responses (test_id, model_config_id, status, response_time, cost)
WHERE status = 'completed';

-- Evaluation dimensions index
-- Improves queries filtering by evaluation dimensions and scores
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_evaluation_dimensions
ON evaluation_results (test_id, dimension, score DESC, completed_at DESC);

-- Status transition tracking index
-- Optimizes queries tracking test and response status changes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_status_transitions
ON model_responses (status, updated_at DESC, test_id);

-- Error analysis index
-- Speeds up error analysis and debugging queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_error_analysis
ON model_responses (status, error_type, test_id, created_at DESC)
WHERE status IN ('failed', 'error', 'timeout');

-- Cost analysis and optimization index
-- Improves cost tracking and optimization queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cost_analysis
ON model_responses (test_id, provider_name, cost DESC, tokens_used DESC, created_at DESC)
WHERE cost IS NOT NULL AND tokens_used IS NOT NULL;

-- ANALYZE tables to update statistics after index creation
ANALYZE model_responses;
ANALYZE evaluation_results;
ANALYZE test_samples;
ANALYZE tests;