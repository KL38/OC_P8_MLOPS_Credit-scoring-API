"""Neon / PostgreSQL layer for prediction logging.

Two tables share the same schema (PredictionLog):
- predictions_log: production logs, read by the monitoring dashboard
- predictions_log_test: dedicated for CI/integration tests, truncated after runs
"""
