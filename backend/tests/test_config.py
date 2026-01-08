"""
Tests for the configuration module.
"""
import pytest
import logging
from backend.app.core.config import (
    setup_logging,
    BASE_DIR,
    DATA_DIR,
    PAPERS_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)


class TestConfiguration:
    """Test suite for configuration settings."""

    def test_directories_exist(self):
        """Test that required directories exist."""
        assert BASE_DIR.exists()
        # DATA_DIR might not exist in test environment
        # but PAPERS_DIR and CHROMA_DIR should be created

    def test_chunk_settings_valid(self):
        """Test that chunk settings are valid."""
        assert CHUNK_SIZE > 0
        assert CHUNK_OVERLAP >= 0
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_retrieval_settings_valid(self):
        """Test that retrieval settings are valid."""
        assert TOP_K_RESULTS > 0
        assert TOP_K_RESULTS <= 100  # Reasonable upper limit


class TestLoggingSetup:
    """Test suite for logging configuration."""

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger."""
        logger = setup_logging("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_logger_has_handlers(self):
        """Test that logger has handlers configured."""
        logger = setup_logging("test_handlers")

        # Should have at least one handler (console)
        assert len(logger.handlers) >= 1

    def test_logger_level_set(self):
        """Test that logger level is properly set."""
        logger = setup_logging("test_level")

        # Should have a valid level
        assert logger.level in [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL
        ]

    def test_same_name_returns_same_logger(self):
        """Test that same name returns the same logger instance."""
        logger1 = setup_logging("same_name")
        logger2 = setup_logging("same_name")

        assert logger1 is logger2

    def test_different_names_different_loggers(self):
        """Test that different names return different loggers."""
        logger1 = setup_logging("name_one")
        logger2 = setup_logging("name_two")

        assert logger1.name != logger2.name
