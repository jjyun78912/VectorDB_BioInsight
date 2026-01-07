"""
Base Agent Class for RNA-seq Pipeline

All agents inherit from this base class for consistent:
- Input/Output handling
- Logging
- Error handling
- Metadata generation
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd


class BaseAgent(ABC):
    """Base class for all pipeline agents."""

    def __init__(
        self,
        agent_name: str,
        input_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_name = agent_name
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = self._setup_logging()

        # Track execution
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.success: bool = False
        self.errors: list = []

    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging."""
        logger = logging.getLogger(self.agent_name)
        logger.setLevel(logging.DEBUG)

        # File handler
        log_file = self.output_dir / f"log_{self.agent_name}.txt"
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def load_csv(self, filename: str, required: bool = True) -> Optional[pd.DataFrame]:
        """Load CSV file from input directory."""
        filepath = self.input_dir / filename

        if not filepath.exists():
            if required:
                raise FileNotFoundError(f"Required input file not found: {filepath}")
            self.logger.warning(f"Optional file not found: {filepath}")
            return None

        self.logger.info(f"Loading {filename}...")
        df = pd.read_csv(filepath)
        self.logger.info(f"  -> {len(df)} rows, {len(df.columns)} columns")
        return df

    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Save DataFrame to CSV in output directory."""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved {filename}: {len(df)} rows")
        return filepath

    def load_json(self, filename: str, required: bool = True) -> Optional[Dict]:
        """Load JSON file from input directory."""
        filepath = self.input_dir / filename

        if not filepath.exists():
            if required:
                raise FileNotFoundError(f"Required input file not found: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_json(self, data: Dict, filename: str) -> Path:
        """Save dictionary to JSON in output directory."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        self.logger.info(f"Saved {filename}")
        return filepath

    def generate_metadata(self, **kwargs) -> Dict[str, Any]:
        """Generate agent metadata."""
        metadata = {
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.end_time and self.start_time else None
            ),
            "success": self.success,
            "errors": self.errors,
            "config_used": self.config,
            **kwargs
        }
        return metadata

    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate that all required inputs are present and valid."""
        pass

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the agent's main logic. Returns results dict."""
        pass

    @abstractmethod
    def validate_outputs(self) -> bool:
        """Validate that all required outputs were generated correctly."""
        pass

    def execute(self) -> Dict[str, Any]:
        """Full execution with validation and error handling."""
        self.start_time = datetime.now()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting {self.agent_name}")
        self.logger.info(f"{'='*60}")

        try:
            # Validate inputs
            self.logger.info("Validating inputs...")
            if not self.validate_inputs():
                raise ValueError("Input validation failed")
            self.logger.info("Input validation passed")

            # Run main logic
            self.logger.info("Running analysis...")
            results = self.run()

            # Validate outputs
            self.logger.info("Validating outputs...")
            if not self.validate_outputs():
                raise ValueError("Output validation failed")
            self.logger.info("Output validation passed")

            self.success = True
            self.logger.info(f"{self.agent_name} completed successfully!")

        except Exception as e:
            self.success = False
            self.errors.append(str(e))
            self.logger.error(f"Error in {self.agent_name}: {e}")
            raise

        finally:
            self.end_time = datetime.now()

            # Save metadata
            metadata = self.generate_metadata(**results if self.success else {})
            self.save_json(metadata, f"meta_{self.agent_name}.json")

        return results


class AgentResult:
    """Container for agent execution results."""

    def __init__(
        self,
        agent_name: str,
        success: bool,
        output_dir: Path,
        metadata: Dict[str, Any],
        errors: Optional[list] = None
    ):
        self.agent_name = agent_name
        self.success = success
        self.output_dir = output_dir
        self.metadata = metadata
        self.errors = errors or []

    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return f"AgentResult({self.agent_name}: {status})"
