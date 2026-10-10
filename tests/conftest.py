"""pytest 公共夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import Settings  # noqa: E402
from healthcare_rag.pipeline import HealthcareRAG  # noqa: E402


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="session")
def pipeline(settings: Settings) -> HealthcareRAG:
    return HealthcareRAG(settings)