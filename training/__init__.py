"""
training package

Training algorithms and utilities.
"""

from training.curriculum import CurriculumManager
from training.curriculum_trainer import CurriculumTrainer

__all__ = [
    "CurriculumManager",
    "CurriculumTrainer",
]