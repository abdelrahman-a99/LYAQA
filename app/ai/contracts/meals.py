"""Structured schemas for generated meals and food substitutions."""

from typing import List

from pydantic import BaseModel, ConfigDict

from app.ai.contracts.nlp import FoodItem


class CandidateMeal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    foods: List[FoodItem]


class SubstituteFood(BaseModel):
    model_config = ConfigDict(extra="forbid")
    food: FoodItem
