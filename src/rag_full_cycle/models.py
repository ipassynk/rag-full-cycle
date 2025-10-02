from pydantic import BaseModel
from typing import List


class QuestionsResponse(BaseModel):
    """Response containing a list of questions"""
    questions: List[str]
