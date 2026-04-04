from pydantic import BaseModel


class Observation(BaseModel):
    subject: str
    body: str
    sender_type: str


class Action(BaseModel):
    category: str
    priority: str
    spam: bool


class Reward(BaseModel):
    score: float