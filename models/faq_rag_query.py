from pydantic import BaseModel


class FAQQueryInput(BaseModel):
    text: str


class FAQQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
