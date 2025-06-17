from pydantic import BaseModel
from typing import List

class NewsletterItem(BaseModel):
    title: str
    desc: str
    url: str

class IngestNewsletterHtml(BaseModel):
    body: str
    email_to: str

class IngestNewsletterItem(BaseModel):
    newsletter: List[NewsletterItem]
    email_to: str # `From:` email field
