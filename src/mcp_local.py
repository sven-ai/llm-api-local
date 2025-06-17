from mcp_shared import *
from utils import *


class Mcp:
    def model_for_email(self, email_to: str) -> str:
        return ""

    def newsletter_ingest(
        self, neural_searcher, item: IngestNewsletterItem
    ) -> bool:
        return False

    def ingest_html(
        self,
        background_tasks: BackgroundTasks,
        item: IngestNewsletterHtml,
        neural_searcher,
    ) -> bool:
        return False
