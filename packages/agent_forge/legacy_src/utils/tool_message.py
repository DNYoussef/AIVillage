from pydantic import BaseModel


class ToolMessage(BaseModel):
    request: str
    purpose: str

    def handle(self) -> None:
        """Handle the message.

        Subclasses can override this with specific behavior.
        """
        return
