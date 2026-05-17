"""Session Service — SQLite chat history storage."""


class SessionService:
    """Manage chat sessions and history."""
    
    def __init__(self):
        # TODO: Initialize SQLite connection
        pass
    
    async def create_session(self, collection_name: str) -> str:
        """Create new session. Returns session_id."""
        pass
    
    async def add_message(self, session_id: str, role: str, content: str, sources: list = [], confidence: float = 0):
        """Add message to session history."""
        pass
    
    async def get_history(self, session_id: str, limit: int = 10):
        """Get last N messages from session."""
        pass
    
    async def list_sessions(self, collection_name: str):
        """List all sessions for a collection."""
        pass
    
    async def delete_session(self, session_id: str):
        """Delete a session."""
        pass
