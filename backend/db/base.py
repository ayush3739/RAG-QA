from backend.core.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = settings.DATABASE_URL

# engine
engine = create_async_engine(DATABASE_URL)

# Session 
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_= AsyncSession,
    expire_on_commit= False
)

#Base
class Base (DeclarativeBase):
    pass

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session