"""Loads settings from the .env file so secrets never live in code."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    SN_INSTANCE: str = os.getenv("SN_INSTANCE", "").rstrip("/")
    SN_USER: str = os.getenv("SN_USER", "")
    SN_PASSWORD: str = os.getenv("SN_PASSWORD", "")


settings = Settings()
