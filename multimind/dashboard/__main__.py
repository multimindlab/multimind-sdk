"""`python -m multimind.dashboard` — start the governance dashboard."""

from .server import DashboardSettings, start

if __name__ == "__main__":
    settings = DashboardSettings.from_env()
    print(f"MultiMind dashboard: http://{settings.host}:{settings.port}")
    start(settings)
