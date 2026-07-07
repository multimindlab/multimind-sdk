"""Local AI governance dashboard (requires the [gateway] extras)."""

__all__ = ["DashboardSettings", "GuardrailsConfig", "create_dashboard_app", "start"]


def __getattr__(name):
    # Lazy so plain `import multimind.dashboard` works without fastapi installed
    if name in __all__:
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
