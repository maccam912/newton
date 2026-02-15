"""Tracing — thin wrapper around OpenTelemetry.

Logfire configures the OTEL TracerProvider automatically via
`logfire.configure()` in main.py.  All spans created here flow
through Logfire and out to whatever backend it's pointed at.

Set LOGFIRE_TOKEN in .env to send traces to Logfire cloud,
or configure send_to_logfire="if-token-present" to stay local.
"""

from __future__ import annotations

from opentelemetry import trace


def get_tracer(name: str) -> trace.Tracer:
    """Get a named tracer for a module."""
    return trace.get_tracer(name)
