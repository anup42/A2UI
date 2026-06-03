from __future__ import annotations

import os
import ssl
import urllib.request
from functools import lru_cache
from typing import Any


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_falsey(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"0", "false", "no", "n", "off"}


@lru_cache(maxsize=8)
def _ssl_context_cached(
    disable_verify: bool,
    relax_x509_strict: bool,
    ssl_cert_file: str,
) -> ssl.SSLContext | None:
    # Default urllib behavior (fully verified TLS with system/OpenSSL defaults).
    if not disable_verify and not relax_x509_strict and not ssl_cert_file:
        return None

    cafile = ssl_cert_file or None
    ctx = ssl.create_default_context(cafile=cafile)

    if relax_x509_strict and hasattr(ssl, "VERIFY_X509_STRICT"):
        try:
            ctx.verify_flags = ctx.verify_flags & ~ssl.VERIFY_X509_STRICT  # type: ignore[attr-defined]
        except Exception:
            pass

    if disable_verify:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    return ctx


def _current_ssl_context() -> ssl.SSLContext | None:
    # Dataset generation often runs on managed GPU hosts with incomplete CA
    # bundles. Default to bypassing TLS verification; set
    # A2UI_DISABLE_SSL_VERIFY=0 to restore verified HTTPS.
    disable_verify = not _is_falsey(os.getenv("A2UI_DISABLE_SSL_VERIFY", "1"))
    relax_x509_strict = _is_truthy(os.getenv("A2UI_RELAX_X509_STRICT"))
    ssl_cert_file = (os.getenv("SSL_CERT_FILE") or "").strip()
    return _ssl_context_cached(disable_verify, relax_x509_strict, ssl_cert_file)


def urlopen(request: Any, timeout: float | int):
    ctx = _current_ssl_context()
    if ctx is None:
        return urllib.request.urlopen(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout, context=ctx)
