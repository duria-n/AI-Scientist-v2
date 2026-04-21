import os
import warnings

PROXY_ENV_VARS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)
_WARNED_PROXY_ENV_VARS: set[str] = set()


def normalize_httpx_proxy_env() -> None:
    """Rewrite unsupported socks:// proxy env vars to socks5:// for httpx."""
    for env_var in PROXY_ENV_VARS:
        proxy_url = os.environ.get(env_var)
        if not proxy_url or not proxy_url.startswith("socks://"):
            continue

        os.environ[env_var] = f"socks5://{proxy_url[len('socks://'):]}"
        if env_var not in _WARNED_PROXY_ENV_VARS:
            warnings.warn(
                f"{env_var}={proxy_url!r} uses unsupported proxy scheme 'socks://'. "
                "Rewriting it to 'socks5://' for OpenAI/httpx compatibility."
            )
            _WARNED_PROXY_ENV_VARS.add(env_var)
