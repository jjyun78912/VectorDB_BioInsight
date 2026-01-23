"""
Base API Client with common functionality.

Provides:
- Rate limiting
- Retry logic
- Caching
- Error handling
"""

import asyncio
import aiohttp
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Standardized API response."""
    success: bool
    data: Any
    source: str
    query: str
    cached: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseAPIClient(ABC):
    """
    Base class for external API clients.

    Features:
    - Rate limiting to respect API limits
    - Automatic retry with exponential backoff
    - Local caching to reduce API calls
    - Async support for parallel requests
    """

    # Override in subclasses
    BASE_URL: str = ""
    API_NAME: str = "BaseAPI"
    RATE_LIMIT_DELAY: float = 0.5  # seconds between requests
    MAX_RETRIES: int = 3
    CACHE_DIR: Path = Path.home() / ".bioinsight_cache"

    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache_ttl: int = 86400,  # 24 hours
        timeout: int = 30
    ):
        self.api_key = api_key
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._last_request_time = 0
        self._session: Optional[aiohttp.ClientSession] = None

        # Initialize cache directory
        if enable_cache:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            await asyncio.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key from endpoint and params."""
        key_str = f"{self.API_NAME}:{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.CACHE_DIR / f"{self.API_NAME}_{cache_key}.json"

    def _read_cache(self, cache_key: str) -> Optional[Dict]:
        """Read from cache if valid."""
        if not self.enable_cache:
            return None

        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)

            # Check TTL
            if time.time() - cached.get('timestamp', 0) > self.cache_ttl:
                cache_path.unlink()  # Remove expired cache
                return None

            return cached.get('data')
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def _write_cache(self, cache_key: str, data: Any):
        """Write to cache."""
        if not self.enable_cache:
            return

        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make HTTP request with retry logic."""
        url = f"{self.BASE_URL}/{endpoint}" if not endpoint.startswith("http") else endpoint
        params = params or {}
        headers = headers or {}

        # Add API key if available
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        session = await self._get_session()

        for attempt in range(self.MAX_RETRIES):
            await self._rate_limit()

            try:
                if method.upper() == "GET":
                    async with session.get(url, params=params, headers=headers) as resp:
                        if resp.status == 200:
                            # Handle non-standard content types (e.g., STRING returns text/json)
                            try:
                                return await resp.json()
                            except aiohttp.ContentTypeError:
                                text = await resp.text()
                                return json.loads(text)
                        elif resp.status == 429:  # Rate limited
                            wait_time = 2 ** attempt
                            logger.warning(f"{self.API_NAME} rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.warning(f"{self.API_NAME} request failed: {resp.status}")
                            return None

                elif method.upper() == "POST":
                    async with session.post(url, json=data, headers=headers) as resp:
                        if resp.status == 200:
                            try:
                                return await resp.json()
                            except aiohttp.ContentTypeError:
                                text = await resp.text()
                                return json.loads(text)
                        else:
                            logger.warning(f"{self.API_NAME} POST failed: {resp.status}")
                            return None

            except asyncio.TimeoutError:
                logger.warning(f"{self.API_NAME} timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"{self.API_NAME} error: {e}")

            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

        return None

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> APIResponse:
        """GET request with caching."""
        params = params or {}

        # Check cache
        if use_cache and self.enable_cache:
            cache_key = self._get_cache_key(endpoint, params)
            cached_data = self._read_cache(cache_key)
            if cached_data is not None:
                return APIResponse(
                    success=True,
                    data=cached_data,
                    source=self.API_NAME,
                    query=endpoint,
                    cached=True
                )

        # Make request
        data = await self._request("GET", endpoint, params=params)

        if data is not None:
            # Cache result
            if use_cache and self.enable_cache:
                self._write_cache(cache_key, data)

            return APIResponse(
                success=True,
                data=data,
                source=self.API_NAME,
                query=endpoint
            )

        return APIResponse(
            success=False,
            data=None,
            source=self.API_NAME,
            query=endpoint,
            error="Request failed"
        )

    @abstractmethod
    async def search_gene(self, gene_symbol: str) -> APIResponse:
        """Search for gene information. Override in subclasses."""
        pass

    @abstractmethod
    async def batch_search(self, gene_symbols: List[str]) -> Dict[str, APIResponse]:
        """Batch search for multiple genes. Override in subclasses."""
        pass
