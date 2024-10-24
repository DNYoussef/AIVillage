"""API integration for MAGI."""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import aiohttp
import logging
import json
from pathlib import Path

from ..core.exceptions import APIError
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class APIManager:
    """
    Manages API interactions for MAGI.
    
    Responsibilities:
    - API request handling
    - Response processing
    - Error handling
    - Rate limiting
    - Authentication
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        rate_limit: int = 60,  # requests per minute
        timeout: float = 30.0  # seconds
    ):
        """
        Initialize API manager.
        
        Args:
            config_path: Path to API configuration file (optional)
            rate_limit: Maximum requests per minute
            timeout: Request timeout in seconds
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.rate_limit = rate_limit
        self.timeout = timeout
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_times: List[datetime] = []
        self.api_keys: Dict[str, str] = self.config.get('api_keys', {})
        self.base_urls: Dict[str, str] = self.config.get('base_urls', {})
        self.headers: Dict[str, Dict[str, str]] = self.config.get('headers', {})
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load API configuration from file."""
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading API config: {e}")
            return {}
    
    async def __aenter__(self):
        """Create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        now = datetime.now()
        # Remove requests older than 1 minute
        self.request_times = [
            t for t in self.request_times
            if (now - t).total_seconds() < 60
        ]
        
        if len(self.request_times) >= self.rate_limit:
            # Wait until oldest request is more than 1 minute old
            wait_time = 60 - (now - self.request_times[0]).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        self.request_times.append(now)
    
    def _get_headers(self, api_name: str) -> Dict[str, str]:
        """Get headers for API request."""
        headers = self.headers.get(api_name, {}).copy()
        
        # Add API key if available
        if api_name in self.api_keys:
            headers['Authorization'] = f"Bearer {self.api_keys[api_name]}"
        
        return headers
    
    def _get_url(self, api_name: str, endpoint: str) -> str:
        """Get full URL for API request."""
        base_url = self.base_urls.get(api_name)
        if not base_url:
            raise APIError(f"No base URL configured for API '{api_name}'")
        
        return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    async def _check_response(self, response: aiohttp.ClientResponse) -> None:
        """Check response for errors."""
        if response.status >= 400:
            error_text = await response.text()
            raise APIError(
                f"API request failed with status {response.status}: {error_text}"
            )
    
    async def get(
        self,
        api_name: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make GET request to API.
        
        Args:
            api_name: Name of API to use
            endpoint: API endpoint
            params: Query parameters (optional)
            
        Returns:
            API response
        """
        await self._check_rate_limit()
        
        if self.session is None:
            raise APIError("Session not initialized")
        
        url = self._get_url(api_name, endpoint)
        headers = self._get_headers(api_name)
        
        try:
            async with self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout
            ) as response:
                await self._check_response(response)
                return await response.json()
        except asyncio.TimeoutError:
            raise APIError(f"Request to {url} timed out")
        except Exception as e:
            raise APIError(f"Error making GET request to {url}: {str(e)}")
    
    async def post(
        self,
        api_name: str,
        endpoint: str,
        data: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make POST request to API.
        
        Args:
            api_name: Name of API to use
            endpoint: API endpoint
            data: Request data
            params: Query parameters (optional)
            
        Returns:
            API response
        """
        await self._check_rate_limit()
        
        if self.session is None:
            raise APIError("Session not initialized")
        
        url = self._get_url(api_name, endpoint)
        headers = self._get_headers(api_name)
        
        try:
            async with self.session.post(
                url,
                json=data,
                params=params,
                headers=headers,
                timeout=self.timeout
            ) as response:
                await self._check_response(response)
                return await response.json()
        except asyncio.TimeoutError:
            raise APIError(f"Request to {url} timed out")
        except Exception as e:
            raise APIError(f"Error making POST request to {url}: {str(e)}")
    
    async def put(
        self,
        api_name: str,
        endpoint: str,
        data: Dict[str, Any],
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make PUT request to API.
        
        Args:
            api_name: Name of API to use
            endpoint: API endpoint
            data: Request data
            params: Query parameters (optional)
            
        Returns:
            API response
        """
        await self._check_rate_limit()
        
        if self.session is None:
            raise APIError("Session not initialized")
        
        url = self._get_url(api_name, endpoint)
        headers = self._get_headers(api_name)
        
        try:
            async with self.session.put(
                url,
                json=data,
                params=params,
                headers=headers,
                timeout=self.timeout
            ) as response:
                await self._check_response(response)
                return await response.json()
        except asyncio.TimeoutError:
            raise APIError(f"Request to {url} timed out")
        except Exception as e:
            raise APIError(f"Error making PUT request to {url}: {str(e)}")
    
    async def delete(
        self,
        api_name: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make DELETE request to API.
        
        Args:
            api_name: Name of API to use
            endpoint: API endpoint
            params: Query parameters (optional)
            
        Returns:
            API response
        """
        await self._check_rate_limit()
        
        if self.session is None:
            raise APIError("Session not initialized")
        
        url = self._get_url(api_name, endpoint)
        headers = self._get_headers(api_name)
        
        try:
            async with self.session.delete(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout
            ) as response:
                await self._check_response(response)
                return await response.json()
        except asyncio.TimeoutError:
            raise APIError(f"Request to {url} timed out")
        except Exception as e:
            raise APIError(f"Error making DELETE request to {url}: {str(e)}")
    
    async def stream(
        self,
        api_name: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> aiohttp.ClientResponse:
        """
        Create streaming connection to API.
        
        Args:
            api_name: Name of API to use
            endpoint: API endpoint
            params: Query parameters (optional)
            
        Returns:
            Streaming response
        """
        await self._check_rate_limit()
        
        if self.session is None:
            raise APIError("Session not initialized")
        
        url = self._get_url(api_name, endpoint)
        headers = self._get_headers(api_name)
        
        try:
            response = await self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            await self._check_response(response)
            return response
        except Exception as e:
            raise APIError(f"Error creating stream to {url}: {str(e)}")

# Example usage
if __name__ == "__main__":
    async def main():
        # Create API manager
        api = APIManager("api_config.json")
        
        # Make requests using context manager
        async with api:
            # Make GET request
            try:
                data = await api.get(
                    "example_api",
                    "users",
                    params={"page": 1}
                )
                print(f"GET response: {data}")
            except APIError as e:
                print(f"Error: {e}")
            
            # Make POST request
            try:
                data = await api.post(
                    "example_api",
                    "users",
                    data={"name": "John Doe"}
                )
                print(f"POST response: {data}")
            except APIError as e:
                print(f"Error: {e}")
            
            # Use streaming endpoint
            try:
                async with await api.stream("example_api", "events") as response:
                    async for line in response.content:
                        print(f"Event: {line.decode()}")
            except APIError as e:
                print(f"Error: {e}")
    
    asyncio.run(main())
