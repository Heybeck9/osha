#!/usr/bin/env python3
"""
WADE Intel Query - Secure information gathering with Tor/proxy support
Enables secure, anonymous querying of various information sources
"""

import os
import json
import logging
import asyncio
import time
import random
import hashlib
import base64
from typing import Dict, List, Optional, Any, Union
import re
import urllib.parse

# Network libraries
import requests
import httpx
import aiohttp

# Tor support
try:
    import socks
    import socket
    from stem import Signal
    from stem.control import Controller
    TOR_AVAILABLE = True
except ImportError:
    TOR_AVAILABLE = False

# Import WADE components
try:
    from settings_manager import settings_manager
    from performance import query_cache, connection_pool
    from security import request_signer, credential_manager
except ImportError:
    # For standalone testing
    from wade_env.settings_manager import settings_manager
    from wade_env.performance import query_cache, connection_pool
    from wade_env.security import request_signer, credential_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intel_query")

class IntelQuery:
    """
    Intelligence query manager with Tor/proxy support
    Enables secure, anonymous querying of various information sources
    """
    
    def __init__(self):
        """Initialize the intel query manager"""
        self.settings = self._load_intel_settings()
        self.tor_enabled = self.settings.get("tor_enabled", False) and TOR_AVAILABLE
        self.proxy_enabled = self.settings.get("proxy_enabled", False)
        self.proxy_url = self.settings.get("proxy_url", "")
        self.user_agents = self._load_user_agents()
        self.tor_port = self.settings.get("tor_port", 9050)
        self.tor_control_port = self.settings.get("tor_control_port", 9051)
        self.tor_password = self.settings.get("tor_password", "")
        self.search_engines = self.settings.get("search_engines", ["duckduckgo", "brave"])
        self.rate_limit = self.settings.get("rate_limit_seconds", 2)
        self.last_request_time = 0
        self.query_history = []
        self.security_events = []
        self.performance_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "query_times": [],
            "errors": 0
        }
        
        # Initialize Tor if enabled
        if self.tor_enabled:
            self._setup_tor()
    
    def _load_intel_settings(self) -> Dict[str, Any]:
        """Load intel query settings from settings manager"""
        try:
            intel_settings = settings_manager.get_settings_dict().get("intel_query", {})
            if not intel_settings:
                # Initialize with defaults if not present
                intel_settings = {
                    "tor_enabled": False,
                    "proxy_enabled": False,
                    "proxy_url": "",
                    "tor_port": 9050,
                    "tor_control_port": 9051,
                    "tor_password": "",
                    "search_engines": ["duckduckgo", "brave"],
                    "rate_limit_seconds": 2,
                    "max_results": 20,
                    "timeout_seconds": 30,
                    "user_agent_rotation": True,
                    "safe_search": True,
                    "allowed_domains": [],
                    "blocked_domains": []
                }
                settings_manager.update_settings("intel_query", intel_settings)
            return intel_settings
        except Exception as e:
            logger.error(f"Error loading intel settings: {e}")
            return {
                "tor_enabled": False,
                "proxy_enabled": False,
                "proxy_url": "",
                "tor_port": 9050,
                "tor_control_port": 9051,
                "tor_password": "",
                "search_engines": ["duckduckgo", "brave"],
                "rate_limit_seconds": 2,
                "max_results": 20,
                "timeout_seconds": 30,
                "user_agent_rotation": True,
                "safe_search": True
            }
    
    def _load_user_agents(self) -> List[str]:
        """Load a list of user agents for rotation"""
        default_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
        ]
        
        custom_agents = self.settings.get("user_agents", [])
        return custom_agents if custom_agents else default_agents
    
    def _setup_tor(self) -> bool:
        """Set up Tor connection if available"""
        if not TOR_AVAILABLE:
            logger.warning("Tor libraries not available. Install stem and PySocks to enable Tor support.")
            return False
        
        try:
            # Configure socket to use SOCKS proxy
            socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", self.tor_port)
            socket.socket = socks.socksocket
            
            # Test Tor connection
            response = requests.get("https://check.torproject.org/api/ip")
            data = response.json()
            
            if data.get("IsTor", False):
                logger.info("Tor connection successful")
                return True
            else:
                logger.warning("Connected to proxy, but not detected as Tor")
                return False
        except Exception as e:
            logger.error(f"Error setting up Tor: {e}")
            return False
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent from the list"""
        if not self.user_agents:
            return "WADE-Intel-Query/1.0"
        return random.choice(self.user_agents)
    
    async def _rotate_tor_identity(self) -> bool:
        """Rotate Tor identity by requesting a new circuit"""
        if not TOR_AVAILABLE or not self.tor_enabled:
            return False
        
        try:
            with Controller.from_port(port=self.tor_control_port) as controller:
                if self.tor_password:
                    controller.authenticate(password=self.tor_password)
                else:
                    controller.authenticate()
                
                controller.signal(Signal.NEWNYM)
                logger.info("Tor identity rotated successfully")
                
                # Wait for the new identity to be established
                await asyncio.sleep(2)
                return True
        except Exception as e:
            logger.error(f"Error rotating Tor identity: {e}")
            return False
    
    def _get_proxy_settings(self) -> Dict[str, str]:
        """Get proxy settings for requests"""
        if self.tor_enabled:
            return {
                "http": "socks5://127.0.0.1:{}".format(self.tor_port),
                "https": "socks5://127.0.0.1:{}".format(self.tor_port)
            }
        elif self.proxy_enabled and self.proxy_url:
            return {
                "http": self.proxy_url,
                "https": self.proxy_url
            }
        return {}
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting to avoid being blocked"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            delay = self.rate_limit - time_since_last
            await asyncio.sleep(delay)
        
        self.last_request_time = time.time()
    
    async def search(self, query: str, search_type: str = "general", 
                   max_results: Optional[int] = None, 
                   safe_search: Optional[bool] = None,
                   bypass_cache: bool = False) -> Dict[str, Any]:
        """
        Perform a search query using configured search engines
        Returns search results from multiple sources
        """
        start_time = time.time()
        
        if max_results is None:
            max_results = self.settings.get("max_results", 20)
        
        if safe_search is None:
            safe_search = self.settings.get("safe_search", True)
        
        # Sanitize query
        sanitized_query = self._sanitize_query(query)
        
        # Generate query ID
        query_id = self._generate_query_id(sanitized_query)
        
        # Check cache first (unless bypass_cache is True)
        if not bypass_cache and search_type != "dark":  # Don't cache dark web searches for security
            cache_key = f"search:{search_type}:{sanitized_query}:{safe_search}:{max_results}"
            cached_result = await query_cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for query: {sanitized_query}")
                # Update timestamp on cached result
                cached_result["timestamp"] = time.time()
                cached_result["cached"] = True
                cached_result["cache_age"] = time.time() - cached_result.get("original_timestamp", time.time())
                return cached_result
        
        # Apply rate limiting
        await self._rate_limit()
        
        # Log the query
        self.query_history.append({
            "query_id": query_id,
            "query": sanitized_query,
            "search_type": search_type,
            "timestamp": time.time(),
            "tor_enabled": self.tor_enabled,
            "proxy_enabled": self.proxy_enabled
        })
        
        # Sign the request for security
        payload = {
            "query": sanitized_query,
            "search_type": search_type,
            "max_results": max_results,
            "safe_search": safe_search
        }
        signed_payload, signature = request_signer.sign_request(payload)
        
        # Determine which search engines to use
        engines_to_use = self._select_search_engines(search_type)
        
        # Collect results from all engines
        all_results = []
        errors = []
        
        # Get HTTP session from connection pool
        http_session = await connection_pool.get_http_session()
        
        for engine in engines_to_use:
            try:
                engine_results = await self._search_with_engine(
                    engine, 
                    sanitized_query, 
                    search_type, 
                    safe_search,
                    http_session
                )
                all_results.extend(engine_results)
            except Exception as e:
                logger.error(f"Error searching with {engine}: {e}")
                errors.append({"engine": engine, "error": str(e)})
        
        # Deduplicate results
        unique_results = self._deduplicate_results(all_results)
        
        # Limit results
        limited_results = unique_results[:max_results]
        
        # Format results
        result = {
            "query_id": query_id,
            "query": sanitized_query,
            "search_type": search_type,
            "timestamp": time.time(),
            "original_timestamp": time.time(),  # For cache age calculation
            "total_results": len(unique_results),
            "returned_results": len(limited_results),
            "results": limited_results,
            "engines_used": engines_to_use,
            "errors": errors,
            "tor_enabled": self.tor_enabled,
            "proxy_enabled": self.proxy_enabled,
            "cached": False,
            "request_time": time.time() - start_time,
            "signature": signature  # Include signature for verification
        }
        
        # Cache the result (except for dark web searches)
        if search_type != "dark":
            cache_ttl = 3600  # 1 hour for general searches
            if search_type == "news":
                cache_ttl = 900  # 15 minutes for news
            elif search_type == "academic":
                cache_ttl = 86400  # 24 hours for academic
                
            cache_key = f"search:{search_type}:{sanitized_query}:{safe_search}:{max_results}"
            await query_cache.set(cache_key, result, ttl=cache_ttl)
        
        return result
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize the search query to prevent injection attacks"""
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>\'";]', '', query)
        # Limit length
        return sanitized[:500]
    
    def _generate_query_id(self, query: str) -> str:
        """Generate a unique ID for the query"""
        query_hash = hashlib.sha256()
        query_hash.update(f"{query}_{time.time()}".encode('utf-8'))
        return query_hash.hexdigest()[:16]
    
    def _select_search_engines(self, search_type: str) -> List[str]:
        """Select appropriate search engines based on search type"""
        if search_type == "dark":
            # For dark web searches, use specialized engines
            return ["torch", "ahmia"]
        elif search_type == "academic":
            return ["scholar", "semantic"]
        elif search_type == "news":
            return ["brave", "duckduckgo"]
        else:  # general
            return self.search_engines
    
    async def _search_with_engine(self, engine: str, query: str, 
                                search_type: str, safe_search: bool,
                                http_session: Optional[aiohttp.ClientSession] = None) -> List[Dict[str, Any]]:
        """Perform a search with a specific engine"""
        try:
            if engine == "duckduckgo":
                return await self._search_duckduckgo(query, safe_search, http_session)
            elif engine == "brave":
                return await self._search_brave(query, safe_search, http_session)
            elif engine == "torch":
                return await self._search_torch(query, http_session)
            elif engine == "ahmia":
                return await self._search_ahmia(query, http_session)
            elif engine == "scholar":
                return await self._search_scholar(query, http_session)
            elif engine == "semantic":
                return await self._search_semantic(query, http_session)
            else:
                logger.warning(f"Unknown search engine: {engine}")
                return []
        except Exception as e:
            logger.error(f"Error in _search_with_engine for {engine}: {e}")
            # Add security event logging
            await self._log_security_event(
                event_type="search_error",
                engine=engine,
                query=query,
                error=str(e)
            )
            return []
    
    async def _search_duckduckgo(self, query: str, safe_search: bool, 
                              http_session: Optional[aiohttp.ClientSession] = None) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        start_time = time.time()
        
        # DuckDuckGo doesn't have an official API, so we'll use the HTML endpoint
        safe = "1" if safe_search else "-1"
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}&kp={safe}"
        
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://duckduckgo.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0"
        }
        
        timeout = self.settings.get("timeout_seconds", 30)
        proxies = self._get_proxy_settings()
        
        # Check cache first
        cache_key = f"duckduckgo:{query}:{safe_search}"
        cached_result = await query_cache.get(cache_key)
        if cached_result:
            self.performance_metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for DuckDuckGo search: {query}")
            return cached_result
        
        self.performance_metrics["cache_misses"] += 1
        self.performance_metrics["total_queries"] += 1
        
        try:
            # Use provided session or get one from connection pool
            if http_session is None:
                http_session = await connection_pool.get_http_session()
            
            # Sign the request for security
            payload = {"query": query, "safe_search": safe_search}
            signed_payload, signature = request_signer.sign_request(payload)
            headers["X-Request-Signature"] = signature
            
            # Use aiohttp instead of httpx for connection pooling
            async with http_session.get(
                url, 
                headers=headers, 
                proxy=proxies.get("http") if proxies else None,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response.raise_for_status()
                response_text = await response.text()
                
                # Parse results (simplified - in a real implementation, use proper HTML parsing)
                results = []
                
                # Very basic extraction - in production, use BeautifulSoup or similar
                for match in re.finditer(r'<a class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', response_text, re.DOTALL):
                    result_url, title, snippet = match.groups()
                    
                    # Clean up the results
                    title = re.sub(r'<[^>]+>', '', title).strip()
                    snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    
                    results.append({
                        "title": title,
                        "url": result_url,
                        "snippet": snippet,
                        "source": "duckduckgo",
                        "timestamp": time.time()
                    })
                
                # Cache the results
                await query_cache.set(cache_key, results, ttl=1800)  # 30 minutes cache
                
                # Update performance metrics
                query_time = time.time() - start_time
                self.performance_metrics["query_times"].append(query_time)
                
                return results
        except Exception as e:
            self.performance_metrics["errors"] += 1
            logger.error(f"Error searching DuckDuckGo: {e}")
            
            # Log security event
            await self._log_security_event(
                event_type="search_error",
                engine="duckduckgo",
                query=query,
                error=str(e)
            )
            
            return []
    
    async def _search_brave(self, query: str, safe_search: bool) -> List[Dict[str, Any]]:
        """Search using Brave Search"""
        # Brave Search doesn't have an official API, so we'll use the JSON endpoint
        safe = "active" if safe_search else "off"
        url = f"https://search.brave.com/api/search?q={urllib.parse.quote(query)}&safesearch={safe}"
        
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://search.brave.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin"
        }
        
        timeout = self.settings.get("timeout_seconds", 30)
        proxies = self._get_proxy_settings()
        
        try:
            async with httpx.AsyncClient(proxies=proxies, timeout=timeout) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                # Extract results from JSON response
                for item in data.get("results", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("description", ""),
                        "source": "brave"
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error searching Brave: {e}")
            return []
    
    async def _search_torch(self, query: str) -> List[Dict[str, Any]]:
        """Search using Torch (Tor hidden service search engine)"""
        if not self.tor_enabled:
            logger.warning("Torch search requires Tor to be enabled")
            return []
        
        # Torch onion address (this is a placeholder - use the current address)
        url = f"http://xmh57jrzrnw6insl.onion/search?query={urllib.parse.quote(query)}"
        
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        timeout = self.settings.get("timeout_seconds", 60)  # Longer timeout for Tor
        
        try:
            # For Tor hidden services, we need to use the SOCKS proxy directly
            async with httpx.AsyncClient(
                proxies={"http://": f"socks5://127.0.0.1:{self.tor_port}"},
                timeout=timeout
            ) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                
                # Parse results (simplified)
                results = []
                
                # Very basic extraction - in production, use BeautifulSoup
                for match in re.finditer(r'<h5><a href="([^"]+)"[^>]*>(.*?)</a></h5>.*?<p>(.*?)</p>', response.text, re.DOTALL):
                    url, title, snippet = match.groups()
                    
                    # Clean up the results
                    title = re.sub(r'<[^>]+>', '', title).strip()
                    snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "torch",
                        "is_onion": ".onion" in url
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error searching Torch: {e}")
            return []
    
    async def _search_ahmia(self, query: str) -> List[Dict[str, Any]]:
        """Search using Ahmia (Tor hidden service search engine)"""
        # Ahmia can be accessed without Tor, but results are better with Tor
        url = f"https://ahmia.fi/search/?q={urllib.parse.quote(query)}"
        
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://ahmia.fi/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        timeout = self.settings.get("timeout_seconds", 30)
        proxies = self._get_proxy_settings()
        
        try:
            async with httpx.AsyncClient(proxies=proxies, timeout=timeout) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                
                # Parse results (simplified)
                results = []
                
                # Very basic extraction - in production, use BeautifulSoup
                for match in re.finditer(r'<h4><a href="([^"]+)"[^>]*>(.*?)</a></h4>.*?<p>(.*?)</p>', response.text, re.DOTALL):
                    url, title, snippet = match.groups()
                    
                    # Clean up the results
                    title = re.sub(r'<[^>]+>', '', title).strip()
                    snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "ahmia",
                        "is_onion": ".onion" in url
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error searching Ahmia: {e}")
            return []
    
    async def _search_scholar(self, query: str) -> List[Dict[str, Any]]:
        """Search using Google Scholar (for academic papers)"""
        url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(query)}&hl=en"
        
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Referer": "https://scholar.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        timeout = self.settings.get("timeout_seconds", 30)
        proxies = self._get_proxy_settings()
        
        try:
            async with httpx.AsyncClient(proxies=proxies, timeout=timeout) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                
                # Parse results (simplified)
                results = []
                
                # Very basic extraction - in production, use BeautifulSoup
                for match in re.finditer(r'<h3 class="gs_rt"><a href="([^"]+)"[^>]*>(.*?)</a></h3>.*?<div class="gs_rs">(.*?)</div>', response.text, re.DOTALL):
                    url, title, snippet = match.groups()
                    
                    # Clean up the results
                    title = re.sub(r'<[^>]+>', '', title).strip()
                    snippet = re.sub(r'<[^>]+>', '', snippet).strip()
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "scholar",
                        "type": "academic"
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
            return []
    
    async def _search_semantic(self, query: str) -> List[Dict[str, Any]]:
        """Search using Semantic Scholar (for academic papers)"""
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit=10"
        
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive"
        }
        
        timeout = self.settings.get("timeout_seconds", 30)
        proxies = self._get_proxy_settings()
        
        try:
            async with httpx.AsyncClient(proxies=proxies, timeout=timeout) as client:
                response = await client.get(url, headers=headers, follow_redirects=True)
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                # Extract results from JSON response
                for paper in data.get("data", []):
                    title = paper.get("title", "")
                    paper_id = paper.get("paperId", "")
                    url = f"https://www.semanticscholar.org/paper/{paper_id}"
                    abstract = paper.get("abstract", "")
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": abstract,
                        "source": "semantic_scholar",
                        "type": "academic",
                        "paper_id": paper_id
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error searching Semantic Scholar: {e}")
            return []
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate search results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            # Normalize URL for comparison
            normalized_url = url.rstrip("/").lower()
            
            if normalized_url and normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)
        
        return unique_results
        
    async def _log_security_event(self, event_type: str, **details) -> None:
        """Log security events for monitoring and auditing"""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "details": details,
            "tor_enabled": self.tor_enabled,
            "proxy_enabled": self.proxy_enabled
        }
        
        # Add to in-memory log
        self.security_events.append(event)
        
        # Limit the size of in-memory log
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        # Log to file
        try:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, "security_events.log")
            with open(log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the intel query system"""
        metrics = self.performance_metrics.copy()
        
        # Calculate average query time
        if metrics["query_times"]:
            metrics["avg_query_time"] = sum(metrics["query_times"]) / len(metrics["query_times"])
        else:
            metrics["avg_query_time"] = 0
        
        # Calculate cache hit rate
        total_queries = metrics["cache_hits"] + metrics["cache_misses"]
        if total_queries > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / total_queries
        else:
            metrics["cache_hit_rate"] = 0
        
        return metrics
    
    async def fetch_content(self, url: str, use_tor: Optional[bool] = None) -> Dict[str, Any]:
        """
        Fetch content from a URL with optional Tor routing
        Returns the content and metadata
        """
        # Determine whether to use Tor
        if use_tor is None:
            use_tor = self.tor_enabled
        elif use_tor and not self.tor_enabled:
            # If Tor is requested but not enabled, try to set it up
            if not self._setup_tor():
                return {
                    "success": False,
                    "error": "Tor requested but not available",
                    "url": url
                }
        
        # Apply rate limiting
        await self._rate_limit()
        
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml,application/json,*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        timeout = self.settings.get("timeout_seconds", 30)
        
        # Set up proxies
        proxies = {}
        if use_tor:
            proxies = {
                "http": f"socks5://127.0.0.1:{self.tor_port}",
                "https": f"socks5://127.0.0.1:{self.tor_port}"
            }
        elif self.proxy_enabled and self.proxy_url:
            proxies = {
                "http": self.proxy_url,
                "https": self.proxy_url
            }
        
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(proxies=proxies, timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                # Get content type
                content_type = response.headers.get("content-type", "").lower()
                
                # Process based on content type
                if "application/json" in content_type:
                    content = response.json()
                    text_content = json.dumps(content, indent=2)
                    content_format = "json"
                elif "text/html" in content_type:
                    text_content = response.text
                    content_format = "html"
                elif "text/plain" in content_type:
                    text_content = response.text
                    content_format = "text"
                else:
                    # For binary content, return base64 encoded data
                    content = base64.b64encode(response.content).decode('utf-8')
                    text_content = f"Binary content ({content_type})"
                    content_format = "binary"
                
                # Extract title from HTML
                title = ""
                if content_format == "html":
                    title_match = re.search(r'<title[^>]*>(.*?)</title>', text_content, re.IGNORECASE | re.DOTALL)
                    if title_match:
                        title = title_match.group(1).strip()
                
                return {
                    "success": True,
                    "url": url,
                    "final_url": str(response.url),
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "content_format": content_format,
                    "title": title,
                    "content": text_content if content_format != "binary" else content,
                    "headers": dict(response.headers),
                    "size_bytes": len(response.content),
                    "load_time_ms": int((time.time() - start_time) * 1000),
                    "timestamp": time.time(),
                    "via_tor": use_tor
                }
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e),
                "timestamp": time.time(),
                "via_tor": use_tor
            }
    
    async def dark_search(self, query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform a search specifically on dark web sources
        Returns search results from dark web search engines
        """
        if not self.tor_enabled:
            # Try to set up Tor
            if not self._setup_tor():
                return {
                    "success": False,
                    "error": "Tor is required for dark web searches but is not available",
                    "query": query
                }
        
        # Perform the search using dark web engines
        search_results = await self.search(query, search_type="dark", max_results=max_results, safe_search=False)
        
        # Add a warning about dark web content
        search_results["warning"] = "Dark web content may include illegal or harmful material. Use responsibly and at your own risk."
        
        return search_results
    
    async def intel_query(self, query: str, sources: List[str] = None, 
                        max_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform a comprehensive intelligence query across multiple sources
        Returns aggregated results with source attribution
        """
        if sources is None:
            sources = ["general", "academic", "news"]
        
        if max_results is None:
            max_results = self.settings.get("max_results", 20)
        
        # Collect results from all requested sources
        all_results = []
        source_stats = {}
        errors = []
        
        for source in sources:
            try:
                if source == "dark":
                    # Special handling for dark web sources
                    if not self.tor_enabled and not self._setup_tor():
                        errors.append({
                            "source": source,
                            "error": "Tor is required for dark web searches but is not available"
                        })
                        continue
                
                # Perform search for this source type
                source_results = await self.search(query, search_type=source, max_results=max_results)
                
                # Add results to the combined list
                all_results.extend(source_results.get("results", []))
                
                # Track statistics
                source_stats[source] = {
                    "count": len(source_results.get("results", [])),
                    "engines": source_results.get("engines_used", [])
                }
            except Exception as e:
                logger.error(f"Error querying source {source}: {e}")
                errors.append({
                    "source": source,
                    "error": str(e)
                })
        
        # Deduplicate and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(unique_results, query)
        
        # Limit results
        limited_results = ranked_results[:max_results]
        
        # Generate a query ID
        query_id = self._generate_query_id(query)
        
        return {
            "query_id": query_id,
            "query": query,
            "timestamp": time.time(),
            "total_results": len(unique_results),
            "returned_results": len(limited_results),
            "results": limited_results,
            "sources": source_stats,
            "errors": errors,
            "tor_enabled": self.tor_enabled,
            "proxy_enabled": self.proxy_enabled
        }
    
    def _rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank results based on relevance to the query"""
        # Simple ranking based on keyword presence in title and snippet
        query_terms = set(query.lower().split())
        
        for result in results:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            
            # Count term occurrences
            title_matches = sum(1 for term in query_terms if term in title)
            snippet_matches = sum(1 for term in query_terms if term in snippet)
            
            # Calculate score (title matches weighted more heavily)
            score = (title_matches * 2) + snippet_matches
            
            # Adjust score based on source
            if result.get("source") == "scholar" or result.get("source") == "semantic_scholar":
                score += 1  # Boost academic sources
            
            # Store score in result
            result["relevance_score"] = score
        
        # Sort by score (descending)
        return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """Get the history of queries performed"""
        return self.query_history
    
    def clear_query_history(self) -> bool:
        """Clear the query history"""
        self.query_history = []
        return True
    
    def is_tor_enabled(self) -> bool:
        """Check if Tor is enabled and available"""
        return self.tor_enabled
    
    def set_tor_enabled(self, enabled: bool) -> bool:
        """Enable or disable Tor routing"""
        if enabled and not TOR_AVAILABLE:
            logger.warning("Cannot enable Tor: libraries not available")
            return False
        
        if enabled and not self.tor_enabled:
            # Try to set up Tor
            if not self._setup_tor():
                logger.warning("Failed to set up Tor connection")
                return False
        
        self.tor_enabled = enabled
        self.settings["tor_enabled"] = enabled
        settings_manager.update_settings("intel_query", self.settings)
        
        logger.info(f"Tor routing {'enabled' if enabled else 'disabled'}")
        return True
    
    def set_proxy_enabled(self, enabled: bool, proxy_url: Optional[str] = None) -> bool:
        """Enable or disable proxy routing"""
        if enabled and proxy_url:
            self.proxy_url = proxy_url
            self.settings["proxy_url"] = proxy_url
        
        self.proxy_enabled = enabled
        self.settings["proxy_enabled"] = enabled
        settings_manager.update_settings("intel_query", self.settings)
        
        logger.info(f"Proxy routing {'enabled' if enabled else 'disabled'}")
        return True

# Create singleton instance
intel_query = IntelQuery()

# For testing
if __name__ == "__main__":
    async def test_intel_query():
        # Test general search
        results = await intel_query.search("artificial intelligence ethics")
        print(f"Found {len(results.get('results', []))} results for general search")
        
        # Test dark search if Tor is available
        if intel_query.is_tor_enabled():
            dark_results = await intel_query.dark_search("secure communication")
            print(f"Found {len(dark_results.get('results', []))} results for dark search")
        else:
            print("Tor not available, skipping dark search test")
        
        # Test content fetching
        content = await intel_query.fetch_content("https://example.com")
        print(f"Fetched content: {content.get('success')}, size: {content.get('size_bytes')} bytes")
    
    asyncio.run(test_intel_query())