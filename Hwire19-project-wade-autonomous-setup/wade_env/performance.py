"""
Performance optimization module for WADE autonomous development environment.
Provides model pre-warming, query caching, and connection pooling.
"""

import asyncio
import functools
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
import aiohttp
import aioredis
import diskcache

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

# Configure logging
logger = logging.getLogger(__name__)


class ModelPrewarmer:
    """Pre-warm models to reduce cold start latency."""
    
    def __init__(self, models: List[str] = None):
        """Initialize with a list of models to pre-warm."""
        self.models = models or []
        self.warmed_models: Dict[str, bool] = {model: False for model in self.models}
        self.warmup_prompts: Dict[str, str] = {}
        self.warmup_lock = asyncio.Lock()
    
    def register_model(self, model_name: str, warmup_prompt: str = None) -> None:
        """Register a model for pre-warming."""
        self.models.append(model_name)
        self.warmed_models[model_name] = False
        if warmup_prompt:
            self.warmup_prompts[model_name] = warmup_prompt
    
    async def warmup_model(self, model_name: str) -> bool:
        """Warm up a specific model."""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not registered for pre-warming")
            return False
        
        if self.warmed_models.get(model_name, False):
            logger.debug(f"Model {model_name} already warmed up")
            return True
        
        async with self.warmup_lock:
            # Double-check to avoid race conditions
            if self.warmed_models.get(model_name, False):
                return True
            
            try:
                # Get the warmup prompt for this model
                prompt = self.warmup_prompts.get(
                    model_name, 
                    "This is a warmup prompt to initialize the model."
                )
                
                # Import here to avoid circular imports
                from wade_env.model_router import model_router
                
                # Send a simple prompt to warm up the model
                logger.info(f"Warming up model {model_name}...")
                start_time = time.time()
                
                # Use the model router to send the warmup prompt
                await model_router.generate_with_model(model_name, prompt)
                
                warmup_time = time.time() - start_time
                logger.info(f"Model {model_name} warmed up in {warmup_time:.2f}s")
                
                self.warmed_models[model_name] = True
                return True
            
            except Exception as e:
                logger.error(f"Error warming up model {model_name}: {e}")
                return False
    
    async def warmup_all_models(self) -> Dict[str, bool]:
        """Warm up all registered models."""
        results = {}
        for model in self.models:
            results[model] = await self.warmup_model(model)
        return results
    
    def is_model_warmed(self, model_name: str) -> bool:
        """Check if a model is warmed up."""
        return self.warmed_models.get(model_name, False)


class QueryCache:
    """Cache query results to improve performance."""
    
    def __init__(self, cache_dir: str = None, ttl: int = 3600):
        """Initialize the query cache."""
        self.memory_cache: Dict[str, Tuple[Any, float]] = {}
        self.disk_cache = diskcache.Cache(cache_dir or "/tmp/wade_cache")
        self.ttl = ttl  # Default TTL: 1 hour
        self.redis_client: Optional[aioredis.Redis] = None
    
    async def setup_redis(self, redis_url: str) -> None:
        """Set up Redis for distributed caching."""
        try:
            self.redis_client = await aioredis.create_redis_pool(redis_url)
            logger.info("Redis cache connection established")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
    
    def _get_cache_key(self, query: str, params: Dict[str, Any] = None) -> str:
        """Generate a cache key from a query and parameters."""
        if params:
            key_data = f"{query}:{json.dumps(params, sort_keys=True)}"
        else:
            key_data = query
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, query: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """Get a cached result for a query."""
        cache_key = self._get_cache_key(query, params)
        
        # Try memory cache first (fastest)
        memory_result = self.memory_cache.get(cache_key)
        if memory_result:
            value, timestamp = memory_result
            if time.time() - timestamp < self.ttl:
                logger.debug(f"Memory cache hit for {cache_key}")
                return value
            else:
                # Expired, remove from memory cache
                del self.memory_cache[cache_key]
        
        # Try Redis cache if available (distributed)
        if self.redis_client:
            try:
                redis_result = await self.redis_client.get(f"wade:cache:{cache_key}")
                if redis_result:
                    value = json.loads(redis_result)
                    logger.debug(f"Redis cache hit for {cache_key}")
                    # Update memory cache
                    self.memory_cache[cache_key] = (value, time.time())
                    return value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Try disk cache (persistent)
        disk_result = self.disk_cache.get(cache_key)
        if disk_result:
            logger.debug(f"Disk cache hit for {cache_key}")
            # Update memory cache
            self.memory_cache[cache_key] = (disk_result, time.time())
            return disk_result
        
        return None
    
    async def set(self, query: str, params: Dict[str, Any] = None, value: Any, ttl: int = None) -> None:
        """Cache a result for a query."""
        cache_key = self._get_cache_key(query, params)
        ttl = ttl or self.ttl
        
        # Update memory cache
        self.memory_cache[cache_key] = (value, time.time())
        
        # Update disk cache
        self.disk_cache.set(cache_key, value, expire=ttl)
        
        # Update Redis cache if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"wade:cache:{cache_key}", 
                    ttl,
                    json.dumps(value)
                )
            except Exception as e:
                logger.warning(f"Redis cache set error: {e}")
    
    async def invalidate(self, query: str, params: Dict[str, Any] = None) -> None:
        """Invalidate a cached result."""
        cache_key = self._get_cache_key(query, params)
        
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove from disk cache
        self.disk_cache.delete(cache_key)
        
        # Remove from Redis cache if available
        if self.redis_client:
            try:
                await self.redis_client.delete(f"wade:cache:{cache_key}")
            except Exception as e:
                logger.warning(f"Redis cache invalidation error: {e}")
    
    def cached(self, ttl: int = None):
        """Decorator to cache function results."""
        def decorator(func: Callable[..., R]) -> Callable[..., R]:
            @functools.wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> R:
                # Generate a cache key from function name, args, and kwargs
                cache_key = f"{func.__module__}.{func.__name__}"
                params = {
                    "args": args,
                    "kwargs": kwargs
                }
                
                # Try to get from cache
                cached_result = await self.get(cache_key, params)
                if cached_result is not None:
                    return cast(R, cached_result)
                
                # Call the function
                result = await func(*args, **kwargs)
                
                # Cache the result
                await self.set(cache_key, params, result, ttl)
                
                return result
            
            return wrapper
        
        return decorator


class ConnectionPool:
    """Manage connection pools for external services."""
    
    def __init__(self, pool_size: int = 10, timeout: float = 30.0):
        """Initialize the connection pool."""
        self.pool_size = pool_size
        self.timeout = timeout
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.db_pools: Dict[str, Any] = {}
    
    async def get_http_session(self) -> aiohttp.ClientSession:
        """Get or create an HTTP session."""
        if self.http_session is None or self.http_session.closed:
            # Create a new session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.pool_size,
                ttl_dns_cache=300,  # Cache DNS results for 5 minutes
                ssl=False  # We'll handle SSL in the request
            )
            
            self.http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        
        return self.http_session
    
    async def close_http_session(self) -> None:
        """Close the HTTP session."""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
            self.http_session = None
    
    async def register_db_pool(self, name: str, pool: Any) -> None:
        """Register a database connection pool."""
        self.db_pools[name] = pool
    
    async def get_db_pool(self, name: str) -> Optional[Any]:
        """Get a database connection pool."""
        return self.db_pools.get(name)
    
    async def close_all(self) -> None:
        """Close all connection pools."""
        await self.close_http_session()
        
        # Close all database pools
        for name, pool in self.db_pools.items():
            if hasattr(pool, 'close'):
                if asyncio.iscoroutinefunction(pool.close):
                    await pool.close()
                else:
                    pool.close()
            elif hasattr(pool, 'terminate'):
                if asyncio.iscoroutinefunction(pool.terminate):
                    await pool.terminate()
                else:
                    pool.terminate()


# Initialize default instances
model_prewarmer = ModelPrewarmer()
query_cache = QueryCache()
connection_pool = ConnectionPool()