# Changelog

## Version 3.0.0 (2025-06-27)

### Added
- **Security Hardening**
  - Added security.py module with CredentialManager, RequestSigner, TLSManager, and TokenManager
  - Implemented request signing for API security
  - Added secure credential storage
  - Created TLS certificate management
  - Added JWT-based authentication
  - Implemented rate limiting

- **Performance Optimization**
  - Added performance.py module with ModelPrewarmer, QueryCache, and ConnectionPool
  - Implemented model pre-warming to reduce cold start latency
  - Added multi-level query caching for improved response times
  - Created connection pooling for network efficiency
  - Added asynchronous processing for non-blocking operations

- **Error Handling**
  - Added error_handler.py module with centralized error handling
  - Implemented error logging and notification
  - Added error recovery mechanisms
  - Created error statistics and reporting

- **WebSocket Integration**
  - Added websocket_manager.py module for real-time updates
  - Implemented connection management
  - Added event broadcasting
  - Created periodic update tasks

- **API Endpoints**
  - Added /api/auth/token for authentication
  - Added /api/security/status for security system status
  - Added /api/models/status for model router status
  - Added /api/models/warmup for model pre-warming
  - Added /api/models/set-override for model selection override
  - Added /api/intel-query for intelligence queries
  - Added /api/intel/performance for intel query performance metrics
  - Added /api/errors for error log access
  - Added /api/errors/stats for error statistics

- **WebSocket Endpoints**
  - Added /ws/updates for real-time system updates

### Updated
- **Model Router**
  - Enhanced with caching and performance tracking
  - Added model pre-warming integration
  - Improved model selection algorithm
  - Added security features

- **Intel Query**
  - Added security features including request signing
  - Implemented connection pooling
  - Added performance metrics collection
  - Enhanced result deduplication

- **UI**
  - Added security status indicators
  - Added model selection indicators
  - Added performance metrics display
  - Added error reporting

### Fixed
- Improved error handling for network failures
- Fixed memory leaks in long-running processes
- Enhanced security against common web vulnerabilities
- Improved performance for repeated queries

### Dependencies
- Added pyjwt for JWT token management
- Added diskcache for disk-based caching
- Added aioredis for Redis integration