# MCP Size Tracking Integration Guide

## Quick Start

### 1. Add to your MCP server

```python
from mcp.server.fastmcp import FastMCP
from mcp_size_tracker import SizeTracker, track_size
import json

# Initialize
mcp = FastMCP("KDB AI")
tracker = SizeTracker("kdbai_size_log.json")

# Wrap your existing tools
@mcp.tool()
async def kdbai_query_data(query: str) -> str:
    """Query data from KDB AI tables"""
    import time
    start = time.time()
    
    # Parse query
    q = json.loads(query) if isinstance(query, str) else query
    
    # Your existing implementation
    result = kx_client.getData(q)
    
    # Track size
    duration_ms = (time.time() - start) * 1000
    tracker.log_call("kdbai_query_data", q, result, duration_ms)
    
    return result
```

### 2. Alternative: Using decorator

```python
@mcp.tool()
@track_size(tracker, "kdbai_query_data")
async def kdbai_query_data(query: str) -> str:
    """Query data from KDB AI tables"""
    result = kx_client.getData(query)
    return result
```

## Viewing Statistics

### Basic usage
```bash
python view_stats.py
```

### Filter by date
```bash
python view_stats.py --since 2026-02-01
```

### Filter by tool
```bash
python view_stats.py --tool kdbai_query_data
```

### Show detailed logs
```bash
python view_stats.py --detail
```

## Log File Format

The tracker creates `mcp_size_log.json` with entries like:

```json
[
  {
    "timestamp": "2026-02-16T04:30:15.123456",
    "tool": "kdbai_query_data",
    "query_size_mb": 0.00015,
    "response_size_mb": 2.45,
    "duration_ms": 234,
    "query_summary": {
      "table": "bloomberg_headlines",
      "limit": -100
    }
  }
]
```

## Adding Custom Tracking

For non-MCP usage, use the tracker directly:

```python
from mcp_size_tracker import SizeTracker

tracker = SizeTracker("my_log.json")

query = {"table": "dOrderReport", "limit": -100}
response = api.call(query)

tracker.log_call("custom_api", query, response)

# Get stats
stats = tracker.get_stats(since_date="2026-02-16T00:00:00")
print(f"Total response data: {stats['total_response_mb']:.2f} MB")
```

## Monitoring Best Practices

1. **Rotate logs periodically** - Archive old logs to prevent file growth
2. **Set alerts** - Monitor for unusually large responses
3. **Track trends** - Compare daily/weekly totals
4. **Optimize queries** - Use aggregations to reduce response sizes

## Log Rotation Script

```python
from pathlib import Path
from datetime import datetime
import json
import shutil

def rotate_logs(log_file="mcp_size_log.json", keep_days=30):
    log_path = Path(log_file)
    if not log_path.exists():
        return
    
    # Archive current log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = log_path.parent / f"{log_path.stem}_{timestamp}.json"
    shutil.copy(log_path, archive_path)
    
    # Keep only recent entries
    with open(log_path) as f:
        logs = json.load(f)
    
    cutoff = datetime.now() - timedelta(days=keep_days)
    cutoff_str = cutoff.isoformat()
    
    recent = [l for l in logs if l["timestamp"] >= cutoff_str]
    
    with open(log_path, 'w') as f:
        json.dump(recent, f, indent=2)
    
    print(f"Archived {len(logs) - len(recent)} old entries")
```

## Example Output

```
================================================================================
MCP API CALL SIZE SUMMARY
================================================================================
Total calls: 128
Date range: 2026-02-10 to 2026-02-16

BY TOOL:
--------------------------------------------------------------------------------

kdbai_query_data:
  Calls:            128
  Total Query:      18.45 KB
  Total Response:   245.67 MB
  Avg Response:     1.92 MB
  Max Response:     15.34 MB
  Avg Duration:     156 ms

================================================================================
OVERALL TOTALS:
  Total Query Data:    18.45 KB
  Total Response Data: 245.67 MB
  Combined:            245.69 MB
================================================================================
```
