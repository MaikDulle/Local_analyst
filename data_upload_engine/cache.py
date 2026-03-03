"""
Caching system for loaded data and analysis results.
Uses file hashing to avoid reloading unchanged files.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, Tuple
from datetime import datetime
import pandas as pd


class DataCache:
    """
    Simple file-based cache for dataframes and analysis results.
    Uses MD5 hash of file content to detect changes.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to .cache in current dir.
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / '.analyst_cache'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for current session
        self._memory_cache: Dict[str, Any] = {}
        self._cache_metadata: Dict[str, Dict] = {}
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file content."""
        hasher = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _get_df_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataframe for result caching."""
        # Use shape + column names + sample of data
        hash_str = f"{df.shape}_{list(df.columns)}_{df.head(10).to_json()}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:16]
    
    def _get_cache_key(self, file_path: str, **options) -> str:
        """Generate cache key from file path and load options."""
        file_hash = self._get_file_hash(file_path)
        options_str = json.dumps(options, sort_keys=True, default=str)
        options_hash = hashlib.md5(options_str.encode()).hexdigest()[:8]
        return f"{file_hash}_{options_hash}"
    
    def get_cached_data(self, file_path: str, **options) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        Retrieve cached dataframe if available and file unchanged.
        
        Returns:
            Tuple of (DataFrame, metadata) if cached, None otherwise
        """
        cache_key = self._get_cache_key(file_path, **options)
        
        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        meta_file = self.cache_dir / f"{cache_key}.meta.json"
        
        if cache_file.exists() and meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                
                # Verify file hasn't changed
                current_hash = self._get_file_hash(file_path)
                if metadata.get('file_hash') == current_hash:
                    df = pd.read_pickle(cache_file)
                    result = (df, metadata.get('load_metadata', {}))
                    self._memory_cache[cache_key] = result
                    return result
            except Exception:
                # Cache corrupted, ignore
                pass
        
        return None
    
    def cache_data(
        self, 
        file_path: str, 
        df: pd.DataFrame, 
        load_metadata: Dict,
        **options
    ) -> None:
        """
        Cache loaded dataframe.
        
        Args:
            file_path: Original file path
            df: Loaded DataFrame
            load_metadata: Metadata from loader
            **options: Load options used
        """
        cache_key = self._get_cache_key(file_path, **options)
        
        # Store in memory
        self._memory_cache[cache_key] = (df, load_metadata)
        
        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        meta_file = self.cache_dir / f"{cache_key}.meta.json"
        
        try:
            df.to_pickle(cache_file)
            
            full_metadata = {
                'file_hash': self._get_file_hash(file_path),
                'file_path': str(file_path),
                'cached_at': datetime.now().isoformat(),
                'load_metadata': load_metadata,
                'options': options
            }
            
            with open(meta_file, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
                
        except Exception as e:
            # Don't fail if caching fails
            print(f"Warning: Failed to cache data: {e}")
    
    def get_analysis_result(
        self, 
        df: pd.DataFrame, 
        analysis_type: str, 
        **params
    ) -> Optional[Any]:
        """
        Retrieve cached analysis result.
        
        Args:
            df: DataFrame used for analysis
            analysis_type: Type of analysis (e.g., 'correlation', 'summary')
            **params: Analysis parameters
        
        Returns:
            Cached result if available, None otherwise
        """
        df_hash = self._get_df_hash(df)
        params_str = json.dumps(params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        cache_key = f"analysis_{analysis_type}_{df_hash}_{params_hash}"
        
        return self._memory_cache.get(cache_key)
    
    def cache_analysis_result(
        self, 
        df: pd.DataFrame, 
        analysis_type: str, 
        result: Any,
        **params
    ) -> None:
        """
        Cache analysis result.
        
        Args:
            df: DataFrame used for analysis
            analysis_type: Type of analysis
            result: Analysis result to cache
            **params: Analysis parameters
        """
        df_hash = self._get_df_hash(df)
        params_str = json.dumps(params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        cache_key = f"analysis_{analysis_type}_{df_hash}_{params_hash}"
        
        self._memory_cache[cache_key] = result
    
    def clear_memory_cache(self) -> None:
        """Clear in-memory cache."""
        self._memory_cache.clear()
    
    def clear_all_cache(self) -> None:
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        
        # Remove all cache files
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        for file in self.cache_dir.glob("*.meta.json"):
            file.unlink()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_files = list(self.cache_dir.glob("*.pkl"))
        disk_size = sum(f.stat().st_size for f in disk_files)
        
        return {
            'memory_entries': len(self._memory_cache),
            'disk_entries': len(disk_files),
            'disk_size_mb': round(disk_size / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir)
        }


# Global cache instance
_global_cache: Optional[DataCache] = None


def get_cache() -> DataCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache()
    return _global_cache
