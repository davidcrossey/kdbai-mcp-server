import logging
from typing import Optional, Dict, Any, List
from mcp_server.utils.embeddings import get_provider
from mcp_server.utils.embeddings_helpers import get_embedding_config
from mcp_server.utils.kdbai import get_table
from mcp_server.utils.filters import parse_temporal_filters
from mcp_server.server import app_settings
from mcp_server.tools.kdbai_data import normalize_result

db_config = app_settings.db
logger = logging.getLogger(__name__)


async def kdbai_sparse_search_impl(table_name: str,
                                    query: str,
                                    sparse_index_name: str,
                                    database_name: Optional[str] = None,
                                    n: Optional[int] = None,
                                    filters: Optional[List[tuple]] = None,
                                    sort_columns: Optional[List[str]] = None,
                                    group_by: Optional[List[str]] = None,
                                    aggs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        if database_name is None:
            database_name = db_config.database_name
        if n is None:
            n = db_config.k

        table = get_table(table_name, database_name)

        _, _, sparse_tokenizer_provider, sparse_tokenizer_model = get_embedding_config(database_name, table_name)

        sparse_provider = get_provider(sparse_tokenizer_provider)
        query_sparse = await sparse_provider.sparse_embed(query, sparse_tokenizer_model)

        search_params = {
            "vectors": {sparse_index_name: [query_sparse]},
            "n": int(n),
            **{k: v for k, v in {
                'filter': parse_temporal_filters(filters, table.schema),
                'sort_columns': sort_columns,
                'group_by': group_by,
                'aggs': aggs
            }.items() if v is not None}
        }

        result = table.search(**search_params)[0]
        result = normalize_result(result, table)

        return {
            "status": "success",
            "database": database_name,
            "table": table_name,
            "recordsCount": len(result),
            "records": result
        }
    except Exception as e:
        logger.error(f"Error performing sparse search on table {table_name}: {e}")
        return {
            "status": "error",
            "message": str(e),
            "database": database_name,
            "table": table_name,
        }


def register_tools(mcp_server):
    @mcp_server.tool()
    async def kdbai_sparse_search(table_name: str,
                                   query: str,
                                   sparse_index_name: str,
                                   database_name: Optional[str] = None,
                                   n: Optional[int] = None,
                                   filters: Optional[List[tuple]] = None,
                                   sort_columns: Optional[List[str]] = None,
                                   group_by: Optional[List[str]] = None,
                                   aggs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform sparse (keyword/BM25) search on a KDB.AI table using only a sparse index.
        Use this when you want keyword-based retrieval without dense vector similarity.
        For search syntax and examples, see: file://kdbai_operations_guidance

        Args:
            table_name: Name of the table to search
            query: Text query to tokenize and search using the sparse index
            sparse_index_name: Name of the sparse index to search against
            database_name: Name of the database containing the table
            n: Number of results to return
            filters: List of filter conditions as q/kdb+ parse tree (operator, filter column name, value).
                Examples:
                 - Simple equality: ("=", "filter_column_name", "value")
                 - Logical AND: [("<", "filter_column_name_1", "value"), (">", "filter_column_name_2", "value")]
            sort_columns: List of column names to sort by, e.g. '["price", "date"]'
            group_by: List of column names to group by, e.g. '["category"]'
            aggs: Dictionary of aggregation rules, e.g. '{"total": ["sum", "amount"]}'. It can use any KDB+ supported aggregation function like avg, max, sum etc.

        Returns:
            Dictionary containing sparse search results or error message.
        """
        return await kdbai_sparse_search_impl(
            table_name,
            query,
            sparse_index_name,
            database_name,
            n,
            filters,
            sort_columns,
            group_by,
            aggs
        )

    return ["kdbai_sparse_search"]
