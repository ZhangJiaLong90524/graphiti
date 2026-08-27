from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    edge_similarity_search,
    node_similarity_search,
)


def _driver_with_search_ops(search_ops: MagicMock) -> MagicMock:
    driver = MagicMock()
    driver.search_interface = None
    driver.search_ops = search_ops
    driver.execute_query = AsyncMock(return_value=([], None, None))
    return driver


@pytest.mark.asyncio
async def test_node_similarity_search_uses_provider_search_ops():
    expected = [object()]
    search_ops = MagicMock()
    search_ops.node_similarity_search = AsyncMock(return_value=expected)
    driver = _driver_with_search_ops(search_ops)
    search_vector = [0.1, 0.2]
    search_filter = SearchFilters()

    result = await node_similarity_search(
        driver,
        search_vector,
        search_filter,
        group_ids=['agent_memory'],
        limit=5,
        min_score=0.6,
    )

    assert result is expected
    search_ops.node_similarity_search.assert_awaited_once_with(
        driver, search_vector, search_filter, ['agent_memory'], 5, 0.6
    )
    driver.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_edge_similarity_search_uses_provider_search_ops():
    expected = [object()]
    search_ops = MagicMock()
    search_ops.edge_similarity_search = AsyncMock(return_value=expected)
    driver = _driver_with_search_ops(search_ops)
    search_vector = [0.3, 0.4]
    search_filter = SearchFilters()

    result = await edge_similarity_search(
        driver,
        search_vector,
        None,
        None,
        search_filter,
        group_ids=['agent_memory'],
        limit=5,
        min_score=0.6,
    )

    assert result is expected
    search_ops.edge_similarity_search.assert_awaited_once_with(
        driver, search_vector, None, None, search_filter, ['agent_memory'], 5, 0.6
    )
    driver.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_node_similarity_search_prefers_custom_search_interface():
    interface_result = [object()]
    ops_result = [object()]
    search_interface = MagicMock()
    search_interface.node_similarity_search = AsyncMock(return_value=interface_result)
    search_ops = MagicMock()
    search_ops.node_similarity_search = AsyncMock(return_value=ops_result)
    driver = _driver_with_search_ops(search_ops)
    driver.search_interface = search_interface
    search_vector = [0.5]
    search_filter = SearchFilters()

    result = await node_similarity_search(
        driver, search_vector, search_filter, group_ids=['agent_memory']
    )

    assert result is interface_result
    search_ops.node_similarity_search.assert_not_called()
    driver.execute_query.assert_not_called()
