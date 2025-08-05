from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.p2p.p2p_node import P2PNode


@pytest.mark.asyncio
@patch("ssl.create_default_context")
@patch("asyncio.start_server", new_callable=AsyncMock)
async def test_p2pnode_tls_start(mock_start_server, mock_ssl_ctx):
    mock_ssl = Mock()
    mock_ssl_ctx.return_value = mock_ssl
    mock_server = AsyncMock()
    mock_server.close = Mock()
    mock_server.wait_closed = AsyncMock()
    mock_start_server.return_value = mock_server
    node = P2PNode(use_tls=True, certfile="cert.pem", keyfile="key.pem")
    await node.start()
    mock_ssl.load_cert_chain.assert_called_with("cert.pem", "key.pem")
    assert mock_start_server.call_args.kwargs["ssl"] is mock_ssl
    await node.stop()
