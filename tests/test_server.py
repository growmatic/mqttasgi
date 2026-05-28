"""
Unit tests for mqttasgi.server.

These tests do NOT require a running MQTT broker. They verify:
- paho-mqtt 1.x / 2.x compatibility shim works correctly
- Callback signatures accept both paho API versions
- Server initialises without errors
- MQTTv5 protocol selection and clean_start handling
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import paho.mqtt.client as mqtt

from mqttasgi.server import Server, _PAHO_MQTT_V2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_server():
    """Return a Server instance without starting the event loop."""
    mock_app = AsyncMock()
    return Server(mock_app, host='localhost', port=1883)


# ---------------------------------------------------------------------------
# paho version detection
# ---------------------------------------------------------------------------

class TestPahoVersionDetection:
    def test_flag_matches_installed_paho(self):
        """_PAHO_MQTT_V2 must agree with the actual installed paho version."""
        if hasattr(mqtt, 'CallbackAPIVersion'):
            assert _PAHO_MQTT_V2 is True, "paho 2.x detected but flag is False"
        else:
            assert _PAHO_MQTT_V2 is False, "paho 1.x detected but flag is True"

    def test_client_type(self):
        """Server.client must be a paho Client instance regardless of version."""
        server = _make_server()
        assert isinstance(server.client, mqtt.Client)


# ---------------------------------------------------------------------------
# on_connect callback — must work with both paho 1.x and 2.x signatures
# ---------------------------------------------------------------------------

class TestOnConnectCompatibility:
    def test_paho_v1_signature(self):
        """Accepts paho 1.x signature: (client, userdata, flags, rc)."""
        server = _make_server()
        server._on_connect(MagicMock(), {}, {}, 0)  # rc as positional arg

    def test_paho_v2_signature(self):
        """Accepts paho 2.x signature: (client, userdata, flags, reason_code, properties)."""
        server = _make_server()
        server._on_connect(MagicMock(), {}, {}, MagicMock(), MagicMock())

    def test_sends_connect_event_to_all_apps(self):
        """_on_connect must enqueue mqtt.connect for every registered application."""
        server = _make_server()

        # Manually register two fake applications with receive queues
        for app_id in (0, 1):
            server.application_data[app_id] = {'receive': asyncio.Queue()}

        server._on_connect(MagicMock(), {}, {}, 0)

        for app_id in (0, 1):
            event = server.application_data[app_id]['receive'].get_nowait()
            assert event['type'] == 'mqtt.connect'


# ---------------------------------------------------------------------------
# on_disconnect callback
# ---------------------------------------------------------------------------

class TestOnDisconnectCompatibility:
    def _make_stopped_server(self):
        server = _make_server()
        server.stop = True  # prevent reconnect attempt
        return server

    def test_paho_v1_signature(self):
        """Accepts paho 1.x signature: (client, userdata, rc)."""
        server = self._make_stopped_server()
        server._on_disconnect(MagicMock(), {}, 0)

    def test_paho_v2_signature(self):
        """Accepts paho 2.x signature: (client, userdata, disconnect_flags, reason_code, properties)."""
        server = self._make_stopped_server()
        server._on_disconnect(MagicMock(), {}, MagicMock(), MagicMock(), MagicMock())

    def test_no_reconnect_when_stopped(self):
        """Does not attempt reconnect if server.stop is True."""
        server = self._make_stopped_server()
        reconnect_called = []
        server._handle_reconnect = lambda: reconnect_called.append(True)
        server._on_disconnect(MagicMock(), {}, 0)
        assert reconnect_called == []


# ---------------------------------------------------------------------------
# Server initialisation
# ---------------------------------------------------------------------------

class TestServerInit:
    def test_default_message_types(self):
        server = _make_server()
        assert server.mqtt_type_pub == 'mqtt.pub'
        assert server.mqtt_type_sub == 'mqtt.sub'
        assert server.mqtt_type_usub == 'mqtt.usub'
        assert server.mqtt_type_msg == 'mqtt.msg'

    def test_custom_message_types(self):
        server = Server(
            AsyncMock(), 'localhost', 1883,
            mqtt_type_pub='custom.pub',
            mqtt_type_sub='custom.sub',
        )
        assert server.mqtt_type_pub == 'custom.pub'
        assert server.mqtt_type_sub == 'custom.sub'

    def test_empty_subscriptions_on_init(self):
        server = _make_server()
        assert server.topics_subscription == {}
        assert server.topic_queues == {}
        assert server.application_data == {}


# ---------------------------------------------------------------------------
# MQTTv5 — constructor
# ---------------------------------------------------------------------------

class TestMQTTv5Init:
    """Server constructor with protocol=mqtt.MQTTv5."""

    def _make_v5_server(self, clean_session=True):
        return Server(AsyncMock(), host='localhost', port=1883,
                      clean_session=clean_session, protocol=mqtt.MQTTv5)

    def test_stores_protocol(self):
        """server.protocol must equal mqtt.MQTTv5."""
        server = self._make_v5_server()
        assert server.protocol == mqtt.MQTTv5

    def test_stores_clean_start_true(self):
        """_clean_start carries the user's clean_session=True for later use in connect()."""
        server = self._make_v5_server(clean_session=True)
        assert server._clean_start is True

    def test_stores_clean_start_false(self):
        """_clean_start carries the user's clean_session=False for later use in connect()."""
        server = self._make_v5_server(clean_session=False)
        assert server._clean_start is False

    def test_paho_client_receives_clean_session_none(self):
        """paho Client.__init__ must receive clean_session=None for MQTTv5.

        From paho docs: 'clean_session is not accepted if MQTT version is v5.0
        — use the clean_start argument on connect() instead.'
        """
        with patch.object(mqtt, 'Client') as MockClient:
            MockClient.return_value = MagicMock()
            self._make_v5_server(clean_session=True)
            _, kwargs = MockClient.call_args
            assert kwargs.get('clean_session') is None

    def test_paho_client_receives_protocol_v5(self):
        """paho Client.__init__ must receive protocol=mqtt.MQTTv5."""
        with patch.object(mqtt, 'Client') as MockClient:
            MockClient.return_value = MagicMock()
            self._make_v5_server()
            _, kwargs = MockClient.call_args
            assert kwargs.get('protocol') == mqtt.MQTTv5

    def test_v311_default_clean_session_unaffected(self):
        """Default v3.1.1 path must still pass clean_session to paho Client."""
        with patch.object(mqtt, 'Client') as MockClient:
            MockClient.return_value = MagicMock()
            Server(AsyncMock(), 'localhost', 1883, clean_session=False)
            _, kwargs = MockClient.call_args
            assert kwargs.get('clean_session') is False
            assert kwargs.get('protocol') == mqtt.MQTTv311


# ---------------------------------------------------------------------------
# MQTTv5 — initial connect in mqtt_receive_loop
# ---------------------------------------------------------------------------

class TestMQTTv5Connect:
    """mqtt_receive_loop passes clean_start= for MQTTv5, omits it for v3.1.1."""

    async def test_v5_connect_passes_clean_start_true(self):
        """connect() must be called with clean_start=True when v5 + clean_session=True."""
        server = Server(AsyncMock(), 'localhost', 1883,
                        clean_session=True, protocol=mqtt.MQTTv5)
        server.stop = True  # exit the while-loop immediately after connect
        server.client.connect = MagicMock()

        await server.mqtt_receive_loop()

        server.client.connect.assert_called_once_with('localhost', 1883, clean_start=True)

    async def test_v5_connect_passes_clean_start_false(self):
        """connect() must be called with clean_start=False when v5 + clean_session=False."""
        server = Server(AsyncMock(), 'localhost', 1883,
                        clean_session=False, protocol=mqtt.MQTTv5)
        server.stop = True
        server.client.connect = MagicMock()

        await server.mqtt_receive_loop()

        server.client.connect.assert_called_once_with('localhost', 1883, clean_start=False)

    async def test_v311_connect_has_no_clean_start(self):
        """connect() must NOT receive clean_start for v3.1.1."""
        server = _make_server()
        server.stop = True
        server.client.connect = MagicMock()

        await server.mqtt_receive_loop()

        server.client.connect.assert_called_once_with('localhost', 1883)


# ---------------------------------------------------------------------------
# MQTTv5 — reconnect
# ---------------------------------------------------------------------------

class TestMQTTv5Reconnect:
    """_handle_reconnect uses clean_start= for MQTTv5 on initial connect, not on reconnect."""

    def test_v5_on_connect_passes_clean_start(self):
        """on_connect=True must call connect(host, port, clean_start=...) for v5."""
        server = Server(AsyncMock(), 'localhost', 1883,
                        clean_session=False, protocol=mqtt.MQTTv5)
        server.client.connect = MagicMock()

        with patch('time.sleep'):
            server._handle_reconnect(on_connect=True)

        server.client.connect.assert_called_once_with('localhost', 1883, clean_start=False)

    def test_v311_on_connect_no_clean_start(self):
        """on_connect=True must call connect(host, port) without clean_start for v3.1.1."""
        server = _make_server()
        server.client.connect = MagicMock()

        with patch('time.sleep'):
            server._handle_reconnect(on_connect=True)

        server.client.connect.assert_called_once_with('localhost', 1883)

    def test_reconnect_after_disconnect_uses_reconnect_not_connect(self):
        """on_connect=False must call client.reconnect() for both protocols, never connect().

        After a disconnect, paho re-uses the existing socket via reconnect().
        clean_start is only meaningful on a fresh connect(), not on reconnect().
        """
        for protocol in (mqtt.MQTTv311, mqtt.MQTTv5):
            server = Server(AsyncMock(), 'localhost', 1883, protocol=protocol)
            server.client.reconnect = MagicMock()
            server.client.connect = MagicMock()

            with patch('time.sleep'):
                server._handle_reconnect(on_connect=False)

            server.client.reconnect.assert_called_once()
            server.client.connect.assert_not_called()
