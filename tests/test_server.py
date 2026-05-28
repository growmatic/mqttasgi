"""
Unit tests for mqttasgi.server.

These tests do NOT require a running MQTT broker. They verify:
- paho-mqtt 1.x / 2.x compatibility shim works correctly
- Callback signatures accept both paho API versions
- Server initialises without errors
- MQTTv5 protocol selection and clean_start handling
- MQTTv5 properties: helper, receive, publish, callbacks
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call

import paho.mqtt.client as mqtt
from paho.mqtt.properties import Properties
from paho.mqtt.packettypes import PacketTypes

from mqttasgi.server import Server, _PAHO_MQTT_V2

# _properties_to_dict is added by the MQTTv5 properties commit.
# Tests that depend on it are skipped until it is importable.
try:
    from mqttasgi.server import _properties_to_dict
    _PROPS_AVAILABLE = True
except ImportError:
    _properties_to_dict = None
    _PROPS_AVAILABLE = False

_needs_props = pytest.mark.skipif(not _PROPS_AVAILABLE,
                                   reason="_properties_to_dict not yet in server.py")


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


# ---------------------------------------------------------------------------
# MQTTv5 properties — _properties_to_dict helper
# ---------------------------------------------------------------------------

@_needs_props
class TestPropertiesHelper:
    """_properties_to_dict converts a paho Properties object to a plain dict.

    Only non-None attributes are included so that consumers can safely call
    .get('CorrelationData') without always checking for key presence.
    """

    def _make_props(self, **kwargs):
        """Create a real paho Properties object for PUBLISH packets."""
        props = Properties(PacketTypes.PUBLISH)
        for key, value in kwargs.items():
            setattr(props, key, value)
        return props

    def test_none_returns_empty_dict(self):
        assert _properties_to_dict(None) == {}

    def test_empty_props_returns_empty_dict(self):
        """A Properties object with no attributes set returns {}."""
        props = Properties(PacketTypes.PUBLISH)
        assert _properties_to_dict(props) == {}

    def test_extracts_correlation_data(self):
        props = self._make_props(CorrelationData=b'req-123')
        result = _properties_to_dict(props)
        assert result['CorrelationData'] == b'req-123'

    def test_extracts_response_topic(self):
        props = self._make_props(ResponseTopic='reply/queue')
        result = _properties_to_dict(props)
        assert result['ResponseTopic'] == 'reply/queue'

    def test_extracts_content_type(self):
        props = self._make_props(ContentType='application/json')
        result = _properties_to_dict(props)
        assert result['ContentType'] == 'application/json'

    def test_extracts_user_property(self):
        props = self._make_props(UserProperty=[('key', 'value')])
        result = _properties_to_dict(props)
        assert result['UserProperty'] == [('key', 'value')]

    def test_extracts_message_expiry_interval(self):
        props = self._make_props(MessageExpiryInterval=60)
        result = _properties_to_dict(props)
        assert result['MessageExpiryInterval'] == 60

    def test_extracts_payload_format_indicator(self):
        props = self._make_props(PayloadFormatIndicator=1)
        result = _properties_to_dict(props)
        assert result['PayloadFormatIndicator'] == 1

    def test_only_set_attrs_are_included(self):
        """A Properties object with only CorrelationData must not contain other keys."""
        props = self._make_props(CorrelationData=b'x')
        result = _properties_to_dict(props)
        assert set(result.keys()) == {'CorrelationData'}

    def test_multiple_attrs_all_extracted(self):
        props = self._make_props(
            CorrelationData=b'id-1',
            ResponseTopic='resp/topic',
        )
        result = _properties_to_dict(props)
        assert result == {'CorrelationData': b'id-1', 'ResponseTopic': 'resp/topic'}


# ---------------------------------------------------------------------------
# MQTTv5 properties — _mqtt_receive propagates to consumers and queues
# ---------------------------------------------------------------------------

class TestMQTTv5Receive:
    """_mqtt_receive must include a 'properties' key in every mqtt.msg event."""

    def _make_subscribed_server(self):
        """Return a server with one app registered on 'test/topic'."""
        server = _make_server()
        server.application_data[0] = {'receive': asyncio.Queue(), 'subscriptions': {}}
        server.topics_subscription['test/topic'] = {'qos': 1, 'apps': {0}}
        return server

    def test_properties_included_in_mqtt_msg_event(self):
        """Delivered message event must contain the properties dict."""
        server = self._make_subscribed_server()
        server._mqtt_receive('test/topic', 'test/topic', b'hello', 1,
                              {'CorrelationData': b'abc'})
        event = server.application_data[0]['receive'].get_nowait()
        assert event['mqtt']['properties'] == {'CorrelationData': b'abc'}

    def test_properties_default_to_empty_dict(self):
        """When no properties are passed, the event must still have properties: {}."""
        server = self._make_subscribed_server()
        server._mqtt_receive('test/topic', 'test/topic', b'hello', 1)
        event = server.application_data[0]['receive'].get_nowait()
        assert event['mqtt']['properties'] == {}

    def test_queued_message_stores_properties(self):
        """Messages arriving before any subscription (sub=-1) must also store properties."""
        server = _make_server()
        server._mqtt_receive(-1, 'unknown/topic', b'data', 0,
                              {'ResponseTopic': 'reply/here'})
        queued = server.topic_queues['unknown/topic'][0]
        assert queued['properties'] == {'ResponseTopic': 'reply/here'}

    def test_queued_message_empty_properties(self):
        """Queue path with no properties stores properties: {}."""
        server = _make_server()
        server._mqtt_receive(-1, 'unknown/topic', b'data', 0)
        queued = server.topic_queues['unknown/topic'][0]
        assert queued['properties'] == {}

    def test_on_message_callback_extracts_paho_properties(self):
        """The on_message lambda registered in __init__ must extract paho properties.

        Simulates paho delivering a message with a real Properties object
        and verifies that _mqtt_receive receives the extracted dict.
        """
        server = _make_server()
        captured = {}

        def fake_receive(subscription, topic, payload, qos, properties=None):
            captured['properties'] = properties

        server._mqtt_receive = fake_receive

        # Simulate a paho v5 message with properties
        fake_msg = MagicMock()
        fake_msg.topic = 'some/topic'
        fake_msg.payload = b'payload'
        fake_msg.qos = 1
        props = Properties(PacketTypes.PUBLISH)
        props.CorrelationData = b'req-42'
        fake_msg.properties = props

        server.client.on_message(None, None, fake_msg)

        assert captured.get('properties', 'NOT_PASSED') != 'NOT_PASSED', \
            "on_message did not pass properties to _mqtt_receive"
        assert captured['properties'].get('CorrelationData') == b'req-42'

    def test_on_message_callback_no_properties_passes_empty_dict(self):
        """on_message with a v3.1.1 message (no properties attr) must pass {}."""
        server = _make_server()
        captured = {}

        def fake_receive(subscription, topic, payload, qos, properties=None):
            captured['properties'] = properties

        server._mqtt_receive = fake_receive

        fake_msg = MagicMock(spec=['topic', 'payload', 'qos'])  # no .properties attr
        fake_msg.topic = 'some/topic'
        fake_msg.payload = b'payload'
        fake_msg.qos = 0

        server.client.on_message(None, None, fake_msg)

        assert captured['properties'] == {}


# ---------------------------------------------------------------------------
# MQTTv5 properties — mqtt_publish builds paho Properties object
# ---------------------------------------------------------------------------

class TestMQTTv5Publish:
    """mqtt_publish must build a paho Properties object for v5 and omit it for v3.1.1."""

    def _make_v5_server(self):
        server = Server(AsyncMock(), 'localhost', 1883, protocol=mqtt.MQTTv5)
        server.client.publish = MagicMock()
        return server

    def _make_v311_server(self):
        server = _make_server()
        server.client.publish = MagicMock()
        return server

    def _pub_msg(self, **props):
        return {
            'type': 'mqtt.pub',
            'mqtt': {'topic': 't', 'payload': b'p', 'qos': 1, 'retain': False,
                     'properties': props},
        }

    async def test_v5_correlation_data_bytes_passed_through(self):
        server = self._make_v5_server()
        await server.mqtt_publish(0, self._pub_msg(CorrelationData=b'req-1'))
        _, kwargs = server.client.publish.call_args
        assert kwargs['properties'].CorrelationData == b'req-1'

    async def test_v5_correlation_data_str_encoded_to_bytes(self):
        """CorrelationData as str must be encoded to bytes before setting on Properties."""
        server = self._make_v5_server()
        await server.mqtt_publish(0, self._pub_msg(CorrelationData='req-1'))
        _, kwargs = server.client.publish.call_args
        assert kwargs['properties'].CorrelationData == b'req-1'

    async def test_v5_response_topic(self):
        server = self._make_v5_server()
        await server.mqtt_publish(0, self._pub_msg(ResponseTopic='reply/q'))
        _, kwargs = server.client.publish.call_args
        assert kwargs['properties'].ResponseTopic == 'reply/q'

    async def test_v5_content_type(self):
        server = self._make_v5_server()
        await server.mqtt_publish(0, self._pub_msg(ContentType='application/json'))
        _, kwargs = server.client.publish.call_args
        assert kwargs['properties'].ContentType == 'application/json'

    async def test_v5_user_property(self):
        server = self._make_v5_server()
        await server.mqtt_publish(0, self._pub_msg(UserProperty=[('k', 'v')]))
        _, kwargs = server.client.publish.call_args
        assert kwargs['properties'].UserProperty == [('k', 'v')]

    async def test_v5_no_properties_publish_still_passes_properties_kwarg(self):
        """v5 publish with empty properties dict must still pass a Properties object."""
        server = self._make_v5_server()
        msg = {'type': 'mqtt.pub',
               'mqtt': {'topic': 't', 'payload': b'p', 'qos': 1, 'retain': False}}
        await server.mqtt_publish(0, msg)
        _, kwargs = server.client.publish.call_args
        assert 'properties' in kwargs
        assert isinstance(kwargs['properties'], Properties)

    async def test_v311_publish_never_passes_properties_kwarg(self):
        """v3.1.1 publish must NOT include a properties kwarg — paho would reject it."""
        server = self._make_v311_server()
        await server.mqtt_publish(0, self._pub_msg(CorrelationData=b'x'))
        _, kwargs = server.client.publish.call_args
        assert 'properties' not in kwargs

    async def test_v311_publish_with_properties_logs_warning(self):
        """v3.1.1 publish with a non-empty properties dict must emit a warning.

        Properties are silently dropped on v3.1.1, so callers must be told
        via a log warning rather than failing silently.
        """
        server = self._make_v311_server()
        with patch.object(server.log, 'warning') as mock_warn:
            await server.mqtt_publish(0, self._pub_msg(CorrelationData=b'x'))
        mock_warn.assert_called_once()
        warning_text = mock_warn.call_args[0][0]
        assert 'properties' in warning_text and 'v3.1.1' in warning_text

    async def test_v311_publish_without_properties_no_warning(self):
        """v3.1.1 publish with no properties must not produce a warning."""
        server = self._make_v311_server()
        msg = {'type': 'mqtt.pub',
               'mqtt': {'topic': 't', 'payload': b'p', 'qos': 1, 'retain': False}}
        with patch.object(server.log, 'warning') as mock_warn:
            await server.mqtt_publish(0, msg)
        mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# Shared subscriptions — $share/<group>/<topic>
# ---------------------------------------------------------------------------

class TestSharedSubscriptions:
    """Shared subscription filters ($share/<group>/<topic>) must be handled correctly.

    The broker strips the $share/<group>/ prefix before delivering messages,
    so the topic in incoming messages is the real topic (e.g. 'sensors/temp'),
    not the full filter ('$share/mygroup/sensors/temp').

    paho's message_callback_add routes by matching message.topic against the
    registered filter.  mqtt.topic_matches_sub('$share/g/t', 't') returns False,
    so callbacks registered with the full $share string are never fired.

    The fix: register message_callback_add with the real (stripped) topic while
    still passing the full $share string to client.subscribe / client.unsubscribe.
    """

    def _make_server_with_app(self):
        server = _make_server()
        server.application_data[0] = {'receive': asyncio.Queue(), 'subscriptions': {}}
        server.client.subscribe = MagicMock()
        server.client.unsubscribe = MagicMock()
        server.client.message_callback_add = MagicMock()
        server.client.message_callback_remove = MagicMock()
        return server

    # --- subscribe ---

    async def test_paho_subscribe_uses_full_shared_topic(self):
        """client.subscribe must receive the full $share/<group>/<topic> string."""
        server = self._make_server_with_app()
        await server.mqtt_subscribe(0, {'mqtt': {'topic': '$share/mygroup/sensors/temp', 'qos': 1}})
        server.client.subscribe.assert_called_once_with('$share/mygroup/sensors/temp', 1)

    async def test_message_callback_uses_real_topic(self):
        """message_callback_add must be registered for the real topic, not $share/…."""
        server = self._make_server_with_app()
        await server.mqtt_subscribe(0, {'mqtt': {'topic': '$share/mygroup/sensors/temp', 'qos': 1}})
        registered = server.client.message_callback_add.call_args[0][0]
        assert registered == 'sensors/temp'

    async def test_topics_subscription_keyed_by_real_topic(self):
        """topics_subscription must be keyed by the real topic, not $share/…."""
        server = self._make_server_with_app()
        await server.mqtt_subscribe(0, {'mqtt': {'topic': '$share/mygroup/sensors/temp', 'qos': 1}})
        assert 'sensors/temp' in server.topics_subscription
        assert '$share/mygroup/sensors/temp' not in server.topics_subscription

    async def test_message_delivered_to_subscribed_app(self):
        """Message arriving on 'sensors/temp' must reach app subscribed via $share/….

        The lambda registered in message_callback_add passes the stripped topic
        as the subscription key to _mqtt_receive.
        """
        server = self._make_server_with_app()
        await server.mqtt_subscribe(0, {'mqtt': {'topic': '$share/mygroup/sensors/temp', 'qos': 1}})

        # After subscribe, internal key is the stripped topic.
        # Simulate paho calling _mqtt_receive with the stripped subscription key.
        server._mqtt_receive('sensors/temp', 'sensors/temp', b'25C', 1)

        event = server.application_data[0]['receive'].get_nowait()
        assert event['type'] == 'mqtt.msg'
        assert event['mqtt']['topic'] == 'sensors/temp'
        assert event['mqtt']['payload'] == b'25C'

    async def test_shared_sub_multi_level_wildcard_real_topic(self):
        """$share filter with a wildcard real topic is registered correctly."""
        server = self._make_server_with_app()
        await server.mqtt_subscribe(0, {'mqtt': {'topic': '$share/grp/sensors/#', 'qos': 1}})
        registered = server.client.message_callback_add.call_args[0][0]
        assert registered == 'sensors/#'

    # --- unsubscribe ---

    async def test_paho_unsubscribe_uses_full_shared_topic(self):
        """client.unsubscribe must receive the full $share/<group>/<topic> string."""
        server = self._make_server_with_app()
        await server.mqtt_subscribe(0, {'mqtt': {'topic': '$share/mygroup/sensors/temp', 'qos': 1}})
        await server.mqtt_unsubscribe(0, {'mqtt': {'topic': '$share/mygroup/sensors/temp'}})
        server.client.unsubscribe.assert_called_once_with('$share/mygroup/sensors/temp')

    async def test_unsubscribe_removes_real_topic_callback(self):
        """message_callback_remove must use the real topic, not $share/…."""
        server = self._make_server_with_app()
        await server.mqtt_subscribe(0, {'mqtt': {'topic': '$share/mygroup/sensors/temp', 'qos': 1}})
        await server.mqtt_unsubscribe(0, {'mqtt': {'topic': '$share/mygroup/sensors/temp'}})
        server.client.message_callback_remove.assert_called_once_with('sensors/temp')

    # --- regular topics unaffected ---

    async def test_regular_topic_message_callback_unchanged(self):
        """Non-shared subscriptions must still register with the topic as-is."""
        server = self._make_server_with_app()
        await server.mqtt_subscribe(0, {'mqtt': {'topic': 'sensors/temp', 'qos': 1}})
        registered = server.client.message_callback_add.call_args[0][0]
        assert registered == 'sensors/temp'

    async def test_regular_topic_unsubscribe_callback_unchanged(self):
        """Non-shared unsubscribe must still remove the topic as-is."""
        server = self._make_server_with_app()
        await server.mqtt_subscribe(0, {'mqtt': {'topic': 'sensors/temp', 'qos': 1}})
        await server.mqtt_unsubscribe(0, {'mqtt': {'topic': 'sensors/temp'}})
        server.client.message_callback_remove.assert_called_once_with('sensors/temp')
