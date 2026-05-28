"""
Integration tests — require a running MQTT broker.

By default these tests start a temporary mosquitto process on port 18883.
If mosquitto is not installed they are skipped automatically.

Run just the integration tests:
    pytest tests/test_integration.py -v

Skip them explicitly:
    pytest --ignore=tests/test_integration.py
"""

import asyncio
import shutil
import subprocess
import tempfile
import threading
import time
import pytest
import paho.mqtt.client as mqtt
from paho.mqtt.properties import Properties
from paho.mqtt.packettypes import PacketTypes

from mqttasgi.server import Server, _PAHO_MQTT_V2
from mqttasgi.consumers import MqttConsumer


# ---------------------------------------------------------------------------
# Broker fixture
# ---------------------------------------------------------------------------

BROKER_PORT = 18883


def _mosquitto_available():
    return shutil.which('mosquitto') is not None


def _broker_reachable(port, timeout=3.0):
    """Return True if something is accepting TCP connections on *port*."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(('localhost', port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope='module')
def mqtt_broker():
    """Start a temporary mosquitto broker for the duration of the module.

    mosquitto 2.x requires a config file with `allow_anonymous true` and an
    explicit `listener` directive; we write one to a temp file.

    If mosquitto is not installed or fails to start (e.g. port conflict or
    a known macOS crash in 2.0.x), the tests are skipped automatically.
    """
    if not _mosquitto_available():
        pytest.skip('mosquitto not installed')

    with tempfile.NamedTemporaryFile('w', suffix='.conf', delete=False) as f:
        f.write(f'listener {BROKER_PORT}\nallow_anonymous true\n')
        conf_path = f.name

    proc = subprocess.Popen(
        ['mosquitto', '-c', conf_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not _broker_reachable(BROKER_PORT, timeout=3.0):
        proc.terminate()
        proc.wait()
        pytest.skip('mosquitto failed to start (check installation / port conflict)')

    yield BROKER_PORT
    proc.terminate()
    proc.wait()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paho_client():
    """Create a paho v3.1.1 client compatible with whichever paho version is installed."""
    if _PAHO_MQTT_V2:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    return mqtt.Client()


def _make_paho_v5_client():
    """Create a paho MQTTv5 client."""
    if _PAHO_MQTT_V2:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    return mqtt.Client(protocol=mqtt.MQTTv5)


# ---------------------------------------------------------------------------
# Basic broker connectivity
# ---------------------------------------------------------------------------

class TestBrokerConnectivity:
    def test_paho_can_connect(self, mqtt_broker):
        """Verify paho itself can connect to the test broker."""
        connected = []

        def on_connect(client, userdata, flags, *args):
            connected.append(True)

        client = _make_paho_client()
        client.on_connect = on_connect
        client.connect('localhost', mqtt_broker)
        client.loop_start()
        deadline = time.time() + 5
        while not connected and time.time() < deadline:
            time.sleep(0.05)
        client.loop_stop()
        client.disconnect()
        assert connected, "paho client failed to connect to test broker"

    def test_publish_subscribe_roundtrip(self, mqtt_broker):
        """A single client can publish and receive on the same topic."""
        received = []

        def on_connect(client, userdata, flags, *args):
            client.subscribe('integration/test', qos=1)

        def on_message(client, userdata, msg):
            received.append(msg.payload)

        client = _make_paho_client()
        client.on_connect = on_connect
        client.on_message = on_message
        client.connect('localhost', mqtt_broker)
        client.loop_start()

        # Wait for connection + subscription
        time.sleep(0.3)
        client.publish('integration/test', b'hello', qos=1)

        deadline = time.time() + 5
        while not received and time.time() < deadline:
            time.sleep(0.05)

        client.loop_stop()
        client.disconnect()
        assert received == [b'hello']


@pytest.fixture(scope='module')
def live_broker():
    """Reuse an already-running broker on BROKER_PORT (e.g. Docker mosquitto).

    Unlike mqtt_broker, this fixture does NOT start a new process — it simply
    checks if a broker is already accepting connections on BROKER_PORT.
    Tests that use this fixture run against the Docker mosquitto and produce
    real broker-side traffic (visible in docker logs).

    Skipped when no broker is reachable on that port.
    """
    if not _broker_reachable(BROKER_PORT, timeout=1.0):
        pytest.skip(f'No MQTT broker reachable on localhost:{BROKER_PORT}')
    yield BROKER_PORT


# ---------------------------------------------------------------------------
# Server + consumer integration
# ---------------------------------------------------------------------------

class TestServerConsumerIntegration:
    """
    Runs the mqttasgi Server in a background thread and verifies that a
    consumer can subscribe, receive, and publish messages end-to-end.
    """

    def _run_server(self, consumer_cls, broker_port, received_messages, published_events):
        """Helper: runs the server until the test signals it to stop."""

        class _TestConsumer(consumer_cls):
            async def receive(self, mqtt_message):
                received_messages.append(mqtt_message)
                await super().receive(mqtt_message)

        server = Server(_TestConsumer.as_asgi(), host='localhost', port=broker_port)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        return server, thread

    async def test_consumer_receives_published_message(self, mqtt_broker):
        """Messages published by an external client reach the consumer."""
        received = []
        # threading.Event is thread-safe — asyncio.Event cannot be set()
        # from a different thread/loop (the Server runs its own event loop).
        ready = threading.Event()

        class ListenerConsumer(MqttConsumer):
            async def connect(self):
                await self.subscribe('srv/test', qos=1)
                ready.set()  # safe to call from any thread

            async def receive(self, mqtt_message):
                received.append(mqtt_message)

            async def disconnect(self):
                pass

        server = Server(ListenerConsumer.as_asgi(), host='localhost', port=mqtt_broker, client_id=None)

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for server + consumer to be ready using run_in_executor so we
        # don't block the test's event loop while waiting on the thread event.
        loop = asyncio.get_event_loop()
        connected = await loop.run_in_executor(None, lambda: ready.wait(timeout=5))
        if not connected:
            pytest.fail("Consumer did not connect and subscribe within 5 seconds")

        # Publish from an external paho client
        pub_client = _make_paho_client()
        pub_client.connect('localhost', mqtt_broker)
        pub_client.loop_start()
        pub_client.publish('srv/test', b'from-outside', qos=1)
        time.sleep(0.5)
        pub_client.loop_stop()
        pub_client.disconnect()

        # Allow the message to propagate
        deadline = time.time() + 3
        while not received and time.time() < deadline:
            await asyncio.sleep(0.05)

        assert received, "Consumer never received the published message"
        assert received[0]['topic'] == 'srv/test'
        assert received[0]['payload'] == b'from-outside'


# ---------------------------------------------------------------------------
# Shared subscriptions — requires live Docker broker
# ---------------------------------------------------------------------------

class TestSharedSubIntegration:
    """End-to-end: consumer subscribes via $share/<group>/<topic>.

    mosquitto 2.x supports shared subscriptions for both MQTTv3.1.1 and MQTTv5.
    These tests verify the full path: broker receives the subscribe, distributes
    a publish to one of the shared subscribers, and the consumer gets the message.
    """

    async def test_shared_sub_receives_message(self, live_broker):
        """Consumer subscribed via $share/... receives messages published to the real topic."""
        received = []
        ready = threading.Event()

        class SharedSubConsumer(MqttConsumer):
            async def connect(self):
                await self.subscribe('$share/loadbalance/integ/shared', qos=1)
                ready.set()

            async def receive(self, mqtt_message):
                received.append(mqtt_message)

            async def disconnect(self):
                pass

        server = Server(SharedSubConsumer.as_asgi(), host='localhost', port=live_broker, client_id=None)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        connected = await loop.run_in_executor(None, lambda: ready.wait(timeout=5))
        if not connected:
            pytest.fail("Consumer did not connect and subscribe within 5 seconds")

        pub_client = _make_paho_client()
        pub_client.connect('localhost', live_broker)
        pub_client.loop_start()
        pub_client.publish('integ/shared', b'shared-payload', qos=1)
        time.sleep(0.5)
        pub_client.loop_stop()
        pub_client.disconnect()

        deadline = time.time() + 3
        while not received and time.time() < deadline:
            await asyncio.sleep(0.05)

        assert received, "Shared subscription consumer never received the message"
        assert received[0]['topic'] == 'integ/shared'
        assert received[0]['payload'] == b'shared-payload'

    async def test_shared_sub_wildcard_receives_message(self, live_broker):
        """Consumer subscribed via $share/grp/sensors/# receives messages on sensors/…."""
        received = []
        ready = threading.Event()

        class WildcardSharedConsumer(MqttConsumer):
            async def connect(self):
                await self.subscribe('$share/grp/integ/sensors/#', qos=1)
                ready.set()

            async def receive(self, mqtt_message):
                received.append(mqtt_message)

            async def disconnect(self):
                pass

        server = Server(WildcardSharedConsumer.as_asgi(), host='localhost', port=live_broker, client_id=None)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        connected = await loop.run_in_executor(None, lambda: ready.wait(timeout=5))
        if not connected:
            pytest.fail("Consumer did not connect and subscribe within 5 seconds")

        pub_client = _make_paho_client()
        pub_client.connect('localhost', live_broker)
        pub_client.loop_start()
        pub_client.publish('integ/sensors/temp', b'42C', qos=1)
        time.sleep(0.5)
        pub_client.loop_stop()
        pub_client.disconnect()

        deadline = time.time() + 3
        while not received and time.time() < deadline:
            await asyncio.sleep(0.05)

        assert received, "Wildcard shared subscription consumer never received the message"
        assert received[0]['topic'] == 'integ/sensors/temp'
        assert received[0]['payload'] == b'42C'


# ---------------------------------------------------------------------------
# MQTTv5 properties — requires live Docker broker
# ---------------------------------------------------------------------------

class TestMQTTv5PropertiesIntegration:
    """End-to-end: MQTTv5 properties survive the broker and reach the consumer."""

    async def test_correlation_data_received(self, live_broker):
        """CorrelationData published by a v5 client is delivered to the consumer."""
        received = []
        ready = threading.Event()

        class V5Consumer(MqttConsumer):
            async def connect(self):
                await self.subscribe('integ/v5/props', qos=1)
                ready.set()

            async def receive(self, mqtt_message):
                received.append(mqtt_message)

            async def disconnect(self):
                pass

        server = Server(V5Consumer.as_asgi(), host='localhost', port=live_broker,
                        protocol=mqtt.MQTTv5, client_id=None)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        connected = await loop.run_in_executor(None, lambda: ready.wait(timeout=5))
        if not connected:
            pytest.fail("Consumer did not connect within 5 seconds")

        pub_client = _make_paho_v5_client()
        pub_client.connect('localhost', live_broker)
        pub_client.loop_start()

        props = Properties(PacketTypes.PUBLISH)
        props.CorrelationData = b'req-integ-001'
        props.ContentType = 'application/json'
        pub_client.publish('integ/v5/props', b'{"ok": true}', qos=1, properties=props)
        time.sleep(0.5)
        pub_client.loop_stop()
        pub_client.disconnect()

        deadline = time.time() + 3
        while not received and time.time() < deadline:
            await asyncio.sleep(0.05)

        assert received, "Consumer never received the v5 message"
        assert received[0]['properties'].get('CorrelationData') == b'req-integ-001'
        assert received[0]['properties'].get('ContentType') == 'application/json'

    async def test_request_reply_with_correlation_data(self, live_broker):
        """Consumer echoes CorrelationData back to ResponseTopic — full request/reply flow."""
        replies = []
        server_ready = threading.Event()

        class ReplyConsumer(MqttConsumer):
            async def connect(self):
                await self.subscribe('integ/v5/requests', qos=1)
                server_ready.set()

            async def receive(self, mqtt_message):
                props = mqtt_message.get('properties', {})
                reply_topic = props.get('ResponseTopic')
                correlation_id = props.get('CorrelationData')
                if reply_topic:
                    await self.publish(
                        reply_topic,
                        b'{"status": "ok"}',
                        qos=1,
                        properties={'CorrelationData': correlation_id},
                    )

            async def disconnect(self):
                pass

        server = Server(ReplyConsumer.as_asgi(), host='localhost', port=live_broker,
                        protocol=mqtt.MQTTv5, client_id=None)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        connected = await loop.run_in_executor(None, lambda: server_ready.wait(timeout=5))
        if not connected:
            pytest.fail("Consumer did not connect within 5 seconds")

        reply_received = threading.Event()

        req_client = _make_paho_v5_client()

        def on_message(client, userdata, msg):
            replies.append(msg)
            reply_received.set()

        req_client.on_message = on_message
        req_client.connect('localhost', live_broker)
        req_client.loop_start()
        req_client.subscribe('integ/v5/replies', qos=1)
        time.sleep(0.2)

        props = Properties(PacketTypes.PUBLISH)
        props.CorrelationData = b'corr-456'
        props.ResponseTopic = 'integ/v5/replies'
        req_client.publish('integ/v5/requests', b'{"action": "ping"}', qos=1, properties=props)

        await loop.run_in_executor(None, lambda: reply_received.wait(timeout=5))
        req_client.loop_stop()
        req_client.disconnect()

        assert replies, "No reply received on ResponseTopic"
        reply_props = replies[0].properties
        assert reply_props.CorrelationData == b'corr-456'
