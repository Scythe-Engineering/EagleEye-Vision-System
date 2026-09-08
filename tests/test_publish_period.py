"""Publisher cadence configuration regression tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.secondary_operations.publish_to_networktables import PublishToNetworktables


def test_publish_period_can_be_changed_live() -> None:
    """Keep duplicate frames while replacing the publisher at the requested cadence."""
    topic = Mock()
    table = SimpleNamespace(getDoubleTopic=lambda key: topic)
    operation = PublishToNetworktables(table, "telemetry", publish_period_seconds=0.1)
    operation.run(1.0)
    initial = topic.publish.call_args.args[0]
    assert initial.periodic == pytest.approx(0.1)
    assert initial.keepDuplicates and initial.sendAll
    old_publisher = topic.publish.return_value
    operation.update_config({"publish_period_seconds": 0.01})
    old_publisher.close.assert_called_once()
    operation.run(2.0)
    assert topic.publish.call_count == 2
    assert topic.publish.call_args.args[0].periodic == pytest.approx(0.01)
