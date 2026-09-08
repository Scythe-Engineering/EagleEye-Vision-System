"""Regression coverage for bounded Settings log snapshots and SSE congestion."""
import queue
import threading
from types import SimpleNamespace

import pytest

from src.utils.logging.logger import MessageEntry, MessageHistory
from src.webui.web_server import EagleEyeInterface
from src.webui.web_server_utils import system_monitor_mixin


def interface_with_history(size=10000):
    interface = EagleEyeInterface.__new__(EagleEyeInterface)
    history = MessageHistory(max_size=size)
    interface.logger = SimpleNamespace(lock=threading.RLock(), message_history=history)
    return interface, history


def test_recent_history_is_bounded_without_changing_download_history(monkeypatch):
    monkeypatch.setattr(system_monitor_mixin, "_request", lambda: SimpleNamespace(args={}))
    interface, history = interface_with_history()
    history.messages = [MessageEntry('now', str(i)) for i in range(15000)]
    data, status = interface.get_log_messages()
    assert status == 200
    assert len(data['messages']) == 200
    assert data['total_count'] == 15000
    assert data['offset'] == 14800
    assert '14800' in data['messages'][0]
    assert len(history.messages) == 15000


def test_log_stream_detects_rollover_and_collapsed_repeats(monkeypatch):
    interface, history = interface_with_history(size=2)
    history.add_message(MessageEntry('now', 'first'))
    history.add_message(MessageEntry('now', 'second'))
    events = []
    interface._publish_event = lambda name, data: events.append(data)
    ticks = 0

    def tick(_seconds):
        nonlocal ticks
        ticks += 1
        if ticks == 1:
            history.add_message(MessageEntry('now', 'third'))  # unchanged length
        elif ticks in (2, 3):
            history.add_message(MessageEntry('now', 'third'))  # collapse/repeat
        else:
            raise KeyboardInterrupt

    monkeypatch.setattr(system_monitor_mixin.time, 'sleep', tick)
    with pytest.raises(KeyboardInterrupt):
        interface._log_monitor_loop()
    assert len(events) == 4
    assert all(event['replace'] for event in events)
    assert 'repeated x3' in ''.join(events[-1]['messages'])


def test_sse_overflow_does_not_generate_a_log_for_every_drop():
    interface, _ = interface_with_history()
    interface._sse_queue = queue.Queue(maxsize=1)
    interface._sse_queue_lock = threading.Lock()
    interface._last_sse_overflow_warning_ts = 0
    warnings = []
    interface.log = warnings.append
    for i in range(1000):
        interface._publish_event('log_update', {'messages': [str(i)]})
    assert len(warnings) == 1
    assert interface._sse_queue.qsize() == 1


def test_pages_are_stable_across_rollover_and_sequence_collapsing(monkeypatch):
    args = {}
    monkeypatch.setattr(system_monitor_mixin, "_request", lambda: SimpleNamespace(args=args))
    interface, history = interface_with_history(size=600)
    history.messages = [MessageEntry('now', str(i)) for i in range(600)]
    tail, _ = interface.get_log_messages()
    args.update(snapshot=tail['snapshot'], offset='0', limit='200')
    for i in range(100):
        history.add_message(MessageEntry('later', 'repeated'))
    page, code = interface.get_log_messages()
    assert code == 200
    assert page['offset'] == 0
    assert page['total_count'] == 600
    assert page['messages'] == [f'[now] {i}' for i in range(200)]
    args.clear()
    latest, _ = interface.get_log_messages()
    assert latest['snapshot'] != tail['snapshot']


def test_history_query_limits_and_expired_snapshots(monkeypatch):
    args = {'offset': '-1'}
    monkeypatch.setattr(system_monitor_mixin, "_request", lambda: SimpleNamespace(args=args))
    interface, history = interface_with_history()
    assert interface.get_log_messages()[1] == 400
    args.update(offset='not-an-integer')
    assert interface.get_log_messages()[1] == 400
    args.clear()
    args['snapshot'] = 'missing'
    assert interface.get_log_messages()[1] == 410
    args.clear()
    for i in range(20):
        history.add_message(MessageEntry('now', str(i)))
        interface.get_log_messages()
    assert len(interface._log_history_snapshots) == 16
    history.messages = [MessageEntry('now', str(i)) for i in range(1000)]
    args['limit'] = '99999'
    data, _ = interface.get_log_messages()
    assert len(data['messages']) == 500
    assert data['total_count'] == 1000


def test_multiline_entries_are_paged_as_physical_rows(monkeypatch):
    monkeypatch.setattr(system_monitor_mixin, "_request", lambda: SimpleNamespace(args={}))
    interface, history = interface_with_history()
    history.add_message(MessageEntry('now', 'first\nsecond\nthird'))
    data, _ = interface.get_log_messages()
    assert data['messages'] == ['[now] first', 'second', 'third']
    assert data['total_count'] == 3
