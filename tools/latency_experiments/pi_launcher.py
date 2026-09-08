"""Temporary live-frame NT4/UDP comparison launcher, never used by normal startup.

Run as a module from the repository root. Configuration is read from
/tmp/eagleeye-latency-experiment.json; the regular publisher source is untouched.
The UDP mirror is diagnostic only and reuses NT4's time synchronization.
"""
from __future__ import annotations
import json
import runpy
import socket
import struct
import threading
import time
from pathlib import Path

import ntcore
from src.secondary_operations import publish_to_networktables as publisher

CONFIG = Path('/tmp/eagleeye-latency-experiment.json')
DESTINATION = ('100.75.14.59', 5810)
PREFIX = 'localization/arducam-ov9281-usb-camera/'
settings = {'periodic': .01, 'flush': False, 'scenario': 0, 'udp': True}
lock = threading.RLock()
pending = {}
sequence = 0
audit = None
audit_period = None
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.setblocking(False)
sent = dropped = 0
original_options = publisher.PubSubOptions
original_create = publisher.PublishToNetworktables._create_publisher
original_publish = publisher.PublishToNetworktables._publish


def config_loop():
    global settings
    while True:
        try:
            data = json.loads(CONFIG.read_text())
            period = float(data['periodic'])
            if period not in {.001, .005, .01}:
                raise ValueError('unsupported period')
            settings = {'periodic': period, 'flush': bool(data.get('flush', False)),
                        'scenario': int(data['scenario']), 'udp': bool(data.get('udp', True))}
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(.2)


def options(*args, **kwargs):
    kwargs['periodic'] = settings['periodic']
    return original_options(*args, **kwargs)


class Proxy:
    def __init__(self, inner, key):
        self.inner = inner
        self.key = key
    def close(self):
        self.inner.close()
    def set(self, value, timestamp=0):
        self.inner.set(value, timestamp)
        if not timestamp or self.key not in (PREFIX + 'pose', PREFIX + 'meta'):
            return
        # Both branches have now published for this exact camera frame.
        with lock:
            global audit, audit_period, sequence, sent, dropped
            parts = pending.setdefault(timestamp, {})
            parts[self.key.rsplit('/', 1)[1]] = value
            if len(parts) != 2:
                return
            del pending[timestamp]
            for old in list(pending):
                if old < timestamp - 1000000:
                    del pending[old]
            inst = ntcore.NetworkTableInstance.getDefault()
            offset = inst.getServerTimeOffset()
            if offset is None:
                return
            pose, meta = parts['pose'], parts['meta']
            if len(meta) != 3:
                return
            cfg = settings
            if audit is None or audit_period != cfg['periodic']:
                if audit is not None:
                    audit.close()
                audit = inst.getDoubleArrayTopic('/EagleEye/audit/latency/frame').publish(
                    original_options(sendAll=True, keepDuplicates=True, periodic=cfg['periodic']))
                audit_period = cfg['periodic']
            q = pose.rotation().getQuaternion()
            sequence += 1
            ready = ntcore._now()
            values = [sequence, timestamp + offset, ready + offset,
                      pose.X(), pose.Y(), pose.Z(), q.W(), q.X(), q.Y(), q.Z(),
                      *meta, cfg['scenario'], offset]
            # Same diagnostic payload through both transports; source data are unchanged.
            audit.set(values, timestamp)
            if cfg['udp']:
                try:
                    udp.sendto(struct.pack('<15d', *values), DESTINATION)
                    sent += 1
                except (BlockingIOError, OSError):
                    dropped += 1
            if cfg['flush']:
                inst.flush()
            if sequence % 3600 == 0:
                print(f'LATENCY_PROBE scenario={cfg["scenario"]} frames={sequence} udp_sent={sent} udp_dropped={dropped}', flush=True)


def create(self, value):
    inner = original_create(self, value)
    return Proxy(inner, self.target_key) if inner is not None else None


def publish(self, value):
    # Recreate only on a test cadence change; discard transition windows in analysis.
    period = settings['periodic']
    if getattr(self, '_experiment_period', None) != period:
        if self._publisher is not None:
            self._publisher.close()
        self._publisher = None
        self._experiment_period = period
    return original_publish(self, value)


if __name__ == '__main__':
    threading.Thread(target=config_loop, daemon=True, name='latency-config').start()
    publisher.PubSubOptions = options
    publisher.PublishToNetworktables._create_publisher = create
    publisher.PublishToNetworktables._publish = publish
    runpy.run_module('src.main_backend', run_name='__main__')
