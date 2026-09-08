"""Read two snapshots of the synthetic estimator benchmark status."""

import json
import time
import ntcore

instance = ntcore.NetworkTableInstance.create()
instance.startClient4("highrate-validation")
instance.setServer("127.0.0.1")
keys = [
    f"HighRate/{lane}/{metric}"
    for lane in ["main20", "thread5"]
    for metric in ["Updates", "Accepted", "ErrorMeters", "MaxGapMs"]
]
subscribers = {
    key: instance.getDoubleTopic("/SmartDashboard/" + key).subscribe(-1) for key in keys
}
try:
    time.sleep(2)
    initial = {key: subscriber.get() for key, subscriber in subscribers.items()}
    time.sleep(3)
    final = {key: subscriber.get() for key, subscriber in subscribers.items()}
    print(
        json.dumps(
            {"connected": instance.isConnected(), "start": initial, "end": final},
            indent=2,
        )
    )
finally:
    for subscriber in subscribers.values():
        subscriber.close()
    instance.stopClient()
    ntcore.NetworkTableInstance.destroy(instance)
