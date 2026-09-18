"""ZMQ publisher backed by the lazy Maxwell replay source.

New code should use ``MaxwellEnv(replay=...)`` directly.  This module remains
for transport compatibility and explicit ZMQ loopback testing.
"""

import time

import zmq

from braindance.core.replay import H5ReplaySource, pack_replay_packet


def run(
    data_path,
    speed=1.0,
    loop=True,
    start_frame=0,
    stop_frame=None,
    channels=None,
    unfiltered_endpoint="tcp://*:7204",
    filtered_endpoint="tcp://*:7205",
    stop_event=None,
    ready_event=None,
    max_frames=None,
):
    """Publish replay frames using Maxwell's three-part dummy packet format."""
    context = zmq.Context()
    publisher_unfiltered = context.socket(zmq.PUB)
    publisher_filtered = context.socket(zmq.PUB)
    source = None
    try:
        publisher_unfiltered.setsockopt(zmq.LINGER, 0)
        publisher_filtered.setsockopt(zmq.LINGER, 0)
        publisher_unfiltered.bind(unfiltered_endpoint)
        publisher_filtered.bind(filtered_endpoint)
        source = H5ReplaySource(
            source=data_path,
            speed=speed,
            loop=loop,
            start_frame=start_frame,
            stop_frame=stop_frame,
            channels=channels,
        )
        if ready_event is not None:
            ready_event.set()
        # PUB/SUB needs a short subscription handshake before first packet.
        time.sleep(0.1)

        sent = 0
        while stop_event is None or not stop_event.is_set():
            batch = source.read(count=1)
            if batch is None:
                break
            parts = pack_replay_packet(
                batch["frame_numbers"][0],
                batch["raw_float32"][0],
                batch["events"][0],
            )
            publisher_unfiltered.send_multipart(parts)
            publisher_filtered.send_multipart(parts)
            sent += 1
            if max_frames is not None and sent >= int(max_frames):
                break
    finally:
        if source is not None:
            source.close()
        publisher_unfiltered.close(linger=0)
        publisher_filtered.close(linger=0)
        context.term()


if __name__ == "__main__":
    raise SystemExit("Use bdreplay or call run(data_path=...) explicitly")
