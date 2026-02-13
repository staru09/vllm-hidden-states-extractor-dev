"""
Real-time in-process hidden states consumer.

Runs as a background thread inside the vLLM worker process, polling _pending_hidden_states for new handles and logging tensor info as each token is generated.

Start it automatically by setting:
    HIDDEN_STATES_REALTIME_CONSUMER=1

"""
import torch
import threading
import time
import logging

logger = logging.getLogger(__name__)


def _polling_loop(stop_event: threading.Event, poll_interval: float = 0.01):
    """
    Background polling loop that watches _pending_hidden_states and
    processes new handles as they appear.

    This runs INSIDE the vLLM process so it has direct access to the
    GPU buffer and pending states dict.
    """
    from vllm_hidden_states_extractor.hidden_activations import _pending_hidden_states
    from vllm_hidden_states_extractor.gpu_buffer import get_global_buffer

    # Track how many handles we've consumed per request
    consumed_counts: dict[str, int] = {}
    
    # Store accumulated tensors per request: req_id -> list[torch.Tensor]
    request_accumulators: dict[str, list[torch.Tensor]] = {}

    logger.info("[RealtimeConsumer] Polling started")

    while not stop_event.is_set():
        # Snapshot current request IDs
        active_req_ids = list(_pending_hidden_states.keys())

        for req_id in active_req_ids:
            handles = _pending_hidden_states.get(req_id, [])
            already_consumed = consumed_counts.get(req_id, 0)

            # Initialize accumulator if new request
            if req_id not in request_accumulators:
                request_accumulators[req_id] = []

            # Process any new handles
            while already_consumed < len(handles):
                handle = handles[already_consumed]
                step = already_consumed

                try:
                    buffer = get_global_buffer()
                    tensor, meta = buffer.get(handle)

                    if tensor is not None:
                        # ──────────────────────────────────────────
                        # SECONDARY MODEL LOGIC:
                        # 1. Clone/detach to keep it after freeing buffer slot
                        # 2. Accumulate in our local list
                        # 3. Log progress
                        # ──────────────────────────────────────────
                        
                        # Clone implies a copy, ensuring we own this data
                        # independent of the buffer slot we're about to free.
                        # (If secondary model runs usually, we'd process here directly)
                        my_copy = tensor.clone().detach()
                        request_accumulators[req_id].append(my_copy)
                        
                        # Free the buffer slot immediately!
                        # This keeps GPU memory usage low (only 1-2 slots active per req).
                        # [MODIFIED] User requested to keep tensors for post-hoc inspection
                        # buffer.free(handle)

                        curr_len = sum(t.shape[0] for t in request_accumulators[req_id])
                        is_prefill = meta.get("is_prefill", False)
                        step_type = "prefill" if is_prefill else "decode"

                        logger.info(
                            f"[RealtimeConsumer] req={req_id[:8]}... "
                            f"step={step} ({step_type}) "
                            f"consumed shape={list(tensor.shape)} "
                            f"-> accumulated tokens={curr_len}"
                        )
                    else:
                        logger.warning(
                            f"[RealtimeConsumer] req={req_id[:8]}... "
                            f"step={step} handle={handle} tensor=None (expired?)"
                        )
                except Exception as e:
                    logger.error(
                        f"[RealtimeConsumer] req={req_id[:8]}... "
                        f"step={step} error: {e}"
                    )

                already_consumed += 1

            consumed_counts[req_id] = already_consumed

        # Clean up finished requests (no longer in _pending_hidden_states)
        finished = [rid for rid in consumed_counts if rid not in _pending_hidden_states]
        for rid in finished:
            total_steps = consumed_counts.pop(rid)
            tensors = request_accumulators.pop(rid, [])
            
            if tensors:
                full_stacked = torch.cat(tensors, dim=0)
                logger.info(
                    f"[RealtimeConsumer] req={rid[:8]}... DONE — "
                    f"Processed {total_steps} steps. "
                    f"Final Stacked Shape: {list(full_stacked.shape)}"
                )
                # Here you would save 'full_stacked' to disk or send it somewhere
            else:
                 logger.info(
                    f"[RealtimeConsumer] req={rid[:8]}... DONE — "
                    f"Processed {total_steps} steps (no tensors captured)."
                )

        time.sleep(poll_interval)

    logger.info("[RealtimeConsumer] Polling stopped")



# Singleton state
_consumer_lock = threading.Lock()
_consumer_stop_fn = None


def start_consumer(poll_interval: float = 0.01) -> callable:
    """
    Start the real-time consumer in a background thread.
    Idempotent: if already running, returns the existing stop function.

    Args:
        poll_interval: seconds between polls (default 10ms)

    Returns:
        stop_fn: call this to stop the consumer thread
    """
    global _consumer_stop_fn

    with _consumer_lock:
        if _consumer_stop_fn is not None:
            logger.info("[RealtimeConsumer] Already running, returning existing stop_fn")
            return _consumer_stop_fn

        stop_event = threading.Event()
        thread = threading.Thread(
            target=_polling_loop,
            args=(stop_event, poll_interval),
            daemon=True,
            name="hidden-states-realtime-consumer",
        )
        thread.start()

        def stop():
            global _consumer_stop_fn
            stop_event.set()
            thread.join(timeout=5.0)
            logger.info("[RealtimeConsumer] Thread joined")
            with _consumer_lock:
                _consumer_stop_fn = None

        _consumer_stop_fn = stop
        return stop


def maybe_auto_start():
    """
    Auto-start consumer if HIDDEN_STATES_REALTIME_CONSUMER=1 is set.
    Called from the connector's __init__.
    """
    import os
    if os.environ.get("HIDDEN_STATES_REALTIME_CONSUMER", "0") == "1":
        logger.info("[RealtimeConsumer] Auto-starting (HIDDEN_STATES_REALTIME_CONSUMER=1)")
        return start_consumer()
    return None
