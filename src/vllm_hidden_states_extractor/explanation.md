# System Explanation: Streaming Hidden States Extractor

This document explains the architecture, concepts, and design decisions behind the real-time hidden states extraction pipeline for vLLM.

## 1. How are we capturing the hidden states?

We use a **PyTorch Forward Hook** injected into a specific layer of the model (e.g., Layer 20).

1.  **Intercept**: When vLLM runs a forward pass (prefill or decode step), the hook intercepts the layer's output tensor.
2.  **Store**: The tensor is immediately copied to a pre-allocated **GPU Ring Buffer**. This copy happens entirely on the GPU (device-to-device), avoiding slow CPU transfers.
3.  **Track**: A unique handle (string ID) is generated for the stored tensor. This handle is appended to a list in a global dictionary (`_pending_hidden_states`) keyed by the request ID.

## 2. Important Jargon & Concepts

- **IPC (Inter-Process Communication)**: We use **CUDA IPC** handles to allow separate processes to access GPU memory pointers without copying data. While our primary real-time solution runs in-process (using threads), the underlying buffer supports IPC for multi-process architectures.
- **Memory Overhead**: Each captured hidden state consumes GPU VRAM. For a `4096` dim model in `bfloat16`, a single token takes `4096 * 2 bytes = 8KB`. A full sequence of 4096 tokens takes ~32MB. We use a **Ring Buffer** (default 512 slots) to recycle this memory automatically.
- **Race Conditions**: When multiple threads access shared data (like the list of handles), race conditions can occur. We solved this by using a **Singleton Pattern** for the consumer thread, ensuring only one poller runs at a time, preventing double-processing or conflicts.
- **Threading**: The real-time consumer runs as a **Daemon Thread** inside the main vLLM worker process. "Daemon" means it runs in the background and automatically shuts down when the main process exits.

## 3. KV Cache vs. Hidden States

They are distinct components serve different purposes:

- **Hidden States**: The _current_ activation vector (output) of a specific layer for the token currently being processed. It represents the model's "thought" at that specific layer and time step.
- **Relation**: Our system captures hidden states _after_ the attention mechanism (which reads the KV cache) has computed its output. We do not modify or read the KV cache directly.

## 4. Lifecycle of Captured Tensors

1.  **Creation**: The forward hook writes the tensor to the `GPUBuffer` and gets a handle.
2.  **Pending**: The handle sits in the `_pending_hidden_states` list. The tensor occupies a slot in GPU VRAM.
3.  **Consumption**: A consumer (real-time thread or API client) retrieves the tensor using the handle.
4.  **Freeing**:
    - **Ideally**: The consumer calls `buffer.free(handle)` immediately after processing. The slot is marked free and can be overwritten by new data.
    - **Fallback**: If not manually freed, the slot remains occupied until its **TTL (Time-To-Live)** expires (default 60s), at which point it is forcibly reclaimed.

## 5. Consumer vs. Realtime Consumer

- **`consumer.py` (Passive)**: A client library for "pulling" data. You typically use this _after_ a request finishes to fetch all handles at once. It's good for offline analysis or dataset creation where latency doesn't matter.
- **`realtime_consumer.py` (Active)**: A background worker thread that "pushes" data to your logic. It polls for new handles constantly (every ~10ms) and processes them _during_ generation. This is essential for applications that need to react to the model's internal state in real-time (e.g., steering, monitoring).

## 6. Why HTTP-based Solution Didn't Work

We initially tried a `GET /hidden_states/consume` API endpoint, but it failed for real-time needs because:

1.  **Blocking IO**: HTTP requests are synchronous and slow. Polling an API 50-100 times per second adds significant latency.
2.  **Serialization Overhead**: You cannot send raw GPU memory over HTTP. Tensors would have to be moved to CPU -> serialized to bytes (e.g., JSON/SafeTensors) -> sent over network -> deserialized. This is extremely slow and blocks the generation loop.
3.  **Process Isolation**: An external HTTP client is a separate process and cannot easily access the GPU memory pointer of the vLLM worker without complex setup.

## 7. Ways to Consume Tensors

You have three main options:

1.  **Real-Time In-Process (Recommended)**:
    - **Mechanism**: Customize `src/vllm_hidden_states_extractor/realtime_consumer.py`.
    - **Pros**: Lowest latency (<10ms), zero-copy access to GPU tensors.
    - **Cons**: Runs inside the vLLM process (shared resources).

2.  **Post-Hoc Batch (Standard)**:
    - **Mechanism**: Read `kv_transfer_params` from the final `/v1/completions` response.
    - **Pros**: Simple, no code changes to vLLM, works with standard HTTP clients.
    - **Cons**: High latency (must wait for full completion), high memory usage (buffers all states until end).

3.  **Multi-Process via CUDA IPC**:
    - **Mechanism**: Use `GPUBuffer.get_ipc_handle()` to pass valid IPC handles to another Python process on the same machine.
    - **Pros**: Isolates your secondary model from the vLLM process (safety, separate GIL).
    - **Cons**: High complexity to implement the side-channel for passing handles.

## 8. Example Logs & Explanation

### Server Logs (Real-Time Consumer)

This is what you see in the terminal running `vllm serve`. Notice how the steps are processed sequentially as they happen.

```text
INFO: [RealtimeConsumer] req=cmpl-b7e... step=0 (prefill) consumed shape=[7, 4096] -> accumulated tokens=7
INFO: [RealtimeConsumer] req=cmpl-b7e... step=1 (decode) consumed shape=[1, 4096] -> accumulated tokens=8
INFO: [RealtimeConsumer] req=cmpl-b7e... step=2 (decode) consumed shape=[1, 4096] -> accumulated tokens=9
INFO: [RealtimeConsumer] req=cmpl-b7e... step=3 (decode) consumed shape=[1, 4096] -> accumulated tokens=10
INFO: [RealtimeConsumer] req=cmpl-b7e... step=4 (decode) consumed shape=[1, 4096] -> accumulated tokens=11
INFO: [RealtimeConsumer] req=cmpl-b7e... DONE — Processed 5 steps. Final Stacked Shape: [11, 4096]
```

**What's happening:**

1.  **Step 0**: The prompt (7 tokens) is processed. The consumer grabs the `[7, 4096]` tensor, logs it, and **frees the memory immediately**.
2.  **Steps 1-4**: New tokens are generated one by one. Each `[1, 4096]` tensor is consumed and freed instantly.
3.  **End**: The request finishes. The consumer stacks all accumulating tensors into a final `[11, 4096]` tensor (7 prefill + 4 generated) and logs the result.

### Client Logs (Test Script)

This is what you see in the terminal running `python test_consumer.py`.

```text
  Prompt:     What is the capital of France?
  Max tokens: 5
  ...
  Generated: Additionally, could you explain...
  Handles received: 5

  ⚠ No per-step data in response (older server version?)
  Step   0 ( prefill): handle=eecb9092-337
  Step   1 (  decode): handle=fd47722f-133
  Step   2 (  decode): handle=66edf1e9-289
  Step   3 (  decode): handle=b475444e-d92
  Step   4 (  decode): handle=cfa7e5ac-af0
```

**Explanation:**

- **"No per-step data"**: This is **expected behavior** when using the Real-Time Consumer!
- Because the server consumer already processed and **freed** the tensors (to save GPU memory), the final report cannot inspect them anymore. They are gone.
- The client still receives the list of handles as a trace record, but the actual data payload was handled entirely on the server side.


1. GPU_Buffer - 

Key design:
- Pre-allocated ring buffer to avoid dynamic allocation during inference
- Handle-based access (integer IDs) returned to clients
- TTL-based expiry for unclaimed tensors
- CUDA IPC handle generation for cross-process access

2. Hidden_Activations - 

Architecture:
    1. Plugin patches the model class to register a forward hook on target layer
    2. Hook captures the layer output and stores in GPUBufferManager (stays on GPU)
    3. Connector's request_finished() returns the buffer handle in kv_transfer_params
    4. Consumer process reads the tensor using the handle (via CUDA IPC or co-process API)