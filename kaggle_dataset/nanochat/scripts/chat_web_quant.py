#!/usr/bin/env python3
"""
Web chat server for quantized nanochat artifacts.

Serves both UI and API from a single FastAPI instance, similar to chat_web.py,
but loads quantized artifacts produced by:
- scripts.chat_quantize.py
- scripts.chat_quant_awq.py

Supports:
- int8/fp16-style quantized exports reconstructed via chat_quant_eval.py
- AWQ-style int4 exports reconstructed via chat_quant_eval.py

Launch examples:

- single GPU or CPU
python -m scripts.chat_web_quant --quant-artifact ~/.cache/nanochat/chatquant_exports/d12_int8_linear/quant_000100.pt

- AWQ artifact
python -m scripts.chat_web_quant --quant-artifact ~/.cache/nanochat/chatquant_exports/d12_awq_int4/quant_000100.pt
"""

import argparse
import asyncio
import json
import logging
import os
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine
from scripts.chat_quant_eval import load_quantized_model

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description="NanoChat Quantized Web Server")
parser.add_argument("--quant-artifact", type=str, required=True, help="Path to a quantized artifact produced by chat_quantize.py or chat_quant_awq.py")
parser.add_argument("-n", "--num-gpus", type=int, default=1, help="Number of GPUs to use (default: 1)")
parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Default temperature for generation")
parser.add_argument("-k", "--top-k", type=int, default=50, help="Default top-k sampling parameter")
parser.add_argument("-m", "--max-tokens", type=int, default=512, help="Default max tokens for generation")
parser.add_argument('-p', '--port', type=int, default=8001, help='Port to run the server on')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)


@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    quant_method: str


class WorkerPool:
    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            num_gpus = torch.cuda.device_count() if device_type == "cuda" else 1
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, quant_artifact: str):
        print(f"Initializing quantized worker pool with {self.num_gpus} workers...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."
        quant_artifact = os.path.abspath(os.path.expanduser(quant_artifact))

        for gpu_id in range(self.num_gpus):
            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading quantized artifact on GPU {gpu_id}...")
            else:
                device = torch.device(device_type)
                print(f"Loading quantized artifact on {device_type}...")

            model, tokenizer, artifact = load_quantized_model(quant_artifact, device)
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=Engine(model, tokenizer),
                tokenizer=tokenizer,
                quant_method=artifact.get("quant_method", "unknown"),
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} quantized workers initialized!")

    async def acquire_worker(self) -> Worker:
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        await self.available_workers.put(worker)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None


def validate_chat_request(request: ChatRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request")

    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")
        if len(message.content) > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message")
        total_length += len(message.content)
    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed")

    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant", "system"]:
            raise HTTPException(status_code=400, detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'")

    expected_role = "user"
    for i, message in enumerate(request.messages):
        if i == 0 and message.role == "system":
            continue
        if message.role != expected_role:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Message {i} has role '{message.role}' but expected '{expected_role}'. "
                    "Messages must alternate user/assistant, with an optional leading system message."
                )
            )
        expected_role = "user" if message.role == "assistant" else "assistant"

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Conversation must end with a user message so the assistant can reply")

    if request.temperature is not None and not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
        raise HTTPException(status_code=400, detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
    if request.top_k is not None and not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
        raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")
    if request.max_tokens is not None and not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
        raise HTTPException(status_code=400, detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading quantized nanochat model(s)...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.quant_artifact)
    print(f"Quantized server ready at http://localhost:{args.port}")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    return FileResponse(os.path.join("nanochat", "logo.svg"), media_type="image/svg+xml")


async def generate_stream(worker: Worker, tokens, temperature=None, max_new_tokens=None, top_k=None) -> AsyncGenerator[str, None]:
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()
    accumulated_tokens = []
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
    ):
        token = token_column[0]
        if token == assistant_end or token == bos:
            break
        accumulated_tokens.append(token)
        current_text = worker.tokenizer.decode(accumulated_tokens)
        if not current_text.endswith('�'):
            new_text = current_text[len(last_clean_text):]
            if new_text:
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    validate_chat_request(request)
    logger.info("=" * 20)
    for message in request.messages:
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-" * 20)

    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()
    try:
        completion_conversation = {
            "messages": [{"role": m.role, "content": m.content} for m in request.messages] +
                        [{"role": "assistant", "content": ""}]
        }
        conversation_tokens = worker.tokenizer.render_for_completion(completion_conversation)

        response_tokens = []

        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                ):
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}, {worker.quant_method}): {''.join(response_tokens)}")
                logger.info("=" * 20)
                await worker_pool.release_worker(worker)

        return StreamingResponse(stream_and_release(), media_type="text/event-stream")
    except Exception:
        await worker_pool.release_worker(worker)
        raise


@app.get("/health")
async def health():
    worker_pool = getattr(app.state, "worker_pool", None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0,
    }


@app.get("/stats")
async def stats():
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device),
                "quant_method": w.quant_method,
            }
            for w in worker_pool.workers
        ],
    }


if __name__ == "__main__":
    import uvicorn

    print("Starting NanoChat Quantized Web Server")
    print(f"Quant artifact: {args.quant_artifact}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
