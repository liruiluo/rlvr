from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, Optional

import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.replica import TokenOutput


@dataclass(slots=True)
class _PendingRequest:
    prompt_ids: list[int]
    sampling_params: dict[str, Any]
    future: asyncio.Future[TokenOutput]


class HFLocalAsyncRollout:
    def __init__(self, *, model, tokenizer, rollout_config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = rollout_config

        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._batch_task: asyncio.Task[None] | None = None
        self._batch_log_once = False
        self._active_batches = 0

        custom = getattr(self.config, "custom", None)
        if OmegaConf.is_config(custom):
            custom = OmegaConf.to_container(custom, resolve=True)
        verl_ext_custom: dict[str, Any] = {}
        if isinstance(custom, dict):
            raw = custom.get("verl_ext")
            if OmegaConf.is_config(raw):
                raw = OmegaConf.to_container(raw, resolve=True)
            if isinstance(raw, dict):
                verl_ext_custom = raw

        self._max_batch_size = int(verl_ext_custom.get("hf_max_batch_size", 8) or 8)
        self._batch_wait_ms = float(verl_ext_custom.get("hf_batch_wait_ms", 2.0) or 2.0)

    async def drain(self) -> None:
        """Wait until all pending batched generations finish.

        This is important when the same (FSDP-wrapped) module is reused for training:
        we must not overlap HF generation with training forward/backward.
        """
        while True:
            if self._queue.empty() and self._active_batches == 0:
                return
            await asyncio.sleep(0.01)

    def _ensure_batch_task(self) -> None:
        if self._batch_task is not None:
            return
        loop = asyncio.get_running_loop()
        self._batch_task = loop.create_task(self._batch_loop())

    def _sampling_key(self, sampling_params: dict[str, Any]) -> tuple[Any, ...]:
        max_new_tokens = sampling_params.get("max_new_tokens", None)
        if max_new_tokens is None:
            max_new_tokens = sampling_params.get("max_tokens", int(getattr(self.config, "response_length", 128)))

        temperature = float(sampling_params.get("temperature", float(getattr(self.config, "temperature", 1.0))))
        top_p = float(sampling_params.get("top_p", float(getattr(self.config, "top_p", 1.0))))
        repetition_penalty = float(
            sampling_params.get("repetition_penalty", float(getattr(self.config, "repetition_penalty", 1.0)))
        )
        ignore_eos = bool(getattr(self.config, "ignore_eos", False))
        return (int(max_new_tokens), temperature, top_p, repetition_penalty, ignore_eos)

    def _pad_left(self, sequences: list[list[int]], *, pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(s) for s in sequences) if sequences else 0
        batch = len(sequences)
        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
        for i, s in enumerate(sequences):
            if not s:
                continue
            n = len(s)
            input_ids[i, -n:] = torch.tensor(s, dtype=torch.long, device=device)
            attention_mask[i, -n:] = 1
        return input_ids, attention_mask

    async def _batch_loop(self) -> None:
        while True:
            first = await self._queue.get()
            batch: list[_PendingRequest] = [first]

            if self._batch_wait_ms > 0:
                await asyncio.sleep(self._batch_wait_ms / 1000.0)

            while len(batch) < self._max_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            # Group by sampling params so we only batch compatible requests.
            groups: dict[tuple[Any, ...], list[_PendingRequest]] = {}
            for req in batch:
                groups.setdefault(self._sampling_key(req.sampling_params), []).append(req)

            for key, reqs in groups.items():
                try:
                    await self._run_batched_generate(key, reqs)
                except Exception as e:
                    for r in reqs:
                        if not r.future.done():
                            r.future.set_exception(e)

    async def _run_batched_generate(self, key: tuple[Any, ...], reqs: list[_PendingRequest]) -> None:
        max_new_tokens, temperature, top_p, repetition_penalty, ignore_eos = key
        do_sample = temperature > 0.0

        param = next(self.model.parameters())
        device = param.device
        autocast_dtype = param.dtype

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        eos_token_id = None if ignore_eos else self.tokenizer.eos_token_id

        input_ids, attention_mask = self._pad_left(
            [r.prompt_ids for r in reqs],
            pad_id=int(pad_token_id),
            device=device,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        generation_config = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            max_new_tokens=int(max_new_tokens),
            num_beams=1,
            repetition_penalty=repetition_penalty,
        )

        if not self._batch_log_once:
            print(
                f"[HFLocalAsyncRollout] batching enabled: max_batch={self._max_batch_size} wait_ms={self._batch_wait_ms}",
                flush=True,
            )
            self._batch_log_once = True

        self.model.eval()
        param_ctx = (
            FSDP.summon_full_params(self.model, writeback=False, recurse=False)
            if isinstance(self.model, FSDP)
            else contextlib.nullcontext()
        )

        self._active_batches += 1
        try:
            with param_ctx, torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
                seq = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    generation_config=generation_config,
                    eos_token_id=eos_token_id,
                    pad_token_id=int(pad_token_id),
                    output_scores=False,
                    return_dict_in_generate=False,
                    use_cache=True,
                )
        finally:
            self._active_batches -= 1

        prompt_len = input_ids.size(1)
        for i, r in enumerate(reqs):
            response = seq[i, prompt_len:].tolist()
            if not r.future.done():
                r.future.set_result(
                    TokenOutput(token_ids=response, log_probs=None, routed_experts=None, stop_reason="completed")
                )

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id
        if image_data is not None or video_data is not None:
            raise NotImplementedError("HFLocalAsyncRollout batching does not support multimodal inputs yet.")

        self._ensure_batch_task()
        fut: asyncio.Future[TokenOutput] = asyncio.get_running_loop().create_future()
        await self._queue.put(_PendingRequest(prompt_ids=prompt_ids, sampling_params=sampling_params, future=fut))
        return await fut
