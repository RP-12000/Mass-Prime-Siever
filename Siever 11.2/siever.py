#!/usr/bin/env python3
"""Siever - Python port of the C++ discrete-log sieve."""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_PATH = "siever.log"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        ],
    )
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)sZ [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
            handler.formatter.converter = time.gmtime


def log_info(message: str) -> None:
    logging.info(message)


def log_fatal(message: str) -> None:
    logging.error(message)
    raise SystemExit(message)


# ---------------------------------------------------------------------------
# Types / config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkItem:
    b: int
    c: int
    P0: int
    P1: int
    N0: int
    N1: int


@dataclass
class PrimeConfig:
    cpu_count: int = 0
    memory_allowed_mb: int = 10240


@dataclass(frozen=True)
class Relation:
    p: int
    M: int
    q: int


@dataclass(frozen=True)
class PrimeChunk:
    low: int
    high: int


def detect_cpu_count() -> int:
    count = os.cpu_count() or 1
    return max(1, count)


def load_prime_config(path: str = "prime.ini") -> PrimeConfig:
    config = PrimeConfig(cpu_count=detect_cpu_count(), memory_allowed_mb=10240)
    has_cpu = False
    has_memory = False
    lines: List[str] = []

    if Path(path).exists():
        lines = Path(path).read_text(encoding="utf-8").splitlines()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key == "CPU_count":
            config.cpu_count = int(value)
            has_cpu = True
        elif key == "MemoryAllowed":
            config.memory_allowed_mb = int(value)
            has_memory = True

    if config.cpu_count <= 0:
        config.cpu_count = detect_cpu_count()
        has_cpu = False
    if config.memory_allowed_mb <= 0:
        config.memory_allowed_mb = 10240
        has_memory = False

    if not has_cpu or not has_memory or not lines:
        Path(path).write_text(
            f"CPU_count={config.cpu_count}\nMemoryAllowed={config.memory_allowed_mb}\n",
            encoding="utf-8",
        )
        log_info(
            f"Wrote prime.ini with CPU_count={config.cpu_count} "
            f"MemoryAllowed={config.memory_allowed_mb} MB"
        )

    return config


def load_worktodo(path: str = "worktodo.txt") -> List[WorkItem]:
    if not Path(path).exists():
        log_fatal(f"Failed to open worktodo file: {path}")

    items: List[WorkItem] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("Factor="):
            line = line[len("Factor=") :]

        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) != 6:
            log_fatal(
                f"worktodo line {line_no} must have 6 comma-separated values (b,c,P0,P1,N0,N1)"
            )

        try:
            item = WorkItem(*(int(part) for part in parts))
        except ValueError as exc:
            log_fatal(f"worktodo line {line_no} parse error: {exc}")

        if item.P0 >= item.P1:
            log_fatal(f"worktodo line {line_no} requires P0 < P1")
        if item.N0 >= item.N1:
            log_fatal(f"worktodo line {line_no} requires N0 < N1")
        if item.N1 - item.N0 > 10_000_000:
            log_fatal(f"worktodo line {line_no} range N1-N0 exceeds 10 million")

        items.append(item)

    if not items:
        log_fatal(f"No work items found in {path}")
    return items


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def mulmod(a: int, b: int, mod: int) -> int:
    return (a * b) % mod


def powmod(base: int, exponent: int, mod: int) -> int:
    if mod == 1:
        return 0
    result = 1
    base %= mod
    while exponent > 0:
        if exponent & 1:
            result = mulmod(result, base, mod)
        base = mulmod(base, base, mod)
        exponent >>= 1
    return result


def mod_inverse(value: int, mod: int) -> int:
    return powmod(value % mod, mod - 2, mod)


def isqrt(n: int) -> int:
    if n <= 0:
        return 0
    return math.isqrt(n)


def isqrt_ceil(n: int) -> int:
    if n <= 0:
        return 0
    root = math.isqrt(n)
    return root if root * root == n else root + 1


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("ceil_div requires positive divisor")
    if a >= 0:
        return (a + b - 1) // b
    return a // b


def floor_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("floor_div requires positive divisor")
    if a >= 0:
        return a // b
    return (a - b + 1) // b


def mod_int(value: int, mod: int) -> int:
    reduced = value % mod
    return reduced if reduced >= 0 else reduced + mod


def mod_neg(value: int, mod: int) -> int:
    positive = mod_int(value, mod)
    return 0 if positive == 0 else mod - positive


def is_prime64(n: int) -> bool:
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    def check(a: int) -> bool:
        d = n - 1
        s = 0
        while d % 2 == 0:
            d >>= 1
            s += 1
        x = powmod(a % n, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(1, s):
            x = mulmod(x, x, n)
            if x == n - 1:
                return True
        return False

    return all(check(a) for a in (2, 3, 5, 7, 11))


def pollard_rho(n: int, rng: random.Random) -> int:
    if n % 2 == 0:
        return 2
    while True:
        c = rng.randint(1, n - 1)
        x = rng.randint(0, n - 1)
        y = x
        d = 1

        def step(v: int) -> int:
            return (mulmod(v, v, n) + c) % n

        while d == 1:
            x = step(x)
            y = step(step(y))
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def factor_integer(n: int, rng: random.Random) -> List[int]:
    stack = [n]
    factors: List[int] = []
    while stack:
        current = stack.pop()
        if current == 1:
            continue
        if is_prime64(current):
            factors.append(current)
            continue
        divisor = pollard_rho(current, rng)
        stack.append(divisor)
        stack.append(current // divisor)
    factors.sort()
    return factors


def multiplicative_order(base: int, prime: int, rng: random.Random) -> int:
    order = prime - 1
    previous = 0
    for factor in factor_integer(prime - 1, rng):
        if factor == previous:
            continue
        previous = factor
        while order % factor == 0 and powmod(base % prime, order // factor, prime) == 1:
            order //= factor
    return order


def run_bsgs(prime: int, item: WorkItem, rng: random.Random) -> Optional[Relation]:
    if prime < 2:
        return None

    b_mod = mod_int(item.b, prime)
    neg_c_mod = mod_neg(item.c, prime)

    if b_mod == 0:
        if neg_c_mod == 0:
            return Relation(prime, item.N0, 1)
        return None

    m = isqrt_ceil(item.N1) if prime > item.N1 else isqrt_ceil(prime)
    if m <= 0:
        return None

    baby: Dict[int, int] = {}
    power = 1
    for i in range(m):
        if power not in baby:
            baby[power] = i
        power = mulmod(power, b_mod, prime)

    step = powmod(b_mod, m, prime)
    step_inverse = mod_inverse(step, prime)

    giant = neg_c_mod
    for j in range(m):
        if giant in baby:
            return Relation(prime, baby[giant] + j * m, multiplicative_order(b_mod, prime, rng))
        giant = mulmod(giant, step_inverse, prime)

    return None


# ---------------------------------------------------------------------------
# Sieve
# ---------------------------------------------------------------------------


def sieve_small_primes(limit: int) -> List[int]:
    if limit < 2:
        return []
    is_prime = bytearray(b"\x01") * (limit + 1)
    is_prime[0] = 0
    if limit >= 1:
        is_prime[1] = 0

    for i in range(2, isqrt(limit) + 1):
        if not is_prime[i]:
            continue
        start = i * i
        is_prime[start : limit + 1 : i] = b"\x00" * (((limit - start) // i) + 1)

    return [i for i in range(2, limit + 1) if is_prime[i]]


def sieve_limit_for(item: WorkItem) -> int:
    return max(isqrt_ceil(item.P1), 2)


def segmented_sieve(low: int, high: int, small_primes: List[int]) -> List[int]:
    if low >= high:
        return []

    segment_size = high - low
    composite = bytearray(segment_size)

    if low <= 1 < high:
        composite[1 - low] = 1

    for prime in small_primes:
        if prime * prime >= high:
            break

        if low <= prime:
            start = prime * 2
        else:
            rem = low % prime
            start = low if rem == 0 else low + (prime - rem)

        value = start
        while value < high:
            composite[value - low] = 1
            value += prime

    primes: List[int] = []
    for offset in range(segment_size):
        if composite[offset]:
            continue
        candidate = low + offset
        if candidate >= 2:
            primes.append(candidate)
    return primes


# ---------------------------------------------------------------------------
# Worker pool / progress
# ---------------------------------------------------------------------------


def worker_chunk_size_bytes(memory_allowed_mb: int, cpu_count: int) -> int:
    total_bytes = memory_allowed_mb * 1024 * 1024
    return max(1, total_bytes // cpu_count)


def iter_prime_chunks(item: WorkItem, chunk_size: int) -> List[PrimeChunk]:
    chunks: List[PrimeChunk] = []
    cursor = item.P0
    while cursor < item.P1:
        step = min(chunk_size, item.P1 - cursor)
        chunks.append(PrimeChunk(cursor, cursor + step))
        cursor += step
    return chunks


class WorkerProgress:
    def __init__(self, total_chunks: int) -> None:
        self.total_chunks = total_chunks
        self.global_chunks = 0
        self.global_relations = 0
        self.start_time = time.monotonic()
        self._lock = threading.Lock()

    def on_chunk_complete(self, relations_in_chunk: int) -> None:
        with self._lock:
            self.global_chunks += 1
            self.global_relations += relations_in_chunk

            percent = 100.0 if self.total_chunks == 0 else 100.0 * self.global_chunks / self.total_chunks
            log_info(
                f"global {self.global_chunks}/{self.total_chunks} chunks ({percent:.1f}%), "
                f"+{relations_in_chunk} relations, {self.global_relations} total relations, "
                f"ETA {self._estimate_finish_time(self.global_chunks)}"
            )

    def _estimate_finish_time(self, completed: int) -> str:
        if completed == 0:
            return "unknown"
        if completed >= self.total_chunks:
            return "complete"

        elapsed = time.monotonic() - self.start_time
        if elapsed <= 0:
            return "unknown"

        remaining = elapsed / completed * (self.total_chunks - completed)
        eta = datetime.now(timezone.utc) + timedelta(seconds=remaining)
        return eta.strftime("%Y-%m-%d %H-%M-%S UTC+0")


def process_chunk(
    chunk: PrimeChunk,
    thread_id: int,
    item: WorkItem,
    small_primes: List[int],
) -> int:
    output_path = f"core_{thread_id}.p"
    relations: List[Relation] = []
    primes = segmented_sieve(chunk.low, chunk.high, small_primes)

    rng = random.Random((thread_id + 1) ^ (chunk.low & ((1 << 64) - 1)))
    for prime in primes:
        relation = run_bsgs(prime, item, rng)
        if relation is not None:
            relations.append(relation)

    if relations:
        with open(output_path, "a", encoding="utf-8") as handle:
            for relation in relations:
                handle.write(f"{relation.p}, {relation.M}, {relation.q}\n")

    return len(relations)


def run_worker_pool(item: WorkItem, cpu_count: int, memory_allowed_mb: int, small_primes: List[int]) -> None:
    chunk_size = worker_chunk_size_bytes(memory_allowed_mb, cpu_count)
    chunks = iter_prime_chunks(item, chunk_size)
    total_chunks = len(chunks)
    log_info(f"Chunk size per task: {chunk_size} integers, {total_chunks} total chunks queued")

    progress = WorkerProgress(total_chunks)
    task_queue: queue.Queue[Optional[PrimeChunk]] = queue.Queue()
    for chunk in chunks:
        task_queue.put(chunk)
    for _ in range(cpu_count):
        task_queue.put(None)

    def worker(thread_id: int) -> None:
        while True:
            chunk = task_queue.get()
            try:
                if chunk is None:
                    return
                relations_found = process_chunk(chunk, thread_id, item, small_primes)
                progress.on_chunk_complete(relations_found)
            finally:
                task_queue.task_done()

    threads = [threading.Thread(target=worker, args=(thread_id,), daemon=True) for thread_id in range(cpu_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    log_info(f"Worker pool complete, {progress.global_relations} total relations")


def format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def count_survivors(candidates: List[List[int]]) -> int:
    return sum(1 for factors in candidates if not factors)


def log_final_status(candidates: List[List[int]], elapsed_seconds: float) -> None:
    total = len(candidates)
    survivors = count_survivors(candidates)
    survival_rate = 0.0 if total == 0 else 100.0 * survivors / total
    log_info(
        f"Final status: {survivors}/{total} candidates survived "
        f"({survival_rate:.2f}% survival rate), elapsed {format_elapsed(elapsed_seconds)}"
    )


def pause_before_exit() -> None:
    try:
        input("\nPress Enter to exit...")
    except EOFError:
        pass


def remove_file_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def merge_core_files(cpu_count: int) -> None:
    with open("relations.p", "wb") as merged:
        for thread_id in range(cpu_count):
            core_path = f"core_{thread_id}.p"
            if not Path(core_path).exists():
                continue
            with open(core_path, "rb") as core:
                merged.write(core.read())
        merged.flush()
        os.fsync(merged.fileno())

    for thread_id in range(cpu_count):
        remove_file_if_exists(f"core_{thread_id}.p")


def load_relations(path: str = "relations.p") -> List[Relation]:
    if not Path(path).exists():
        return []

    relations: List[Relation] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            log_fatal(f"Malformed relation line: {line}")
        relations.append(Relation(int(parts[0]), int(parts[1]), int(parts[2])))
    return relations


def apply_relations(item: WorkItem, relations: List[Relation]) -> List[List[int]]:
    count = item.N1 - item.N0
    candidates: List[List[int]] = [[] for _ in range(count)]

    for relation in relations:
        q = relation.q
        m_value = relation.M
        x0 = ceil_div(item.N0 - m_value, q)
        x1 = floor_div(item.N1 - 1 - m_value, q)

        x = x0
        while x <= x1:
            n = q * x + m_value
            index = n - item.N0
            if 0 <= index < count:
                candidates[index].append(relation.p)
            x += 1

    return candidates


def json_number(value: int):
    if -(2**63) <= value <= 2**63 - 1:
        return value
    return str(value)


def append_results_json(item: WorkItem, candidates: List[List[int]], path: str = "results.json") -> None:
    entry = {
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        "b": json_number(item.b),
        "c": json_number(item.c),
        "P0": json_number(item.P0),
        "P1": json_number(item.P1),
        "N0": json_number(item.N0),
        "N1": json_number(item.N1),
        "prp": [],
        "factored": [],
    }

    for index, factors in enumerate(candidates):
        n = item.N0 + index
        if not factors:
            entry["prp"].append(json_number(n))
        else:
            entry["factored"].append(
                {"n": json_number(n), "factors": [json_number(factor) for factor in factors]}
            )

    all_results: List[dict] = []
    results_path = Path(path)
    if results_path.exists():
        content = results_path.read_text(encoding="utf-8").strip()
        if content:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                log_fatal(f"Failed to parse results.json: {exc}")
            if not isinstance(parsed, list):
                log_fatal("results.json must contain a JSON array")
            all_results = parsed

    all_results.append(entry)
    results_path.write_text(json.dumps(all_results, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> int:
    setup_logging()
    program_start = time.monotonic()
    log_info("Siever starting")

    config = load_prime_config()
    work_items = load_worktodo()

    log_info(f"Loaded {len(work_items)} work item(s)")
    log_info(f"CPU_count={config.cpu_count} MemoryAllowed={config.memory_allowed_mb} MB")

    for work_index, item in enumerate(work_items, start=1):
        log_info(
            f"Processing work item {work_index}/{len(work_items)} b={item.b} c={item.c}"
        )

        for thread_id in range(config.cpu_count):
            remove_file_if_exists(f"core_{thread_id}.p")
        remove_file_if_exists("relations.p")

        sieve_limit = sieve_limit_for(item)
        log_info(f"Sieving small primes up to {sieve_limit}")
        small_primes = sieve_small_primes(sieve_limit)

        run_worker_pool(item, config.cpu_count, config.memory_allowed_mb, small_primes)
        log_info("Worker pool complete, merging core files")
        merge_core_files(config.cpu_count)

        relations = load_relations()
        log_info(f"Loaded {len(relations)} relation(s)")

        candidates = apply_relations(item, relations)
        append_results_json(item, candidates)

        item_elapsed = time.monotonic() - program_start
        log_final_status(candidates, item_elapsed)
        log_info(f"Finished work item {work_index}")

    total_elapsed = time.monotonic() - program_start
    log_info(f"Siever completed successfully, total elapsed {format_elapsed(total_elapsed)}")
    pause_before_exit()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        log_fatal(f"Unhandled exception: {exc}")
