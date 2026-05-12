# Production deployment

This document covers the operational concerns for running
`micro-expert-router` as a long-lived service. It complements the
top-level `README.md` (which is focused on the architecture and the
benchmark CLI) with the bits an SRE actually needs: tuning the host,
running it under a process supervisor, dashboarding it, load-testing
it, and scaling it horizontally.

## 1. Host tuning

The hot path is dominated by NVMe reads and L2/L3 cache pressure;
both are sensitive to OS-level knobs that are off by default on most
distros. Apply these as one-shot `sysctl` (`/etc/sysctl.d/99-mer.conf`):

```ini
# Faster dirty-page writeback so streaming writes (logs, metrics) don't
# back up into the page cache the engine relies on for hot experts.
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5

# Allow the engine to mlock its hot-expert + KV-cache buffers.
vm.swappiness = 1

# Larger receive/send buffers improve high-throughput streaming SSE.
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
```

For NVMe specifically (replace `nvme0n1` with your actual device name
— check `lsblk -d -o NAME,ROTA,TYPE` to confirm which device the
expert files live on):

```bash
# `none` scheduler is preferred for NVMe; `mq-deadline` is also fine.
echo none | sudo tee /sys/block/nvme0n1/queue/scheduler

# Drop read-ahead (the engine does its own O_DIRECT reads).
sudo blockdev --setra 0 /dev/nvme0n1
```

If you build with `--features io_uring`, also raise:

```bash
ulimit -l unlimited  # for pinned/registered buffers
ulimit -n 65535      # one fd per expert file when warmup is enabled
```

NUMA-pinning: set `MER_PIN_CORES=N` in the environment to ask the
engine to bind its tokio runtime to the first `N` cores of the
process's allowed CPU set. On a 2-socket machine, prefer running one
instance per NUMA node (with `numactl --cpunodebind=0 --membind=0
…`) over a single process spanning both sockets.

## 2. Running under systemd

```ini
# /etc/systemd/system/micro-expert-router.service
[Unit]
Description=Micro-Expert Router (SSD-streamed MoE)
After=network-online.target

[Service]
Type=simple
User=mer
Group=mer
ExecStart=/usr/local/bin/micro-expert-router serve --config /etc/mer/config.toml
ExecReload=/bin/kill -HUP $MAINPID

# Capacity & filesystem isolation.
LimitNOFILE=65535
LimitMEMLOCK=infinity
ProtectSystem=strict
ReadWritePaths=/var/lib/mer
ReadOnlyPaths=/etc/mer

# Crash policy: never give up.
Restart=always
RestartSec=1

[Install]
WantedBy=multi-user.target
```

`systemctl reload micro-expert-router` will send `SIGHUP`; the engine
re-reads the config and logs which fields would have been applied.
See "Configuration reload" below for which fields require a full
restart.

## 3. Configuration reload (SIGHUP)

The engine installs a `SIGHUP` handler that re-reads the configured
TOML file and validates it. The result is logged structurally:

* **Live-applicable fields** (effective on next request): currently
  `sampling.*` and `server.max_tokens`.
* **Restart-required fields** (logged with `WARN`): `storage.predict_fanout`,
  `real_transformer.batch_timeout_ms`, `storage.predict_min_prob`,
  `storage.partial_load_fraction`. Changing these requires
  `systemctl restart micro-expert-router`.

An invalid file is rejected with a `WARN`; the running config is left
untouched.

## 4. Admin / health endpoints

In addition to the OpenAI-shaped `/v1/completions`,
`/v1/chat/completions`, `/v1/sessions/{id}`, the server exposes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe (200 OK if the process is up). |
| `GET` | `/metrics` | Prometheus text exposition. |
| `GET` | `/v1/admin/health/experts` | Readiness probe: `200` when healthy, `503` when one or more expert reads have failed since startup. |
| `POST` | `/v1/admin/evict` | Trigger a one-shot reclaim of any heap-backed paged-KV overflow blocks. Useful after a transient burst. |

Use `/v1/admin/health/experts` (not just `/health`) for Kubernetes
*readiness* — it short-circuits routing when the SSD has begun to
fail.

## 5. Observability

### Prometheus

All counters live under `mer_*`. Recommended alert seeds:

```yaml
- alert: MERHighFailureRate
  expr: rate(mer_expert_read_failures_total[5m]) > 0
  for: 2m
- alert: MERSlowP99
  expr: histogram_quantile(0.99, rate(mer_request_latency_seconds_bucket[5m])) > 2
  for: 5m
- alert: MERFrequent503s
  expr: rate(mer_requests_total{status="503"}[5m]) > 1
  for: 5m
```

### Grafana dashboard

A starter dashboard JSON lives at `deploy/grafana/dashboard.json`. It
covers: requests/sec by status code, p50/p95/p99 latency, tokens/sec
throughput, cache hit ratio, paged-KV occupancy, SSD read bandwidth,
predictor accuracy. Import it via *Dashboards → Import → upload JSON*.

### Tracing

Every request is assigned a UUID (rendered as 32-char lowercase hex)
and propagated to:

* the `request_id` field of every `tracing` event for that request,
* the `X-Request-Id` response header.

For now, `RUST_LOG=mer=info,tower_http=info` plus a Loki / journald
collector is the recommended path. A future OTLP exporter can be
wired behind a `otel` Cargo feature.

## 6. Security

### API keys

Set `[security] api_keys = ["sk-…", "sk-…"]` in the config to require
either `Authorization: Bearer <key>` or `X-API-Key: <key>` on every
request. An empty list disables the gate. Keys are compared by exact
string match; for stronger guarantees front the engine with an
OAuth-aware proxy.

### Rate limiting

`[security] rate_limit_rps = 0` disables; any positive value enables
an in-process per-key token bucket with `rate_limit_burst` capacity.
Returns `429 Too Many Requests` when exhausted. This is intended as a
last line of defence; do high-RPS shaping at the LB tier.

### TLS

The engine does **not** terminate TLS itself. The recommended
deployment puts an `nginx`, `envoy`, or `caddy` reverse proxy in
front for TLS, OCSP, HTTP/2, and certificate rotation. A bare-bones
`nginx` example:

```nginx
server {
    listen 443 ssl http2;
    server_name mer.example.com;
    ssl_certificate     /etc/letsencrypt/live/mer.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mer.example.com/privkey.pem;

    location / {
        proxy_pass         http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;   # streaming SSE
    }
}
```

### Memory zeroing

`DELETE /v1/sessions/{id}` overwrites every KV-cache buffer with
zeros before releasing it, so a subsequent tenant cannot read a
former tenant's attention state from re-used heap memory. This is
automatic and requires no configuration.

## 7. Admission control

`[server] max_concurrent_requests = N` enables a counting semaphore;
requests beyond `N` are rejected with `503 Service Unavailable`.
`[server] admission_min_free_blocks = M` additionally rejects when
the paged-KV pool has fewer than `M` free blocks left — useful when
each request needs ~K blocks and you want to avoid starting requests
that will OOM the pool mid-stream. Set both to `0` to disable.

## 8. Load testing

`oha` (https://github.com/hatoo/oha):

```bash
oha -c 32 -z 60s --no-tui \
    -H 'Content-Type: application/json' \
    -d '{"model":"mer","prompt":"hello","max_tokens":64}' \
    http://localhost:8080/v1/completions
```

`wrk`:

```bash
wrk -t8 -c64 -d60s -s scripts/wrk_completion.lua http://localhost:8080
```

## 9. Horizontal scaling (distributed serving)

The engine is currently a single-process service. There are two
documented patterns for sharding it:

1. **Replication**: front N identical engines with a round-robin load
   balancer. KV-cache pinning is local to each replica, so set
   session affinity on the LB (haproxy `balance source`, nginx
   `ip_hash`) to keep follow-up requests on the replica that holds
   the cached session.

2. **Expert partitioning** (planned, not yet wired): shard expert ids
   `id % num_nodes`; each node holds its slice of the expert files
   and the gating layer dispatches over RPC. The design is captured
   in `docs/distributed.md` and is intentionally documented ahead of
   the implementation so the surface area is visible to operators.

A starter Helm chart with replication and HPA against
`mer_requests_total` lives at `deploy/helm/mer/`.
