-- wrk script for /v1/completions. Usage:
--   wrk -t8 -c64 -d60s -s scripts/wrk_completion.lua http://localhost:8080
wrk.method  = "POST"
wrk.path    = "/v1/completions"
wrk.headers["Content-Type"] = "application/json"
wrk.body    = [[{"model":"mer","prompt":"hello world","max_tokens":64}]]
