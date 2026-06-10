//! Native terminal dashboard for `mer-cli monitor` — Phase 4 of the
//! three-tier (SSD → RAM → VRAM) memory hierarchy.
//!
//! This module is compiled only when the `tui` cargo feature is on
//! (the default), in which case it pulls in `ratatui` + `crossterm`
//! and exposes [`run_monitor`] for `main.rs`'s subcommand dispatch.
//! The dashboard is a pure consumer of the running `serve` instance:
//! it polls `GET /v1/admin/health/experts` and `GET /metrics` at a
//! configurable interval and renders the result as a live ratatui
//! view. It never mutates engine state.
//!
//! Visual language is borrowed from the "Amalgafy" reference image
//! attached to the original gist — high-density, monochromatic
//! tech-noir, crisp single-pixel borders, high-contrast status bars:
//!
//! * Header strip with status pill, uptime, and per-second throughput
//!   (with restart-recovery: TPS resets to zero on a backwards jump
//!   of `tokens_generated`);
//! * 3-tier hit-grid (VRAM / RAM / SSD) rendered as three side-by-side
//!   sparklines plotting the **delta** of each tier's hit counter
//!   per refresh tick — i.e. pulse/load, not a cumulative staircase;
//! * VRAM and RAM utilisation bars (anchor + LRU regions for the
//!   VRAM tier; logical occupancy for the RAM tier);
//! * I/O reactor pulse — a sparkline of the per-tick miss delta
//!   (i.e. SSD reads required this tick), surfacing backpressure
//!   and stall on the inference critical path.
//!
//! The HTTP fetch path is dependency-free: we hand-roll a minimal
//! HTTP/1.1 GET over `tokio::net::TcpStream` so the dashboard does
//! not pull in `reqwest` (which would inflate the build noticeably
//! for a single endpoint).

use std::collections::VecDeque;
use std::io;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline},
    Terminal,
};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Subset of the `/v1/admin/health/experts` response we render in the
/// dashboard. Mirrors the `ExpertHealth` struct in `server.rs`; extra
/// fields are silently ignored thanks to serde's default behaviour.
#[derive(Debug, Default, Deserialize)]
struct HealthSnapshot {
    status: String,
    cache_hits: u64,
    cache_misses: u64,
    cache_pinned: usize,
    cache_capacity: usize,
    tokens_generated: u64,
    #[serde(default)]
    gpu_cache_enabled: bool,
    #[serde(default)]
    gpu_cache_hits: u64,
    #[serde(default)]
    gpu_cache_misses: u64,
    #[serde(default)]
    gpu_promotions_total: u64,
    #[serde(default)]
    vram_used_bytes: u64,
    #[serde(default)]
    vram_capacity_bytes: u64,
    #[serde(default)]
    gpu_anchor_count: usize,
    #[serde(default)]
    gpu_lru_count: usize,
}

struct AppState {
    url_base: String,
    last: HealthSnapshot,
    last_tokens: u64,
    tokens_per_sec: u64,
    /// Per-tier hit *delta* history (cap 60). Each push is
    /// `current_counter - prev_counter` so the sparkline shows the
    /// pulse/load per refresh tick rather than the cumulative
    /// staircase the previous revision rendered. Phase 1 telemetry
    /// bug fix.
    vram_hits_history: VecDeque<u64>,
    ram_hits_history: VecDeque<u64>,
    ssd_hits_history: VecDeque<u64>,
    /// Snapshots of the cumulative counters captured on the previous
    /// poll, used to compute the delta. Initialised lazily — on the
    /// first poll we record the *current* values without pushing a
    /// data point so the first frame doesn't render a spurious spike
    /// equal to the cumulative-since-startup total.
    prev_vram_hits: Option<u64>,
    prev_ram_hits: Option<u64>,
    prev_ssd_hits: Option<u64>,
    start: std::time::Instant,
    error: Option<String>,
}

/// Maximum number of points kept in each rolling sparkline buffer.
/// Caps memory growth (gist "Telemetry" constraint: "Ensure the
/// sparkline history is capped (e.g., 60 points) to prevent
/// memory growth").
const HISTORY_CAP: usize = 60;

impl AppState {
    fn new(url_base: String) -> Self {
        Self {
            url_base,
            last: HealthSnapshot::default(),
            last_tokens: 0,
            tokens_per_sec: 0,
            vram_hits_history: VecDeque::with_capacity(HISTORY_CAP),
            ram_hits_history: VecDeque::with_capacity(HISTORY_CAP),
            ssd_hits_history: VecDeque::with_capacity(HISTORY_CAP),
            prev_vram_hits: None,
            prev_ram_hits: None,
            prev_ssd_hits: None,
            start: std::time::Instant::now(),
            error: None,
        }
    }

    /// Record one polled sample into the rolling histories. Stores the
    /// **delta** (current − prev) for each tier so the sparkline
    /// represents per-tick load, not a cumulative staircase
    /// (Phase 1 bug fix). On a counter regression — which happens on
    /// server restart, since the engine resets its counters to zero —
    /// the delta is treated as zero and the previous snapshot is
    /// rebased to the new value so subsequent ticks resume reading
    /// real load instead of a single huge negative spike.
    fn record_history(&mut self) {
        push_delta(
            &mut self.vram_hits_history,
            &mut self.prev_vram_hits,
            self.last.gpu_cache_hits,
        );
        push_delta(
            &mut self.ram_hits_history,
            &mut self.prev_ram_hits,
            self.last.cache_hits,
        );
        // SSD tier activity is driven by a RAM miss, so this history
        // stores the RAM-miss / SSD-read delta from `cache_misses`,
        // not a literal SSD-hit counter. The reactor stall pane in
        // `draw_pulse` reads from this same history at render time,
        // so we don't keep a second buffer of identical values.
        let ssd_misses = self.last.cache_misses;
        push_delta(
            &mut self.ssd_hits_history,
            &mut self.prev_ssd_hits,
            ssd_misses,
        );
    }
}

/// Push `current - prev` onto `buf`, rebase `prev` to `current`, and
/// cap `buf` at [`HISTORY_CAP`] entries. On the very first call
/// (`prev` is `None`) the delta is **not** pushed; we only seed
/// `prev` so the next tick's delta is a true per-interval pulse
/// rather than the cumulative since startup.
fn push_delta(buf: &mut VecDeque<u64>, prev: &mut Option<u64>, current: u64) {
    match *prev {
        None => {
            *prev = Some(current);
        }
        Some(p) => {
            // Counter regression (e.g. server restart): rebase prev
            // and emit a zero this tick so the sparkline doesn't
            // pretend the negative jump was real load.
            let delta = current.saturating_sub(p);
            // On a regression `current < p` ⇒ delta == 0 via
            // saturating_sub; rebase prev to current so the next
            // delta is computed against the new baseline.
            buf.push_back(delta);
            if buf.len() > HISTORY_CAP {
                buf.pop_front();
            }
            *prev = Some(current);
        }
    }
}

/// Entry point invoked from `main.rs` for `mer-cli monitor`.
///
/// Sets up the alternate-screen terminal, polls the configured
/// endpoint every `refresh_ms`, and renders the dashboard until the
/// user presses `q` / `Esc` / `Ctrl-C`. Always restores the terminal
/// even on error.
pub async fn run_monitor(url_base: &str, refresh_ms: u64) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut app = AppState::new(url_base.trim_end_matches('/').to_string());
    let result = event_loop(&mut terminal, &mut app, refresh_ms).await;
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut AppState,
    refresh_ms: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let interval = Duration::from_millis(refresh_ms.max(50));
    let mut last_tick = std::time::Instant::now();
    loop {
        // Drain pending key events without blocking the render loop.
        if event::poll(Duration::from_millis(16))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    match k.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char('c')
                            if k.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) =>
                        {
                            return Ok(())
                        }
                        _ => {}
                    }
                }
            }
        }
        if last_tick.elapsed() >= interval {
            last_tick = std::time::Instant::now();
            match fetch_health(&app.url_base).await {
                Ok(snap) => {
                    let dt = interval.as_secs_f64().max(0.001);
                    // TPS guard with restart recovery (gist Phase 1
                    // bug fix). When `tokens_generated` jumps
                    // **backwards** the server has been restarted —
                    // the engine resets its counter to zero on boot.
                    // Treat that as a fresh epoch: zero the rate and
                    // rebase `last_tokens` to the new snapshot so the
                    // next tick computes a real delta instead of a
                    // negative one (which would wrap around as a
                    // u64 underflow).
                    if snap.tokens_generated < app.last_tokens {
                        app.tokens_per_sec = 0;
                        app.last_tokens = snap.tokens_generated;
                    } else {
                        app.tokens_per_sec =
                            ((snap.tokens_generated - app.last_tokens) as f64 / dt) as u64;
                        app.last_tokens = snap.tokens_generated;
                    }
                    app.last = snap;
                    app.record_history();
                    app.error = None;
                }
                Err(e) => {
                    app.error = Some(e.to_string());
                }
            }
        }
        terminal.draw(|f| draw(f, app))?;
    }
}

fn draw(f: &mut ratatui::Frame, app: &AppState) {
    let area = f.area();
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(8),
            Constraint::Length(7),
            Constraint::Min(5),
        ])
        .split(area);

    draw_header(f, outer[0], app);
    draw_tiers(f, outer[1], app);
    draw_vram(f, outer[2], app);
    draw_pulse(f, outer[3], app);
}

const ACCENT: Color = Color::LightGreen;
const DIM: Color = Color::DarkGray;

fn draw_header(f: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let uptime = app.start.elapsed().as_secs();
    let status_color = match app.last.status.as_str() {
        "ok" => ACCENT,
        "" => DIM,
        _ => Color::LightRed,
    };
    let status = if app.last.status.is_empty() {
        "PENDING".to_string()
    } else {
        app.last.status.to_uppercase()
    };
    let mut spans = vec![
        Span::styled(
            "MICRO-EXPERT ROUTER · 3-TIER TELEMETRY",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::raw("   "),
        Span::styled(format!("[{status}]"), Style::default().fg(status_color)),
        Span::raw("   "),
        Span::styled(
            format!("UPTIME {uptime}s"),
            Style::default().fg(Color::White),
        ),
        Span::raw("   "),
        Span::styled(
            format!("{} TPS", app.tokens_per_sec),
            Style::default().fg(Color::White),
        ),
        Span::raw("   "),
        Span::styled(format!("{}", app.url_base), Style::default().fg(DIM)),
    ];
    if let Some(err) = app.error.as_ref() {
        spans.push(Span::raw("   "));
        spans.push(Span::styled(
            format!("ERR: {err}"),
            Style::default().fg(Color::LightRed),
        ));
    }
    let p = Paragraph::new(Line::from(spans)).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(ACCENT)),
    );
    f.render_widget(p, area);
}

fn draw_tiers(f: &mut ratatui::Frame, area: Rect, app: &AppState) {
    // Cumulative counters (header labels) — but the body of the
    // panel renders **delta** sparklines below (gist Phase 1).
    let vram_hits = app.last.gpu_cache_hits;
    let ram_hits = app.last.cache_hits;
    let ssd_hits = app.last.cache_misses;

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " 3-TIER HIT GRID · Δ per tick ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(block, area);

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .margin(1)
        .split(area);

    draw_tier_sparkline(
        f,
        columns[0],
        "VRAM",
        vram_hits,
        &app.vram_hits_history,
        ACCENT,
    );
    draw_tier_sparkline(
        f,
        columns[1],
        "RAM ",
        ram_hits,
        &app.ram_hits_history,
        Color::LightCyan,
    );
    draw_tier_sparkline(
        f,
        columns[2],
        "SSD ",
        ssd_hits,
        &app.ssd_hits_history,
        Color::LightYellow,
    );
}

/// Render one tier column: a label line ("VRAM 12345 Δ7") above a
/// sparkline drawn from the supplied delta history. Each panel owns
/// its own border so the three tiers visually segment cleanly even
/// in narrow terminals.
fn draw_tier_sparkline(
    f: &mut ratatui::Frame,
    area: Rect,
    label: &str,
    cumulative: u64,
    history: &VecDeque<u64>,
    fg: Color,
) {
    let last_delta = history.back().copied().unwrap_or(0);
    let title = format!(" {label} · total {cumulative} · Δ {last_delta} ");
    let data: Vec<u64> = history.iter().copied().collect();
    let sp = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(DIM))
                .title(Span::styled(
                    title,
                    Style::default().fg(fg).add_modifier(Modifier::BOLD),
                )),
        )
        .data(&data)
        .style(Style::default().fg(fg));
    f.render_widget(sp, area);
}

fn draw_vram(f: &mut ratatui::Frame, area: Rect, app: &AppState) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " VRAM UTILISATION ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(block, area);
    let inner = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    };
    let vram_pct = if app.last.vram_capacity_bytes == 0 {
        0
    } else {
        ((app.last.vram_used_bytes as f64 / app.last.vram_capacity_bytes as f64) * 100.0) as u16
    };
    let ram_pct = if app.last.cache_capacity == 0 {
        0
    } else {
        ((app.last.cache_pinned as f64 / app.last.cache_capacity as f64) * 100.0) as u16
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Length(2), Constraint::Length(1)])
        .split(inner);

    let vram_label = format!(
        "VRAM  {} / {} MiB   anchor={}  lru={}  promotions={}",
        app.last.vram_used_bytes / (1024 * 1024),
        app.last.vram_capacity_bytes / (1024 * 1024),
        app.last.gpu_anchor_count,
        app.last.gpu_lru_count,
        app.last.gpu_promotions_total
    );
    let vram = Gauge::default()
        .label(vram_label)
        .gauge_style(Style::default().fg(ACCENT).bg(Color::Black))
        .percent(vram_pct.min(100));
    let ram_label = format!(
        "RAM   pinned {} / {} slots",
        app.last.cache_pinned, app.last.cache_capacity
    );
    let ram = Gauge::default()
        .label(ram_label)
        .gauge_style(Style::default().fg(Color::LightCyan).bg(Color::Black))
        .percent(ram_pct.min(100));

    f.render_widget(vram, rows[0]);
    f.render_widget(ram, rows[1]);
    let footer = if app.last.gpu_cache_enabled {
        "[gpu_cache] ACTIVE — RAM→VRAM promotions on hit threshold"
    } else {
        "[gpu_cache] DISABLED — running 2-tier SSD→RAM"
    };
    f.render_widget(
        Paragraph::new(Span::styled(footer, Style::default().fg(DIM))),
        rows[2],
    );
}

fn draw_pulse(f: &mut ratatui::Frame, area: Rect, app: &AppState) {
    // Reactor pulse — per-tick SSD-miss delta sourced from
    // `ssd_hits_history` (a RAM miss == an SSD read). A flat-zero
    // line means every routed expert was served out of RAM/VRAM
    // (no I/O stall); tall bars are direct evidence of backpressure
    // on the inference critical path.
    let last_delta = app.ssd_hits_history.back().copied().unwrap_or(0);
    let title = format!(" I/O REACTOR · stall pulse · last Δ {last_delta} ");
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            title,
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let data: Vec<u64> = app.ssd_hits_history.iter().copied().collect();
    let sp = Sparkline::default()
        .block(block)
        .data(&data)
        .style(Style::default().fg(Color::LightYellow));
    f.render_widget(sp, area);
}

fn parse_http_base(url_base: &str) -> Result<(String, String, String), Box<dyn std::error::Error>> {
    let trimmed = url_base.trim();
    let without_scheme = if let Some(rest) = trimmed.strip_prefix("http://") {
        rest
    } else if trimmed.starts_with("https://") {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "https:// is not supported by fetch_health without TLS support",
        )));
    } else {
        trimmed
    };

    let (authority, base_path) = match without_scheme.split_once('/') {
        Some((auth, path)) => (auth, format!("/{}", path.trim_start_matches('/'))),
        None => (without_scheme, String::new()),
    };

    if authority.is_empty() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "missing host in url_base",
        )));
    }

    let (connect_addr, host_header) = match authority.rsplit_once(':') {
        Some((host, port_str)) if !host.is_empty() && port_str.parse::<u16>().is_ok() => (
            format!("{host}:{port_str}"),
            authority.to_string(),
        ),
        _ => (format!("{authority}:8080"), authority.to_string()),
    };

    let request_path = format!(
        "{}/v1/admin/health/experts",
        base_path.trim_end_matches('/')
    );

    Ok((connect_addr, host_header, request_path))
}

/// Minimal HTTP/1.1 GET over `tokio::net::TcpStream`. Avoids pulling
/// in a full HTTP client crate for a single-endpoint poll. Parses an
/// `application/json` body and decodes it through serde.
async fn fetch_health(url_base: &str) -> Result<HealthSnapshot, Box<dyn std::error::Error>> {
    let (connect_addr, host_header, request_path) = parse_http_base(url_base)?;
    let request = format!(
        "GET {request_path} HTTP/1.1\r\nHost: {host_header}\r\nConnection: close\r\n\r\n"
    );
    let mut socket = TcpStream::connect(&connect_addr).await?;
    socket.write_all(request.as_bytes()).await?;
    let mut buf = Vec::with_capacity(4096);
    socket.read_to_end(&mut buf).await?;
    // Find header / body boundary.
    let needle = b"\r\n\r\n";
    let body_start = buf
        .windows(needle.len())
        .position(|w| w == needle)
        .ok_or("malformed HTTP response (no header terminator)")?
        + needle.len();
    let head = &buf[..body_start - needle.len()];
    let body = &buf[body_start..];

    // Validate status line — only 2xx responses carry a meaningful JSON
    // body for this endpoint; anything else is surfaced as an error so
    // the dashboard does not attempt to parse an HTML error page.
    let head_str = std::str::from_utf8(head)
        .map_err(|_| "malformed HTTP response (non-utf8 header)")?;
    let mut header_lines = head_str.split("\r\n");
    let status_line = header_lines
        .next()
        .ok_or("malformed HTTP response (missing status line)")?;
    let mut status_parts = status_line.splitn(3, ' ');
    let _http_version = status_parts.next().unwrap_or("");
    let status_code: u16 = status_parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or("malformed HTTP response (missing status code)")?;
    if !(200..300).contains(&status_code) {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::Other,
            format!("health endpoint returned HTTP {status_code}"),
        )));
    }

    // Parse headers for Content-Length and Transfer-Encoding so we can
    // pick the correct framing for the response body.
    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in header_lines {
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim();
            let value = value.trim();
            if name.eq_ignore_ascii_case("content-length") {
                content_length = value.parse().ok();
            } else if name.eq_ignore_ascii_case("transfer-encoding")
                && value.eq_ignore_ascii_case("chunked")
            {
                chunked = true;
            }
        }
    }

    let payload: Vec<u8> = if chunked {
        decode_chunked(body)?
    } else if let Some(len) = content_length {
        if body.len() < len {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "HTTP body shorter than Content-Length",
            )));
        }
        body[..len].to_vec()
    } else {
        body.to_vec()
    };

    let snap: HealthSnapshot = serde_json::from_slice(&payload)?;
    Ok(snap)
}

/// Minimal RFC 7230 `Transfer-Encoding: chunked` decoder. Reads
/// `<hex-size>\r\n<bytes>\r\n` frames until a zero-sized chunk and
/// returns the concatenated payload. Trailer headers (if any) are
/// ignored.
fn decode_chunked(mut body: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut out = Vec::with_capacity(body.len());
    loop {
        let line_end = body
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or("malformed chunked body (missing chunk size CRLF)")?;
        let size_line = std::str::from_utf8(&body[..line_end])?;
        // Chunk extensions (after `;`) are ignored per RFC 7230 §4.1.1.
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| "malformed chunked body (invalid chunk size)")?;
        body = &body[line_end + 2..];
        if size == 0 {
            break;
        }
        if body.len() < size + 2 {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "chunked body shorter than declared chunk size",
            )));
        }
        out.extend_from_slice(&body[..size]);
        if &body[size..size + 2] != b"\r\n" {
            return Err("malformed chunked body (missing chunk trailer CRLF)".into());
        }
        body = &body[size + 2..];
    }
    Ok(out)
}
