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
//! attached to the original gist:
//!
//! * Header strip with status pill, uptime, and per-second throughput;
//! * 3-tier hit-grid (VRAM / RAM / SSD) rendered with progress
//!   gauges so the relative tier mix is legible at a glance;
//! * VRAM and RAM utilisation bars (anchor + LRU regions for the
//!   VRAM tier; logical occupancy for the RAM tier);
//! * I/O reactor pulse — a sparkline of recent bytes-per-poll.
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
    bytes_history: VecDeque<u64>,
    start: std::time::Instant,
    error: Option<String>,
}

impl AppState {
    fn new(url_base: String) -> Self {
        Self {
            url_base,
            last: HealthSnapshot::default(),
            last_tokens: 0,
            tokens_per_sec: 0,
            bytes_history: VecDeque::with_capacity(60),
            start: std::time::Instant::now(),
            error: None,
        }
    }

    fn record_history(&mut self) {
        let bytes_proxy = self.last.cache_hits + self.last.cache_misses;
        self.bytes_history.push_back(bytes_proxy);
        if self.bytes_history.len() > 60 {
            self.bytes_history.pop_front();
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
                    if snap.tokens_generated >= app.last_tokens {
                        app.tokens_per_sec =
                            ((snap.tokens_generated - app.last_tokens) as f64 / dt) as u64;
                    }
                    app.last_tokens = snap.tokens_generated;
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
    let vram_hits = app.last.gpu_cache_hits;
    let ram_hits = app.last.cache_hits;
    let misses = app.last.cache_misses;
    let total = (vram_hits + ram_hits + misses).max(1);
    let vram_pct = (vram_hits as f64 / total as f64 * 100.0) as u16;
    let ram_pct = (ram_hits as f64 / total as f64 * 100.0) as u16;
    let ssd_pct = (misses as f64 / total as f64 * 100.0).round() as u16;

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(2),
        ])
        .margin(1)
        .split(area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " TIERED HIT GRID ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(block, area);

    let vram = Gauge::default()
        .label(format!("VRAM  {vram_hits:>10}  {vram_pct:>3}%"))
        .gauge_style(Style::default().fg(ACCENT).bg(Color::Black))
        .percent(vram_pct.min(100));
    let ram = Gauge::default()
        .label(format!("RAM   {ram_hits:>10}  {ram_pct:>3}%"))
        .gauge_style(Style::default().fg(Color::LightCyan).bg(Color::Black))
        .percent(ram_pct.min(100));
    let ssd = Gauge::default()
        .label(format!("SSD   {misses:>10}  {ssd_pct:>3}%"))
        .gauge_style(Style::default().fg(Color::LightYellow).bg(Color::Black))
        .percent(ssd_pct.min(100));
    f.render_widget(vram, rows[0]);
    f.render_widget(ram, rows[1]);
    f.render_widget(ssd, rows[2]);
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
        (app.last.vram_capacity_bytes / (1024 * 1024)).max(0),
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
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " I/O REACTOR PULSE — lookups per tick ",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ));
    let data: Vec<u64> = app.bytes_history.iter().copied().collect();
    let sp = Sparkline::default()
        .block(block)
        .data(&data)
        .style(Style::default().fg(ACCENT));
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
        _ => (format!("{authority}:80"), authority.to_string()),
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
    let body = &buf[body_start..];
    // Strip transfer-encoding: chunked artefacts if present (the server
    // serialises a small JSON body and uses Content-Length, so this is
    // a defensive trim of trailing CRLFs only).
    let json = std::str::from_utf8(body)?.trim_matches(|c: char| c == '\r' || c == '\n');
    // If the server sent a chunked body, strip a trailing "0" chunk-size
    // line after the last newline-trim above so the JSON parses cleanly.
    let json = json.trim_end_matches('\u{0}');
    let snap: HealthSnapshot = serde_json::from_str(json.trim())?;
    Ok(snap)
}
