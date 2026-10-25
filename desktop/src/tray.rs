//! NoMusic.exe: runs in the background with a tray icon, removing background music from everything the
//! laptop plays (through VB-Audio Virtual Cable, see live.rs).
//!
//! Right-click menu: on/off, strength, start with Windows, install VB-Cable, quit. Quitting, logging off
//! or shutting down routes audio back to the speakers. Settings live in %LOCALAPPDATA%\NoMusic.

#![windows_subsystem = "windows"]

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use nomusic::live::{self, Controls, Options, Session};
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};
use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{GetLastError, ERROR_ALREADY_EXISTS, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Registry::{
    RegDeleteKeyValueW, RegGetValueW, RegSetKeyValueW, HKEY_CURRENT_USER, REG_SZ, RRF_RT_REG_SZ,
};
use windows::Win32::System::Threading::CreateMutexW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, MessageBoxW, PostQuitMessage,
    RegisterClassW, SetTimer, TranslateMessage, MB_ICONERROR, MB_ICONINFORMATION, MB_OK, MSG,
    WINDOW_EX_STYLE, WM_ENDSESSION, WM_TIMER, WNDCLASSW, WS_OVERLAPPED,
};

const ICON_PNG: &[u8] = include_bytes!("../../icons/icon32.png");
const LEVELS: [u32; 4] = [100, 75, 50, 25];
const RUN_KEY: PCWSTR = w!("Software\\Microsoft\\Windows\\CurrentVersion\\Run");
const RUN_VALUE: PCWSTR = w!("NoMusic");
const VB_CABLE_URL: &str = "https://vb-audio.com/Cable/";
const REFRESH_TIMER_MS: u32 = 3000;

thread_local! {
    // The running filter. Lives on the UI thread so the end-of-session handler can drop it (which puts the
    // speakers back as the default device) before Windows logs off.
    static SESSION: RefCell<Option<Session>> = const { RefCell::new(None) };
}

fn message_box(text: &str, error: bool) {
    let text: Vec<u16> = text.encode_utf16().chain(Some(0)).collect();
    let style = MB_OK | if error { MB_ICONERROR } else { MB_ICONINFORMATION };
    unsafe {
        MessageBoxW(None, PCWSTR(text.as_ptr()), w!("NoMusic"), style);
    }
}

// ── Settings ────────────────────────────────────────────────────────────────

struct Settings {
    enabled: bool,
    level: u32,
}

fn settings_path() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("NoMusic").join("settings.txt"))
}

impl Settings {
    fn load() -> Self {
        let mut s = Settings { enabled: true, level: 100 };
        let text = settings_path().and_then(|p| std::fs::read_to_string(p).ok()).unwrap_or_default();
        for line in text.lines() {
            match line.split_once('=') {
                Some(("enabled", v)) => s.enabled = v.trim() == "1",
                Some(("level", v)) => s.level = v.trim().parse().unwrap_or(100).min(100),
                _ => {}
            }
        }
        s
    }

    fn save(&self) {
        if let Some(p) = settings_path() {
            let _ = std::fs::create_dir_all(p.parent().unwrap());
            let _ = std::fs::write(p, format!("enabled={}\nlevel={}\n", self.enabled as u8, self.level));
        }
    }
}

// ── Start with Windows (HKCU Run key; no admin rights needed) ───────────────

fn exe_path() -> String {
    std::env::current_exe().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()
}

fn autostart_enabled() -> bool {
    let mut buf = [0u16; 1024];
    let mut len = (buf.len() * 2) as u32;
    let ok = unsafe {
        RegGetValueW(HKEY_CURRENT_USER, RUN_KEY, RUN_VALUE, RRF_RT_REG_SZ, None, Some(buf.as_mut_ptr().cast()), Some(&mut len))
    };
    if ok.is_err() {
        return false;
    }
    let value = String::from_utf16_lossy(&buf[..(len as usize / 2).saturating_sub(1)]);
    value.trim_matches('"').eq_ignore_ascii_case(&exe_path())
}

fn set_autostart(on: bool) {
    unsafe {
        if on {
            let value: Vec<u16> = format!("\"{}\"", exe_path()).encode_utf16().chain(Some(0)).collect();
            let _ = RegSetKeyValueW(
                HKEY_CURRENT_USER,
                RUN_KEY,
                RUN_VALUE,
                REG_SZ.0,
                Some(value.as_ptr().cast()),
                (value.len() * 2) as u32,
            );
        } else {
            let _ = RegDeleteKeyValueW(HKEY_CURRENT_USER, RUN_KEY, RUN_VALUE);
        }
    }
}

// ── Icon ────────────────────────────────────────────────────────────────────

/// Decodes the PNG to RGBA; `dim` halves the alpha (the "off" look).
fn icon(dim: bool) -> Option<Icon> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(ICON_PNG));
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let px = &buf[..info.buffer_size()];
    let mut rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => px.to_vec(),
        png::ColorType::Rgb => px.chunks_exact(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => px.chunks_exact(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Grayscale => px.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => return None, // EXPAND turns palettes into RGB(A)
    };
    if dim {
        rgba.chunks_exact_mut(4).for_each(|p| p[3] /= 3);
    }
    Icon::from_rgba(rgba, info.width, info.height).ok()
}

// ── App ─────────────────────────────────────────────────────────────────────

struct App {
    tray: TrayIcon,
    status: MenuItem,
    toggle: CheckMenuItem,
    levels: Vec<CheckMenuItem>,
    autostart: CheckMenuItem,
    install: MenuItem,
    quit: MenuItem,
    controls: Option<std::sync::Arc<Controls>>,
    settings: Settings,
    last_error: Option<String>,
}

impl App {
    fn start(&mut self) {
        if SESSION.with_borrow(Option::is_some) {
            return;
        }
        let controls = Controls::new(self.settings.level);
        let opts = Options { input: None, output: None, rnnoise: true, wet: 1.0, dry_run: false, seconds: None };
        match Session::start(&opts, controls.clone()) {
            Ok(session) => {
                SESSION.with_borrow_mut(|s| *s = Some(session));
                self.controls = Some(controls);
                self.last_error = None;
            }
            Err(e) => self.last_error = Some(e),
        }
    }

    fn stop(&mut self) {
        SESSION.with_borrow_mut(|s| *s = None); // dropping restores the speakers as default device
        self.controls = None;
    }

    /// Brings the menu, tooltip and icon in line with the current state.
    fn refresh(&mut self) {
        let cable = live::cable_installed();
        if self.settings.enabled && cable && SESSION.with_borrow(Option::is_none) && self.last_error.is_none() {
            self.start(); // e.g. VB-Cable was just installed
        }
        let running = SESSION.with_borrow(Option::is_some);
        let text = if running {
            let out = SESSION.with_borrow(|s| s.as_ref().map(|s| s.output_name.clone())).unwrap_or_default();
            format!("NoMusic is on: removing music, playing on {out}")
        } else if !cable {
            "Needs VB-Audio Virtual Cable (free)".to_string()
        } else if let Some(e) = &self.last_error {
            format!("Could not start: {e}")
        } else {
            "NoMusic is off".to_string()
        };
        self.status.set_text(&text);
        self.toggle.set_checked(self.settings.enabled);
        self.install.set_enabled(!cable);
        for (item, level) in self.levels.iter().zip(LEVELS) {
            item.set_checked(level == self.settings.level);
        }
        let _ = self.tray.set_tooltip(Some(if running { "NoMusic: on" } else { "NoMusic: off" }));
        let _ = self.tray.set_icon(icon(!running));
    }

    fn handle(&mut self, event: &MenuEvent) -> bool {
        let id = &event.id;
        if id == self.quit.id() {
            self.stop();
            return false;
        }
        if id == self.toggle.id() {
            self.settings.enabled = !self.settings.enabled;
            self.settings.save();
            self.last_error = None;
            if self.settings.enabled {
                if live::cable_installed() {
                    self.start();
                } else {
                    open_vb_cable_page();
                }
            } else {
                self.stop();
            }
        } else if id == self.autostart.id() {
            set_autostart(!autostart_enabled());
            self.autostart.set_checked(autostart_enabled());
        } else if id == self.install.id() {
            open_vb_cable_page();
        } else if let Some(i) = self.levels.iter().position(|item| item.id() == id) {
            self.settings.level = LEVELS[i];
            self.settings.save();
            if let Some(c) = &self.controls {
                c.level.store(self.settings.level, Ordering::Relaxed);
            }
        }
        self.refresh();
        true
    }
}

fn open_vb_cable_page() {
    let _ = std::process::Command::new("explorer.exe").arg(VB_CABLE_URL).spawn();
    message_box(
        "NoMusic needs the free VB-Audio Virtual Cable to filter games and other apps.\n\n\
         Download it from the page that just opened, run its setup as administrator, and reboot. \
         NoMusic switches on by itself once the cable is installed.",
        false,
    );
}

// A hidden window: receives the periodic refresh timer, and WM_ENDSESSION so that logging off or shutting
// down routes audio back to the speakers (the console handler doesn't apply to a windowed app).
unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if msg == WM_ENDSESSION && wparam.0 != 0 {
        SESSION.with_borrow_mut(|s| *s = None);
        PostQuitMessage(0);
        return LRESULT(0);
    }
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

fn create_hidden_window() -> Option<HWND> {
    unsafe {
        let instance = GetModuleHandleW(None).ok()?;
        let class = WNDCLASSW { lpfnWndProc: Some(wndproc), hInstance: instance.into(), lpszClassName: w!("NoMusicHidden"), ..Default::default() };
        RegisterClassW(&class);
        // A top-level (not message-only) window: only those receive WM_ENDSESSION. It is never shown.
        CreateWindowExW(WINDOW_EX_STYLE(0), w!("NoMusicHidden"), w!("NoMusic"), WS_OVERLAPPED, 0, 0, 0, 0, None, None, Some(instance.into()), None).ok()
    }
}

fn main() {
    // One instance at a time: a second NoMusic would fight over the default device.
    let _mutex = unsafe { CreateMutexW(None, true, w!("Local\\NoMusicTray")) };
    if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        message_box("NoMusic is already running. Look for its icon in the tray, next to the clock.", false);
        return;
    }

    let status = MenuItem::new("Starting...", false, None);
    let toggle = CheckMenuItem::new("Remove music", true, false, None);
    let levels: Vec<CheckMenuItem> =
        LEVELS.iter().map(|l| CheckMenuItem::new(format!("{l}%"), true, false, None)).collect();
    let strength = Submenu::new("Strength", true);
    for item in &levels {
        let _ = strength.append(item);
    }
    let autostart = CheckMenuItem::new("Start with Windows", true, autostart_enabled(), None);
    let install = MenuItem::new("Install VB-Audio Virtual Cable...", true, None);
    let quit = MenuItem::new("Quit NoMusic", true, None);

    let menu = Menu::new();
    let _ = menu.append_items(&[
        &status,
        &PredefinedMenuItem::separator(),
        &toggle,
        &strength,
        &autostart,
        &PredefinedMenuItem::separator(),
        &install,
        &quit,
    ]);

    let tray = match TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("NoMusic")
        .with_icon(icon(true).expect("embedded icon decodes"))
        .build()
    {
        Ok(t) => t,
        Err(e) => {
            message_box(&format!("Could not create the tray icon: {e}"), true);
            return;
        }
    };

    let hidden = create_hidden_window();
    if let Some(hwnd) = hidden {
        unsafe {
            SetTimer(Some(hwnd), 1, REFRESH_TIMER_MS, None);
        }
    }

    let mut app = App {
        tray,
        status,
        toggle,
        levels,
        autostart,
        install,
        quit,
        controls: None,
        settings: Settings::load(),
        last_error: None,
    };
    if app.settings.enabled && live::cable_installed() {
        app.start();
    }
    app.refresh();

    let menu_events = MenuEvent::receiver();
    let mut msg = MSG::default();
    'outer: loop {
        if !unsafe { GetMessageW(&mut msg, None, 0, 0) }.as_bool() {
            break; // WM_QUIT
        }
        unsafe {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        while let Ok(event) = menu_events.try_recv() {
            if !app.handle(&event) {
                break 'outer;
            }
        }
        if msg.message == WM_TIMER {
            app.refresh();
        }
    }
    app.stop();
}
