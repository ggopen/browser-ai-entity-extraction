// Lightweight diagnostic logger used across the app.
const MAX_LINES = 200;

class DiagLogger {
  constructor() {
    this.body = null;
    this.header = null;
    this.panel = null;
    this.lines = [];
  }

  attach(bodyEl, headerEl, panelEl) {
    this.body = bodyEl;
    this.header = headerEl;
    this.panel = panelEl;
    this.header.addEventListener('click', () => {
      this.header.classList.toggle('collapsed');
      this.panel.classList.toggle('expanded');
    });
  }

  show() {
    if (this.panel) {
      this.panel.classList.add('expanded');
      this.header?.classList.remove('collapsed');
    }
  }

  _write(level, msg) {
    const now = new Date();
    const t = now.toTimeString().slice(0, 8);
    const line = { level, time: t, msg };
    this.lines.push(line);
    if (this.lines.length > MAX_LINES) this.lines.shift();
    if (this.body) {
      const el = document.createElement('div');
      el.className = `diag-line ${level}`;
      el.textContent = `${t} ${msg}`;
      this.body.appendChild(el);
      // Trim DOM
      while (this.body.children.length > MAX_LINES) {
        this.body.removeChild(this.body.firstChild);
      }
      this.body.scrollTop = this.body.scrollHeight;
    }
    if ((level === 'err' || level === 'warn') && this.panel) {
      // Auto-show panel for errors/warnings
      this.show();
    }
    // Also log to console
    const fn = level === 'err' ? console.error : level === 'warn' ? console.warn : console.log;
    fn(`[${t}] ${msg}`);
  }

  info(msg)  { this._write('info', msg); }
  ok(msg)    { this._write('ok', msg); }
  warn(msg)  { this._write('warn', msg); }
  err(msg)   { this._write('err', msg); }
  time(msg)  { this._write('time', msg); }
}

export const logger = new DiagLogger();
