# -*- coding: utf-8 -*-
"""HUD flutuante de legendas, com modo click-through.

Isolamento
----------

O overlay vive no processo principal e nunca toca no áudio, no VAD nem nos
modelos. A única ligação ao resto do pipeline é uma ``multiprocessing.Queue``
de onde lê sem bloquear. Se a engine parar, o overlay continua responsivo; se o
overlay congelar, a engine continua a transcrever.

Ausência de cintilação
----------------------

Três decisões evitam o piscar típico de legendas em tempo real:

1. **Um repaint por frame, não por mensagem.** Um ``QTimer`` esvazia a fila
   inteira a cada 33 ms e aplica apenas o estado mais recente. Vinte revisões
   num intervalo dão na mesma um só repaint.
2. **Escritas idempotentes.** O texto só é atribuído ao widget se tiver mudado.
   Um ``setText`` com o mesmo valor marcaria a região como suja à mesma.
3. **Geometria fixa.** O fundo é desenhado no ``paintEvent`` sobre uma janela
   translúcida de tamanho constante, pelo que o texto a crescer nunca
   redimensiona a janela nem despoleta um novo ciclo de layout.

Click-through no Windows
------------------------

``Qt.WindowType.WindowTransparentForInput`` sozinha não é fiável em janelas
sem moldura e translúcidas no Windows. O que funciona de facto é acrescentar
``WS_EX_TRANSPARENT`` ao estilo estendido da janela nativa. Aplicam-se as duas.
"""

from __future__ import annotations

import ctypes
import html
import logging
import multiprocessing as mp
import queue
import sys
from ctypes import wintypes
from typing import Final

from pipeline.config import PipelineConfig
from pipeline.protocol import Fatal, Level, Status, Transcript, Translation

try:  # pragma: no cover - depende do ambiente
    from PyQt6 import QtCore, QtGui, QtWidgets

    QT_BINDING: Final[str] = "PyQt6"
except ImportError:  # pragma: no cover
    try:
        from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore[no-redef]

        QT_BINDING = "PySide6"
    except ImportError as _exc:  # noqa: N816
        raise ImportError(
            "O overlay precisa do PyQt6 ou do PySide6, e nenhum está instalado.\n"
            "Execute: pip install PyQt6\n"
            "(o resto do pipeline funciona sem eles; use python main.py --check)"
        ) from _exc

__all__ = ["QT_BINDING", "SubtitleOverlay", "run_overlay"]

LOG = logging.getLogger(__name__)

_GWL_EXSTYLE: Final[int] = -20
_WS_EX_TRANSPARENT: Final[int] = 0x0000_0020
_WS_EX_LAYERED: Final[int] = 0x0008_0000
_WS_EX_NOACTIVATE: Final[int] = 0x0800_0000
_WS_EX_TOOLWINDOW: Final[int] = 0x0000_0080


def _set_click_through(window: QtWidgets.QWidget, enabled: bool) -> None:
    """Liga ou desliga a transparência a cliques ao nível do Win32.

    Sem efeito fora do Windows, onde a flag do Qt basta.
    """
    if sys.platform != "win32":
        return
    handle = int(window.winId())
    user32 = ctypes.windll.user32
    getter = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
    setter = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
    getter.restype = ctypes.c_longlong
    getter.argtypes = [wintypes.HWND, ctypes.c_int]
    setter.restype = ctypes.c_longlong
    setter.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]

    style = getter(handle, _GWL_EXSTYLE)
    mask = _WS_EX_TRANSPARENT | _WS_EX_LAYERED | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW
    style = style | mask if enabled else (style & ~_WS_EX_TRANSPARENT) | _WS_EX_LAYERED
    setter(handle, _GWL_EXSTYLE, style)


class SubtitleOverlay(QtWidgets.QWidget):
    """Janela de legendas sempre no topo.

    Args:
        inbox: Fila alimentada pelos processos de engine e tradução.
        config: Configuração completa (línguas e aparência).
        on_close: Chamado quando o utilizador fecha o overlay.
    """

    def __init__(
        self,
        inbox: "mp.Queue[object]",
        config: PipelineConfig,
        *,
        on_close: object | None = None,
    ) -> None:
        super().__init__()
        self._inbox = inbox
        self._config = config
        self._style = config.overlay
        self._on_close = on_close

        self._source_text = ""
        self._target_text = ""
        self._status = "A iniciar..."
        self._latency_ms = 0.0
        self._level = 0.0
        self._drag_origin: QtCore.QPoint | None = None

        self._build()
        self._restore_geometry()
        self.set_click_through(self._style.click_through)

        self._poll = QtCore.QTimer(self)
        self._poll.timeout.connect(self._drain)
        self._poll.start(max(16, self._style.refresh_ms))

        self._idle = QtCore.QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.timeout.connect(self._on_idle)

    # -- construção ------------------------------------------------------

    def _build(self) -> None:
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowTitle("Legenda ao vivo")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 14, 22, 14)
        layout.setSpacing(4)

        self._header = QtWidgets.QLabel(self)
        self._header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header_font = self.font()
        header_font.setPointSize(max(7, self._style.font_size // 3))
        self._header.setFont(header_font)
        layout.addWidget(self._header)

        self._source = QtWidgets.QLabel(self)
        self._source.setWordWrap(True)
        self._source.setTextFormat(QtCore.Qt.TextFormat.RichText)
        source_font = self.font()
        source_font.setPointSize(max(8, int(self._style.font_size * 0.55)))
        self._source.setFont(source_font)
        self._source.setVisible(self._style.show_source)
        layout.addWidget(self._source)

        self._target = QtWidgets.QLabel(self)
        self._target.setWordWrap(True)
        self._target.setTextFormat(QtCore.Qt.TextFormat.RichText)
        target_font = self.font()
        target_font.setPointSize(self._style.font_size)
        target_font.setBold(True)
        self._target.setFont(target_font)
        layout.addWidget(self._target, stretch=1)

        for label in (self._header, self._source, self._target):
            label.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
            )

        self._grip = QtWidgets.QSizeGrip(self)
        grip_row = QtWidgets.QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        grip_row.addWidget(self._grip)
        layout.addLayout(grip_row)

        self.resize(max(420, self._style.width), max(110, self._style.height))
        self.setMinimumSize(360, 96)
        self._render()

    def _restore_geometry(self) -> None:
        if self._style.pos_x is None or self._style.pos_y is None:
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                area = screen.availableGeometry()
                self.move(
                    area.center().x() - self.width() // 2,
                    area.bottom() - self.height() - 80,
                )
            return
        self.move(int(self._style.pos_x), int(self._style.pos_y))

    # -- pintura ---------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - API Qt
        """Desenha o painel arredondado por baixo do texto."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        colour = QtGui.QColor(self._style.bg_color)
        colour.setAlpha(int(max(0, min(100, self._style.opacity)) * 2.55))
        painter.setBrush(colour)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1), 16, 16)
        painter.end()

    def _render(self) -> None:
        """Reescreve as etiquetas, evitando writes redundantes."""
        pair = f"{self._config.source_lang.upper()} → {self._config.target_lang.upper()}"
        metrics = (
            f" · {self._latency_ms:.0f} ms"
            if self._style.show_metrics and self._latency_ms
            else ""
        )
        header = (
            f'<span style="color:{self._style.interim_color}">'
            f"{html.escape(pair)} · {html.escape(self._status)}{metrics}</span>"
        )
        if self._header.text() != header:
            self._header.setText(header)

        if self._style.show_source:
            source = (
                f'<span style="color:{self._style.source_color}">'
                f"{html.escape(self._source_text)}</span>"
            )
            if self._source.text() != source:
                self._source.setText(source)

        target = (
            f'<span style="color:{self._style.text_color}">'
            f"{html.escape(self._target_text)}</span>"
        )
        if self._target.text() != target:
            self._target.setText(target)

    # -- fila ------------------------------------------------------------

    def _drain(self) -> None:
        """Esvazia a fila e aplica apenas o estado final. Nunca bloqueia."""
        latest_translation: Translation | None = None
        latest_transcript: Transcript | None = None
        dirty = False

        while True:
            try:
                message = self._inbox.get_nowait()
            except (queue.Empty, OSError, EOFError):
                break
            if isinstance(message, Translation):
                latest_translation = message
            elif isinstance(message, Transcript):
                latest_transcript = message
            elif isinstance(message, Status):
                self._status = message.text
                dirty = True
            elif isinstance(message, Fatal):
                self._status = f"⚠ {message.text}"
                dirty = True
            elif isinstance(message, Level):
                self._level = message.rms

        if latest_transcript is not None:
            self._source_text = _compose(
                latest_transcript.committed, latest_transcript.interim
            )
            dirty = True
        if latest_translation is not None:
            self._target_text = _compose(
                latest_translation.committed, latest_translation.interim
            )
            self._latency_ms = latest_translation.end_to_end_ms
            dirty = True

        if dirty:
            self._render()
            if self._style.autohide_seconds > 0:
                self._idle.start(int(self._style.autohide_seconds * 1000))

    def _on_idle(self) -> None:
        """Limpa o texto após silêncio prolongado, sem esconder a janela."""
        self._source_text = ""
        self._target_text = ""
        self._latency_ms = 0.0
        self._render()

    # -- interacção ------------------------------------------------------

    def set_click_through(self, enabled: bool) -> None:
        """Alterna a transparência a cliques do rato."""
        self._style.click_through = enabled
        self.setWindowFlag(
            QtCore.Qt.WindowType.WindowTransparentForInput, enabled
        )
        self.show()
        _set_click_through(self, enabled)
        self._grip.setVisible(not enabled)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_origin = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._drag_origin = None
        event.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == QtCore.Qt.Key.Key_Escape:
            self.close()
        elif key == QtCore.Qt.Key.Key_L and (
            event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.set_click_through(not self._style.click_through)
        elif key in (QtCore.Qt.Key.Key_Plus, QtCore.Qt.Key.Key_Equal):
            self._rescale(2)
        elif key == QtCore.Qt.Key.Key_Minus:
            self._rescale(-2)
        else:
            super().keyPressEvent(event)

    def _rescale(self, delta: int) -> None:
        self._style.font_size = max(10, min(72, self._style.font_size + delta))
        font = self._target.font()
        font.setPointSize(self._style.font_size)
        self._target.setFont(font)
        source_font = self._source.font()
        source_font.setPointSize(max(8, int(self._style.font_size * 0.55)))
        self._source.setFont(source_font)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Guarda a geometria e avisa o coordenador."""
        self._style.pos_x = self.x()
        self._style.pos_y = self.y()
        self._style.width = self.width()
        self._style.height = self.height()
        try:
            self._config.save()
        except Exception:  # pragma: no cover
            LOG.exception("não foi possível gravar a configuração do overlay")
        if callable(self._on_close):
            self._on_close()
        event.accept()


def _compose(committed: str, interim: str) -> str:
    """Junta texto confirmado e especulativo numa única linha legível."""
    if committed and interim:
        return f"{committed} {interim}"
    return committed or interim


def run_overlay(
    inbox: "mp.Queue[object]",
    config: PipelineConfig,
    *,
    on_close: object | None = None,
) -> int:
    """Corre o ciclo de eventos do Qt até o overlay fechar."""
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    overlay = SubtitleOverlay(inbox, config, on_close=on_close)
    overlay.show()
    LOG.info("overlay activo (%s)", QT_BINDING)
    return int(application.exec())
