# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QColor, QFont, QCursor

class FloatingOverlayWindow(QWidget):
    """
    Janela flutuante de legendas translúcida, sem bordas e sempre no topo (Always-on-Top).
    Ideal para sobrepor em jogos, reuniões (Zoom, Meet, Discord) e vídeos.
    """
    def __init__(self, settings_manager, parent=None):
        super().__init__(parent)
        self.settings = settings_manager
        self.drag_position = QPoint()
        self.locked = False
        
        self.source_lang = self.settings.get("source_lang", "en")
        self.target_lang = self.settings.get("target_lang", "pt")
        self._init_ui()

    def set_languages(self, source_lang: str, target_lang: str):
        self.source_lang = source_lang
        self.target_lang = target_lang
        if self.lbl_translated.text().startswith("A tradução em") or not self.lbl_translated.text():
            self.lbl_translated.setText("A tradução aparecerá aqui...")
        if self.lbl_original.text().startswith("O áudio original") or not self.lbl_original.text():
            self.lbl_original.setText("O áudio original captado aparecerá aqui...")
        
    def _init_ui(self):
        # Configurações de janela flutuante
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool | 
            Qt.SubWindow
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        
        self.resize(750, 130)
        self.setMinimumSize(400, 90)

        # Layout Principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 10, 15, 10)

        # Container estilizado
        self.container = QWidget(self)
        self.container.setObjectName("overlayContainer")
        
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(16, 10, 16, 10)
        container_layout.setSpacing(4)

        # Barra Superior de Controles (Sutil)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_badge = QLabel("🎙️ LEGENDA AO VIVO", self)
        self.lbl_badge.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold; letter-spacing: 1px;")

        self.btn_lock = QPushButton("🔓", self)
        self.btn_lock.setFixedSize(24, 24)
        self.btn_lock.setToolTip("Travar posição da janela")
        self.btn_lock.setStyleSheet(self._button_style())
        self.btn_lock.clicked.connect(self._toggle_lock)

        self.btn_close = QPushButton("✕", self)
        self.btn_close.setFixedSize(24, 24)
        self.btn_close.setToolTip("Fechar modo legenda flutuante")
        self.btn_close.setStyleSheet(self._button_style(is_close=True))
        self.btn_close.clicked.connect(self.hide)

        header_layout.addWidget(self.lbl_badge)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_lock)
        header_layout.addWidget(self.btn_close)
        container_layout.addLayout(header_layout)

        # Texto Traduzido Principal (idioma de destino, padrão: português)
        self.lbl_translated = QLabel("A tradução em português aparecerá aqui...", self)
        self.lbl_translated.setWordWrap(True)
        self.lbl_translated.setAlignment(Qt.AlignCenter)
        self.lbl_translated.setStyleSheet("color: #ffffff; font-size: 20px; font-weight: 700; font-family: 'Segoe UI', sans-serif;")

        # Sombra para contraste perfeito sobre qualquer live/jogo/fundo
        shadow_translated = QGraphicsDropShadowEffect(self)
        shadow_translated.setBlurRadius(8)
        shadow_translated.setColor(QColor(0, 0, 0, 220))
        shadow_translated.setOffset(1, 2)
        self.lbl_translated.setGraphicsEffect(shadow_translated)
        container_layout.addWidget(self.lbl_translated)

        # Texto Original Secundário (idioma de origem, padrão: inglês da live)
        self.lbl_original = QLabel("O áudio original captado aparecerá aqui...", self)
        self.lbl_original.setWordWrap(True)
        self.lbl_original.setAlignment(Qt.AlignCenter)
        self.lbl_original.setStyleSheet("color: #94a3b8; font-size: 13px; font-style: italic;")
        container_layout.addWidget(self.lbl_original)

        main_layout.addWidget(self.container)

        self.update_appearance()

    def update_appearance(self):
        opacity = self.settings.get("overlay_opacity", 85) / 100.0
        alpha = int(opacity * 255)
        self.container.setStyleSheet(f"""
            #overlayContainer {{
                background-color: rgba(15, 23, 42, {alpha});
                border: 1.5px solid rgba(56, 189, 248, 140);
                border-radius: 12px;
            }}
        """)
        font_size = self.settings.get("overlay_font_size", 20)
        self.lbl_translated.setStyleSheet(f"color: #ffffff; font-size: {font_size}px; font-weight: 700; font-family: 'Segoe UI', sans-serif;")

    def _button_style(self, is_close=False):
        hover_bg = "rgba(239, 68, 68, 180)" if is_close else "rgba(56, 189, 248, 180)"
        return f"""
            QPushButton {{
                background-color: rgba(30, 41, 59, 150);
                color: #e2e8f0;
                border: none;
                border-radius: 6px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
                color: white;
            }}
        """

    def _toggle_lock(self):
        self.locked = not self.locked
        self.btn_lock.setText("🔒" if self.locked else "🔓")
        self.btn_lock.setToolTip("Destravar posição da janela" if self.locked else "Travar posição da janela")

    def update_subtitles(self, original_text: str, translated_text: str):
        self.lbl_original.setText(f"{self.source_lang.upper()}: {original_text}")
        self.lbl_translated.setText(translated_text)

    # Permitir arrastar a janela livremente pela tela
    def mousePressEvent(self, event):
        if not self.locked and event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not self.locked and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()
