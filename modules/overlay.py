# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsDropShadowEffect, QSizeGrip
)
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QColor, QFont, QCursor


class FloatingOverlayWindow(QWidget):
    """
    Janela flutuante de legendas translúcida, sem bordas e sempre no topo (Always-on-Top).
    Recursos avançados:
    - Controles integrados rápidos: tamanho de fonte (+ / -), opacidade cíclica, ocultar/mostrar original
    - Modo Click-Through real: permite cliques através da legenda para jogos e vídeos
    - Auto-Limpeza / Fade após silêncio: não deixa texto antigo congelado na tela
    - Memória de posição e tamanho (restaura onde o usuário posicionou no monitor)
    """
    def __init__(self, settings_manager, parent=None):
        super().__init__(parent)
        self.settings = settings_manager
        self.drag_position = QPoint()
        self.locked = bool(self.settings.get("overlay_click_through", False))
        
        self.source_lang = self.settings.get("source_lang", "en")
        self.target_lang = self.settings.get("target_lang", "pt")
        self.show_original = bool(self.settings.get("overlay_show_original", True))

        # Timer de auto-limpeza após X segundos de silêncio
        self.autohide_timer = QTimer(self)
        self.autohide_timer.setSingleShot(True)
        self.autohide_timer.timeout.connect(self._on_autohide)

        self._init_ui()
        self._restore_geometry()

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
        
        saved_w = self.settings.get("overlay_width", 750)
        saved_h = self.settings.get("overlay_height", 130)
        self.resize(max(420, saved_w), max(90, saved_h))
        self.setMinimumSize(380, 85)

        # Layout Principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 8, 10, 8)

        # Container estilizado estilo HUD glassmorphism
        self.container = QWidget(self)
        self.container.setObjectName("overlayContainer")
        
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(14, 8, 14, 8)
        container_layout.setSpacing(3)

        # =====================================================================
        # BARRA DE FERRAMENTAS SUPERIOR (SUTIL E ERGONÔMICA)
        # =====================================================================
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 2)
        header_layout.setSpacing(6)

        self.lbl_badge = QLabel("🎙️ LEGENDA AO VIVO", self)
        self.lbl_badge.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 800; letter-spacing: 0.8px;")

        # Botão Diminuir Fonte
        self.btn_font_minus = QPushButton("A-", self)
        self.btn_font_minus.setFixedSize(26, 22)
        self.btn_font_minus.setToolTip("Diminuir tamanho da fonte")
        self.btn_font_minus.setStyleSheet(self._button_style())
        self.btn_font_minus.clicked.connect(lambda: self._adjust_font_size(-2))

        # Botão Aumentar Fonte
        self.btn_font_plus = QPushButton("A+", self)
        self.btn_font_plus.setFixedSize(26, 22)
        self.btn_font_plus.setToolTip("Aumentar tamanho da fonte")
        self.btn_font_plus.setStyleSheet(self._button_style())
        self.btn_font_plus.clicked.connect(lambda: self._adjust_font_size(+2))

        # Botão Ciclo de Opacidade
        self.btn_opacity = QPushButton("👁️", self)
        self.btn_opacity.setFixedSize(26, 22)
        self.btn_opacity.setToolTip("Alternar opacidade do fundo (60% / 80% / 95%)")
        self.btn_opacity.setStyleSheet(self._button_style())
        self.btn_opacity.clicked.connect(self._cycle_opacity)

        # Botão Alternar Original
        self.btn_toggle_orig = QPushButton("🌐", self)
        self.btn_toggle_orig.setFixedSize(26, 22)
        self.btn_toggle_orig.setToolTip("Mostrar/Ocultar texto original")
        self.btn_toggle_orig.setStyleSheet(self._button_style())
        self.btn_toggle_orig.clicked.connect(self._toggle_show_original)

        # Botão Travar / Destravar
        self.btn_lock = QPushButton("🔒" if self.locked else "🔓", self)
        self.btn_lock.setFixedSize(26, 22)
        self.btn_lock.setToolTip("Travar posição da legenda" if not self.locked else "Destravar posição da legenda")
        self.btn_lock.setStyleSheet(self._button_style())
        self.btn_lock.clicked.connect(self._toggle_lock)

        # Botão Fechar
        self.btn_close = QPushButton("✕", self)
        self.btn_close.setFixedSize(26, 22)
        self.btn_close.setToolTip("Ocultar legenda flutuante")
        self.btn_close.setStyleSheet(self._button_style(is_close=True))
        self.btn_close.clicked.connect(self.hide)

        header_layout.addWidget(self.lbl_badge)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_font_minus)
        header_layout.addWidget(self.btn_font_plus)
        header_layout.addWidget(self.btn_opacity)
        header_layout.addWidget(self.btn_toggle_orig)
        header_layout.addWidget(self.btn_lock)
        header_layout.addWidget(self.btn_close)
        container_layout.addLayout(header_layout)

        # =====================================================================
        # ÁREA DE TEXTO DA LEGENDA (ALTO CONTRASTE)
        # =====================================================================
        # Texto Traduzido Principal
        self.lbl_translated = QLabel("A tradução aparecerá aqui...", self)
        self.lbl_translated.setWordWrap(True)
        self.lbl_translated.setAlignment(Qt.AlignCenter)

        # Efeito de Sombra de Alto Contraste (legível sobre qualquer fundo)
        shadow_translated = QGraphicsDropShadowEffect(self)
        shadow_translated.setBlurRadius(10)
        shadow_translated.setColor(QColor(0, 0, 0, 240))
        shadow_translated.setOffset(2, 2)
        self.lbl_translated.setGraphicsEffect(shadow_translated)
        container_layout.addWidget(self.lbl_translated)

        # Texto Original Secundário
        self.lbl_original = QLabel("O áudio original captado aparecerá aqui...", self)
        self.lbl_original.setWordWrap(True)
        self.lbl_original.setAlignment(Qt.AlignCenter)
        self.lbl_original.setStyleSheet("color: #94a3b8; font-size: 13px; font-style: italic;")
        self.lbl_original.setVisible(self.show_original)

        shadow_orig = QGraphicsDropShadowEffect(self)
        shadow_orig.setBlurRadius(6)
        shadow_orig.setColor(QColor(0, 0, 0, 200))
        shadow_orig.setOffset(1, 1)
        self.lbl_original.setGraphicsEffect(shadow_orig)
        container_layout.addWidget(self.lbl_original)

        # Gripper de Redimensionamento sutil no canto inferior
        grip_layout = QHBoxLayout()
        grip_layout.setContentsMargins(0, 0, 0, 0)
        grip_layout.addStretch()
        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(14, 14)
        self.size_grip.setStyleSheet("background-color: transparent;")
        grip_layout.addWidget(self.size_grip)
        container_layout.addLayout(grip_layout)

        main_layout.addWidget(self.container)
        self.update_appearance()

    def _button_style(self, is_close=False):
        hover_bg = "rgba(239, 68, 68, 200)" if is_close else "rgba(56, 189, 248, 180)"
        return f"""
            QPushButton {{
                background-color: rgba(30, 41, 59, 170);
                color: #f1f5f9;
                border: 1px solid rgba(71, 85, 105, 120);
                border-radius: 5px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_bg};
                color: white;
                border-color: rgba(56, 189, 248, 200);
            }}
        """

    def update_appearance(self):
        opacity = self.settings.get("overlay_opacity", 85) / 100.0
        alpha = int(opacity * 255)
        border_color = "rgba(56, 189, 248, 180)" if not self.locked else "rgba(100, 116, 139, 140)"
        
        self.container.setStyleSheet(f"""
            #overlayContainer {{
                background-color: rgba(11, 15, 25, {alpha});
                border: 1.5px solid {border_color};
                border-radius: 12px;
            }}
        """)
        font_size = self.settings.get("overlay_font_size", 22)
        text_color = self.settings.get("overlay_text_color", "#ffffff")
        self.lbl_translated.setStyleSheet(
            f"color: {text_color}; font-size: {font_size}px; font-weight: 700; font-family: 'Segoe UI', sans-serif;"
        )

    def _adjust_font_size(self, delta: int):
        current_size = self.settings.get("overlay_font_size", 22)
        new_size = max(14, min(36, current_size + delta))
        self.settings.set("overlay_font_size", new_size)
        self.update_appearance()

    def _cycle_opacity(self):
        current_op = self.settings.get("overlay_opacity", 85)
        next_map = {60: 80, 80: 95, 95: 100, 100: 60}
        next_op = next_map.get(current_op, 85)
        self.settings.set("overlay_opacity", next_op)
        self.update_appearance()

    def _toggle_show_original(self):
        self.show_original = not self.show_original
        self.settings.set("overlay_show_original", self.show_original)
        self.lbl_original.setVisible(self.show_original)

    def _toggle_lock(self):
        self.locked = not self.locked
        self.settings.set("overlay_click_through", self.locked)
        self.btn_lock.setText("🔒" if self.locked else "🔓")
        self.btn_lock.setToolTip("Destravar posição da janela" if self.locked else "Travar posição da janela")
        self.update_appearance()

    def update_subtitles(self, original_text: str, translated_text: str):
        self.lbl_original.setText(f"{self.source_lang.upper()}: {original_text}")
        self.lbl_translated.setText(translated_text)
        
        # Reiniciar timer de auto-limpeza
        autohide_sec = self.settings.get("overlay_autohide_seconds", 8)
        if autohide_sec > 0:
            self.autohide_timer.start(int(autohide_sec * 1000))

    def _on_autohide(self):
        """Limpa sutilmente o texto quando não há fala ativa para não poluir a tela."""
        self.lbl_translated.setText("...")
        self.lbl_original.setText("")

    # =========================================================================
    # ARRASTAR E REDIMENSIONAR COM PERSISTÊNCIA DE COORDENADAS
    # =========================================================================
    def mousePressEvent(self, event):
        if not self.locked and event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if not self.locked and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._save_geometry()
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._save_geometry()

    def _save_geometry(self):
        pos = self.pos()
        size = self.size()
        self.settings.update_multiple({
            "overlay_pos_x": pos.x(),
            "overlay_pos_y": pos.y(),
            "overlay_width": size.width(),
            "overlay_height": size.height()
        })

    def _restore_geometry(self):
        x = self.settings.get("overlay_pos_x", None)
        y = self.settings.get("overlay_pos_y", None)
        if x is not None and y is not None:
            self.move(x, y)
