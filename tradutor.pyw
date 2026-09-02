# -*- coding: utf-8 -*-
"""
=============================================================================
 TRADUTOR DE LIVE EM TEMPO REAL PROFISSIONAL
 - Legenda Flutuante (HUD Overlay): Sobreposta em jogos, vídeos e reuniões
 - Painel Esquerdo: Transcrição em tempo real do áudio da live com timestamps
 - Painel Direito: Tradução contínua ultra-rápida com efeito de digitação fluida
 - Seletor Rápido de Dispositivo no Topo (Microfone ou Som do PC)
 - VAD Adaptativo Ultra-Sensível para Fala Natural e Suave
 - Exportação para Texto (.txt) e Legendas Sincronizadas (.srt)
=============================================================================
"""

import sys
import os
import time
import json
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QSlider, QCheckBox, QSplitter,
    QFrame, QGroupBox, QFileDialog, QMessageBox, QComboBox, QDialog,
    QLineEdit, QTabWidget, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QIcon, QTextCursor

# Módulos do sistema
from modules.settings_manager import SettingsManager
from modules.translator import translator_engine, format_subtitle_text
from modules.tts_engine import tts_engine
from modules.audio_engine import AudioEngine, AVAILABLE_LANGUAGES
from modules.visualizer import AudioVisualizerWidget
from modules.overlay import FloatingOverlayWindow


class SettingsDialog(QDialog):
    """Janela modal organizada com abas para configuração completa de áudio, HUD e voz."""
    def __init__(self, audio_engine, settings, overlay_window, parent=None):
        super().__init__(parent)
        self.audio_engine = audio_engine
        self.settings = settings
        self.overlay = overlay_window
        self.setWindowTitle("⚙️ Configurações Avançadas")
        self.resize(560, 430)
        self.setStyleSheet("""
            QDialog { background-color: #0b0f19; color: #f8fafc; font-size: 13px; font-family: 'Segoe UI', sans-serif; }
            QTabWidget::pane { border: 1px solid #1e293b; background-color: #111827; border-radius: 8px; }
            QTabBar::tab { background: #0f172a; color: #94a3b8; padding: 8px 16px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
            QTabBar::tab:selected { background: #1e293b; color: #38bdf8; font-weight: bold; }
            QGroupBox { border: 1px solid #334155; border-radius: 8px; margin-top: 10px; font-weight: bold; color: #38bdf8; padding: 12px; }
            QComboBox { background-color: #1e293b; border: 1px solid #475569; border-radius: 6px; padding: 6px 10px; color: #f8fafc; font-weight: 500; }
            QPushButton { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 8px 14px; font-weight: bold; }
            QPushButton:hover { background-color: #334155; border-color: #38bdf8; }
            QSlider::groove:horizontal { height: 6px; background: #1e293b; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #38bdf8; border-radius: 3px; }
            QSlider::handle:horizontal { background: #f8fafc; border: 2px solid #38bdf8; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
        """)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        tabs = QTabWidget(self)

        # ---------------------------------------------------------------------
        # ABA 1: DISPOSITIVOS & ÁUDIO
        # ---------------------------------------------------------------------
        tab_audio = QWidget()
        audio_layout = QVBoxLayout(tab_audio)
        audio_layout.setSpacing(12)

        # Seletor de Dispositivo
        grp_dev = QGroupBox("🎙️ Dispositivo de Entrada de Áudio", tab_audio)
        grp_dev_layout = QVBoxLayout(grp_dev)
        
        self.combo_mics = QComboBox(grp_dev)
        devices = self.audio_engine.get_input_devices()
        saved_idx = self.settings.get("mic_device_index", None)
        selected_i = 0

        for i, dev in enumerate(devices):
            self.combo_mics.addItem(dev["display_name"], dev["index"])
            if (saved_idx is None and dev["index"] is None) or (saved_idx is not None and dev["index"] == saved_idx):
                selected_i = i

        if devices:
            self.combo_mics.setCurrentIndex(selected_i)
        grp_dev_layout.addWidget(self.combo_mics)

        lbl_tip = QLabel(
            "💡 Dica: Para traduzir lives com som direto do PC, selecione 🔊 [Áudio do PC / Live].\n"
            "Para traduzir sua voz ou falar pelo fone, selecione 🎙️ Microfone ou 🎧 Headset.",
            grp_dev
        )
        lbl_tip.setStyleSheet("color: #38bdf8; font-size: 11px; margin-top: 4px;")
        grp_dev_layout.addWidget(lbl_tip)
        audio_layout.addWidget(grp_dev)

        # Fatiamento Contínuo
        grp_vad = QGroupBox("⚡ Ajustes de Fala para Lives", tab_audio)
        vad_layout = QVBoxLayout(grp_vad)
        
        lbl_dur = QLabel("Duração Máxima de Fala Contínua:", grp_vad)
        vad_layout.addWidget(lbl_dur)
        self.slider_dur = QSlider(Qt.Horizontal, grp_vad)
        self.slider_dur.setRange(4, 12)
        saved_dur = int(self.settings.get("max_speech_duration", 7.0))
        self.slider_dur.setValue(saved_dur)
        self.lbl_dur_val = QLabel(f"{saved_dur} segundos (ideal para lives)", grp_vad)
        self.lbl_dur_val.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.slider_dur.valueChanged.connect(lambda v: self.lbl_dur_val.setText(f"{v} segundos"))
        vad_layout.addWidget(self.slider_dur)
        vad_layout.addWidget(self.lbl_dur_val)
        audio_layout.addWidget(grp_vad)

        audio_layout.addStretch()
        tabs.addTab(tab_audio, "🎙️ Áudio & Microfone")

        # ---------------------------------------------------------------------
        # ABA 2: LEGENDA FLUTUANTE (OVERLAY HUD)
        # ---------------------------------------------------------------------
        tab_overlay = QWidget()
        ov_layout = QVBoxLayout(tab_overlay)
        ov_layout.setSpacing(12)

        grp_ov = QGroupBox("🪟 Estilo e Comportamento da Legenda", tab_overlay)
        grp_ov_layout = QVBoxLayout(grp_ov)

        # Tamanho da Fonte
        grp_ov_layout.addWidget(QLabel("Tamanho da Fonte da Tradução:"))
        self.slider_font = QSlider(Qt.Horizontal, grp_ov)
        self.slider_font.setRange(16, 36)
        saved_font = self.settings.get("overlay_font_size", 22)
        self.slider_font.setValue(saved_font)
        self.lbl_font_val = QLabel(f"{saved_font}px", grp_ov)
        self.slider_font.valueChanged.connect(lambda v: self.lbl_font_val.setText(f"{v}px"))
        grp_ov_layout.addWidget(self.slider_font)
        grp_ov_layout.addWidget(self.lbl_font_val)

        # Opacidade
        grp_ov_layout.addWidget(QLabel("Opacidade do Fundo Transparente:"))
        self.slider_opacity = QSlider(Qt.Horizontal, grp_ov)
        self.slider_opacity.setRange(40, 100)
        saved_op = self.settings.get("overlay_opacity", 85)
        self.slider_opacity.setValue(saved_op)
        self.lbl_op_val = QLabel(f"{saved_op}%", grp_ov)
        self.slider_opacity.valueChanged.connect(lambda v: self.lbl_op_val.setText(f"{v}%"))
        grp_ov_layout.addWidget(self.slider_opacity)
        grp_ov_layout.addWidget(self.lbl_op_val)

        # Auto-limpeza
        grp_ov_layout.addWidget(QLabel("Tempo de Auto-Limpeza após Silêncio:"))
        self.slider_autohide = QSlider(Qt.Horizontal, grp_ov)
        self.slider_autohide.setRange(0, 20)
        saved_ah = self.settings.get("overlay_autohide_seconds", 8)
        self.slider_autohide.setValue(saved_ah)
        self.lbl_ah_val = QLabel(f"{saved_ah}s (0 = nunca limpar)", grp_ov)
        self.slider_autohide.valueChanged.connect(lambda v: self.lbl_ah_val.setText(f"{v}s (0 = nunca limpar)" if v > 0 else "Desativado"))
        grp_ov_layout.addWidget(self.slider_autohide)
        grp_ov_layout.addWidget(self.lbl_ah_val)

        # Checkboxes
        self.chk_show_orig = QCheckBox("Mostrar texto original abaixo da tradução", grp_ov)
        self.chk_show_orig.setChecked(self.settings.get("overlay_show_original", True))
        grp_ov_layout.addWidget(self.chk_show_orig)

        ov_layout.addWidget(grp_ov)
        ov_layout.addStretch()
        tabs.addTab(tab_overlay, "🪟 Legenda Flutuante")

        # ---------------------------------------------------------------------
        # ABA 3: VOZ & SÍNTESE TTS
        # ---------------------------------------------------------------------
        tab_tts = QWidget()
        tts_layout = QVBoxLayout(tab_tts)
        tts_layout.setSpacing(12)

        grp_tts = QGroupBox("🔊 Voz do Sistema (TTS)", tab_tts)
        grp_tts_layout = QVBoxLayout(grp_tts)

        self.combo_voices = QComboBox(grp_tts)
        voices = tts_engine.get_voices()
        saved_voice = self.settings.get("tts_voice_id", tts_engine.selected_voice_id)
        selected_v = 0
        for i, v in enumerate(voices):
            self.combo_voices.addItem(v["name"], v["id"])
            if saved_voice and v["id"] == saved_voice:
                selected_v = i
        if voices:
            self.combo_voices.setCurrentIndex(selected_v)
        grp_tts_layout.addWidget(self.combo_voices)

        # Velocidade
        grp_tts_layout.addWidget(QLabel("Velocidade da Narração (Rate):"))
        self.slider_rate = QSlider(Qt.Horizontal, grp_tts)
        self.slider_rate.setRange(100, 240)
        saved_rate = self.settings.get("tts_rate", 160)
        self.slider_rate.setValue(saved_rate)
        self.lbl_rate_val = QLabel(f"{saved_rate}", grp_tts)
        self.slider_rate.valueChanged.connect(lambda v: self.lbl_rate_val.setText(f"{v}"))
        grp_tts_layout.addWidget(self.slider_rate)
        grp_tts_layout.addWidget(self.lbl_rate_val)

        # Botão Testar Voz
        btn_test_voice = QPushButton("▶ Testar Pronúncia da Voz", grp_tts)
        btn_test_voice.clicked.connect(self._test_voice)
        grp_tts_layout.addWidget(btn_test_voice)

        tts_layout.addWidget(grp_tts)
        tts_layout.addStretch()
        tabs.addTab(tab_tts, "🔊 Voz & TTS")

        main_layout.addWidget(tabs)

        # Botões Rodapé
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_cancel = QPushButton("Cancelar", self)
        btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Salvar Alterações", self)
        btn_save.setStyleSheet("background-color: #0284c7; color: white; border: none;")
        btn_save.clicked.connect(self._save_and_close)
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_save)
        main_layout.addLayout(btn_box)

    def _test_voice(self):
        if self.combo_voices.currentIndex() >= 0:
            voice_id = self.combo_voices.itemData(self.combo_voices.currentIndex())
            tts_engine.set_voice(voice_id)
        tts_engine.set_rate(self.slider_rate.value())
        tts_engine.speak("Olá! Este é um teste da voz de narração do tradutor em tempo real.", priority=True)

    def _save_and_close(self):
        if self.combo_mics.currentIndex() >= 0:
            mic_idx = self.combo_mics.itemData(self.combo_mics.currentIndex())
            self.settings.set("mic_device_index", mic_idx)
        
        self.settings.set("max_speech_duration", float(self.slider_dur.value()))
        self.audio_engine.max_speech_duration = float(self.slider_dur.value())

        self.settings.update_multiple({
            "overlay_font_size": self.slider_font.value(),
            "overlay_opacity": self.slider_opacity.value(),
            "overlay_autohide_seconds": self.slider_autohide.value(),
            "overlay_show_original": self.chk_show_orig.isChecked()
        })
        self.overlay.update_appearance()
        self.overlay.show_original = self.chk_show_orig.isChecked()
        self.overlay.lbl_original.setVisible(self.chk_show_orig.isChecked())

        if self.combo_voices.currentIndex() >= 0:
            voice_id = self.combo_voices.itemData(self.combo_voices.currentIndex())
            tts_engine.set_voice(voice_id)
            self.settings.set("tts_voice_id", voice_id)
        rate = self.slider_rate.value()
        tts_engine.set_rate(rate)
        self.settings.set("tts_rate", rate)

        self.accept()


class RealtimeSplitTranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = SettingsManager()
        self.history_items = []
        self.start_session_time = time.time()

        # Variáveis de animação de escrita (Typewriter effect)
        self.typewriter_text = ""
        self.typewriter_index = 0
        self.typewriter_timer = QTimer(self)
        self.typewriter_timer.setInterval(10)
        self.typewriter_timer.timeout.connect(self._on_typewriter_step)

        # Motores
        self.audio_engine = AudioEngine(self.settings, translator_engine, tts_engine)
        self.overlay_window = FloatingOverlayWindow(self.settings)

        # Sinais do motor de áudio
        self.audio_engine.level_changed.connect(self._on_audio_level)
        self.audio_engine.speech_started.connect(self._on_speech_started)
        self.audio_engine.speech_ended.connect(self._on_speech_ended)
        self.audio_engine.transcription_ready.connect(self._on_transcription_ready)
        self.audio_engine.translation_ready_with_metrics.connect(self._on_translation_ready_with_metrics)
        self.audio_engine.status_changed.connect(self._on_status_changed)
        self.audio_engine.error_occurred.connect(self._on_error)

        self._init_ui()
        self._apply_theme()

    def _init_ui(self):
        self.resize(1140, 760)
        self.setMinimumSize(880, 580)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(10)

        # =====================================================================
        # 1. BARRA SUPERIOR DE CONTROLE E VISUALIZAÇÃO
        # =====================================================================
        top_bar = QFrame(self)
        top_bar.setObjectName("topBarFrame")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(14, 10, 14, 10)
        top_layout.setSpacing(12)

        # Botão Principal de Iniciar/Parar Gravação
        self.btn_mic = QPushButton("🎙️ INICIAR ESCUTA", top_bar)
        self.btn_mic.setObjectName("btnMicStart")
        self.btn_mic.setMinimumHeight(44)
        self.btn_mic.setMinimumWidth(180)
        self.btn_mic.setCursor(Qt.PointingHandCursor)
        self.btn_mic.setToolTip("Iniciar captura e tradução em tempo real (Atalho: Espaço)")
        self.btn_mic.clicked.connect(self._toggle_microphone)
        top_layout.addWidget(self.btn_mic)

        # SELETOR RÁPIDO DE DISPOSITIVO (ENXUTO E DIRETO NO TOPO)
        self.combo_quick_device = QComboBox(top_bar)
        self.combo_quick_device.setObjectName("comboQuickDev")
        self.combo_quick_device.setMinimumHeight(38)
        self.combo_quick_device.setMinimumWidth(230)
        self.combo_quick_device.setToolTip("Escolha o que o app vai escutar: seu Microfone, Headset ou o Som do PC")
        self._populate_quick_devices()
        self.combo_quick_device.currentIndexChanged.connect(self._on_quick_device_changed)
        top_layout.addWidget(self.combo_quick_device)

        # Visualizador de Áudio Reativo com Marcador de Limiar
        self.visualizer = AudioVisualizerWidget(top_bar)
        self.visualizer.setMinimumHeight(50)
        top_layout.addWidget(self.visualizer, 1)

        # Controles Rápidos no Topo
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(3)

        # Sensibilidade
        sens_box = QHBoxLayout()
        lbl_sens_title = QLabel("Sensibilidade:", top_bar)
        lbl_sens_title.setStyleSheet("font-size: 11px; font-weight: bold; color: #94a3b8;")
        sens_box.addWidget(lbl_sens_title)
        
        self.slider_sens = QSlider(Qt.Horizontal, top_bar)
        self.slider_sens.setRange(15, 95)
        self.slider_sens.setValue(self.settings.get("mic_sensitivity", 75))
        self.slider_sens.setToolTip("Ajuste de sensibilidade: 75% é ideal para voz normal/baixa. Se pegar muito ruído, reduza para 60%.")
        self.slider_sens.valueChanged.connect(self._on_sens_changed)
        sens_box.addWidget(self.slider_sens)
        
        self.lbl_sens = QLabel(f"{self.slider_sens.value()}%", top_bar)
        self.lbl_sens.setStyleSheet("font-size: 11px; font-weight: bold; color: #38bdf8;")
        sens_box.addWidget(self.lbl_sens)
        ctrl_layout.addLayout(sens_box)

        # Opção Narrar a tradução em voz alta
        self.chk_tts = QCheckBox("🔊 Narrar tradução (TTS)", top_bar)
        self.chk_tts.setStyleSheet("font-size: 11px;")
        self.chk_tts.setChecked(self.settings.get("auto_tts", False))
        self.chk_tts.toggled.connect(lambda val: self.settings.set("auto_tts", val))
        ctrl_layout.addWidget(self.chk_tts)

        top_layout.addLayout(ctrl_layout)

        # Botões de Ação Topo
        top_btns = QVBoxLayout()
        top_btns.setSpacing(4)

        self.btn_overlay = QPushButton("🪟 Legenda Flutuante (HUD)", top_bar)
        self.btn_overlay.setObjectName("btnOverlayTop")
        self.btn_overlay.setToolTip("Sobrepor legenda translúcida na tela da live/jogo (Atalho: Ctrl+L)")
        self.btn_overlay.clicked.connect(self._toggle_overlay)
        top_btns.addWidget(self.btn_overlay)

        self.btn_settings = QPushButton("⚙️ Configurações", top_bar)
        self.btn_settings.clicked.connect(self._open_settings)
        top_btns.addWidget(self.btn_settings)

        top_layout.addLayout(top_btns)
        main_layout.addWidget(top_bar)

        # =====================================================================
        # 1.5. BARRA DE SELEÇÃO E INVERSÃO DE IDIOMAS (EX: EN ⇄ PT-BR)
        # =====================================================================
        lang_bar = QFrame(self)
        lang_bar.setObjectName("langBarFrame")
        lang_layout = QHBoxLayout(lang_bar)
        lang_layout.setContentsMargins(14, 8, 14, 8)
        lang_layout.setSpacing(10)

        lbl_src_title = QLabel("🎙️ Idioma de Entrada (Live):", lang_bar)
        lbl_src_title.setObjectName("lblLangTitle")
        self.combo_src_lang = QComboBox(lang_bar)
        self.combo_src_lang.setObjectName("comboLang")
        self.combo_src_lang.setMinimumWidth(165)

        for code, info in AVAILABLE_LANGUAGES.items():
            self.combo_src_lang.addItem(f"{info['flag']} {info['name']}", code)

        self.btn_swap_lang = QPushButton("⇄ INVERTER IDIOMAS", lang_bar)
        self.btn_swap_lang.setObjectName("btnSwapLang")
        self.btn_swap_lang.setCursor(Qt.PointingHandCursor)
        self.btn_swap_lang.setToolTip("Inverter idiomas de entrada e saída instantaneamente (Atalho: Ctrl+I / Alt+I)")
        self.btn_swap_lang.clicked.connect(self._swap_languages)

        lbl_tgt_title = QLabel("🌐 Idioma de Saída (Legenda):", lang_bar)
        lbl_tgt_title.setObjectName("lblLangTitle")
        self.combo_tgt_lang = QComboBox(lang_bar)
        self.combo_tgt_lang.setObjectName("comboLang")
        self.combo_tgt_lang.setMinimumWidth(165)

        for code, info in AVAILABLE_LANGUAGES.items():
            self.combo_tgt_lang.addItem(f"{info['flag']} {info['name']}", code)

        self.lbl_lang_direction = QLabel("", lang_bar)
        self.lbl_lang_direction.setObjectName("badgeLangDirection")

        self.lbl_latency = QLabel("⚡ Latência: --", lang_bar)
        self.lbl_latency.setObjectName("badgeLatency")
        self.lbl_latency.setToolTip("Tempo total de processamento: voz captada ➔ traduzida")

        lang_layout.addWidget(lbl_src_title)
        lang_layout.addWidget(self.combo_src_lang)
        lang_layout.addWidget(self.btn_swap_lang)
        lang_layout.addWidget(lbl_tgt_title)
        lang_layout.addWidget(self.combo_tgt_lang)
        lang_layout.addStretch()
        lang_layout.addWidget(self.lbl_latency)
        lang_layout.addWidget(self.lbl_lang_direction)
        main_layout.addWidget(lang_bar)

        # =====================================================================
        # 2. TELA DIVIDIDA EM DOIS PAINÉIS (QSplitter)
        # =====================================================================
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setObjectName("mainSplitter")
        self.splitter.setHandleWidth(8)

        # PAINEL DA ESQUERDA: ÁUDIO ORIGINAL
        panel_pt = QFrame(self.splitter)
        panel_pt.setObjectName("panelPT")
        layout_pt = QVBoxLayout(panel_pt)
        layout_pt.setContentsMargins(14, 12, 14, 12)
        layout_pt.setSpacing(8)

        head_pt = QHBoxLayout()
        self.lbl_head_pt = QLabel("🎧 Áudio Original", panel_pt)
        self.lbl_head_pt.setObjectName("paneTitlePT")
        self.badge_pt = QLabel("AO VIVO", panel_pt)
        self.badge_pt.setObjectName("badgeLivePT")
        head_pt.addWidget(self.lbl_head_pt)
        head_pt.addWidget(self.badge_pt)
        head_pt.addStretch()
        layout_pt.addLayout(head_pt)

        self.lbl_desc_pt = QLabel("O áudio da live captado pelo dispositivo é transcrito abaixo:", panel_pt)
        self.lbl_desc_pt.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout_pt.addWidget(self.lbl_desc_pt)

        self.txt_transcription = QTextEdit(panel_pt)
        self.txt_transcription.setObjectName("txtTranscription")
        self.txt_transcription.setReadOnly(True)
        layout_pt.addWidget(self.txt_transcription, 1)

        bar_pt = QHBoxLayout()
        self.lbl_stats_pt = QLabel("0 palavras • 0 caracteres", panel_pt)
        self.lbl_stats_pt.setStyleSheet("color: #64748b; font-size: 11px;")
        bar_pt.addWidget(self.lbl_stats_pt)
        bar_pt.addStretch()

        self.btn_copy_pt = QPushButton("📋 Copiar Original", panel_pt)
        self.btn_copy_pt.clicked.connect(lambda: self._copy_text(self.txt_transcription.toPlainText()))
        btn_clear_pt = QPushButton("🗑️ Limpar", panel_pt)
        btn_clear_pt.clicked.connect(self._clear_all)

        bar_pt.addWidget(self.btn_copy_pt)
        bar_pt.addWidget(btn_clear_pt)
        layout_pt.addLayout(bar_pt)

        # PAINEL DA DIREITA: TRADUÇÃO EM TEMPO REAL
        panel_en = QFrame(self.splitter)
        panel_en.setObjectName("panelEN")
        layout_en = QVBoxLayout(panel_en)
        layout_en.setContentsMargins(14, 12, 14, 12)
        layout_en.setSpacing(8)

        head_en = QHBoxLayout()
        self.lbl_head_en = QLabel("🌐 Tradução em Tempo Real", panel_en)
        self.lbl_head_en.setObjectName("paneTitleEN")
        self.badge_en = QLabel("TRADUÇÃO", panel_en)
        self.badge_en.setObjectName("badgeLiveEN")
        head_en.addWidget(self.lbl_head_en)
        head_en.addWidget(self.badge_en)
        head_en.addStretch()
        layout_en.addLayout(head_en)

        self.lbl_desc_en = QLabel("A tradução é escrita automaticamente em tempo real:", panel_en)
        self.lbl_desc_en.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout_en.addWidget(self.lbl_desc_en)

        self.txt_writing_en = QTextEdit(panel_en)
        self.txt_writing_en.setObjectName("txtWritingEN")
        self.txt_writing_en.setReadOnly(True)
        layout_en.addWidget(self.txt_writing_en, 1)

        bar_en = QHBoxLayout()
        self.lbl_stats_en = QLabel("0 palavras traduzidas", panel_en)
        self.lbl_stats_en.setStyleSheet("color: #64748b; font-size: 11px;")
        bar_en.addWidget(self.lbl_stats_en)
        bar_en.addStretch()

        self.btn_speak_last = QPushButton("🔊 Ouvir", panel_en)
        self.btn_speak_last.clicked.connect(self._speak_all_english)
        self.btn_copy_en = QPushButton("📋 Copiar Tradução", panel_en)
        self.btn_copy_en.clicked.connect(lambda: self._copy_text(self.txt_writing_en.toPlainText()))
        
        btn_save_txt = QPushButton("💾 Salvar .TXT", panel_en)
        btn_save_txt.setToolTip("Salvar transcrição e tradução em arquivo de texto")
        btn_save_txt.clicked.connect(self._export_both)

        btn_save_srt = QPushButton("🎬 Exportar .SRT", panel_en)
        btn_save_srt.setToolTip("Exportar legendas sincronizadas com marcações de tempo (.srt)")
        btn_save_srt.setStyleSheet("background-color: #0284c7; color: white; border: none;")
        btn_save_srt.clicked.connect(self._export_srt)

        bar_en.addWidget(self.btn_speak_last)
        bar_en.addWidget(self.btn_copy_en)
        bar_en.addWidget(btn_save_txt)
        bar_en.addWidget(btn_save_srt)
        layout_en.addLayout(bar_en)

        self.splitter.addWidget(panel_pt)
        self.splitter.addWidget(panel_en)
        self.splitter.setSizes([550, 550])
        main_layout.addWidget(self.splitter, 1)

        # =====================================================================
        # 3. BARRA INFERIOR DE STATUS E ATALHOS
        # =====================================================================
        bottom_bar = QFrame(self)
        bottom_bar.setObjectName("bottomBar")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(12, 6, 12, 6)

        self.lbl_status = QLabel("🟢 Pronto. Escolha o dispositivo acima e clique em 'INICIAR ESCUTA' (ou aperte Espaço).", self)
        self.lbl_status.setObjectName("statusLabel")

        self.lbl_shortcuts_info = QLabel("Atalhos: Espaço (Escuta) • Ctrl+I (Inverter) • Ctrl+L (Legenda HUD) • Ctrl+S (Exportar)", self)
        self.lbl_shortcuts_info.setStyleSheet("color: #64748b; font-size: 11px;")

        self.lbl_indicator = QLabel("● MICROFONE DESLIGADO", self)
        self.lbl_indicator.setObjectName("indOff")

        bottom_layout.addWidget(self.lbl_status)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.lbl_shortcuts_info)
        bottom_layout.addSpacing(16)
        bottom_layout.addWidget(self.lbl_indicator)
        main_layout.addWidget(bottom_bar)

        # Sincronização inicial de idiomas
        saved_src = self.settings.get("source_lang", "en")
        saved_tgt = self.settings.get("target_lang", "pt")
        self._set_combo_code(self.combo_src_lang, saved_src)
        self._set_combo_code(self.combo_tgt_lang, saved_tgt)

        self.combo_src_lang.currentIndexChanged.connect(self._on_combo_language_changed)
        self.combo_tgt_lang.currentIndexChanged.connect(self._on_combo_language_changed)

        self._update_languages(saved_src, saved_tgt, save_settings=False)

    # -------------------------------------------------------------------------
    # SELETOR RÁPIDO DE DISPOSITIVO
    # -------------------------------------------------------------------------
    def _populate_quick_devices(self):
        self.combo_quick_device.blockSignals(True)
        self.combo_quick_device.clear()
        
        devices = self.audio_engine.get_input_devices()
        saved_idx = self.settings.get("mic_device_index", None)
        selected_i = 0

        for i, dev in enumerate(devices):
            self.combo_quick_device.addItem(dev["display_name"], dev["index"])
            if (saved_idx is None and dev["index"] is None) or (saved_idx is not None and dev["index"] == saved_idx):
                selected_i = i

        if devices:
            self.combo_quick_device.setCurrentIndex(selected_i)

        self.combo_quick_device.blockSignals(False)

    def _on_quick_device_changed(self, idx):
        if idx >= 0:
            dev_idx = self.combo_quick_device.itemData(idx)
            self.settings.set("mic_device_index", dev_idx)
            if self.audio_engine.is_recording:
                self.audio_engine.stop_listening()
                self.audio_engine.start_listening(dev_idx)
            else:
                dev_name = self.combo_quick_device.currentText()
                self.lbl_status.setText(f"🎧 Dispositivo selecionado: {dev_name}")

    # -------------------------------------------------------------------------
    # GERENCIAMENTO E INVERSÃO DE IDIOMAS
    # -------------------------------------------------------------------------
    def _set_combo_code(self, combo: QComboBox, code: str):
        for i in range(combo.count()):
            if combo.itemData(i) == code:
                combo.setCurrentIndex(i)
                break

    def _swap_languages(self):
        src_idx = self.combo_src_lang.currentIndex()
        tgt_idx = self.combo_tgt_lang.currentIndex()
        src_code = self.combo_src_lang.itemData(src_idx) if src_idx >= 0 else "en"
        tgt_code = self.combo_tgt_lang.itemData(tgt_idx) if tgt_idx >= 0 else "pt"

        if src_code == tgt_code:
            return

        self.combo_src_lang.blockSignals(True)
        self.combo_tgt_lang.blockSignals(True)

        self._set_combo_code(self.combo_src_lang, tgt_code)
        self._set_combo_code(self.combo_tgt_lang, src_code)

        self.combo_src_lang.blockSignals(False)
        self.combo_tgt_lang.blockSignals(False)

        self._update_languages(tgt_code, src_code, save_settings=True)

    def _on_combo_language_changed(self):
        src_idx = self.combo_src_lang.currentIndex()
        tgt_idx = self.combo_tgt_lang.currentIndex()
        src_code = self.combo_src_lang.itemData(src_idx) if src_idx >= 0 else "en"
        tgt_code = self.combo_tgt_lang.itemData(tgt_idx) if tgt_idx >= 0 else "pt"
        self._update_languages(src_code, tgt_code, save_settings=True)

    def _update_languages(self, src_code: str, tgt_code: str, save_settings: bool = True):
        if save_settings:
            self.settings.set("source_lang", src_code)
            self.settings.set("target_lang", tgt_code)

        self.audio_engine.set_languages(src_code, tgt_code)
        self.overlay_window.set_languages(src_code, tgt_code)
        tts_engine.auto_select_voice_for_language(tgt_code)

        src_info = AVAILABLE_LANGUAGES.get(src_code, {"name": src_code, "flag": "🌐"})
        tgt_info = AVAILABLE_LANGUAGES.get(tgt_code, {"name": tgt_code, "flag": "🌐"})

        self.lbl_lang_direction.setText(f"{src_info['flag']} {src_code.upper()} ➔ {tgt_info['flag']} {tgt_code.upper()}")
        self.setWindowTitle(f"Tradutor em Tempo Real • {src_info['flag']} {src_info['name']} ➔ {tgt_info['flag']} {tgt_info['name']}")

        self.lbl_head_pt.setText(f"🎧 Áudio Original ({src_info['name']})")
        self.lbl_desc_pt.setText(f"O áudio captado ({src_info['name']}) é transcrito abaixo:")
        self.txt_transcription.setPlaceholderText(
            f"Clique em 'INICIAR ESCUTA' acima...\n\nO áudio captado ({src_info['name']}) aparecerá aqui continuamente."
        )
        self.btn_copy_pt.setText(f"📋 Copiar {src_info['name']}")

        self.lbl_head_en.setText(f"{tgt_info['flag']} Tradução em {tgt_info['name']} (Tempo Real)")
        self.lbl_desc_en.setText(f"A tradução em {tgt_info['name']} é escrita automaticamente em tempo real:")
        self.txt_writing_en.setPlaceholderText(
            f"A tradução em {tgt_info['name']} será escrita aqui em tempo real conforme o áudio é detectado..."
        )
        self.btn_copy_en.setText(f"📋 Copiar {tgt_info['name']}")
        self.btn_speak_last.setText(f"🔊 Ouvir em {tgt_info['name']}")

        if save_settings:
            self.lbl_status.setText(f"🔄 Idiomas invertidos: {src_info['flag']} {src_info['name']} ➔ {tgt_info['flag']} {tgt_info['name']}")

    # -------------------------------------------------------------------------
    # CONTROLE DE MICROFONE & VOZ
    # -------------------------------------------------------------------------
    def _toggle_microphone(self):
        if not self.audio_engine.is_recording:
            selected_device = self.combo_quick_device.currentData()
            self.audio_engine.start_listening(selected_device)
            self.visualizer.set_recording(True)
            self.btn_mic.setText("🛑 PARAR ESCUTA")
            self.btn_mic.setObjectName("btnMicStop")
            self.lbl_indicator.setText("● OUVINDO...")
            self.lbl_indicator.setObjectName("indOn")
            self.badge_pt.setText("ESCUTANDO")
            self.badge_pt.setStyleSheet("background-color: #10b981; color: white;")
            
            if not self.overlay_window.isVisible():
                self._toggle_overlay()
        else:
            self.audio_engine.stop_listening()
            self.visualizer.set_recording(False)
            self.btn_mic.setText("🎙️ INICIAR ESCUTA")
            self.btn_mic.setObjectName("btnMicStart")
            self.lbl_indicator.setText("● MICROFONE DESLIGADO")
            self.lbl_indicator.setObjectName("indOff")
            self.badge_pt.setText("AO VIVO")
            self.badge_pt.setStyleSheet("background-color: #0284c7; color: white;")
        self._apply_theme()

    def _on_audio_level(self, level, waveform, threshold):
        self.visualizer.set_level(level, waveform, threshold)

    def _on_speech_started(self):
        src_name = AVAILABLE_LANGUAGES.get(self.audio_engine.source_lang, {}).get("name", "áudio")
        self.lbl_status.setText(f"🎧 Voz detectada... transcrevendo em {src_name.lower()}...")
        self.lbl_indicator.setText("● GRAVANDO")
        self.badge_pt.setStyleSheet("background-color: #f59e0b; color: #000; font-weight: bold;")

    def _on_speech_ended(self):
        tgt_name = AVAILABLE_LANGUAGES.get(self.audio_engine.target_lang, {}).get("name", "destino")
        self.lbl_status.setText(f"⚡ Traduzindo para {tgt_name.lower()}...")
        self.lbl_indicator.setText("● TRADUZINDO")
        self.badge_en.setText("ESCREVENDO...")
        self.badge_en.setStyleSheet("background-color: #8b5cf6; color: white;")

    def _on_transcription_ready(self, text_original):
        current_pt = self.txt_transcription.toPlainText().strip()
        timestamp = datetime.now().strftime("%H:%M:%S")

        if current_pt:
            new_pt = f"{current_pt}\n[{timestamp}] {text_original}"
        else:
            new_pt = f"[{timestamp}] {text_original}"

        self.txt_transcription.setPlainText(new_pt)
        self.txt_transcription.moveCursor(QTextCursor.End)

        chars = len(new_pt)
        words = len(new_pt.split())
        self.lbl_stats_pt.setText(f"{words} palavras • {chars} caracteres")

    def _on_translation_ready_with_metrics(self, text_original, text_translated, latency_ms):
        timestamp = datetime.now().strftime("%H:%M:%S")
        rel_seconds = time.time() - self.start_session_time

        self.history_items.append({
            "time": timestamp,
            "rel_sec": rel_seconds,
            "original": text_original,
            "translated": text_translated
        })

        self.overlay_window.update_subtitles(text_original, text_translated)

        lat_int = int(latency_ms)
        color = "#10b981" if lat_int < 700 else ("#38bdf8" if lat_int < 1200 else "#f59e0b")
        self.lbl_latency.setText(f"⚡ {lat_int}ms")
        self.lbl_latency.setStyleSheet(f"background-color: #1e293b; color: {color}; border: 1px solid #334155; border-radius: 6px; padding: 4px 8px; font-weight: bold; font-size: 11px;")

        if self.typewriter_timer.isActive():
            self.typewriter_timer.stop()
            if self.typewriter_index < len(self.typewriter_text):
                self.txt_writing_en.insertPlainText(self.typewriter_text[self.typewriter_index:])
                self.txt_writing_en.moveCursor(QTextCursor.End)

        entry_header = f"\n[{timestamp}] " if self.txt_writing_en.toPlainText().strip() else f"[{timestamp}] "
        self.typewriter_text = entry_header + text_translated
        self.typewriter_index = 0
        self.typewriter_timer.start(10)

        self.lbl_status.setText(f"✅ Traduzido com sucesso em {lat_int}ms!")
        self.badge_en.setText("TRADUÇÃO")
        self.badge_en.setStyleSheet("background-color: #6366f1; color: white;")

    def _on_typewriter_step(self):
        if self.typewriter_index < len(self.typewriter_text):
            end_idx = min(len(self.typewriter_text), self.typewriter_index + 2)
            chunk = self.typewriter_text[self.typewriter_index:end_idx]
            self.txt_writing_en.insertPlainText(chunk)
            self.txt_writing_en.moveCursor(QTextCursor.End)
            self.typewriter_index = end_idx
        else:
            self.typewriter_timer.stop()
            total_text = self.txt_writing_en.toPlainText()
            words = len(total_text.split())
            self.lbl_stats_en.setText(f"{words} palavras traduzidas")

    def _on_status_changed(self, msg):
        self.lbl_status.setText(msg)

    def _on_error(self, err_msg):
        self.lbl_status.setText(f"⚠️ {err_msg}")

    def _on_sens_changed(self, val):
        self.audio_engine.set_sensitivity(val)
        self.lbl_sens.setText(f"{val}%")

    def _toggle_overlay(self):
        if self.overlay_window.isVisible():
            self.overlay_window.hide()
            self.btn_overlay.setText("🪟 Legenda Flutuante (HUD)")
        else:
            self.overlay_window.show()
            self.btn_overlay.setText("🪟 Ocultar Legenda")

    def _open_settings(self):
        dlg = SettingsDialog(self.audio_engine, self.settings, self.overlay_window, self)
        if dlg.exec_():
            self._populate_quick_devices()

    def _speak_all_english(self):
        text = self.txt_writing_en.toPlainText()
        if text.strip():
            clean_lines = []
            for line in text.splitlines():
                if "]" in line:
                    clean_lines.append(line.split("]", 1)[1].strip())
                else:
                    clean_lines.append(line.strip())
            speech_text = " ".join(clean_lines)
            tts_engine.speak(speech_text, priority=True)

    def _copy_text(self, text: str):
        if text.strip():
            QApplication.clipboard().setText(text.strip())
            self.lbl_status.setText("📋 Texto copiado com sucesso!")

    def _clear_all(self):
        self.txt_transcription.clear()
        self.txt_writing_en.clear()
        self.history_items = []
        self.lbl_stats_pt.setText("0 palavras • 0 caracteres")
        self.lbl_stats_en.setText("0 palavras traduzidas")
        self.lbl_status.setText("🗑️ Painéis limpos.")

    def _export_both(self):
        original_text = self.txt_transcription.toPlainText()
        translated_text = self.txt_writing_en.toPlainText()
        if not original_text.strip() and not translated_text.strip():
            QMessageBox.information(self, "Vazio", "Não há conteúdo para exportar.")
            return

        src_code = self.audio_engine.source_lang
        tgt_code = self.audio_engine.target_lang
        src_name = AVAILABLE_LANGUAGES.get(src_code, {}).get("name", src_code.upper())
        tgt_name = AVAILABLE_LANGUAGES.get(tgt_code, {}).get("name", tgt_code.upper())

        path, _ = QFileDialog.getSaveFileName(self, "Salvar Transcrição e Tradução", "transcricao_e_traducao.txt", "Text Files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"=== TRANSCRIÇÃO E TRADUÇÃO EM TEMPO REAL ({datetime.now().strftime('%d/%m/%Y %H:%M')}) ===\n\n")
                f.write(f"--- ÁUDIO ORIGINAL ({src_name.upper()}) ---\n")
                f.write(original_text + "\n\n")
                f.write(f"--- TRADUÇÃO ({tgt_name.upper()}) ---\n")
                f.write(translated_text + "\n")
            QMessageBox.information(self, "Salvo", f"Arquivo salvo com sucesso em:\n{os.path.basename(path)}")

    def _export_srt(self):
        if not self.history_items:
            QMessageBox.information(self, "Vazio", "Não há legendas gravadas no histórico para exportar.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Exportar Legenda Sincronizada", "legenda_live.srt", "SubRip Subtitle (*.srt)")
        if not path:
            return

        def format_srt_time(seconds: float) -> str:
            millis = int((seconds - int(seconds)) * 1000)
            seconds = int(seconds)
            mins, secs = divmod(seconds, 60)
            hours, mins = divmod(mins, 60)
            return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

        with open(path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(self.history_items, 1):
                start_sec = max(0.0, item["rel_sec"] - 3.0)
                end_sec = item["rel_sec"] + 1.5
                f.write(f"{idx}\n")
                f.write(f"{format_srt_time(start_sec)} --> {format_srt_time(end_sec)}\n")
                f.write(f"{item['translated']}\n\n")

        QMessageBox.information(self, "Exportado", f"Arquivo de legenda .SRT exportado com sucesso!\n{os.path.basename(path)}")

    def keyPressEvent(self, event):
        if (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)) and event.key() == Qt.Key_I:
            self._swap_languages()
            event.accept()
            return

        if (event.modifiers() & Qt.ControlModifier) and event.key() == Qt.Key_L:
            self._toggle_overlay()
            event.accept()
            return

        if (event.modifiers() & Qt.ControlModifier) and event.key() == Qt.Key_S:
            self._export_srt()
            event.accept()
            return

        if event.key() == Qt.Key_Space:
            self._toggle_microphone()
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.audio_engine.stop_listening()
        self.overlay_window.close()
        event.accept()

    # -------------------------------------------------------------------------
    # ESTILOS E DESIGN SYSTEM
    # -------------------------------------------------------------------------
    def _apply_theme(self):
        qss = """
        QMainWindow, QWidget {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
            font-size: 13px;
        }

        #topBarFrame {
            background-color: #111827;
            border: 1px solid #1e293b;
            border-radius: 12px;
        }

        #btnMicStart {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
            color: white;
            font-size: 13px;
            font-weight: 800;
            border: none;
            border-radius: 10px;
            padding: 8px 16px;
        }
        #btnMicStart:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #047857, stop:1 #059669);
        }

        #btnMicStop {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #dc2626, stop:1 #ef4444);
            color: white;
            font-size: 13px;
            font-weight: 800;
            border: none;
            border-radius: 10px;
            padding: 8px 16px;
        }
        #btnMicStop:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #b91c1c, stop:1 #dc2626);
        }

        #comboQuickDev {
            background-color: #1e293b;
            border: 1.5px solid #334155;
            border-radius: 8px;
            padding: 4px 8px;
            color: #f8fafc;
            font-size: 11px;
            font-weight: 600;
        }
        #comboQuickDev:hover {
            border-color: #38bdf8;
        }

        #btnOverlayTop {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #8b5cf6);
            color: white;
            border: none;
            border-radius: 7px;
            padding: 6px 12px;
            font-weight: bold;
            font-size: 11px;
        }
        #btnOverlayTop:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #7c3aed);
        }

        /* BARRA DE SELEÇÃO E INVERSÃO DE IDIOMAS */
        #langBarFrame {
            background-color: #111827;
            border: 1px solid #1e293b;
            border-radius: 10px;
        }

        #lblLangTitle {
            color: #94a3b8;
            font-size: 12px;
            font-weight: 600;
        }

        #comboLang {
            background-color: #1e293b;
            border: 1.5px solid #334155;
            border-radius: 8px;
            padding: 5px 10px;
            color: #f8fafc;
            font-weight: 600;
            font-size: 12px;
        }
        #comboLang:hover {
            border-color: #38bdf8;
        }
        #comboLang::drop-down {
            border: none;
            padding-right: 6px;
        }
        #comboLang QAbstractItemView {
            background-color: #1e293b;
            color: #f8fafc;
            selection-background-color: #0284c7;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 4px;
        }

        #btnSwapLang {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
            color: #0f172a;
            font-size: 12px;
            font-weight: 800;
            border: none;
            border-radius: 8px;
            padding: 7px 14px;
        }
        #btnSwapLang:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0369a1, stop:1 #0284c7);
            color: #ffffff;
        }
        #btnSwapLang:pressed {
            background: #0284c7;
        }

        #badgeLangDirection {
            background-color: #1e293b;
            border: 1px solid #334155;
            color: #38bdf8;
            border-radius: 6px;
            padding: 5px 12px;
            font-size: 12px;
            font-weight: bold;
            letter-spacing: 0.5px;
        }

        #badgeLatency {
            background-color: #1e293b;
            border: 1px solid #334155;
            color: #10b981;
            border-radius: 6px;
            padding: 5px 10px;
            font-size: 11px;
            font-weight: bold;
        }

        /* PAINÉIS DA TELA DIVIDIDA */
        #panelPT {
            background-color: #111827;
            border: 1.5px solid #1e293b;
            border-radius: 12px;
        }
        #panelEN {
            background-color: #111827;
            border: 1.5px solid #1e293b;
            border-radius: 12px;
        }

        #paneTitlePT {
            color: #38bdf8;
            font-size: 15px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }
        #paneTitleEN {
            color: #a855f7;
            font-size: 15px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }

        #badgeLivePT {
            background-color: #0284c7;
            color: white;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 10px;
            font-weight: bold;
        }
        #badgeLiveEN {
            background-color: #6366f1;
            color: white;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 10px;
            font-weight: bold;
        }

        /* CAIXAS DE TEXTO */
        #txtTranscription {
            background-color: #0b0f19;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 12px;
            color: #e2e8f0;
            font-size: 14px;
            line-height: 1.6;
            selection-background-color: #0284c7;
        }
        #txtTranscription:focus {
            border: 1.5px solid #38bdf8;
        }

        #txtWritingEN {
            background-color: #0b0f19;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 12px;
            color: #38bdf8;
            font-size: 15px;
            font-weight: 600;
            line-height: 1.6;
            selection-background-color: #7c3aed;
        }
        #txtWritingEN:focus {
            border: 1.5px solid #a855f7;
        }

        /* BOTÕES GERAIS */
        QPushButton {
            background-color: #1f2937;
            color: #f1f5f9;
            border: 1px solid #374151;
            border-radius: 7px;
            padding: 6px 12px;
            font-weight: 600;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #374151;
            border-color: #4b5563;
        }
        QPushButton:pressed {
            background-color: #111827;
        }

        /* SLIDERS */
        QSlider::groove:horizontal {
            height: 6px;
            background: #1f2937;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #38bdf8;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #f8fafc;
            border: 2px solid #38bdf8;
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }

        /* CHECKBOX */
        QCheckBox {
            color: #cbd5e1;
            font-weight: 500;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
            border-radius: 4px;
            border: 1px solid #4b5563;
            background-color: #0b0f19;
        }
        QCheckBox::indicator:checked {
            background-color: #38bdf8;
            border-color: #38bdf8;
        }

        /* DIVISOR SPLITTER */
        QSplitter::handle {
            background-color: #1f2937;
            border-radius: 4px;
        }
        QSplitter::handle:hover {
            background-color: #38bdf8;
        }

        /* BARRA INFERIOR */
        #bottomBar {
            background-color: #111827;
            border: 1px solid #1e293b;
            border-radius: 8px;
        }
        #statusLabel {
            color: #94a3b8;
            font-size: 12px;
        }
        #indOff {
            color: #64748b;
            font-size: 11px;
            font-weight: bold;
        }
        #indOn {
            color: #10b981;
            font-size: 11px;
            font-weight: bold;
        }
        """
        self.setStyleSheet(qss)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = RealtimeSplitTranslatorApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
