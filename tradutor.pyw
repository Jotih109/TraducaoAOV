# -*- coding: utf-8 -*-
"""
=============================================================================
 TRADUTOR DE LIVE EM TEMPO REAL (Inglês ➔ Português)
 - Legenda Flutuante: sobreposta na tela, por cima da live, com a tradução
 - Painel Esquerdo: transcrição ao vivo do áudio original captado (inglês)
 - Painel Direito: tradução contínua para português em tempo real
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
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QIcon, QTextCursor

# Módulos do sistema
from modules.settings_manager import SettingsManager
from modules.translator import translator_engine
from modules.tts_engine import tts_engine
from modules.audio_engine import AudioEngine, AVAILABLE_LANGUAGES
from modules.visualizer import AudioVisualizerWidget
from modules.overlay import FloatingOverlayWindow


class SettingsDialog(QDialog):
    """Janela modal para seleção de microfone e voz TTS."""
    def __init__(self, audio_engine, settings, parent=None):
        super().__init__(parent)
        self.audio_engine = audio_engine
        self.settings = settings
        self.setWindowTitle("Configurações de Dispositivos e Voz")
        self.resize(480, 320)
        self.setStyleSheet("""
            QDialog { background-color: #0f172a; color: #f8fafc; font-size: 13px; }
            QGroupBox { border: 1px solid #334155; border-radius: 8px; margin-top: 12px; font-weight: bold; color: #38bdf8; padding: 12px; }
            QComboBox { background-color: #1e293b; border: 1px solid #475569; border-radius: 6px; padding: 6px; color: #f8fafc; }
            QPushButton { background-color: #334155; color: #f8fafc; border-radius: 6px; padding: 8px 14px; font-weight: bold; }
            QPushButton:hover { background-color: #475569; }
        """)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Microfone
        grp_mic = QGroupBox("🎙️ Selecionar Microfone de Entrada", self)
        mic_layout = QVBoxLayout(grp_mic)
        self.combo_mics = QComboBox(grp_mic)
        devices = self.audio_engine.get_input_devices()
        saved_idx = self.settings.get("mic_device_index", None)
        selected_i = 0
        for i, dev in enumerate(devices):
            tag = " (Padrão)" if dev["is_default"] else ""
            self.combo_mics.addItem(f"{dev['index']}: {dev['name']}{tag}", dev["index"])
            if saved_idx is not None and dev["index"] == saved_idx:
                selected_i = i
        if devices:
            self.combo_mics.setCurrentIndex(selected_i)
        mic_layout.addWidget(self.combo_mics)
        layout.addWidget(grp_mic)

        # Voz TTS
        grp_tts = QGroupBox("🔊 Voz para Narrar a Tradução (TTS)", self)
        tts_layout = QVBoxLayout(grp_tts)
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
        tts_layout.addWidget(self.combo_voices)
        layout.addWidget(grp_tts)

        # Botão Salvar
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_save = QPushButton("Salvar e Fechar", self)
        btn_save.clicked.connect(self._save_and_close)
        btn_box.addWidget(btn_save)
        layout.addLayout(btn_box)

    def _save_and_close(self):
        if self.combo_mics.currentIndex() >= 0:
            mic_idx = self.combo_mics.itemData(self.combo_mics.currentIndex())
            self.settings.set("mic_device_index", mic_idx)
        if self.combo_voices.currentIndex() >= 0:
            voice_id = self.combo_voices.itemData(self.combo_voices.currentIndex())
            tts_engine.set_voice(voice_id)
            self.settings.set("tts_voice_id", voice_id)
        self.accept()


class RealtimeSplitTranslatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = SettingsManager()
        self.history_items = []

        # Variáveis de animação de escrita (Typewriter effect)
        self.typewriter_text = ""
        self.typewriter_index = 0
        self.typewriter_timer = QTimer(self)
        self.typewriter_timer.setInterval(22) # ~45 chars/sec
        self.typewriter_timer.timeout.connect(self._on_typewriter_step)

        # Motores
        self.audio_engine = AudioEngine(self.settings, translator_engine, tts_engine)
        self.overlay_window = FloatingOverlayWindow(self.settings)

        # Sinais
        self.audio_engine.level_changed.connect(self._on_audio_level)
        self.audio_engine.speech_started.connect(self._on_speech_started)
        self.audio_engine.speech_ended.connect(self._on_speech_ended)
        self.audio_engine.transcription_ready.connect(self._on_transcription_ready)
        self.audio_engine.translation_ready.connect(self._on_translation_ready)
        self.audio_engine.status_changed.connect(self._on_status_changed)
        self.audio_engine.error_occurred.connect(self._on_error)

        self._init_ui()
        self._apply_theme()

    def _init_ui(self):
        self.resize(1120, 750)
        self.setMinimumSize(880, 580)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(18, 14, 18, 14)
        main_layout.setSpacing(10)

        # =====================================================================
        # 1. BARRA SUPERIOR DE CONTROLE E VISUALIZAÇÃO
        # =====================================================================
        top_bar = QFrame(self)
        top_bar.setObjectName("topBarFrame")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 10, 16, 10)
        top_layout.setSpacing(14)

        # Botão Principal de Iniciar/Parar Gravação
        self.btn_mic = QPushButton("🎙️ INICIAR MICROFONE", top_bar)
        self.btn_mic.setObjectName("btnMicStart")
        self.btn_mic.setMinimumHeight(46)
        self.btn_mic.setMinimumWidth(220)
        self.btn_mic.setCursor(Qt.PointingHandCursor)
        self.btn_mic.setToolTip(
            "Dica: para captar o áudio de uma live com qualidade, use um cabo de áudio\n"
            "virtual (ex: VB-CABLE, gratuito) e selecione-o como microfone em ⚙️ Configurações."
        )
        self.btn_mic.clicked.connect(self._toggle_microphone)
        top_layout.addWidget(self.btn_mic)

        # Visualizador de Áudio Reativo
        self.visualizer = AudioVisualizerWidget(top_bar)
        self.visualizer.setMinimumHeight(55)
        top_layout.addWidget(self.visualizer, 1)

        # Controles Rápidos no Topo
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(4)

        # Sensibilidade
        sens_box = QHBoxLayout()
        sens_box.addWidget(QLabel("Sensibilidade:", top_bar))
        self.slider_sens = QSlider(Qt.Horizontal, top_bar)
        self.slider_sens.setRange(5, 95)
        self.slider_sens.setValue(self.settings.get("mic_sensitivity", 50))
        self.slider_sens.valueChanged.connect(self._on_sens_changed)
        sens_box.addWidget(self.slider_sens)
        self.lbl_sens = QLabel(f"{self.slider_sens.value()}%", top_bar)
        sens_box.addWidget(self.lbl_sens)
        ctrl_layout.addLayout(sens_box)

        # Opção Narrar a tradução em voz alta
        self.chk_tts = QCheckBox("🔊 Narrar tradução em voz alta (TTS)", top_bar)
        self.chk_tts.setChecked(self.settings.get("auto_tts", False))
        self.chk_tts.toggled.connect(lambda val: self.settings.set("auto_tts", val))
        ctrl_layout.addWidget(self.chk_tts)

        top_layout.addLayout(ctrl_layout)

        # Botões de Ação Topo
        top_btns = QVBoxLayout()
        top_btns.setSpacing(4)

        self.btn_overlay = QPushButton("🪟 Legenda Flutuante (Recomendado)", top_bar)
        self.btn_overlay.setObjectName("btnOverlayTop")
        self.btn_overlay.setToolTip("Sobrepõe a tradução por cima da tela da live, como uma legenda.")
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

        # Entrada (Áudio da Live)
        lbl_src_title = QLabel("🎙️ Idioma de Entrada (Áudio):", lang_bar)
        lbl_src_title.setObjectName("lblLangTitle")
        self.combo_src_lang = QComboBox(lang_bar)
        self.combo_src_lang.setObjectName("comboLang")
        self.combo_src_lang.setMinimumWidth(165)

        for code, info in AVAILABLE_LANGUAGES.items():
            self.combo_src_lang.addItem(f"{info['flag']} {info['name']}", code)

        # Botão Central de Inverter
        self.btn_swap_lang = QPushButton("⇄ INVERTER IDIOMAS", lang_bar)
        self.btn_swap_lang.setObjectName("btnSwapLang")
        self.btn_swap_lang.setCursor(Qt.PointingHandCursor)
        self.btn_swap_lang.setToolTip("Inverter idioma de entrada e idioma de saída (Atalho: Ctrl+I ou Alt+I)")
        self.btn_swap_lang.clicked.connect(self._swap_languages)

        # Saída (Legenda / Tradução)
        lbl_tgt_title = QLabel("🌐 Idioma de Saída (Legenda):", lang_bar)
        lbl_tgt_title.setObjectName("lblLangTitle")
        self.combo_tgt_lang = QComboBox(lang_bar)
        self.combo_tgt_lang.setObjectName("comboLang")
        self.combo_tgt_lang.setMinimumWidth(165)

        for code, info in AVAILABLE_LANGUAGES.items():
            self.combo_tgt_lang.addItem(f"{info['flag']} {info['name']}", code)

        # Badge indicador da direção atual
        self.lbl_lang_direction = QLabel("", lang_bar)
        self.lbl_lang_direction.setObjectName("badgeLangDirection")

        lang_layout.addWidget(lbl_src_title)
        lang_layout.addWidget(self.combo_src_lang)
        lang_layout.addWidget(self.btn_swap_lang)
        lang_layout.addWidget(lbl_tgt_title)
        lang_layout.addWidget(self.combo_tgt_lang)
        lang_layout.addStretch()
        lang_layout.addWidget(self.lbl_lang_direction)
        main_layout.addWidget(lang_bar)

        # =====================================================================
        # 2. TELA DIVIDIDA EM DOIS PAINÉIS (QSplitter)
        # =====================================================================
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setObjectName("mainSplitter")
        self.splitter.setHandleWidth(8)

        # ---------------------------------------------------------------------
        # PAINEL DA ESQUERDA: ÁUDIO ORIGINAL CAPTADO DA LIVE
        # ---------------------------------------------------------------------
        panel_pt = QFrame(self.splitter)
        panel_pt.setObjectName("panelPT")
        layout_pt = QVBoxLayout(panel_pt)
        layout_pt.setContentsMargins(14, 12, 14, 12)
        layout_pt.setSpacing(10)

        # Header Esquerdo
        head_pt = QHBoxLayout()
        self.lbl_head_pt = QLabel("🎧 Áudio Original", panel_pt)
        self.lbl_head_pt.setObjectName("paneTitlePT")
        self.badge_pt = QLabel("AO VIVO", panel_pt)
        self.badge_pt.setObjectName("badgeLivePT")
        head_pt.addWidget(self.lbl_head_pt)
        head_pt.addWidget(self.badge_pt)
        head_pt.addStretch()
        layout_pt.addLayout(head_pt)

        # Subtítulo explicativo
        self.lbl_desc_pt = QLabel("O áudio da live captado pelo microfone é transcrito abaixo:", panel_pt)
        self.lbl_desc_pt.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout_pt.addWidget(self.lbl_desc_pt)

        # Caixa de Texto Principal do Áudio Original
        self.txt_transcription = QTextEdit(panel_pt)
        self.txt_transcription.setObjectName("txtTranscription")
        self.txt_transcription.setReadOnly(True)
        layout_pt.addWidget(self.txt_transcription, 1)

        # Barra de Ações do Painel Esquerdo
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

        # ---------------------------------------------------------------------
        # PAINEL DA DIREITA: TRADUÇÃO EM TEMPO REAL (LEGENDA DA LIVE)
        # ---------------------------------------------------------------------
        panel_en = QFrame(self.splitter)
        panel_en.setObjectName("panelEN")
        layout_en = QVBoxLayout(panel_en)
        layout_en.setContentsMargins(14, 12, 14, 12)
        layout_en.setSpacing(10)

        # Header Direito
        head_en = QHBoxLayout()
        self.lbl_head_en = QLabel("🌐 Tradução em Tempo Real", panel_en)
        self.lbl_head_en.setObjectName("paneTitleEN")
        self.badge_en = QLabel("TRADUÇÃO", panel_en)
        self.badge_en.setObjectName("badgeLiveEN")
        head_en.addWidget(self.lbl_head_en)
        head_en.addWidget(self.badge_en)
        head_en.addStretch()
        layout_en.addLayout(head_en)

        # Subtítulo explicativo
        self.lbl_desc_en = QLabel("A tradução é escrita automaticamente em tempo real:", panel_en)
        self.lbl_desc_en.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout_en.addWidget(self.lbl_desc_en)

        # Caixa de Texto Principal da Tradução
        self.txt_writing_en = QTextEdit(panel_en)
        self.txt_writing_en.setObjectName("txtWritingEN")
        self.txt_writing_en.setReadOnly(True)
        layout_en.addWidget(self.txt_writing_en, 1)

        # Barra de Ações do Painel Direito
        bar_en = QHBoxLayout()
        self.lbl_stats_en = QLabel("0 palavras traduzidas", panel_en)
        self.lbl_stats_en.setStyleSheet("color: #64748b; font-size: 11px;")
        bar_en.addWidget(self.lbl_stats_en)
        bar_en.addStretch()

        self.btn_speak_last = QPushButton("🔊 Ouvir Tradução", panel_en)
        self.btn_speak_last.clicked.connect(self._speak_all_english)
        self.btn_copy_en = QPushButton("📋 Copiar Tradução", panel_en)
        self.btn_copy_en.clicked.connect(lambda: self._copy_text(self.txt_writing_en.toPlainText()))
        btn_save = QPushButton("💾 Salvar Arquivo", panel_en)
        btn_save.clicked.connect(self._export_both)

        bar_en.addWidget(self.btn_speak_last)
        bar_en.addWidget(self.btn_copy_en)
        bar_en.addWidget(btn_save)
        layout_en.addLayout(bar_en)

        # Adicionar painéis ao Splitter com divisão de 50% / 50%
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

        self.lbl_status = QLabel("🟢 Pronto para traduzir a live. Clique em 'INICIAR MICROFONE' ou pressione Espaço.", self)
        self.lbl_status.setObjectName("statusLabel")

        self.lbl_indicator = QLabel("● MICROFONE DESLIGADO", self)
        self.lbl_indicator.setObjectName("indOff")

        bottom_layout.addWidget(self.lbl_status)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.lbl_indicator)
        main_layout.addWidget(bottom_bar)

        # Inicializar seleção salva de idiomas
        saved_src = self.settings.get("source_lang", "en")
        saved_tgt = self.settings.get("target_lang", "pt")
        self._set_combo_code(self.combo_src_lang, saved_src)
        self._set_combo_code(self.combo_tgt_lang, saved_tgt)

        self.combo_src_lang.currentIndexChanged.connect(self._on_combo_language_changed)
        self.combo_tgt_lang.currentIndexChanged.connect(self._on_combo_language_changed)

        self._update_languages(saved_src, saved_tgt, save_settings=False)

    # -------------------------------------------------------------------------
    # GERENCIAMENTO E INVERSÃO DE IDIOMAS
    # -------------------------------------------------------------------------
    def _set_combo_code(self, combo: QComboBox, code: str):
        for i in range(combo.count()):
            if combo.itemData(i) == code:
                combo.setCurrentIndex(i)
                break

    def _swap_languages(self):
        """Inverte imediatamente os idiomas de entrada e saída (ex: EN ➔ PT vira PT ➔ EN)."""
        src_idx = self.combo_src_lang.currentIndex()
        tgt_idx = self.combo_tgt_lang.currentIndex()
        src_code = self.combo_src_lang.itemData(src_idx) if src_idx >= 0 else "en"
        tgt_code = self.combo_tgt_lang.itemData(tgt_idx) if tgt_idx >= 0 else "pt"

        # Se forem iguais, nada a inverter
        if src_code == tgt_code:
            return

        # Bloquear sinais para atualização atômica
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
        """Sincroniza todos os componentes do sistema com os novos idiomas selecionados."""
        if save_settings:
            self.settings.set("source_lang", src_code)
            self.settings.set("target_lang", tgt_code)

        # Atualizar AudioEngine, Overlay e TTS
        self.audio_engine.set_languages(src_code, tgt_code)
        self.overlay_window.set_languages(src_code, tgt_code)
        tts_engine.auto_select_voice_for_language(tgt_code)

        src_info = AVAILABLE_LANGUAGES.get(src_code, {"name": src_code, "flag": "🌐"})
        tgt_info = AVAILABLE_LANGUAGES.get(tgt_code, {"name": tgt_code, "flag": "🌐"})

        # 1. Badge de Direção
        self.lbl_lang_direction.setText(f"{src_info['flag']} {src_code.upper()} ➔ {tgt_info['flag']} {tgt_code.upper()}")

        # 2. Título da Janela
        self.setWindowTitle(f"Tradutor em Tempo Real • {src_info['flag']} {src_info['name']} ➔ {tgt_info['flag']} {tgt_info['name']}")

        # 3. Painel Esquerdo (Áudio Original)
        self.lbl_head_pt.setText(f"🎧 Áudio Original ({src_info['name']})")
        self.lbl_desc_pt.setText(f"O áudio captado ({src_info['name']}) é transcrito abaixo:")
        self.txt_transcription.setPlaceholderText(
            f"Clique em 'INICIAR MICROFONE' acima...\n\nO áudio captado ({src_info['name']}) aparecerá aqui continuamente."
        )
        self.btn_copy_pt.setText(f"📋 Copiar {src_info['name']}")

        # 4. Painel Direito (Tradução)
        self.lbl_head_en.setText(f"{tgt_info['flag']} Tradução em {tgt_info['name']} (Tempo Real)")
        self.lbl_desc_en.setText(f"A tradução em {tgt_info['name']} é escrita automaticamente em tempo real:")
        self.txt_writing_en.setPlaceholderText(
            f"A tradução em {tgt_info['name']} será escrita aqui em tempo real conforme o áudio é detectado..."
        )
        self.btn_copy_en.setText(f"📋 Copiar {tgt_info['name']}")
        self.btn_speak_last.setText(f"🔊 Ouvir em {tgt_info['name']}")

        if save_settings:
            self.lbl_status.setText(f"🔄 Idiomas invertidos/atualizados: {src_info['flag']} {src_info['name']} ➔ {tgt_info['flag']} {tgt_info['name']}")

    # -------------------------------------------------------------------------
    # CONTROLE DE MICROFONE & VOZ
    # -------------------------------------------------------------------------
    def _toggle_microphone(self):
        if not self.audio_engine.is_recording:
            self.audio_engine.start_listening()
            self.btn_mic.setText("🛑 PARAR MICROFONE")
            self.btn_mic.setObjectName("btnMicStop")
            self.lbl_indicator.setText("● OUVINDO...")
            self.lbl_indicator.setObjectName("indOn")
            self.badge_pt.setText("ESCUTANDO")
            self.badge_pt.setStyleSheet("background-color: #10b981; color: white;")
            # A legenda flutuante é a interface principal para assistir à live traduzida
            if not self.overlay_window.isVisible():
                self._toggle_overlay()
        else:
            self.audio_engine.stop_listening()
            self.btn_mic.setText("🎙️ INICIAR MICROFONE")
            self.btn_mic.setObjectName("btnMicStart")
            self.lbl_indicator.setText("● MICROFONE DESLIGADO")
            self.lbl_indicator.setObjectName("indOff")
            self.badge_pt.setText("AO VIVO")
            self.badge_pt.setStyleSheet("background-color: #0284c7; color: white;")
        self._apply_theme()

    def _on_audio_level(self, level, waveform):
        self.visualizer.set_level(level, waveform)

    def _on_speech_started(self):
        src_name = AVAILABLE_LANGUAGES.get(self.audio_engine.source_lang, {}).get("name", "áudio")
        self.lbl_status.setText(f"🎧 Áudio detectado... transcrevendo em {src_name.lower()}...")
        self.lbl_indicator.setText("● GRAVANDO")
        self.badge_pt.setStyleSheet("background-color: #f59e0b; color: #000; font-weight: bold;")

    def _on_speech_ended(self):
        tgt_name = AVAILABLE_LANGUAGES.get(self.audio_engine.target_lang, {}).get("name", "destino")
        self.lbl_status.setText(f"⚡ Traduzindo para {tgt_name.lower()} e escrevendo...")
        self.lbl_indicator.setText("● TRADUZINDO")
        self.badge_en.setText("ESCREVENDO...")
        self.badge_en.setStyleSheet("background-color: #8b5cf6; color: white;")

    def _on_transcription_ready(self, text_original):
        # Adicionar áudio original transcrito ao painel esquerdo
        current_pt = self.txt_transcription.toPlainText().strip()
        timestamp = datetime.now().strftime("%H:%M:%S")

        if current_pt:
            new_pt = f"{current_pt}\n[{timestamp}] {text_original}"
        else:
            new_pt = f"[{timestamp}] {text_original}"

        self.txt_transcription.setPlainText(new_pt)
        # Rolar automaticamente para o final
        self.txt_transcription.moveCursor(QTextCursor.End)

        # Atualizar contadores
        chars = len(new_pt)
        words = len(new_pt.split())
        self.lbl_stats_pt.setText(f"{words} palavras • {chars} caracteres")

    def _on_translation_ready(self, text_original, text_translated):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_items.append({"time": timestamp, "original": text_original, "translated": text_translated})

        # Atualizar HUD de legenda flutuante
        self.overlay_window.update_subtitles(text_original, text_translated)

        # Se houver digitação em andamento, descarregar restante instantaneamente
        if self.typewriter_timer.isActive():
            self.typewriter_timer.stop()
            if self.typewriter_index < len(self.typewriter_text):
                self.txt_writing_en.insertPlainText(self.typewriter_text[self.typewriter_index:])
                self.txt_writing_en.moveCursor(QTextCursor.End)

        # Iniciar escrita ultra-rápida
        entry_header = f"\n[{timestamp}] " if self.txt_writing_en.toPlainText().strip() else f"[{timestamp}] "
        self.typewriter_text = entry_header + text_translated
        self.typewriter_index = 0
        self.typewriter_timer.start(10) # 10ms por caractere (100 chars/seg)

        self.lbl_status.setText("✅ Frase transcrita e traduzida!")
        self.badge_en.setText("TRADUÇÃO")
        self.badge_en.setStyleSheet("background-color: #6366f1; color: white;")

    def _on_typewriter_step(self):
        """Escreve o texto traduzido caractere por caractere para efeito dinâmico."""
        if self.typewriter_index < len(self.typewriter_text):
            # Escrever em blocos de até 2 caracteres para máxima fluidez
            end_idx = min(len(self.typewriter_text), self.typewriter_index + 2)
            chunk = self.typewriter_text[self.typewriter_index:end_idx]
            self.txt_writing_en.insertPlainText(chunk)
            self.txt_writing_en.moveCursor(QTextCursor.End)
            self.typewriter_index = end_idx
        else:
            self.typewriter_timer.stop()
            # Atualizar contador de palavras traduzidas
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
            self.btn_overlay.setText("🪟 Legenda Flutuante (Recomendado)")
        else:
            self.overlay_window.show()
            self.btn_overlay.setText("🪟 Ocultar Legenda")

    def _open_settings(self):
        dlg = SettingsDialog(self.audio_engine, self.settings, self)
        dlg.exec_()

    def _speak_all_english(self):
        text = self.txt_writing_en.toPlainText()
        if text.strip():
            # Limpar timestamps [00:00:00] antes de falar
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
            self.lbl_status.setText("📋 Texto copiado para a área de transferência!")

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

    def keyPressEvent(self, event):
        # Tecla de Atalho: Ctrl+I ou Alt+I para inverter idiomas rapidamente
        if (event.modifiers() & (Qt.ControlModifier | Qt.AltModifier)) and event.key() == Qt.Key_I:
            self._swap_languages()
            event.accept()
            return

        # Tecla de Atalho: Espaço inicia/pausa microfone se não estiver editando texto
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
            font-family: 'Segoe UI', 'Inter', sans-serif;
            font-size: 13px;
        }

        #topBarFrame {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
        }

        #btnMicStart {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981);
            color: white;
            font-size: 14px;
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
            font-size: 14px;
            font-weight: 800;
            border: none;
            border-radius: 10px;
            padding: 8px 16px;
        }
        #btnMicStop:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #b91c1c, stop:1 #dc2626);
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
            border: 1px solid #1f2937;
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
            font-size: 16px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }
        #paneTitleEN {
            color: #a855f7;
            font-size: 16px;
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

        /* CAIXAS DE TEXTO GRANDES */
        #txtTranscription {
            background-color: #0b0f19;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 14px;
            color: #e2e8f0;
            font-size: 15px;
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
            padding: 14px;
            color: #38bdf8;
            font-size: 16px;
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
            padding: 7px 13px;
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
            border: 1px solid #1f2937;
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
