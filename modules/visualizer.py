# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QPainterPath, QFont
from PyQt5.QtCore import Qt, QTimer, QPointF
import numpy as np


class AudioVisualizerWidget(QWidget):
    """
    Widget visualizador de áudio de alta precisão e performance:
    - Ondas senoidais neon curvas ultra-suaves
    - Medidor VU profissional com marcador visual do LIMIAR DE VOZ (Threshold Line)
    - Indicador em tempo real: mostra exatamente quando o som ultrapassa o limiar
    - Peak-Hold com decaimento gradual
    - 0% de uso de CPU em modo Standby (pausa animações quando desligado)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.level = 0.0
        self.target_level = 0.0
        self.peak_level = 0.0
        self.peak_hold_frames = 0
        self.waveform = [0.0] * 32
        self.phase = 0.0
        self.threshold = 0.005
        self.target_threshold = 0.005
        self.is_active = False
        self.is_recording = False
        
        # Timer de animação suave a 60 FPS
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_step)

    def set_recording(self, recording: bool):
        self.is_recording = recording
        if recording:
            if not self.anim_timer.isActive():
                self.anim_timer.start(16)
        else:
            self.anim_timer.stop()
            self.level = 0.0
            self.target_level = 0.0
            self.peak_level = 0.0
            self.waveform = [0.0] * 32
            self.is_active = False
            self.update()

    def set_level(self, level: float, waveform_samples: list, threshold: float = 0.005):
        # Ganho visual para a barra VU ficar nítida e responsiva
        self.target_level = max(0.0, min(1.0, level * 5.0))
        self.target_threshold = max(0.01, min(0.95, threshold * 5.0))
        
        if waveform_samples and len(waveform_samples) > 0:
            self.waveform = waveform_samples
        
        self.is_active = (level > threshold)

        if self.target_level > self.peak_level:
            self.peak_level = self.target_level
            self.peak_hold_frames = 22
        
        if not self.anim_timer.isActive() and self.is_recording:
            self.anim_timer.start(16)

    def _animate_step(self):
        # Interpolação suave
        self.level += (self.target_level - self.level) * 0.30
        self.threshold += (self.target_threshold - self.threshold) * 0.20
        self.phase += 0.09
        if self.phase > 6.283:
            self.phase -= 6.283

        if self.peak_hold_frames > 0:
            self.peak_hold_frames -= 1
        else:
            self.peak_level = max(0.0, self.peak_level - 0.018)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cy = h / 2.0

        # Fundo escuro elegante
        bg_brush = QBrush(QColor(11, 15, 25, 240))
        painter.setBrush(bg_brush)
        painter.setPen(QPen(QColor(30, 41, 59), 1))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)

        # Standby se o microfone estiver desligado (0% CPU)
        if not self.is_recording:
            painter.setPen(QPen(QColor(51, 65, 85, 90), 1, Qt.DashLine))
            painter.drawLine(12, int(cy), w - 12, int(cy))

            font = QFont("Segoe UI", 9)
            painter.setFont(font)
            painter.setPen(QColor(100, 116, 139, 180))
            painter.drawText(self.rect(), Qt.AlignCenter, "● MICROFONE EM ESPERA • CLIQUE PARA INICIAR")
            return

        # Linha central guia
        painter.setPen(QPen(QColor(51, 65, 85, 70), 1, Qt.DashLine))
        painter.drawLine(12, int(cy), w - 12, int(cy))

        # Onda Sonora Líquida
        pts_count = len(self.waveform)
        if pts_count > 1:
            dx = (w - 30) / float(pts_count - 1)
            amp_mult = max(0.10, self.level * 4.0)

            path = QPainterPath()
            start_y = cy - (self.waveform[0] * (h * 0.38) * amp_mult)
            path.moveTo(15, start_y)

            for i in range(1, pts_count):
                prev_x = 15 + (i - 1) * dx
                prev_y = cy - (self.waveform[i - 1] * (h * 0.38) * amp_mult + np.sin(self.phase + (i - 1) * 0.35) * self.level * (h * 0.10))
                curr_x = 15 + i * dx
                curr_y = cy - (self.waveform[i] * (h * 0.38) * amp_mult + np.sin(self.phase + i * 0.35) * self.level * (h * 0.10))
                
                prev_y = max(6, min(h - 16, prev_y))
                curr_y = max(6, min(h - 16, curr_y))

                mid_x = (prev_x + curr_x) / 2.0
                path.cubicTo(mid_x, prev_y, mid_x, curr_y, curr_x, curr_y)

            # Se fala estiver detectada, cor mais vibrante
            grad = QLinearGradient(0, 0, w, 0)
            if self.is_active:
                grad.setColorAt(0.0, QColor(52, 211, 153, 240))  # Emerald 400
                grad.setColorAt(0.5, QColor(56, 189, 248, 255))  # Cyan 400
                grad.setColorAt(1.0, QColor(168, 85, 247, 240))  # Purple 400
            else:
                grad.setColorAt(0.0, QColor(56, 189, 248, 140))
                grad.setColorAt(0.5, QColor(99, 102, 241, 160))
                grad.setColorAt(1.0, QColor(147, 51, 234, 140))

            if self.level > 0.03:
                glow_pen = QPen(QColor(56, 189, 248, 50 if not self.is_active else 90), 5.0)
                glow_pen.setCapStyle(Qt.RoundCap)
                glow_pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(glow_pen)
                painter.drawPath(path)

            pen = QPen(QBrush(grad), 2.2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path)

        # Barra de Nível VU no Rodapé
        bar_w = w - 30
        bar_h = 5
        bar_x = 15
        bar_y = h - 11

        # Calha de fundo
        painter.setBrush(QBrush(QColor(30, 41, 59)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 2, 2)

        # Preenchimento dinâmico
        current_vu_w = int(bar_w * min(1.0, self.level))
        if current_vu_w > 0:
            vu_grad = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
            if self.is_active:
                vu_grad.setColorAt(0.0, QColor(16, 185, 129))   # Verde vibrante
                vu_grad.setColorAt(0.60, QColor(56, 189, 248))  # Cyan
                vu_grad.setColorAt(0.85, QColor(245, 158, 11))  # Amarelo
                vu_grad.setColorAt(1.0, QColor(239, 68, 68))    # Vermelho
            else:
                vu_grad.setColorAt(0.0, QColor(71, 85, 105))    # Slate discreto quando abaixo do limiar
                vu_grad.setColorAt(1.0, QColor(100, 116, 139))

            painter.setBrush(QBrush(vu_grad))
            painter.drawRoundedRect(bar_x, bar_y, current_vu_w, bar_h, 2, 2)

        # Peak-Hold (pico máximo)
        peak_x = bar_x + int(bar_w * min(1.0, self.peak_level))
        if peak_x > bar_x + 2:
            painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
            painter.drawLine(peak_x, bar_y - 1, peak_x, bar_y + bar_h + 1)

        # Marcador Visual do Limiar (Threshold Marker): mostra a linha onde a voz dispara
        thresh_x = bar_x + int(bar_w * min(0.95, self.threshold))
        painter.setPen(QPen(QColor(245, 158, 11, 230), 1.5, Qt.DashLine)) # Linha amarela indicativa
        painter.drawLine(thresh_x, bar_y - 3, thresh_x, bar_y + bar_h + 3)

        # Indicador de fala detectada no canto superior
        if self.is_active:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(16, 185, 129, 230)))
            painter.drawEllipse(w - 55, 8, 7, 7)
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(52, 211, 153))
            painter.drawText(w - 44, 15, "VOZ")
