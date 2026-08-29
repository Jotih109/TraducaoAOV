# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient, QRadialGradient, QPainterPath
from PyQt5.QtCore import Qt, QTimer, QPointF
import numpy as np

class AudioVisualizerWidget(QWidget):
    """
    Widget visualizador de áudio de alta performance com ondas neon osciloscópio,
    gradientes dinâmicos e medidor de nível VU estético.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)
        self.level = 0.0
        self.target_level = 0.0
        self.waveform = [0.0] * 32
        self.phase = 0.0
        self.is_active = False
        
        # Timer de animação suave a 60 FPS
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_step)
        self.anim_timer.start(16)

    def set_level(self, level: float, waveform_samples: list):
        self.target_level = max(0.0, min(1.0, level * 2.5))
        if waveform_samples and len(waveform_samples) > 0:
            self.waveform = waveform_samples
        self.is_active = (self.target_level > 0.02)

    def _animate_step(self):
        # Interpolação suave do nível (lerp)
        self.level += (self.target_level - self.level) * 0.25
        self.phase += 0.08
        if self.phase > 6.283:
            self.phase -= 6.283
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        cy = h / 2.0

        # Fundo escuro elegante com cantos arredondados
        bg_brush = QBrush(QColor(15, 23, 42, 220)) # Dark Slate
        painter.setBrush(bg_brush)
        painter.setPen(QPen(QColor(30, 41, 59), 1))
        painter.drawRoundedRect(0, 0, w, h, 10, 10)

        # Linha central guia sutil
        painter.setPen(QPen(QColor(51, 65, 85, 90), 1, Qt.DashLine))
        painter.drawLine(10, int(cy), w - 10, int(cy))

        # Se houver fala/áudio, desenhar onda viva e brilhante
        pts_count = len(self.waveform)
        if pts_count > 1:
            dx = (w - 30) / float(pts_count - 1)
            
            # Onda principal (Neon Cyan/Blue)
            path_top = QPainterPath()
            path_top.moveTo(15, cy)

            amp_mult = max(0.1, self.level * 4.5)

            for i in range(pts_count):
                x = 15 + i * dx
                # Mistura sinal de áudio real com senoide suave decorativa
                synth = np.sin(self.phase + i * 0.35) * self.level * (h * 0.15)
                val = self.waveform[i] * (h * 0.42) * amp_mult + synth
                y = cy - val
                # Restringir aos limites da caixa
                y = max(8, min(h - 8, y))
                path_top.lineTo(x, y)

            # Gradiente de brilho neon
            grad = QLinearGradient(0, 0, w, 0)
            grad.setColorAt(0.0, QColor(56, 189, 248, 220))   # Cyan 400
            grad.setColorAt(0.5, QColor(99, 102, 241, 255))   # Indigo 500
            grad.setColorAt(1.0, QColor(168, 85, 247, 220))   # Purple 500

            pen = QPen(QBrush(grad), 2.5)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(path_top)

            # Onda espelhada sutil no fundo (Glow effect)
            if self.level > 0.05:
                glow_path = QPainterPath()
                glow_path.moveTo(15, cy)
                for i in range(pts_count):
                    x = 15 + i * dx
                    val = -self.waveform[i] * (h * 0.28) * amp_mult
                    y = cy - val
                    glow_path.lineTo(x, y)

                glow_pen = QPen(QColor(56, 189, 248, 80), 1.5)
                painter.setPen(glow_pen)
                painter.drawPath(glow_path)

        # Barra de Nível VU Estética no rodapé
        bar_w = w - 40
        bar_h = 5
        bar_x = 20
        bar_y = h - 14

        # Fundo da barra VU
        painter.setBrush(QBrush(QColor(30, 41, 59)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)

        # Preenchimento dinâmico do medidor
        current_vu_w = int(bar_w * min(1.0, self.level))
        if current_vu_w > 0:
            vu_grad = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
            vu_grad.setColorAt(0.0, QColor(16, 185, 129))   # Verde Emerald
            vu_grad.setColorAt(0.65, QColor(56, 189, 248))  # Cyan
            vu_grad.setColorAt(0.85, QColor(245, 158, 11))  # Amarelo/Laranja
            vu_grad.setColorAt(1.0, QColor(239, 68, 68))    # Vermelho Red

            painter.setBrush(QBrush(vu_grad))
            painter.drawRoundedRect(bar_x, bar_y, current_vu_w, bar_h, 3, 3)
