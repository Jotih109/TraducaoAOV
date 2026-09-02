# -*- coding: utf-8 -*-
import json
import os

DEFAULT_SETTINGS = {
    "source_lang": "en",
    "target_lang": "pt",
    "mic_device_index": None,
    "mic_sensitivity": 75,  # 0 to 100 (75 is ideal for natural speech)
    "auto_tts": False,
    "tts_voice_id": "",
    "tts_rate": 160,
    "tts_volume": 0.9,
    "clipboard_monitor": False,
    "overlay_font_size": 22,
    "overlay_opacity": 85,
    "overlay_bg_color": "#111827",
    "overlay_text_color": "#38bdf8",
    "overlay_pt_color": "#9ca3af",
    "overlay_show_original": True,
    "overlay_click_through": False,
    "overlay_autohide_seconds": 8,
    "overlay_pos_x": None,
    "overlay_pos_y": None,
    "overlay_width": 750,
    "overlay_height": 130,
    "always_on_top": True,
    "max_speech_duration": 6.0,
    "highpass_filter": True
}

class SettingsManager:
    def __init__(self, filepath="settings.json"):
        # Put settings next to the app
        base_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(base_dir)
        self.filepath = os.path.join(app_dir, filepath)
        self.settings = DEFAULT_SETTINGS.copy()
        self.load()

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    self.settings.update(saved)
            except Exception as e:
                print(f"Erro ao carregar settings: {e}")
        return self.settings

    def save(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Erro ao salvar settings: {e}")

    def get(self, key, default=None):
        if key in self.settings:
            return self.settings[key]
        return DEFAULT_SETTINGS.get(key, default)

    def set(self, key, value):
        self.settings[key] = value
        self.save()

    def update_multiple(self, mapping: dict):
        self.settings.update(mapping)
        self.save()

