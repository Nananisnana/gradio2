"""Gradio UI ve olay bağlama. Çalıştırma: python -m app.main [--config ...]"""
from __future__ import annotations

import argparse
import sys

import gradio as gr

from .annotate import annotate_frame
from .backends import ModelManager, ModelStatus
from .config import AppConfig, ConfigError, load_config
from .frames import FrameExtractionError, extract_frame
from .ui_text import TR
from .vqa_client import VqaError

_BADGE_STYLE = {
    ModelStatus.ACTIVE: ("#1a7f37", TR["status_active"]),
    ModelStatus.STARTING: ("#b58105", TR["status_starting"]),
    ModelStatus.STOPPED: ("#c93c37", TR["status_stopped"]),
    ModelStatus.UNKNOWN: ("#6e7781", TR["status_unknown"]),
}

# ANA AKIŞ: Enter/Sor anında oynatıcının o anki saniyesi yakalanır. Bu JS,
# on_ask ile AYNI olayda ön-işleme olarak koşar (Gradio: js'in döndürdüğü dizi
# fn'e girdi olur — js-only olay + .then zinciri Gradio 5.31'de tetiklenmediği
# için bu desen şart). Saniyeyi fn'e taşıyan Number bileşeni görünmezdir;
# oynatıcı yoksa/okunamazsa 0. saniyeye düşülür. elem_id="video_in" sabitine
# ve pinlenmiş Gradio sürümünün DOM yapısına bağlıdır.
JS_FREEZE_TIME = """
(video, second, model, prompt) => {
  const v = document.querySelector('#video_in video');
  second = (v && !isNaN(v.currentTime)) ? Math.round(v.currentTime * 10) / 10 : 0;
  return [video, second, model, prompt];
}
"""


def render_status(statuses: dict, cfg: AppConfig, extra_line: str | None = None) -> str:
    rows = []
    for key, model in cfg.models.items():
        color, label = _BADGE_STYLE[statuses.get(key, ModelStatus.UNKNOWN)]
        rows.append(
            f'<div style="margin:2px 0">'
            f'<span style="background:{color};color:#fff;border-radius:4px;'
            f'padding:1px 8px;font-size:0.8em;font-weight:600">{label}</span> '
            f'<span>{model.display_name}</span></div>'
        )
    if extra_line:
        rows.append(f'<div style="margin-top:6px;font-style:italic">{extra_line}</div>')
    return "".join(rows)


def build_app(cfg: AppConfig, mgr: ModelManager) -> gr.Blocks:
    def refresh_status():
        if mgr.switching:
            # aktivasyon generator'ının yazdığı ilerleme satırını ezme
            return gr.update()
        return render_status(mgr.statuses(), cfg)

    def on_model_change(key: str):
        # mock modda aktivasyon yok; dual/switch modda seçilen model otomatik
        # ayağa kaldırılır (dual: yalnız aynı motorun slotu değişir)
        if cfg.mode == "mock":
            yield render_status(mgr.statuses(), cfg), gr.update(), gr.update()
            return
        yield (
            render_status(mgr.statuses(), cfg),
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
        for ev in mgr.activate(key):
            yield (
                render_status(mgr.statuses(), cfg, extra_line=ev.message),
                gr.update(),
                gr.update(),
            )
        yield (
            render_status(mgr.statuses(), cfg),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    def on_ask(video_path, second, key, prompt):
        if not video_path:
            return None, TR["err_no_video"]
        prompt = (prompt or "").strip()
        if not prompt:
            return None, TR["err_no_prompt"]
        if mgr.switching:
            return None, TR["err_switch_in_progress"]
        if cfg.mode != "mock" and not mgr.is_active(key):
            status = mgr.statuses().get(key, ModelStatus.UNKNOWN)
            _, label = _BADGE_STYLE[status]
            return None, TR["err_model_not_active"].format(status=label)
        try:
            fr = extract_frame(
                video_path,
                float(second or 0.0),
                max_dim=cfg.frame.max_dim,
                jpeg_quality=cfg.frame.jpeg_quality,
            )
        except FrameExtractionError as e:
            return None, e.user_message
        try:
            res = mgr.client_for(key).ask(prompt, fr.data_uri)
        except VqaError as e:
            # kare yine gösterilir: model düşse bile akış görünür kalır
            return fr.frame_rgb, e.user_message
        # grounding: model kutu döndürdüyse frame işaretlenir, yoksa düz frame
        out_frame = annotate_frame(fr.frame_rgb, res.boxes)
        answer = TR["answer_prefix"].format(second=fr.actual_second, answer=res.answer)
        if res.boxes:
            answer += TR["boxes_note"].format(n=len(res.boxes))
        return out_frame, answer

    with gr.Blocks(title=TR["app_title"]) as demo:
        gr.Markdown(f"## {TR['app_title']}")
        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label=TR["video_label"], sources=["upload"], elem_id="video_in")
                # görünmez taşıyıcı: JS_FREEZE_TIME'ın yakaladığı saniye buradan fn'e gider
                second_num = gr.Number(value=0, visible=False)
            with gr.Column():
                model_dd = gr.Dropdown(
                    label=TR["backend_label"],
                    choices=[(m.display_name, key) for key, m in cfg.models.items()],
                    value=cfg.default_model,
                )
                status_html = gr.HTML()
                prompt_tb = gr.Textbox(label=TR["prompt_label"])
                ask_btn = gr.Button(TR["ask_btn"], variant="primary")
        with gr.Row():
            frame_out = gr.Image(label=TR["frame_label"], interactive=False)
            answer_out = gr.Textbox(label=TR["answer_label"], lines=6, interactive=False)

        timer = gr.Timer(cfg.status_refresh_s)

        demo.load(refresh_status, None, status_html)
        timer.tick(refresh_status, None, status_html, concurrency_limit=1)
        model_dd.change(
            on_model_change, model_dd, [status_html, ask_btn, model_dd],
            concurrency_limit=1,
        )
        ask_inputs = [video_in, second_num, model_dd, prompt_tb]
        ask_outputs = [frame_out, answer_out]
        # JS_FREEZE_TIME aynı olayda ön-işleme: Enter/Sor anındaki saniye fn'e gider
        ask_btn.click(
            on_ask, ask_inputs, ask_outputs, js=JS_FREEZE_TIME, concurrency_limit=1
        )
        prompt_tb.submit(
            on_ask, ask_inputs, ask_outputs, js=JS_FREEZE_TIME, concurrency_limit=1
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=TR["app_title"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"HATA: {e}", file=sys.stderr)
        sys.exit(1)

    mgr = ModelManager(cfg)
    demo = build_app(cfg, mgr)
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
