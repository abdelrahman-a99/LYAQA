"""Launch a headless Gradio Space; regular REST routes remain available."""

from app.config import load_environment

load_environment(".env")

from app.ai.runtime import get_runtime, initialize_runtime  # noqa: E402

# ZeroGPU initialization imports spaces before torch or Gradio.
initialize_runtime(allow_zerogpu=True)

if get_runtime() == "zerogpu":
    from app.ai.inference import warm_zero_models

    warm_zero_models()

from app.hosting import create_space_app  # noqa: E402

demo = create_space_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssr_mode=False,
        _frontend=False,
    )
