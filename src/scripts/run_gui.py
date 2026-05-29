import sys
from pathlib import Path

import hydra
import requests
import streamlit as st
from PIL import Image

parent_dir = Path(__file__).resolve().parents[1]
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from infer import infer

from data.transforms import get_val_transform


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
)
def run_streamlit_server(cfg):
    """Launches a Streamlit web interface for interactive image classification inference.

    This function configures and serves a graphical user interface that allows users
    to upload images and receive real-time classification results from a remote
    inference server. It executes the following lifecycle steps:
    1. Adjusts the system path to allow local package imports.
    2. Initializes the Streamlit application layout, title, and page configurations.
    3. Sets up validation image transformations and constructs the model serving
       HTTP REST endpoint.
    4. Provides a file upload interface restricted to preconfigured allowed file types.
    5. Upon user submission, invokes the decoupled inference routine and renders
       interactive progress indicators and prediction classes.

    Args:
        cfg (DictConfig): A Hydra configuration object containing interface rules,
            network host/port configurations, transform criteria, and decision thresholds.
    """
    st.set_page_config(page_title="HymenopteraDINOv3🐜🐝🦖", page_icon="🦖")
    st.title("HymenopteraDINOv3🐜🐝🦖\nImage classificator")

    transform = get_val_transform(cfg.dataset.transforms)

    uploaded_file = st.file_uploader(
        "Выбери изображение...", type=list(cfg.tracking.allowed_types)
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженная картинка", use_container_width=True)

        if st.button("Предсказать класс 🔎", type="primary"):
            with st.spinner("Запрос отправлен..."):
                try:
                    probability = infer(
                        image, cfg.tracking.inference.endpoint, transform
                    )

                    st.success("Ответ успешно получен!")
                    st.metric(
                        label="Вероятность положительного класса",
                        value=f"{probability:.2%}",
                    )
                    st.progress(probability)

                    if probability > cfg.model.threshold:
                        st.info("🐝 Обнаружен пчол!")
                    else:
                        st.info("🐜 Это муравей!")

                except requests.exceptions.RequestException as e:
                    st.error(f"Не удалось связаться с сервером инференса: {e}")
                except RuntimeError as e:
                    st.error(str(e))


if __name__ == "__main__":
    run_streamlit_server()
