import os
import sys
from pathlib import Path

import hydra
import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.transforms import get_val_transform


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent / "configs"),
    config_name="config",
)
def run_streamlit_server(cfg):
    st.set_page_config(page_title="HymenopteraDINOv3🐜🐝🦖", page_icon="🦖")
    st.title("HymenopteraDINOv3🐜🐝🦖\nImage classificator")

    transform = get_val_transform(cfg.dataset.transforms)
    endpoint = f"http://{cfg.tracking.host}:{cfg.tracking.port}/invocations"

    uploaded_file = st.file_uploader(
        "Выбери изображение...", type=list(cfg.tracking.allowed_types)
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженная картинка", use_container_width=True)

        if st.button("Предсказать класс 🔮🧙", type="primary"):
            with st.spinner("Запрос отправлен..."):
                try:
                    tensor_numpy = transform(image).cpu().numpy()
                    payload = {"inputs": np.expand_dims(tensor_numpy, axis=0).tolist()}

                    response = requests.post(endpoint, json=payload, timeout=10)

                    if response.status_code == 200:
                        logits = response.json()["predictions"]
                        output = (
                            logits["output"] if isinstance(logits, dict) else logits
                        )

                        probability = float(torch.sigmoid(torch.tensor(output)).item())

                        st.success("Ответ успешно получен!")

                        st.metric(
                            label="Вероятность положительного класса",
                            value=f"{probability:.2%}",
                        )

                        st.progress(probability)

                        if probability > cfg.model.threshold:
                            st.warning("⚠️ Обнаружен пчол!")
                        else:
                            st.info("✅ Это муравей!")

                    else:
                        st.error(
                            f"Ошибка сервера (Код {response.status_code}): {response.text}"
                        )

                except requests.exceptions.RequestException as e:
                    st.error(f"Не удалось связаться с сервером инференса: {e}")


if __name__ == "__main__":
    run_streamlit_server()
