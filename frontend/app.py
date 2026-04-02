import pandas as pd
import streamlit as st
from utils.api_client import (
    BackendUnavailable,
    check_roi_config,
    delete_video,
    get_counts_summary,
    get_restaurants,
    get_video_frame,
    get_video_status,
    list_videos,
    process_video,
    save_roi_config,
    upload_video,
)

st.set_page_config(page_title="Contador de Cervezas — Gambooza", layout="wide")

# -- Gambooza brand styling --
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@500;600;700&family=Archivo:wght@400;600&display=swap');

    /* Global text */
    html, body, [class*="css"] {
        font-family: 'Archivo', sans-serif;
        color: #2E3D34;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6,
    .stTabs [data-baseweb="tab"] {
        font-family: 'Fira Sans', sans-serif !important;
        color: #2E3D34 !important;
    }

    /* Page background */
    .stApp {
        background-color: #DDF6F3;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F4FFF8;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #F4FFF8;
        border: 1px solid #A8D4B9;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        color: #2E3D34 !important;
        font-family: 'Fira Sans', sans-serif !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #87ACA7 !important;
        font-family: 'Archivo', sans-serif !important;
        font-weight: 600 !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #A8D4B9 !important;
        color: #2E3D34 !important;
        border: none !important;
        font-family: 'Archivo', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #8BAB87 !important;
        color: #F4FFF8 !important;
    }

    /* Secondary/default buttons */
    .stButton > button {
        background-color: #F4FFF8 !important;
        border: 1px solid #A8D4B9 !important;
        color: #2E3D34 !important;
        font-family: 'Archivo', sans-serif !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        border-color: #8BAB87 !important;
        background-color: #E5F5E3 !important;
        color: #2E3D34 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F4FFF8;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #A8D4B9 !important;
        color: #2E3D34 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #A8D4B9 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-family: 'Fira Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #2E3D34 !important;
        background-color: #F4FFF8;
        border-radius: 6px;
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid #A8D4CF;
        border-radius: 6px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-color: #A8D4B9 !important;
    }

    /* Selectbox */
    [data-baseweb="select"] > div {
        border-color: #A8D4B9 !important;
    }

    /* Success/warning/error messages */
    .stSuccess { border-left-color: #A8D4B9 !important; }
    .stWarning { border-left-color: #8BAB87 !important; }

    /* Dividers */
    hr {
        border-color: #A8D4CF !important;
    }

    /* Title styling */
    .gambooza-title {
        font-family: 'Fira Sans', sans-serif;
        font-weight: 700;
        color: #2E3D34;
        font-size: 2.4rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="gambooza-title">Contador de Cervezas</div>', unsafe_allow_html=True)

# Help text
_HELP_TEXT = (
    "Esta aplicacion analiza videos de dispensadores de cerveza para contar "
    "automaticamente las cervezas servidas en cada grifo (Tap A y Tap B).\n\n"
    "**Restaurante y camara:** Al subir un video se selecciona el nombre del restaurante "
    "y un identificador de camara. Esto permite guardar configuraciones separadas "
    "segun la resolucion o posicion de cada camara. Todo queda almacenado para "
    "futuros videos de la misma camara.\n\n"
    "**Regiones de interes (ROI):** La primera vez que se sube un video para una "
    "combinacion de restaurante y camara, se abre un asistente de 5 pasos para "
    "definir las regiones de interes:\n\n"
    "- **Regiones de filtro 1 y 2 (pasos 1 y 2):** Se selecciona un recuadro pequeno "
    "sobre cada grifo, en la zona donde cae el liquido. Estas regiones se usan para "
    "detectar rapidamente en que momentos hay actividad, sin necesidad de analizar "
    "el video completo.\n"
    "- **Area general (paso 3):** Se selecciona la region completa que incluye los "
    "grifos, las manijas y el area donde se sirven las cervezas. Esta es la zona "
    "que el modelo YOLO analiza en detalle.\n"
    "- **Manijas de cada grifo (pasos 4 y 5):** Se marcan especificamente las manijas "
    "de Tap A y Tap B dentro de la region recortada. Esto permite identificar con "
    "precision cual grifo esta sirviendo en cada momento.\n\n"
    "Con estas cinco regiones el sistema tiene control completo de la posicion "
    "de cada elemento y puede funcionar correctamente con distintas camaras y angulos."
)

if "show_help" not in st.session_state:
    st.session_state.show_help = True

if st.button("Ayuda", key="help_button"):
    st.session_state.show_help = True

if st.session_state.show_help:
    with st.container():
        st.info(_HELP_TEXT)
        if st.button("Cerrar", key="close_help"):
            st.session_state.show_help = False
            st.rerun()

# Check backend connectivity before rendering the app
try:
    list_videos()
except BackendUnavailable:
    st.error(
        "**Backend no disponible** — no se puede conectar al servidor API. "
        "Asegurate de que el backend este corriendo e intenta de nuevo."
    )
    if st.button("Reintentar conexion"):
        st.rerun()
    st.stop()

tab_upload, tab_dashboard = st.tabs(["Subir y Procesar", "Panel"])

# --- Tab 1: Upload & Process ---
with tab_upload:
    if st.button("Actualizar", key="refresh_upload"):
        st.rerun()

    # Recent videos
    recent = list_videos()[:5]
    if recent:
        st.caption("Videos recientes")
        for v in recent:
            status_icon = {
                "completed": "✅",
                "processing": "⏳",
                "pending": "⬚",
                "error": "❌",
            }.get(v["status"], "")
            st.text(f"{status_icon} {v['original_name']} — {v['status']}")
        st.divider()

    active_id = st.session_state.get("active_video_id")
    _active_status = get_video_status(active_id)["status"] if active_id else None
    _active_busy = _active_status in ("pending",) or (_active_status or "").startswith("processing")

    # Only show uploader when not processing and not in ROI selection/confirm
    if (
        not st.session_state.get("roi_selection_active")
        and not st.session_state.get("roi_confirm_active")
        and not _active_busy
    ):
        # Restaurant + Camera selection
        rest_data = get_restaurants()
        restaurant_list = rest_data.get("restaurants", [])
        cameras_map = rest_data.get("cameras", {})

        ADD_NEW = "➕ Agregar nuevo..."
        restaurant_name = None
        camera_id = None

        # --- Restaurant picker ---
        if st.session_state.get("new_restaurant_confirmed"):
            restaurant_name = st.session_state.new_restaurant_confirmed
            st.text_input("Restaurante", value=restaurant_name, disabled=True)
            if st.button("Cambiar restaurante", key="change_rest"):
                del st.session_state.new_restaurant_confirmed
                st.session_state.pop("new_camera_confirmed", None)
                st.rerun()
        else:
            rest_options = restaurant_list + [ADD_NEW]
            selected_rest = st.selectbox(
                "Restaurante", rest_options, index=None, placeholder="Seleccionar restaurante..."
            )
            if selected_rest == ADD_NEW:
                new_rest = st.text_input(
                    "Nombre del nuevo restaurante", placeholder="ej. mikes_pub"
                )
                if new_rest:
                    st.session_state.new_restaurant_confirmed = new_rest
                    st.rerun()
            elif selected_rest:
                restaurant_name = selected_rest

        # --- Camera picker ---
        if restaurant_name:
            if st.session_state.get("new_camera_confirmed"):
                camera_id = st.session_state.new_camera_confirmed
                st.text_input("ID de camara", value=camera_id, disabled=True)
                if st.button("Cambiar camara", key="change_cam"):
                    del st.session_state.new_camera_confirmed
                    st.rerun()
            else:
                cam_options = cameras_map.get(restaurant_name, []) + [ADD_NEW]
                selected_cam = st.selectbox(
                    "ID de camara", cam_options, index=None, placeholder="Seleccionar camara..."
                )
                if selected_cam == ADD_NEW:
                    new_cam = st.text_input("Nuevo ID de camara", placeholder="ej. cam1")
                    if new_cam:
                        st.session_state.new_camera_confirmed = new_cam
                        st.rerun()
                elif selected_cam:
                    camera_id = selected_cam

        # --- Preview ROI checkbox (only when config exists) ---
        preview_roi = False
        roi_config_data = None
        if restaurant_name and camera_id:
            roi_check = check_roi_config(restaurant_name, camera_id)
            if roi_check["exists"]:
                preview_roi = st.checkbox("Previsualizar ROI en el primer frame antes de procesar")
                roi_config_data = roi_check.get("roi_data")

        uploaded_file = st.file_uploader("Subir un video", type=["mp4", "mov"])

        if uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_name"):
            with st.spinner("Subiendo..."):
                result = upload_video(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    restaurant_name=restaurant_name,
                    camera_id=camera_id,
                )
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.active_video_id = result["id"]
            # Clear confirmed new names for next upload
            st.session_state.pop("new_restaurant_confirmed", None)
            st.session_state.pop("new_camera_confirmed", None)

            # Check if ROI config exists for this combo
            has_roi = roi_config_data is not None
            if not has_roi and restaurant_name and camera_id:
                roi_recheck = check_roi_config(restaurant_name, camera_id)
                has_roi = roi_recheck["exists"]

            if not has_roi:
                # Enter ROI selection mode
                st.session_state.roi_selection_active = True
                st.session_state.roi_restaurant = restaurant_name
                st.session_state.roi_camera = camera_id
                st.toast(f"Subido: {result['original_name']} — se necesita configurar ROI")
                st.rerun()
            elif preview_roi:
                # Enter preview confirmation mode
                st.session_state.roi_confirm_active = True
                st.session_state.roi_confirm_data = roi_config_data
                st.session_state.roi_confirm_restaurant = restaurant_name
                st.session_state.roi_confirm_camera = camera_id
                st.rerun()
            else:
                # Auto-trigger processing
                any_processing = any(v["status"].startswith("processing") for v in list_videos())
                if any_processing:
                    st.toast(
                        f"Subido: {result['original_name']} — en cola (otro video se esta procesando)"
                    )
                else:
                    process_video(result["id"])
                    st.toast(f"Subido y procesando: {result['original_name']}")
                st.rerun()

    # ROI preview confirmation — show first frame with boxes, confirm before processing
    if st.session_state.get("roi_confirm_active"):
        import io

        from PIL import Image, ImageDraw

        video_id = st.session_state.active_video_id
        restaurant_name = st.session_state.roi_confirm_restaurant
        camera_id = st.session_state.roi_confirm_camera
        roi_data = st.session_state.roi_confirm_data

        st.subheader(f"Vista previa ROI: {restaurant_name} / {camera_id}")

        # Always re-fetch ROI data from backend to ensure it's available
        if restaurant_name and camera_id:
            roi_refetch = check_roi_config(restaurant_name, camera_id)
            if roi_refetch["exists"] and roi_refetch.get("roi_data"):
                roi_data = roi_refetch["roi_data"]
                st.session_state.roi_confirm_data = roi_data

        if roi_data:
            try:
                frame_bytes = get_video_frame(video_id)
                frame_image = Image.open(io.BytesIO(frame_bytes))
                img_w, img_h = frame_image.size
                draw = ImageDraw.Draw(frame_image)

                yolo = roi_data.get("yolo", {})
                tap_roi = yolo.get("tap_roi", [])
                sam3_bboxes = yolo.get("sam3_tap_bboxes", [])

                # Draw filter ROIs if present
                simple = roi_data.get("simple", {})
                for key, label in [("roi_1", "FILTER 1"), ("roi_2", "FILTER 2")]:
                    froi = simple.get(key)
                    if froi and len(froi) == 4:
                        fx1 = int(froi[0] * img_w)
                        fy1 = int(froi[1] * img_h)
                        fx2 = int(froi[2] * img_w)
                        fy2 = int(froi[3] * img_h)
                        draw.rectangle([fx1, fy1, fx2, fy2], outline="#FF0000", width=2)
                        draw.text((fx1 + 4, fy1 + 4), label, fill="#FF0000")

                if len(tap_roi) == 4:
                    x1 = int(tap_roi[0] * img_w)
                    y1 = int(tap_roi[1] * img_h)
                    x2 = int(tap_roi[2] * img_w)
                    y2 = int(tap_roi[3] * img_h)
                    draw.rectangle([x1, y1, x2, y2], outline="#FF6600", width=3)
                    draw.text((x1 + 4, y1 + 4), "CROP", fill="#FF6600")

                    colors = ["#0088FF", "#00CC44"]
                    labels = ["TAP A", "TAP B"]
                    for i, bbox in enumerate(sam3_bboxes[:2]):
                        if len(bbox) == 4:
                            bx1 = x1 + int(bbox[0])
                            by1 = y1 + int(bbox[1])
                            bx2 = x1 + int(bbox[2])
                            by2 = y1 + int(bbox[3])
                            draw.rectangle([bx1, by1, bx2, by2], outline=colors[i], width=2)
                            draw.text((bx1 + 4, by1 + 4), labels[i], fill=colors[i])

                st.image(frame_image, caption="Primer frame con ROI superpuesto", width="stretch")
            except Exception as e:
                st.error(f"No se pudo cargar la vista previa: {e}")
        else:
            st.warning("Datos de configuracion ROI no disponibles. Intenta subir de nuevo.")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Confirmar y Procesar", type="primary"):
                for k in [
                    "roi_confirm_active",
                    "roi_confirm_data",
                    "roi_confirm_restaurant",
                    "roi_confirm_camera",
                ]:
                    st.session_state.pop(k, None)
                any_processing = any(v["status"].startswith("processing") for v in list_videos())
                if any_processing:
                    st.toast("En cola — otro video se esta procesando")
                else:
                    process_video(video_id)
                    st.toast("Procesamiento iniciado")
                st.rerun()
        with col2:
            if st.button("Redibujar ROI", key="redraw_roi"):
                # Enter ROI wizard — will overwrite the config on save
                st.session_state.roi_selection_active = True
                st.session_state.roi_restaurant = restaurant_name
                st.session_state.roi_camera = camera_id
                for k in [
                    "roi_confirm_active",
                    "roi_confirm_data",
                    "roi_confirm_restaurant",
                    "roi_confirm_camera",
                ]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col3:
            if st.button("Cancelar", key="cancel_confirm"):
                for k in [
                    "roi_confirm_active",
                    "roi_confirm_data",
                    "roi_confirm_restaurant",
                    "roi_confirm_camera",
                ]:
                    st.session_state.pop(k, None)
                st.rerun()

    # ROI Selection flow — multi-step wizard
    if st.session_state.get("roi_selection_active"):
        import io

        from PIL import Image
        from streamlit_cropper import st_cropper

        video_id = st.session_state.active_video_id
        restaurant_name = st.session_state.roi_restaurant
        camera_id = st.session_state.roi_camera
        roi_step = st.session_state.get("roi_step", 1)

        st.subheader(f"Configuracion ROI: {restaurant_name} / {camera_id}")

        # Load frame once and cache in session
        if "roi_frame_bytes" not in st.session_state:
            st.session_state.roi_frame_bytes = get_video_frame(video_id)

        frame_image = Image.open(io.BytesIO(st.session_state.roi_frame_bytes))
        img_w, img_h = frame_image.size

        def _cancel_roi():
            st.session_state.roi_selection_active = False
            for k in [
                "roi_step",
                "roi_frame_bytes",
                "roi_tap_roi",
                "roi_cropped_bytes",
                "roi_tap_a_bbox",
                "roi_filter_1",
                "roi_filter_2",
                "roi_restaurant",
                "roi_camera",
            ]:
                st.session_state.pop(k, None)
            st.rerun()

        # ── Step 1: Select FILTER ROI 1 on full frame ──
        if roi_step == 1:
            st.markdown(
                "**Paso 1 de 5** — Seleccionar **Region de Filtro 1** (area del grifo TAP A)"
            )
            st.info(
                "Arrastra el recuadro rojo para cubrir ajustadamente el grifo TAP A. "
                "Esta pequena region se usa para deteccion rapida de actividad en videos largos."
            )

            _, filter_box_1 = st_cropper(
                frame_image,
                box_color="#FF0000",
                return_type="both",
                key="filter_roi_1",
            )

            if (
                filter_box_1
                and filter_box_1.get("width", 0) > 5
                and filter_box_1.get("height", 0) > 5
            ):
                roi_f1 = [
                    round(filter_box_1["left"] / img_w, 4),
                    round(filter_box_1["top"] / img_h, 4),
                    round((filter_box_1["left"] + filter_box_1["width"]) / img_w, 4),
                    round((filter_box_1["top"] + filter_box_1["height"]) / img_h, 4),
                ]
                st.caption(f"Filter ROI 1 (normalized): {roi_f1}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Confirmar y Continuar", type="primary", key="confirm_f1"):
                        st.session_state.roi_filter_1 = roi_f1
                        st.session_state.roi_step = 2
                        st.rerun()
                with col2:
                    if st.button("Cancelar", key="cancel_step1"):
                        _cancel_roi()

        # ── Step 2: Select FILTER ROI 2 on full frame ──
        elif roi_step == 2:
            st.markdown(
                "**Paso 2 de 5** — Seleccionar **Region de Filtro 2** (area del grifo TAP B)"
            )
            st.info(
                "Arrastra el recuadro rojo para cubrir ajustadamente el grifo TAP B. "
                "Esta pequena region se usa para deteccion rapida de actividad en videos largos."
            )

            _, filter_box_2 = st_cropper(
                frame_image,
                box_color="#FF0000",
                return_type="both",
                key="filter_roi_2",
            )

            if (
                filter_box_2
                and filter_box_2.get("width", 0) > 5
                and filter_box_2.get("height", 0) > 5
            ):
                roi_f2 = [
                    round(filter_box_2["left"] / img_w, 4),
                    round(filter_box_2["top"] / img_h, 4),
                    round((filter_box_2["left"] + filter_box_2["width"]) / img_w, 4),
                    round((filter_box_2["top"] + filter_box_2["height"]) / img_h, 4),
                ]
                st.caption(f"Filter ROI 2 (normalized): {roi_f2}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Confirmar y Continuar", type="primary", key="confirm_f2"):
                        st.session_state.roi_filter_2 = roi_f2
                        st.session_state.roi_step = 3
                        st.rerun()
                with col2:
                    if st.button("Volver a Filtro ROI 1", key="back_step2"):
                        st.session_state.roi_step = 1
                        st.rerun()

        # ── Step 3: Select crop region on full frame ──
        elif roi_step == 3:
            st.markdown(
                "**Paso 3 de 5** — Seleccionar el **area de grifos** (region de recorte para YOLO)"
            )
            st.info(
                "Arrastra el recuadro naranja para cubrir el area donde se ven los grifos y vasos."
            )

            cropped_img, crop_box = st_cropper(
                frame_image,
                box_color="#FF6600",
                return_type="both",
                key="crop_roi",
            )

            if crop_box and crop_box.get("width", 0) > 10 and crop_box.get("height", 0) > 10:
                tap_roi = [
                    round(crop_box["left"] / img_w, 4),
                    round(crop_box["top"] / img_h, 4),
                    round((crop_box["left"] + crop_box["width"]) / img_w, 4),
                    round((crop_box["top"] + crop_box["height"]) / img_h, 4),
                ]
                st.caption(f"Crop ROI (normalized): {tap_roi}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Confirmar Recorte y Continuar", type="primary"):
                        st.session_state.roi_tap_roi = tap_roi
                        crop_l = int(crop_box["left"])
                        crop_t = int(crop_box["top"])
                        crop_r = int(crop_box["left"] + crop_box["width"])
                        crop_b = int(crop_box["top"] + crop_box["height"])
                        cropped_pil = frame_image.crop((crop_l, crop_t, crop_r, crop_b))
                        buf = io.BytesIO()
                        cropped_pil.save(buf, format="JPEG")
                        st.session_state.roi_cropped_bytes = buf.getvalue()
                        st.session_state.roi_step = 4
                        st.rerun()
                with col2:
                    if st.button("Volver a Filtro ROI 2", key="back_step3"):
                        st.session_state.roi_step = 2
                        st.rerun()

        # ── Step 4: Select TAP A on cropped image ──
        elif roi_step == 4:
            st.markdown("**Paso 4 de 5** — Seleccionar grifo **TAP A** en la imagen recortada")
            st.info("Arrastra el recuadro azul para cubrir el grifo TAP A (grifo izquierdo).")

            cropped_img = Image.open(io.BytesIO(st.session_state.roi_cropped_bytes))

            _, tap_a_box = st_cropper(
                cropped_img,
                box_color="#0088FF",
                return_type="both",
                key="tap_a_roi",
            )

            if tap_a_box and tap_a_box.get("width", 0) > 5 and tap_a_box.get("height", 0) > 5:
                bbox_a = [
                    round(tap_a_box["left"], 2),
                    round(tap_a_box["top"], 2),
                    round(tap_a_box["left"] + tap_a_box["width"], 2),
                    round(tap_a_box["top"] + tap_a_box["height"], 2),
                ]
                st.caption(f"TAP A bbox (pixels on crop): {bbox_a}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Confirmar TAP A y Continuar", type="primary"):
                        st.session_state.roi_tap_a_bbox = bbox_a
                        st.session_state.roi_step = 5
                        st.rerun()
                with col2:
                    if st.button("Volver a Recorte", key="back_step4"):
                        st.session_state.roi_step = 3
                        st.rerun()

        # ── Step 5: Select TAP B on cropped image ──
        elif roi_step == 5:
            st.markdown("**Paso 5 de 5** — Seleccionar grifo **TAP B** en la imagen recortada")
            st.info("Arrastra el recuadro verde para cubrir el grifo TAP B (grifo derecho).")

            cropped_img = Image.open(io.BytesIO(st.session_state.roi_cropped_bytes))

            _, tap_b_box = st_cropper(
                cropped_img,
                box_color="#00CC44",
                return_type="both",
                key="tap_b_roi",
            )

            if tap_b_box and tap_b_box.get("width", 0) > 5 and tap_b_box.get("height", 0) > 5:
                bbox_b = [
                    round(tap_b_box["left"], 2),
                    round(tap_b_box["top"], 2),
                    round(tap_b_box["left"] + tap_b_box["width"], 2),
                    round(tap_b_box["top"] + tap_b_box["height"], 2),
                ]
                st.caption(f"TAP B bbox (pixels on crop): {bbox_b}")

                # Show summary before saving
                st.divider()
                st.markdown("**Resumen**")
                st.caption(f"Filter ROI 1: {st.session_state.roi_filter_1}")
                st.caption(f"Filter ROI 2: {st.session_state.roi_filter_2}")
                st.caption(f"Crop ROI: {st.session_state.roi_tap_roi}")
                st.caption(f"TAP A: {st.session_state.roi_tap_a_bbox}")
                st.caption(f"TAP B: {bbox_b}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Guardar ROI e Iniciar Procesamiento", type="primary"):
                        roi_data = {
                            "simple": {
                                "roi_1": st.session_state.roi_filter_1,
                                "roi_2": st.session_state.roi_filter_2,
                            },
                            "yolo": {
                                "tap_roi": st.session_state.roi_tap_roi,
                                "sam3_tap_bboxes": [
                                    st.session_state.roi_tap_a_bbox,
                                    bbox_b,
                                ],
                            },
                        }
                        save_roi_config(restaurant_name, camera_id, roi_data)
                        st.toast(f"Configuracion ROI guardada para {restaurant_name}/{camera_id}")

                        # Clear all ROI selection state
                        for k in [
                            "roi_selection_active",
                            "roi_step",
                            "roi_frame_bytes",
                            "roi_tap_roi",
                            "roi_cropped_bytes",
                            "roi_tap_a_bbox",
                            "roi_filter_1",
                            "roi_filter_2",
                            "roi_restaurant",
                            "roi_camera",
                        ]:
                            st.session_state.pop(k, None)

                        # Start processing
                        any_processing = any(
                            v["status"].startswith("processing") for v in list_videos()
                        )
                        if not any_processing:
                            process_video(video_id)
                            st.toast("Procesamiento iniciado")
                        st.rerun()
                with col2:
                    if st.button("Volver a TAP A", key="back_step5"):
                        st.session_state.roi_step = 4
                        st.rerun()

    # Show status for active video
    if (
        st.session_state.get("active_video_id")
        and not st.session_state.get("roi_selection_active")
        and not st.session_state.get("roi_confirm_active")
    ):
        video_id = st.session_state.active_video_id
        status = get_video_status(video_id)

        if status["status"] == "pending":
            any_processing = any(v["status"].startswith("processing") for v in list_videos())
            if any_processing:
                st.warning(f"En cola: {status['original_name']} — esperando que termine otro video")
                if st.button("Actualizar Estado", key="refresh_pending"):
                    st.rerun()
            else:
                process_video(video_id)
                st.toast(f"Procesamiento iniciado: {status['original_name']}")
                st.rerun()

        elif status["status"].startswith("processing"):
            stage_labels = {
                "processing": "Procesando (YOLO)...",
                "processing_filter": "Etapa 1/3: Ejecutando filtro de actividad...",
                "processing_clips": "Etapa 2/3: Extrayendo clips de actividad...",
                "processing_yolo": "Etapa 3/3: Ejecutando YOLO en clips...",
            }
            label = stage_labels.get(status["status"], "Procesando...")
            st.warning(f"{label} — {status['original_name']}")
            if st.button("Actualizar Estado", key="refresh_processing"):
                st.rerun()

        elif status["status"] in ("completed", "error"):
            if status["status"] == "completed":
                st.success(f"Completado: {status['original_name']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Tap A", status["tap_a_count"])
                c2.metric("Tap B", status["tap_b_count"])
                c3.metric("Total", status["total"])

                # Show timing breakdown if filter was used
                if status.get("num_clips") is not None:
                    st.caption(
                        f"Filtro: {status.get('filter_time_s', 0):.1f}s | "
                        f"Clips: {status.get('num_clips', 0)} "
                        f"({status.get('filtered_duration_s', 0):.0f}s de "
                        f"{status.get('duration_sec', 0):.0f}s de video) | "
                        f"YOLO: {status.get('yolo_time_s', 0):.1f}s"
                    )
                elif status.get("yolo_time_s"):
                    st.caption(f"Procesamiento YOLO: {status['yolo_time_s']:.1f}s")

                if status["events"]:
                    st.subheader("Eventos de Servido")
                    events_df = pd.DataFrame(status["events"])
                    events_df = events_df[["tap", "timestamp_start", "timestamp_end", "count"]]
                    events_df.columns = ["Grifo", "Inicio (s)", "Fin (s)", "Cantidad"]
                    st.dataframe(events_df, width="stretch")
            else:
                st.error(
                    f"Procesamiento fallido: {status.get('error_message', 'Error desconocido')}"
                )

            # Auto-start next pending video in queue
            all_videos = list_videos()
            next_pending = next((v for v in all_videos if v["status"] == "pending"), None)
            if next_pending:
                st.session_state.active_video_id = next_pending["id"]
                process_video(next_pending["id"])
                st.toast(f"Procesando siguiente: {next_pending['original_name']}")
                st.rerun()

# --- Tab 2: Dashboard ---
with tab_dashboard:
    if st.button("Actualizar", key="refresh_dashboard"):
        st.rerun()

    # Global summary
    summary = get_counts_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tap A Total", summary["tap_a_total"])
    c2.metric("Tap B Total", summary["tap_b_total"])
    c3.metric("Total General", summary["grand_total"])
    c4.metric("Videos Procesados", summary["video_count"])

    st.divider()

    # Video list — each video is an expander
    videos = list_videos()
    if videos:
        for video in videos:
            label = video["status"]
            name = video["original_name"]
            date = video["upload_date"][:10] if video["upload_date"] else ""
            rest = video.get("restaurant_name") or ""
            cam = video.get("camera_id") or ""
            location = f" [{rest}/{cam}]" if rest else ""

            with st.expander(f"{name}{location} — {label} — {date}"):
                if video["status"] == "completed":
                    detail = get_video_status(video["id"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tap A", detail["tap_a_count"])
                    c2.metric("Tap B", detail["tap_b_count"])
                    c3.metric("Total", detail["total"])

                    if detail["events"]:
                        events_df = pd.DataFrame(detail["events"])
                        events_df = events_df[["tap", "timestamp_start", "timestamp_end", "count"]]
                        events_df.columns = ["Grifo", "Inicio (s)", "Fin (s)", "Cantidad"]
                        st.dataframe(events_df, width="stretch")
                elif video["status"] == "error":
                    detail = get_video_status(video["id"])
                    st.error(detail.get("error_message", "Error desconocido"))
                else:
                    st.info(f"Estado: {label}")

                if st.button("Eliminar", key=f"del_{video['id']}"):
                    delete_video(video["id"])
                    st.toast(f"Eliminado: {name}")
                    st.rerun()
    else:
        st.info("No hay videos subidos aun.")
