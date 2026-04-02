import time

import pandas as pd
import streamlit as st
from utils.api_client import (
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

st.set_page_config(page_title="Beer Tap Counter", layout="wide")
st.title("Beer Tap Counter")

tab_upload, tab_dashboard = st.tabs(["Upload & Process", "Dashboard"])

# --- Tab 1: Upload & Process ---
with tab_upload:
    # Recent videos
    recent = list_videos()[:5]
    if recent:
        st.caption("Recent videos")
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

        ADD_NEW = "➕ Add new..."
        restaurant_name = None
        camera_id = None

        # --- Restaurant picker ---
        if st.session_state.get("new_restaurant_confirmed"):
            restaurant_name = st.session_state.new_restaurant_confirmed
            st.text_input("Restaurant", value=restaurant_name, disabled=True)
            if st.button("Change restaurant", key="change_rest"):
                del st.session_state.new_restaurant_confirmed
                st.session_state.pop("new_camera_confirmed", None)
                st.rerun()
        else:
            rest_options = restaurant_list + [ADD_NEW]
            selected_rest = st.selectbox(
                "Restaurant", rest_options, index=None, placeholder="Select restaurant..."
            )
            if selected_rest == ADD_NEW:
                new_rest = st.text_input("New restaurant name", placeholder="e.g. mikes_pub")
                if new_rest:
                    st.session_state.new_restaurant_confirmed = new_rest
                    st.rerun()
            elif selected_rest:
                restaurant_name = selected_rest

        # --- Camera picker ---
        if restaurant_name:
            if st.session_state.get("new_camera_confirmed"):
                camera_id = st.session_state.new_camera_confirmed
                st.text_input("Camera ID", value=camera_id, disabled=True)
                if st.button("Change camera", key="change_cam"):
                    del st.session_state.new_camera_confirmed
                    st.rerun()
            else:
                cam_options = cameras_map.get(restaurant_name, []) + [ADD_NEW]
                selected_cam = st.selectbox(
                    "Camera ID", cam_options, index=None, placeholder="Select camera..."
                )
                if selected_cam == ADD_NEW:
                    new_cam = st.text_input("New camera ID", placeholder="e.g. cam1")
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
                preview_roi = st.checkbox("Preview ROI on first frame before processing")
                roi_config_data = roi_check.get("roi_data")

        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])

        if uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_name"):
            with st.spinner("Uploading..."):
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
                st.toast(f"Uploaded: {result['original_name']} — ROI config needed")
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
                        f"Uploaded: {result['original_name']} — queued (another video is processing)"
                    )
                else:
                    process_video(result["id"])
                    st.toast(f"Uploaded & processing: {result['original_name']}")
                st.rerun()

    # ROI preview confirmation — show first frame with boxes, confirm before processing
    if st.session_state.get("roi_confirm_active"):
        import io

        from PIL import Image, ImageDraw

        video_id = st.session_state.active_video_id
        restaurant_name = st.session_state.roi_confirm_restaurant
        camera_id = st.session_state.roi_confirm_camera
        roi_data = st.session_state.roi_confirm_data

        st.subheader(f"ROI Preview: {restaurant_name} / {camera_id}")

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

                st.image(frame_image, caption="First frame with ROI overlay", width="stretch")
            except Exception as e:
                st.error(f"Could not load frame preview: {e}")
        else:
            st.warning("ROI config data not available. Try re-uploading.")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Confirm & Process", type="primary"):
                for k in [
                    "roi_confirm_active",
                    "roi_confirm_data",
                    "roi_confirm_restaurant",
                    "roi_confirm_camera",
                ]:
                    st.session_state.pop(k, None)
                any_processing = any(v["status"].startswith("processing") for v in list_videos())
                if any_processing:
                    st.toast("Queued — another video is processing")
                else:
                    process_video(video_id)
                    st.toast("Processing started")
                st.rerun()
        with col2:
            if st.button("Re-draw ROI", key="redraw_roi"):
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
            if st.button("Cancel", key="cancel_confirm"):
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

        st.subheader(f"ROI Setup: {restaurant_name} / {camera_id}")

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
            st.markdown("**Step 1 of 5** — Select **Filter Region 1** (TAP A handle area)")
            st.info(
                "Drag the red box to tightly cover the TAP A handle. "
                "This small region is used for fast activity detection on long videos."
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
                    if st.button("Confirm & Continue", type="primary", key="confirm_f1"):
                        st.session_state.roi_filter_1 = roi_f1
                        st.session_state.roi_step = 2
                        st.rerun()
                with col2:
                    if st.button("Cancel", key="cancel_step1"):
                        _cancel_roi()

        # ── Step 2: Select FILTER ROI 2 on full frame ──
        elif roi_step == 2:
            st.markdown("**Step 2 of 5** — Select **Filter Region 2** (TAP B handle area)")
            st.info(
                "Drag the red box to tightly cover the TAP B handle. "
                "This small region is used for fast activity detection on long videos."
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
                    if st.button("Confirm & Continue", type="primary", key="confirm_f2"):
                        st.session_state.roi_filter_2 = roi_f2
                        st.session_state.roi_step = 3
                        st.rerun()
                with col2:
                    if st.button("Back to Filter ROI 1", key="back_step2"):
                        st.session_state.roi_step = 1
                        st.rerun()

        # ── Step 3: Select crop region on full frame ──
        elif roi_step == 3:
            st.markdown("**Step 3 of 5** — Select the **tap area** (crop region for YOLO)")
            st.info("Drag the orange box to cover the area where taps and cups are visible.")

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
                    if st.button("Confirm Crop & Continue", type="primary"):
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
                    if st.button("Back to Filter ROI 2", key="back_step3"):
                        st.session_state.roi_step = 2
                        st.rerun()

        # ── Step 4: Select TAP A on cropped image ──
        elif roi_step == 4:
            st.markdown("**Step 4 of 5** — Select **TAP A** handle on cropped image")
            st.info("Drag the blue box to cover the TAP A (left tap) handle.")

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
                    if st.button("Confirm TAP A & Continue", type="primary"):
                        st.session_state.roi_tap_a_bbox = bbox_a
                        st.session_state.roi_step = 5
                        st.rerun()
                with col2:
                    if st.button("Back to Crop", key="back_step4"):
                        st.session_state.roi_step = 3
                        st.rerun()

        # ── Step 5: Select TAP B on cropped image ──
        elif roi_step == 5:
            st.markdown("**Step 5 of 5** — Select **TAP B** handle on cropped image")
            st.info("Drag the green box to cover the TAP B (right tap) handle.")

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
                st.markdown("**Summary**")
                st.caption(f"Filter ROI 1: {st.session_state.roi_filter_1}")
                st.caption(f"Filter ROI 2: {st.session_state.roi_filter_2}")
                st.caption(f"Crop ROI: {st.session_state.roi_tap_roi}")
                st.caption(f"TAP A: {st.session_state.roi_tap_a_bbox}")
                st.caption(f"TAP B: {bbox_b}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Save ROI & Start Processing", type="primary"):
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
                        st.toast(f"ROI config saved for {restaurant_name}/{camera_id}")

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
                            st.toast("Processing started")
                        st.rerun()
                with col2:
                    if st.button("Back to TAP A", key="back_step5"):
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
                st.warning(
                    f"Queued: {status['original_name']} — waiting for another video to finish"
                )
                if st.button("Refresh Status"):
                    st.rerun()
                # Auto-poll: check every 10s so queued videos start automatically
                time.sleep(10)
                st.rerun()
            else:
                process_video(video_id)
                st.toast(f"Processing started: {status['original_name']}")
                st.rerun()

        elif status["status"].startswith("processing"):
            stage_labels = {
                "processing": "Processing (YOLO)...",
                "processing_filter": "Stage 1/3: Running activity filter...",
                "processing_clips": "Stage 2/3: Extracting activity clips...",
                "processing_yolo": "Stage 3/3: Running YOLO on clips...",
            }
            label = stage_labels.get(status["status"], "Processing...")
            st.warning(f"{label} — {status['original_name']}")
            if st.button("Refresh Status"):
                st.rerun()
            # Auto-poll: check every 10s for completion
            time.sleep(10)
            st.rerun()

        elif status["status"] in ("completed", "error"):
            if status["status"] == "completed":
                st.success(f"Completed: {status['original_name']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Tap A", status["tap_a_count"])
                c2.metric("Tap B", status["tap_b_count"])
                c3.metric("Total", status["total"])

                # Show timing breakdown if filter was used
                if status.get("num_clips") is not None:
                    st.caption(
                        f"Filter: {status.get('filter_time_s', 0):.1f}s | "
                        f"Clips: {status.get('num_clips', 0)} "
                        f"({status.get('filtered_duration_s', 0):.0f}s of "
                        f"{status.get('duration_sec', 0):.0f}s video) | "
                        f"YOLO: {status.get('yolo_time_s', 0):.1f}s"
                    )
                elif status.get("yolo_time_s"):
                    st.caption(f"YOLO processing: {status['yolo_time_s']:.1f}s")

                if status["events"]:
                    st.subheader("Pour Events")
                    events_df = pd.DataFrame(status["events"])
                    events_df = events_df[["tap", "timestamp_start", "timestamp_end", "count"]]
                    events_df.columns = ["Tap", "Start (s)", "End (s)", "Count"]
                    st.dataframe(events_df, width="stretch")
            else:
                st.error(f"Processing failed: {status.get('error_message', 'Unknown error')}")

            # Auto-start next pending video in queue
            all_videos = list_videos()
            next_pending = next((v for v in all_videos if v["status"] == "pending"), None)
            if next_pending:
                st.session_state.active_video_id = next_pending["id"]
                process_video(next_pending["id"])
                st.toast(f"Processing next: {next_pending['original_name']}")
                st.rerun()

# --- Tab 2: Dashboard ---
with tab_dashboard:
    if st.button("Refresh", key="refresh_dashboard"):
        st.rerun()

    # Global summary
    summary = get_counts_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tap A Total", summary["tap_a_total"])
    c2.metric("Tap B Total", summary["tap_b_total"])
    c3.metric("Grand Total", summary["grand_total"])
    c4.metric("Videos Processed", summary["video_count"])

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
                        events_df.columns = ["Tap", "Start (s)", "End (s)", "Count"]
                        st.dataframe(events_df, width="stretch")
                elif video["status"] == "error":
                    detail = get_video_status(video["id"])
                    st.error(detail.get("error_message", "Unknown error"))
                else:
                    st.info(f"Status: {label}")

                if st.button("Delete", key=f"del_{video['id']}"):
                    delete_video(video["id"])
                    st.toast(f"Deleted: {name}")
                    st.rerun()
    else:
        st.info("No videos uploaded yet.")
