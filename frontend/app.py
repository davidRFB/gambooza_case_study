import pandas as pd
import streamlit as st
from utils.api_client import (
    delete_video,
    get_counts_summary,
    get_video_status,
    list_videos,
    process_video,
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

    # Only show uploader when not processing
    if not active_id or get_video_status(active_id)["status"] not in ("pending", "processing"):
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])

        if uploaded_file and uploaded_file.name != st.session_state.get("last_uploaded_name"):
            with st.spinner("Uploading..."):
                result = upload_video(uploaded_file.name, uploaded_file.getvalue())
            st.session_state.last_uploaded_name = uploaded_file.name
            st.session_state.active_video_id = result["id"]
            # Auto-trigger processing only if nothing else is running
            any_processing = any(v["status"] == "processing" for v in list_videos())
            if any_processing:
                st.toast(
                    f"Uploaded: {result['original_name']} — queued (another video is processing)"
                )
            else:
                process_video(result["id"])
                st.toast(f"Uploaded & processing: {result['original_name']}")
            st.rerun()

    # Show status for active video
    if st.session_state.get("active_video_id"):
        video_id = st.session_state.active_video_id
        status = get_video_status(video_id)

        if status["status"] == "pending":
            any_processing = any(v["status"] == "processing" for v in list_videos())
            if any_processing:
                st.warning(
                    f"Queued: {status['original_name']} — waiting for another video to finish"
                )
                if st.button("Refresh Status"):
                    st.rerun()
            else:
                process_video(video_id)
                st.toast(f"Processing started: {status['original_name']}")
                st.rerun()

        elif status["status"] == "processing":
            st.warning(f"Processing: {status['original_name']}")
            if st.button("Refresh Status"):
                st.rerun()

        elif status["status"] == "completed":
            st.success(f"Completed: {status['original_name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tap A", status["tap_a_count"])
            c2.metric("Tap B", status["tap_b_count"])
            c3.metric("Total", status["total"])

            if status["events"]:
                st.subheader("Pour Events")
                events_df = pd.DataFrame(status["events"])
                events_df = events_df[["tap", "timestamp_start", "timestamp_end", "count"]]
                events_df.columns = ["Tap", "Start (s)", "End (s)", "Count"]
                st.dataframe(events_df, use_container_width=True)

        elif status["status"] == "error":
            st.error(f"Processing failed: {status.get('error_message', 'Unknown error')}")

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

            with st.expander(f"{name} — {label} — {date}"):
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
                        st.dataframe(events_df, use_container_width=True)
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
