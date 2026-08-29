"""
media_handler.py - Unified media processing that routes through the agentic system.

This module provides a clean interface for handling all media types (images, voice notes,
documents, videos, stickers) by routing them through the agent loop with proper context.
"""

import base64
import os
import tempfile
import re
from typing import Optional, Dict, Any

# These will be injected by whatsapp_agent
_client = None
_supabase = None
_client_ai = None
_client_gemini = None
_MODEL_NAME = None
_GEMINI_MODEL = None
_BUSINESS_KNOWLEDGE = ""
_DEFAULT_TIMEZONE = "Asia/Karachi"
_admin_commands = None
_get_contacts_maps = None
_get_user_timezone = None
_set_user_timezone = None
_get_tzinfo = None
_get_group_memory = None
_send_proactive_message = None
_fetch_chat_history = None
_insert_chat_message = None
_detect_media_on_proto = None
_file_magic_kind = None
_guess_doc_meta = None
_extract_document_text = None


def init_media_handler(
    client,
    supabase,
    client_ai,
    client_gemini,
    model_name,
    gemini_model,
    business_knowledge,
    default_timezone,
    admin_commands,
    get_contacts_maps,
    get_user_timezone,
    set_user_timezone,
    get_tzinfo,
    get_group_memory,
    send_proactive_message,
    fetch_chat_history,
    insert_chat_message,
    detect_media_on_proto,
    file_magic_kind,
    guess_doc_meta,
    extract_document_text,
):
    """Initialize the media handler with all required dependencies."""
    global _client, _supabase, _client_ai, _client_gemini
    global _MODEL_NAME, _GEMINI_MODEL, _BUSINESS_KNOWLEDGE, _DEFAULT_TIMEZONE
    global _admin_commands, _get_contacts_maps, _get_user_timezone, _set_user_timezone
    global _get_tzinfo, _get_group_memory, _send_proactive_message
    global _fetch_chat_history, _insert_chat_message
    global _detect_media_on_proto, _file_magic_kind, _guess_doc_meta, _extract_document_text

    _client = client
    _supabase = supabase
    _client_ai = client_ai
    _client_gemini = client_gemini
    _MODEL_NAME = model_name
    _GEMINI_MODEL = gemini_model
    _BUSINESS_KNOWLEDGE = business_knowledge
    _DEFAULT_TIMEZONE = default_timezone
    _admin_commands = admin_commands
    _get_contacts_maps = get_contacts_maps
    _get_user_timezone = get_user_timezone
    _set_user_timezone = set_user_timezone
    _get_tzinfo = get_tzinfo
    _get_group_memory = get_group_memory
    _send_proactive_message = send_proactive_message
    _fetch_chat_history = fetch_chat_history
    _insert_chat_message = insert_chat_message
    _detect_media_on_proto = detect_media_on_proto
    _file_magic_kind = file_magic_kind
    _guess_doc_meta = guess_doc_meta
    _extract_document_text = extract_document_text


def _download_media(message, media_proto, suffix=".bin"):
    """Download media from WhatsApp to a temp file."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        candidates = []
        if media_proto is not None:
            candidates.append(media_proto)
            # Try inner media sub-message
            for sub_name in (
                "audioMessage", "AudioMessage",
                "imageMessage", "ImageMessage",
                "videoMessage", "VideoMessage",
                "documentMessage", "DocumentMessage",
                "stickerMessage", "StickerMessage",
            ):
                sub = getattr(media_proto, sub_name, None)
                if sub is not None:
                    candidates.append(sub)
                    break
        candidates.append(message.Message)

        dl_err = None
        for cand in candidates:
            try:
                _client.download_any(cand, path=tmp_path)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    dl_err = None
                    break
            except Exception as e:
                dl_err = e
                continue
        if dl_err and (not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0):
            raise dl_err

        return tmp_path
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


def _get_mime_type(media_kind, doc_src=None, tmp_path=None):
    """Determine the MIME type for a media file."""
    if media_kind == "document" and doc_src:
        mime = getattr(doc_src, "mimetype", None) or "application/octet-stream"
        # Check file magic
        if tmp_path and _file_magic_kind:
            magic = _file_magic_kind(tmp_path)
            if magic == "pdf":
                return "application/pdf"
        return mime
    
    mappings = {
        "image": "image/jpeg",
        "sticker": "image/webp",
        "user_created_sticker": "image/webp",
        "video": "video/mp4",
        "gif": "video/mp4",
        "audio": "audio/ogg",
        "ptt": "audio/ogg",
        "document": "application/octet-stream",
    }
    return mappings.get(media_kind, "application/octet-stream")


def _get_suffix(media_kind, doc_src=None):
    """Get the file suffix for a media type."""
    if media_kind == "document" and doc_src:
        orig_name = getattr(doc_src, "fileName", None) or getattr(doc_src, "title", None) or "file.pdf"
        _, ext = os.path.splitext(orig_name)
        return ext.lower() if ext else ".pdf"
    
    mappings = {
        "image": ".jpg",
        "sticker": ".webp",
        "user_created_sticker": ".webp",
        "video": ".mp4",
        "gif": ".mp4",
        "audio": ".ogg",
        "ptt": ".ogg",
        "document": ".pdf",
    }
    return mappings.get(media_kind, ".bin")


def process_media_message(
    message,
    media_kind,
    chat_id,
    sender_id,
    text_content="",
    target_media_msg=None,
    msg_time=None,
    history_limit=20,
    is_reaction_to_bot=False,
):
    """
    Process a media message by routing it through the agent loop.
    
    This function:
    1. Checks feature flags
    2. Downloads the media
    3. Extracts relevant information
    4. Routes to the agent loop with proper context
    
    Returns the AI's response or None if no response should be sent.
    """
    # Feature flag checks
    chat_has_any_feature = _admin_commands.has_any_feature_enabled(chat_id) if _admin_commands else False

    feature_checks = {
        "document": ("documents", "\ud83d\udcc4 Document reading is currently turned off for this chat."),
        "video": ("videos", "\ud83c\udfa5 Video processing is currently turned off for this chat."),
        "gif": ("videos", "\ud83c\udfa5 Video processing is currently turned off for this chat."),
        "image": ("images", "\ud83d\uddbc\ufe0f Image processing is currently turned off for this chat."),
        "sticker": ("images", "\ud83d\uddbc\ufe0f Image processing is currently turned off for this chat."),
        "user_created_sticker": ("images", "\ud83d\uddbc\ufe0f Image processing is currently turned off for this chat."),
        "audio": ("audio", "\ud83c\udfa4 Voice notes are currently turned off for this chat."),
        "ptt": ("audio", "\ud83c\udfa4 Voice notes are currently turned off for this chat."),
    }

    if media_kind in feature_checks:
        feature_name, message_text = feature_checks[media_kind]
        if not _admin_commands.is_feature_enabled(chat_id, feature_name):
            print(f"[FEATURE OFF] '{feature_name}' disabled for {chat_id}.")
            return message_text if chat_has_any_feature else None

    # Download the media
    tmp_path = None
    try:
        suffix = _get_suffix(media_kind, 
            target_media_msg.documentMessage if target_media_msg and getattr(target_media_msg, "documentMessage", None) 
            else getattr(message.Message, "documentMessage", None)
        )
        tmp_path = _download_media(message, target_media_msg, suffix)

        # Get MIME type and other metadata
        doc_src = (
            target_media_msg.documentMessage
            if target_media_msg and getattr(target_media_msg, "documentMessage", None)
            else getattr(message.Message, "documentMessage", None)
        )
        mime_type = _get_mime_type(media_kind, doc_src, tmp_path)
        
        # For documents, get filename
        filename = ""
        if media_kind == "document" and doc_src:
            filename = getattr(doc_src, "fileName", None) or getattr(doc_src, "title", None) or "file.pdf"
        
        # Read media as base64
        with open(tmp_path, "rb") as f:
            media_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Handle different media types
        user_prompt = text_content.strip() if text_content and text_content.strip() else None

        # For documents, check if it's a PDF or extractable text
        if media_kind == "document":
            magic = _file_magic_kind(tmp_path) if _file_magic_kind else None
            if magic == "pdf" or mime_type == "application/pdf":
                # PDF - use analyze_media tool with vision
                media_context = {
                    "type": "document",
                    "base64_data": media_b64,
                    "mime_type": "application/pdf",
                    "filename": filename,
                    "user_prompt": user_prompt or f"Summarize or explain this PDF ({filename}).",
                }
            else:
                # Try to extract text first
                try:
                    body = _extract_document_text(tmp_path, mime_type, os.path.splitext(filename)[1].lower())
                    if body:
                        # Text extracted successfully - route through agent with text
                        context_str = f"[Document: {filename}]\n{body}"
                        if user_prompt:
                            context_str = f"User said: {user_prompt}\n\n{context_str}"
                        
                        _insert_chat_message(chat_id, "document_text", "user", context_str)
                        
                        # Route through agent loop
                        from agent_loop import run_agent
                        is_group = "g.us" in (chat_id or "")
                        return run_agent(
                            chat_id=chat_id,
                            sender_id=sender_id,
                            history_limit=history_limit,
                            is_group=is_group,
                            msg_time=msg_time,
                            extra_user_note=context_str,
                        )
                    else:
                        # Can't extract text - use analyze_media
                        media_context = {
                            "type": "document",
                            "base64_data": media_b64,
                            "mime_type": mime_type,
                            "filename": filename,
                            "user_prompt": user_prompt or "What is in this document?",
                        }
                except Exception as e:
                    print(f"[DOC EXTRACT ERROR] {filename}: {e}")
                    media_context = {
                        "type": "document",
                        "base64_data": media_b64,
                        "mime_type": mime_type,
                        "filename": filename,
                        "user_prompt": user_prompt or "What is in this document?",
                    }

        elif media_kind in ("audio", "ptt"):
            # Voice note - use transcribe_audio tool
            media_context = {
                "type": "audio",
                "base64_data": media_b64,
                "mime_type": mime_type,
                "text_content": user_prompt or "",
            }

        elif media_kind in ("image", "sticker", "user_created_sticker", "gif", "video"):
            # Visual media - use analyze_media tool
            # For stickers, use image/webp
            if media_kind in ("sticker", "user_created_sticker"):
                mime_type = "image/webp"
            elif media_kind in ("gif", "video"):
                mime_type = "video/mp4"
            else:
                mime_type = "image/jpeg"
            
            media_context = {
                "type": media_kind,
                "base64_data": media_b64,
                "mime_type": mime_type,
                "user_prompt": user_prompt or "What is in this media?",
            }
        else:
            return None

        # Route through agent loop with media context
        if _admin_commands and (
            _admin_commands.is_feature_enabled(chat_id, "ai_chat") or 
            _admin_commands.is_feature_enabled(chat_id, "reminders")
        ):
            from agent_loop import run_agent
            is_group = "g.us" in (chat_id or "")
            return run_agent(
                chat_id=chat_id,
                sender_id=sender_id,
                sender_num=None,
                history_limit=history_limit,
                is_group=is_group,
                msg_time=msg_time,
                extra_user_note=f"[Media: {media_kind}] User said: {user_prompt or ''}",
                latest_user_text=user_prompt or "",
                media_context=media_context,
            )
        
        return None

    except Exception as e:
        print(f"Error processing media: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
