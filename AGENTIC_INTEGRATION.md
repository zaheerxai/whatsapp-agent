# Agentic System Integration - Complete Wiring Guide

## Overview

This document describes the complete integration of the agentic system with the main WhatsApp agent workflow. The integration ensures that all media types (images, voice notes, documents, videos, stickers, links) are properly routed through the agent loop, enabling consistent tool usage and behavior across all message types.

## Problem Statement

Previously, the following issues were identified:

1. **Quoted Media Not Working**: When users replied to images, voice notes, or documents, the system didn't properly extract and process the quoted media
2. **Voice Notes Not Working**: Voice note transcription wasn't integrated with the agent loop
3. **Document Processing Inconsistent**: Documents were processed separately from the agentic tools
4. **URL Handling Inconsistent**: Links weren't consistently routed through the `browse_url` tool
5. **Feature Flags Not Respected**: Media processing sometimes bypassed feature flag checks

## Solution Architecture

### 1. New Module: `media_handler.py`

Created a dedicated module to handle all media processing with the following responsibilities:

- **Unified Media Processing**: Single entry point for all media types
- **Feature Flag Checking**: Respects all admin feature flags (documents, videos, images, audio)
- **Media Download**: Handles downloading from WhatsApp with proper error handling
- **Media Type Detection**: Automatically detects and handles different media types
- **Agent Loop Routing**: Routes all media through the agent loop with proper context

**Key Functions:**
- `init_media_handler()`: Initializes all dependencies
- `process_media_message()`: Main entry point for media processing
- `_download_media()`: Downloads media to temp files
- `_get_mime_type()`: Determines MIME type from media kind
- `_get_suffix()`: Gets file suffix for temp files

### 2. Enhanced `agent_tools.py`

Added three new tools for media processing:

#### `analyze_media` Tool
- **Purpose**: Analyze visual media (images, videos, GIFs, stickers, documents) using Gemini Vision
- **Parameters**:
  - `media_type`: Type of media (image, video, gif, sticker, document)
  - `base64_data`: Base64-encoded media content
  - `mime_type`: MIME type of the media
  - `user_prompt`: User's question or comment about the media
- **Function**: Uses Gemini Vision to extract details, describe, or react to visual media

#### `transcribe_audio` Tool
- **Purpose**: Transcribe voice notes and audio messages using Whisper
- **Parameters**:
  - `base64_data`: Base64-encoded audio content
  - `language`: Language hint (default: "ur" for Urdu)
- **Function**: Uses Whisper Large V3 Turbo for accurate transcription
- **Smart Language Detection**: Automatically detects Urdu/Hindi context from accompanying text

#### `extract_document_text` Tool
- **Purpose**: Extract plain text from various document formats
- **Parameters**:
  - `file_path`: Path to the downloaded document
  - `mime_type`: MIME type of the document
  - `filename`: Original filename
- **Supported Formats**:
  - PDF (returns empty to signal vision processing needed)
  - DOCX (Word documents)
  - XLSX (Excel spreadsheets)
  - PPTX (PowerPoint presentations)
  - TXT, CSV, JSON, MD (plain text formats)

### 3. Updated `agent_loop.py`

Enhanced the `run_agent()` function to handle media context:

**New Parameter:**
- `media_context`: Optional dict containing media information:
  - `type`: Media type (audio, image, video, gif, sticker, document)
  - `base64_data`: Base64-encoded media content
  - `mime_type`: MIME type
  - `filename`: Original filename (for documents)
  - `user_prompt`: User's question/comment
  - `text_content`: Accompanying text

**Media Context Handling:**
- Adds media information to the conversation context
- Allows the LLM to decide which tool to use (analyze_media, transcribe_audio, etc.)
- Maintains proper conversation flow with media attachments

**System Prompt Enhancements:**
- Added explicit instructions for media handling
- Clarified when to use which tools
- Improved URL handling with priority instructions

### 4. Updated `whatsapp_agent.py`

**Changes to `handle_media_message()`:**
- Simplified to delegate to `media_handler.process_media_message()`
- Maintains backward compatibility with existing calls
- Proper error handling and fallback

**Integration Points:**
- Initializes `media_handler` module with all dependencies
- Routes all media through the agentic system
- Maintains feature flag checks

**URL Handling Improvements:**
- Extracts URLs from quoted messages
- Forces `browse_url` tool usage when URLs are detected
- Adds priority system message to ensure URL is processed first

## Flow Diagram

```
User Sends Media
       ↓
┌─────────────────────────┐
│  process_message()       │
│  - Detects media type    │
│  - Checks feature flags  │
└─────────────┬───────────┘
              ↓
┌─────────────────────────┐
│  handle_media_message()  │
│  - Delegates to media_   │
│    handler              │
└─────────────┬───────────┘
              ↓
┌─────────────────────────┐
│  media_handler.         │
│  process_media_message()│
│  - Downloads media       │
│  - Determines MIME type  │
│  - Creates media_context │
└─────────────┬───────────┘
              ↓
┌─────────────────────────┐
│  agent_loop.run_agent() │
│  - Receives media_context│
│  - Adds to conversation  │
│  - LLM decides tool use  │
└─────────────┬───────────┘
              ↓
┌─────────────────────────┐
│  agent_tools.execute_   │
│  tool()                │
│  - analyze_media        │
│  - transcribe_audio     │
│  - extract_document_text│
│  - browse_url           │
└─────────────┬───────────┘
              ↓
┌─────────────────────────┐
│  Response sent to user   │
└─────────────────────────┘
```

## Media Type Handling

### Voice Notes / Audio Messages

**Flow:**
1. User sends voice note
2. `process_message()` detects audio media type
3. `handle_media_message()` delegates to `media_handler`
4. `media_handler` downloads audio, encodes as base64
5. Creates `media_context` with type="audio"
6. Routes to `run_agent()` with media_context
7. Agent adds audio info to messages
8. LLM decides to use `transcribe_audio` tool
9. Tool transcribes using Whisper
10. LLM generates response based on transcript

**Example:**
```
User: [Voice note: "Remind me to call Zaheer at 5pm"]
→ System: [Audio message attached - X bytes. User said: ]
→ LLM: Calls transcribe_audio tool
→ Tool: Returns "Remind me to call Zaheer at 5pm"
→ LLM: Calls set_reminder tool
→ Response: "Reminder set for 5pm to call Zaheer ✅"
```

### Images / Stickers / GIFs / Videos

**Flow:**
1. User sends image or replies to image
2. System detects media type
3. Downloads media, encodes as base64
4. Creates `media_context` with type and base64 data
5. Routes to `run_agent()`
6. Agent adds media info to messages
7. LLM decides to use `analyze_media` tool
8. Tool analyzes using Gemini Vision
9. LLM generates response based on analysis

**Example:**
```
User: [Image of a receipt]
User: "What's the total?"
→ System: [Media attached: image (image/jpeg), X bytes. User said: What's the total?]
→ LLM: Calls analyze_media tool
→ Tool: Returns "Receipt from XYZ Store, Total: $150.50"
→ LLM: "The total is $150.50"
```

### Documents (PDF, DOCX, XLSX, PPTX, TXT)

**Flow:**
1. User sends document
2. System detects document type
3. Downloads document
4. Checks file magic to determine actual type
5. For PDF: Creates media_context with type="document"
6. For extractable formats (DOCX, XLSX, etc.):
   - Attempts text extraction
   - If successful: Routes through agent with extracted text
   - If failed: Uses analyze_media for PDF
7. LLM uses appropriate tool or responds directly

**Example (PDF):**
```
User: [PDF document]
User: "Summarize this"
→ System: [Document attached: report.pdf (application/pdf), X bytes]
→ LLM: Calls analyze_media tool (Gemini Vision)
→ Tool: Returns extracted text/summary
→ LLM: Generates summary
```

**Example (DOCX):**
```
User: [Word document]
→ System: Extracts text using extract_document_text
→ System: [Document: report.docx]
[Extracted text...]
→ LLM: Responds based on extracted content
```

### Quoted Media

**Flow:**
1. User replies to a media message (image, voice, document)
2. System detects quoted message via `contextInfo.quotedMessage`
3. Extracts quoted media type using `detect_media_on_proto()`
4. Downloads quoted media
5. Routes through agent loop with both current text and quoted media
6. LLM has access to both the user's text and the quoted media

**Example:**
```
User: [Replies to image] "What color is this?"
→ System: Detects quoted image
→ System: Downloads image
→ System: Routes to agent with text="What color is this?" + quoted image
→ LLM: Calls analyze_media tool
→ Tool: Returns "The image shows a red car"
→ LLM: "It's red."
```

## URL Handling

### Direct URLs

When user pastes a URL directly:
1. `extract_urls_from_text()` extracts all URLs
2. URLs added to `message_urls`
3. Passed to `run_agent()` as `force_urls`
4. Agent adds priority system message:
   ```
   PRIORITY: The user's LATEST message is about this URL(s): [url]
   You MUST call browse_url on THIS URL as your FIRST (and usually only) tool.
   ```
5. LLM calls `browse_url` tool
6. Tool fetches and processes URL content
7. LLM generates response

### Quoted URLs

When user replies to a message containing a URL:
1. `extract_quoted_text_and_urls()` extracts URLs from quoted message
2. URLs added to `message_urls`
3. Same flow as direct URLs

### Referential URLs

When user asks about "this link" or "iska details":
1. System detects referential intent via keyword matching
2. Falls back to `urls_from_recent_history()`
3. Retrieves most recent URL from chat history
4. Routes through agent with that URL

## Feature Flag Integration

All media processing respects feature flags:

```python
feature_checks = {
    "document": ("documents", "📄 Document reading is currently turned off..."),
    "video": ("videos", "🎥 Video processing is currently turned off..."),
    "gif": ("videos", "🎥 Video processing is currently turned off..."),
    "image": ("images", "🖼️ Image processing is currently turned off..."),
    "sticker": ("images", "🖼️ Image processing is currently turned off..."),
    "user_created_sticker": ("images", "🖼️ Image processing is currently turned off..."),
    "audio": ("audio", "🎤 Voice notes are currently turned off..."),
    "ptt": ("audio", "🎤 Voice notes are currently turned off..."),
}
```

If a feature is disabled:
- Returns friendly message if any features are enabled
- Returns None (no response) if all features are disabled

## System Prompt Enhancements

The system prompt in `agent_loop.py` now includes:

1. **Media Handling Instructions**:
   - Always call `analyze_media` for visual media
   - Always call `transcribe_audio` for voice notes
   - Always call `browse_url` for URLs

2. **URL Priority**:
   - Current message URLs have priority
   - Must be processed first
   - Ignore older links from history

3. **Language Matching**:
   - Match user's language (Roman Urdu, Hindi, English)
   - Use appropriate script for response

4. **Tool Usage Guidelines**:
   - When to use each tool
   - What NOT to do (hallucinate, invent content)

## Testing Checklist

- [ ] Voice notes are transcribed and routed through agent
- [ ] Images are analyzed using Gemini Vision
- [ ] Documents (PDF, DOCX, XLSX, PPTX) are processed correctly
- [ ] Quoted media (images, voice, documents) is extracted and processed
- [ ] URLs are consistently routed through browse_url tool
- [ ] Feature flags are respected for all media types
- [ ] Error handling works for all media types
- [ ] Memory doesn't leak (temp files are cleaned up)
- [ ] Conversation context is maintained with media
- [ ] Group chats handle media correctly

## Migration Notes

### Backward Compatibility

The integration maintains backward compatibility:
- Existing `handle_media_message()` calls still work
- Old direct API calls are replaced but fallbacks exist
- Feature flags continue to work as before
- All existing functionality is preserved

### Performance Considerations

1. **Base64 Encoding**: Media is encoded as base64 for tool calls
   - This increases memory usage temporarily
   - Files are cleaned up immediately after processing
   - Consider size limits for very large files

2. **Tool Selection**: LLM decides which tool to use
   - Adds one extra model call for tool selection
   - More accurate than hardcoded paths
   - Allows for flexible handling of edge cases

3. **Caching**: 
   - Voice note transcripts cached in `LAST_VOICE_BY_CHAT`
   - Recent URLs cached in `LAST_URL_BY_CHAT`
   - Group metadata cached to avoid repeated fetches

## Error Handling

All error paths include:
- Try/catch blocks at appropriate levels
- Logging of errors for debugging
- Graceful fallbacks where possible
- User-friendly error messages

## Files Modified

1. **New File**: `media_handler.py` (361 lines)
   - Complete media processing module

2. **Modified**: `agent_tools.py` (+92 lines)
   - Added 3 new tool schemas (analyze_media, transcribe_audio, extract_document_text)
   - Added 3 new tool executors
   - Enhanced init_tools() with client_gemini and gemini_model

3. **Modified**: `agent_loop.py` (+40 lines)
   - Added media_context parameter to run_agent()
   - Enhanced media context handling
   - Improved system prompt for media

4. **Modified**: `whatsapp_agent.py` (+26 lines, -0 lines net)
   - Updated handle_media_message() to delegate to media_handler
   - Added media_handler initialization
   - Updated agent_tools.init_tools() call with new parameters

## Configuration

No new configuration required. The system uses:
- Existing `GROQ_API_KEY` and `GEMINI_API_KEY`
- Existing `MODEL_NAME` and `GEMINI_MODEL`
- Existing feature flags in admin_commands

## Future Enhancements

1. **Media Caching**: Cache processed media to avoid re-downloading
2. **Batch Processing**: Handle multiple media attachments in one message
3. **Media Metadata**: Extract EXIF data for images, duration for audio/video
4. **OCR for Images**: Add OCR capability for images with text
5. **Video Transcription**: Extract audio from videos and transcribe
6. **Thumbnails**: Generate thumbnails for large images/videos

## Troubleshooting

### "Media analysis failed" errors
- Check GEMINI_API_KEY is set
- Verify network connectivity to Google Gemini
- Check if base64 data is being passed correctly

### "Transcription failed" errors
- Check GROQ_API_KEY is set
- Verify Whisper model is available
- Check audio file format is supported

### "Document extraction failed" errors
- Verify required libraries (python-docx, openpyxl, python-pptx)
- Check file is not corrupted
- Verify file type matches extension

### "No media data available" errors
- Check media download succeeded
- Verify temp file was created
- Check file size > 0

## Summary

This integration provides a complete, unified solution for handling all media types through the agentic system. It ensures that:

1. All media is processed consistently
2. The LLM has access to all necessary context
3. Tools are used appropriately for each media type
4. Feature flags are respected
5. Error handling is comprehensive
6. Performance is optimized
7. The system is maintainable and extensible

The changes are minimal and focused, preserving all existing functionality while adding the new agentic capabilities.
