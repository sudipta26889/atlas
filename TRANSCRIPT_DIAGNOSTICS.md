# YouTube Transcript Fetching - Diagnostic Guide

## Common Reasons Why Transcript Fetching Fails

Based on the code analysis and real-world usage, here are the most common reasons why YouTube transcript fetching fails:

### 1. **Transcripts Disabled by Creator** ⚠️
- **Error Message**: "Transcripts are disabled for this video"
- **Cause**: Video creator has disabled captions/transcripts
- **Solution**: Cannot be fixed - video doesn't have transcripts available
- **Frequency**: Common (especially for older videos or certain channels)

### 2. **No Transcripts Available** 📭
- **Error Message**: "No transcripts available for this video (transcript list is empty)"
- **Cause**: Video has never had transcripts generated (no auto-captions, no manual captions)
- **Solution**: Cannot be fixed - video doesn't have transcripts
- **Frequency**: Very common (especially for videos without auto-captions enabled)

### 3. **IP Blocking by YouTube** 🚫
- **Error Message**: Various (often "Video is unavailable" or connection errors)
- **Cause**: YouTube blocks requests from cloud server IPs (AWS, GCP, Azure, etc.)
- **Solution**: 
  - Use a proxy (configure `YOUTUBE_PROXY` environment variable)
  - Use residential IPs instead of datacenter IPs
  - Add delays between requests
- **Frequency**: Very common when running on cloud servers

### 4. **Language Mismatch** 🌍
- **Error Message**: "No transcript found in [language] or en"
- **Cause**: Video only has transcripts in languages other than requested/default
- **Solution**: 
  - The code now tries the first available language as fallback
  - Check available languages in the logs
- **Frequency**: Less common (most videos have English transcripts)

### 5. **Video Unavailable/Restricted** 🔒
- **Error Message**: "Video is unavailable"
- **Cause**: 
  - Video is private/unlisted and requires authentication
  - Video is region-restricted
  - Video has been deleted
- **Solution**: Cannot be fixed - video is not accessible
- **Frequency**: Occasional

### 6. **Rate Limiting** ⏱️
- **Error Message**: Connection timeouts or "Too many requests"
- **Cause**: Making too many requests too quickly
- **Solution**: 
  - Add delays between requests
  - Reduce number of parallel workers
  - Use sequential processing instead of parallel
- **Frequency**: Occasional when processing many videos

### 7. **Network/Connection Issues** 🌐
- **Error Message**: Connection errors, timeouts
- **Cause**: Network instability, DNS issues
- **Solution**: 
  - Check network connectivity
  - Retry the request
  - Check DNS settings
- **Frequency**: Occasional

## How to Diagnose Issues

### Step 1: Check the Error Messages
With the updated code, you'll now see specific error messages:
- Look for messages like "No transcripts available" vs "Transcripts are disabled"
- Check if it says "Video is unavailable" (access issue) vs "No transcript found" (language issue)

### Step 2: Check the Logs
The improved code now logs:
- Available languages: `[TRANSCRIPT-API] Available languages for {video_id}: en, es, fr`
- Which method failed: `[TRANSCRIPT-API] list() failed` or `[TRANSCRIPT-API] Direct fetch succeeded`

### Step 3: Check Your Environment
- **Are you on a cloud server?** → Likely IP blocking, use proxy
- **Are you processing many videos?** → Likely rate limiting, add delays
- **Are videos old or from small channels?** → Likely no transcripts available

### Step 4: Test Individual Videos
Try fetching transcripts for individual videos to isolate the issue:
```python
from src.fetch_youtube_transcript import YouTubeTranscriptFetcher

fetcher = YouTubeTranscriptFetcher()
success, error = fetcher.fetch_transcript("https://www.youtube.com/watch?v=VIDEO_ID")
print(f"Success: {success}, Error: {error}")
```

## Solutions by Issue Type

### For IP Blocking (Cloud Servers)
1. **Set up a proxy**:
   ```bash
   export YOUTUBE_PROXY="socks5://user:pass@proxy-host:port"
   ```

2. **Use residential proxies** instead of datacenter IPs

3. **Add delays** between requests:
   ```python
   import time
   time.sleep(2)  # 2 second delay between requests
   ```

### For Rate Limiting
1. **Reduce parallel workers**:
   ```python
   pipeline = YouTubePipeline(num_workers=1)  # Sequential processing
   ```

2. **Add delays** in the fetching code

3. **Process videos in smaller batches**

### For Language Issues
1. **Check available languages** in the logs
2. **The code automatically falls back** to the first available language
3. **Manually specify language** if needed:
   ```python
   fetcher = YouTubeTranscriptFetcher(language="es")  # Spanish
   ```

## Expected Success Rates

Based on typical YouTube video statistics:
- **Videos with transcripts**: ~60-70% of videos have some form of transcript
- **Auto-generated captions**: Most videos uploaded after 2010 have auto-captions
- **Manual captions**: ~20-30% of videos have manual captions
- **No transcripts**: ~30-40% of videos have no transcripts at all

So a **60-70% success rate is normal**. If you're getting 10% (1/10), that's unusually low and likely indicates:
- IP blocking (if on cloud server)
- Rate limiting (if processing many videos)
- Bad video selection (old videos, small channels)

## Debugging Checklist

- [ ] Check error messages - are they specific?
- [ ] Check logs for available languages
- [ ] Verify you're not on a blocked IP (test from local machine)
- [ ] Check if videos are accessible (can you watch them?)
- [ ] Verify video age (newer videos more likely to have transcripts)
- [ ] Check channel size (larger channels more likely to have transcripts)
- [ ] Test with a known-good video (e.g., a recent popular video)
- [ ] Check if proxy is configured correctly (if using one)
- [ ] Verify network connectivity
- [ ] Check for rate limiting (reduce workers, add delays)

## Next Steps

1. **Run the pipeline again** with the improved error messages
2. **Review the specific error messages** for each failed video
3. **Check the logs** for available languages and method failures
4. **Identify patterns** - are all failures the same type?
5. **Apply appropriate solutions** based on the error types
