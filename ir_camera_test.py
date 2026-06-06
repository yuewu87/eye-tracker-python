"""通过 Windows.Media.Capture.Frames API 获取 IR 摄像头帧"""
import asyncio
import cv2
import numpy as np
from winrt.windows.media.capture.frames import (
    MediaFrameSourceGroup, MediaFrameSourceKind, MediaFrameReaderAcquisitionMode,
)
from winrt.windows.media.capture import (
    MediaCapture, MediaCaptureInitializationSettings,
    MediaCaptureSharingMode, StreamingCaptureMode, MediaCaptureMemoryPreference,
)
from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap


async def find_ir():
    groups = await MediaFrameSourceGroup.find_all_async()
    for g in groups:
        for info in g.source_infos:
            if info.source_kind == MediaFrameSourceKind.INFRARED:
                return g
    return None


async def main():
    group = await find_ir()
    if group is None:
        print("未找到 IR 摄像头")
        return

    cap = MediaCapture()
    settings = MediaCaptureInitializationSettings()
    settings.source_group = group
    settings.sharing_mode = MediaCaptureSharingMode.EXCLUSIVE_CONTROL
    settings.streaming_capture_mode = StreamingCaptureMode.VIDEO
    settings.memory_preference = MediaCaptureMemoryPreference.CPU
    await cap.initialize_async(settings)

    fs = cap.frame_sources
    if fs.size == 0:
        print("无帧源")
        return

    frame_source = list(fs.values())[0]
    fmt = frame_source.supported_formats[0]
    await frame_source.set_format_async(fmt)
    w, h = fmt.video_format.width, fmt.video_format.height
    print(f"IR: {w}x{h} @ {fmt.frame_rate.numerator}/{fmt.frame_rate.denominator}fps")

    reader = await cap.create_frame_reader_async(frame_source)
    reader.acquisition_mode = MediaFrameReaderAcquisitionMode.REALTIME

    latest_frame = [None]

    def on_frame(sender, args):
        ref = sender.try_acquire_latest_frame()
        if ref and ref.video_media_frame:
            bmp = ref.video_media_frame.software_bitmap
            if bmp:
                if bmp.bitmap_pixel_format != BitmapPixelFormat.BGRA8:
                    bmp = SoftwareBitmap.convert(bmp, BitmapPixelFormat.BGRA8)
                latest_frame[0] = SoftwareBitmap.copy(bmp)

    reader.add_frame_arrived(on_frame)
    await reader.start_async()

    print("按 Esc 退出...")
    while True:
        await asyncio.sleep(0.03)
        bmp = latest_frame[0]
        if bmp:
            w, h = bmp.pixel_width, bmp.pixel_height
            buf = bytearray(w * h * 4)
            bmp.copy_to_buffer(buf)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            # BGRA8 -> BGR
            frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.putText(frame, f"mean={gray.mean():.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("IR Camera", frame)
        if cv2.waitKey(1) == 27:
            break

    await reader.stop_async()
    cap.close()
    cv2.destroyAllWindows()


asyncio.run(main())
