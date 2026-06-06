using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Media.Capture;
using Windows.Media.Capture.Frames;
using Windows.Graphics.Imaging;

var port = args.Length > 0 ? int.Parse(args[0]) : 9876;
Console.WriteLine($"IR Bridge on port {port}...");

// Find IR camera
var groups = await MediaFrameSourceGroup.FindAllAsync();
MediaFrameSourceGroup? irGroup = null;
foreach (var g in groups)
    foreach (var info in g.SourceInfos)
        if (info.SourceKind == MediaFrameSourceKind.Infrared) { irGroup = g; break; }

if (irGroup == null) { Console.Error.WriteLine("No IR camera"); return; }
Console.WriteLine($"IR: {irGroup.DisplayName}");

// TCP listener
var listener = new TcpListener(IPAddress.Loopback, port);
listener.Start();
Console.WriteLine($"Waiting on localhost:{port}...");
var client = await listener.AcceptTcpClientAsync();
var stream = client.GetStream();
Console.WriteLine("Python connected!");

// MediaCapture
var capture = new MediaCapture();
await capture.InitializeAsync(new MediaCaptureInitializationSettings
{
    SourceGroup = irGroup,
    SharingMode = MediaCaptureSharingMode.ExclusiveControl,
    StreamingCaptureMode = StreamingCaptureMode.Video,
    MemoryPreference = MediaCaptureMemoryPreference.Cpu
});

var frameSource = capture.FrameSources.First().Value;
var format = frameSource.SupportedFormats[0];
await frameSource.SetFormatAsync(format);
int w = (int)format.VideoFormat.Width, h = (int)format.VideoFormat.Height;

var reader = await capture.CreateFrameReaderAsync(frameSource);
reader.AcquisitionMode = MediaFrameReaderAcquisitionMode.Realtime;

// Send header: [w:4][h:4]
var header = new byte[8];
BitConverter.GetBytes(w).CopyTo(header, 0);
BitConverter.GetBytes(h).CopyTo(header, 4);
await stream.WriteAsync(header);

Console.WriteLine($"Streaming {w}x{h}. Ctrl+C to stop.");

SoftwareBitmap? latest = null;
var evt = new ManualResetEventSlim(false);

reader.FrameArrived += (_, _) =>
{
    var f = reader.TryAcquireLatestFrame();
    var vf = f?.VideoMediaFrame;
    // 只发送 IR LED 点亮帧，跳过暗帧
    if (vf?.InfraredMediaFrame?.IsIlluminated != true) return;
    var b = vf.SoftwareBitmap;
    if (b is null) return;
    if (b.BitmapPixelFormat != BitmapPixelFormat.Bgra8)
        b = SoftwareBitmap.Convert(b, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
    Interlocked.Exchange(ref latest, SoftwareBitmap.Copy(b));
    evt.Set();
};

await reader.StartAsync();

var cts = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

try
{
    while (!cts.IsCancellationRequested)
    {
        evt.Wait(100);
        evt.Reset();
        var b = Interlocked.Exchange(ref latest, null);
        if (b is null) continue;
        int size = w * h * 4;
        await stream.WriteAsync(BitConverter.GetBytes(size));
        var buf = new byte[size];
        b.CopyToBuffer(buf.AsBuffer());
        await stream.WriteAsync(buf);
        await stream.FlushAsync();
        b.Dispose();
    }
}
finally
{
    await reader.StopAsync();
    capture.Dispose();
    client.Close();
    listener.Stop();
    Console.WriteLine("\nStopped.");
}
