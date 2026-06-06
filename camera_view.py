"""简单摄像头预览 — 按 Esc 退出"""
import sys, cv2

idx = 0
cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"摄像头 [{idx}] 无法打开")
    sys.exit(1)

print(f"摄像头 [{idx}] 预览中… 按 Esc 退出")
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Camera [{idx}]", cv2.flip(frame, 1))
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
