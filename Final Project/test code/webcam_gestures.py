"""Run: python webcam_gestures.py [--camera 1] [--scroll | --preview-scroll]."""

import argparse
import math
import os
from pathlib import Path
import time
import urllib.request

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
WINDOW = "MediaPipe Gestures - Q / Esc to quit"


def is_pointing_down(landmarks, width, height):
    """Heuristic: extended index pointing down, other three fingers bent."""
    points = [(p.x * width, p.y * height) for p in landmarks]

    def distance(a, b):
        return math.dist(points[a], points[b])

    def angle(a, b, c):
        ax, ay = points[a][0] - points[b][0], points[a][1] - points[b][1]
        cx, cy = points[c][0] - points[b][0], points[c][1] - points[b][1]
        size = math.hypot(ax, ay) * math.hypot(cx, cy)
        if size < 1e-6:
            return 0.0
        cosine = max(-1.0, min(1.0, (ax * cx + ay * cy) / size))
        return math.degrees(math.acos(cosine))

    palm = distance(0, 9)
    if palm < 15:  # Too small in the image for dependable control.
        return False
    index_extended = angle(5, 6, 7) > 150 and angle(6, 7, 8) > 150
    index_down = points[8][1] - points[5][1] > palm * 0.4
    index_down = index_down and points[8][1] - points[6][1] > palm * 0.15
    others_bent = all(
        angle(mcp, pip, dip) < 150 or distance(0, tip) < distance(0, pip) * 1.05
        for mcp, pip, dip, tip in ((9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    )
    return index_extended and index_down and others_bent


class DownScrollTracker:
    """Convert downward fingertip travel to wheel steps without emitting input."""

    def __init__(self, step=0.035, clicks=2):
        self.step = step  # Fraction of camera-frame height per wheel event.
        self.clicks = clicks
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_time = None
        self.started = None
        self.filtered_y = None
        self.anchor_y = None
        self.last_scroll = float("-inf")

    def update(self, position, now):
        if position is None:
            self.reset()
            return 0
        if self.previous is not None:
            gap = now - self.previous_time
            if gap <= 0 or gap > 0.25 or math.dist(position, self.previous) > 0.15:
                self.reset()
        if self.previous is None:
            self.previous = position
            self.previous_time = self.started = now
            self.filtered_y = self.anchor_y = position[1]
            return 0

        dt = now - self.previous_time
        self.previous = position
        self.previous_time = now
        alpha = 1.0 - math.exp(-dt / 0.06)
        self.filtered_y += alpha * (position[1] - self.filtered_y)
        if now - self.started < 0.2:
            self.anchor_y = self.filtered_y
            return 0

        # Upward travel only moves the reference; it never scrolls up.
        self.anchor_y = min(self.anchor_y, self.filtered_y)
        travel = self.filtered_y - self.anchor_y
        if travel >= self.step and now - self.last_scroll >= 0.1:
            steps = min(3, int(travel / self.step))
            self.anchor_y = self.filtered_y  # Avoid queued scrolls after stopping.
            self.last_scroll = now
            return -self.clicks * steps
        return 0


def make_global_toggle():
    """Poll F8 on Windows, including while the target app has focus."""
    if os.name != "nt":
        return lambda: False
    import ctypes
    from ctypes import wintypes

    get_key_state = ctypes.WinDLL("user32").GetAsyncKeyState
    get_key_state.argtypes = [wintypes.INT]
    get_key_state.restype = wintypes.SHORT
    was_down = False

    def pressed():
        nonlocal was_down
        down = bool(get_key_state(0x77) & 0x8000)  # VK_F8
        edge = down and not was_down
        was_down = down
        return edge

    return pressed


def get_model():
    path = Path(__file__).resolve().with_name("gesture_recognizer.task")
    if not path.is_file():
        print("Downloading the gesture model (first run only)...", flush=True)
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def open_camera(index):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        cap.release()
    raise RuntimeError(
        "Cannot open webcam. Close other camera apps, check Windows camera "
        "permissions, or try: python webcam_gestures.py --camera 1"
    )


def annotate(frame, result):
    height, width = frame.shape[:2]
    for index, landmarks in enumerate(result.hand_landmarks):
        points = [
            (int(p.x * (width - 1)), int(p.y * (height - 1))) for p in landmarks
        ]
        for start, end in CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 220, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 100, 255), -1)

        categories = result.gestures[index] if index < len(result.gestures) else []
        gesture = categories[0] if categories else None
        if gesture and gesture.category_name != "None" and gesture.score >= 0.5:
            label = f"{gesture.category_name} ({gesture.score:.0%})"
        else:
            label = "Unknown gesture"
        cv2.putText(frame, f"Hand {index + 1}: {label}", (10, 65 + index * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    if not result.hand_landmarks:
        cv2.putText(frame, "Show your hand to the camera", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--scroll", action="store_true", help="Enable actual downward scrolling")
    modes.add_argument("--preview-scroll", action="store_true", help="Detect scroll gestures without sending input")
    parser.add_argument("--scroll-step", type=float, default=0.035,
                        help="Downward travel per event / image height (default: 0.035)")
    parser.add_argument("--scroll-clicks", type=int, default=2,
                        help="Wheel clicks per event (default: 2)")
    args = parser.parse_args()
    if not math.isfinite(args.scroll_step) or not 0.005 <= args.scroll_step <= 0.3:
        parser.error("--scroll-step must be between 0.005 and 0.3")
    if not 1 <= args.scroll_clicks <= 10:
        parser.error("--scroll-clicks must be between 1 and 10")

    pyautogui = None
    if args.scroll:
        try:
            import pyautogui
        except ImportError:
            parser.error("Install scrolling support: python -m pip install pyautogui")
        pyautogui.PAUSE = 0.01
        # Keep PyAutoGUI's corner fail-safe enabled.
    scroll_mode = args.scroll or args.preview_scroll
    scroll_enabled = scroll_mode
    tracker = DownScrollTracker(args.scroll_step, args.scroll_clicks)
    toggle_pressed = make_global_toggle() if scroll_mode else lambda: False
    previous_hand = None
    last_event = float("-inf")

    options = mp.tasks.vision.GestureRecognizerOptions(
        # Reading bytes also works when the folder name contains Korean characters.
        base_options=mp.tasks.BaseOptions(model_asset_buffer=get_model().read_bytes()),
        # Process successive webcam frames synchronously for matching overlays.
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = None
    try:
        with mp.tasks.vision.GestureRecognizer.create_from_options(options) as recognizer:
            cap = open_camera(args.camera)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            timestamp_ms = -1
            print("Webcam started. Press Q or Escape in the video window to quit.")
            if scroll_mode:
                print("Point ONE index finger DOWN, hold briefly, then move it down.")
                print("Bend the finger to reposition. F8 (Windows) or Space (video window): toggle.")
                print("For actual scrolling, put the mouse pointer over the target page.")
            while True:
                if toggle_pressed():
                    scroll_enabled = not scroll_enabled
                    tracker.reset()
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Cannot read webcam frames. Check camera access.")
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                result = recognizer.recognize_for_video(image, timestamp_ms)
                annotate(frame, result)
                if scroll_mode:
                    now = time.monotonic()
                    height, width = frame.shape[:2]
                    position = None
                    hand = None
                    pointing = False
                    if len(result.hand_landmarks) == 1:
                        landmarks = result.hand_landmarks[0]
                        pointing = is_pointing_down(landmarks, width, height)
                        if result.handedness and result.handedness[0]:
                            hand = result.handedness[0][0].category_name
                        if pointing and scroll_enabled:
                            position = (landmarks[8].x, landmarks[8].y)
                    if hand != previous_hand:
                        tracker.reset()
                    previous_hand = hand
                    wheel = tracker.update(position, now)
                    if wheel:
                        if args.scroll:
                            try:
                                pyautogui.scroll(wheel)
                            except pyautogui.FailSafeException:
                                scroll_enabled = False
                                tracker.reset()
                                print("Scrolling paused by mouse-corner fail-safe. F8 to resume.")
                            else:
                                last_event = now
                        else:
                            last_event = now
                    state = "ON" if scroll_enabled else "PAUSED"
                    mode = "SCROLL" if args.scroll else "PREVIEW"
                    if not scroll_enabled:
                        message = "Paused"
                    elif now - last_event < 0.3:
                        message = "DOWN" if args.scroll else "DOWN (preview only)"
                    elif pointing and len(result.hand_landmarks) == 1:
                        message = "Pointing down - move down to scroll"
                    else:
                        message = "Use ONE hand: index down, other fingers bent"
                    cv2.putText(frame, f"{mode}: {state} | F8 / Space: Toggle", (10, height - 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(frame, message, (10, height - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, "Q / Esc: Quit", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if scroll_mode and key == ord(" "):
                    scroll_enabled = not scroll_enabled
                    tracker.reset()
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
