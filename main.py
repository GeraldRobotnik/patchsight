import cv2
import depthai as dai

print("PatchSight — Camera anchor (DepthAI v3)")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()

    rgbQ = cam.requestOutput(
        size=(640, 480),
        type=dai.ImgFrame.Type.BGR888p,
        resizeMode=dai.ImgResizeMode.CROP,   # <-- CORRECT NAME
        fps=30,
        enableUndistortion=True
    ).createOutputQueue()

    pipeline.start()

    while pipeline.isRunning():
        frame = rgbQ.get().getCvFrame()
        cv2.imshow("PatchSight RGB (Undistorted)", frame)

        if cv2.waitKey(1) == ord("q"):
            break