# test_dxcam.py
import dxcam
import numpy as np

camera = dxcam.create(output_idx=0, output_color="RGB")
print("camera:", camera)

frame = camera.grab()
print("frame type:", type(frame))

if frame is None:
    print("frame is None (화면을 못 잡고 있음)")
else:
    print("shape:", frame.shape)
    print("dtype:", frame.dtype)
    print("mean:", np.mean(frame))

camera.release()
