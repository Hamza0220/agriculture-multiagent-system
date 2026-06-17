from agents.crop_doctor import analyze_image_with_vision
import base64
with open('assets/splash.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')
print("VISION START")
print(analyze_image_with_vision(b64))
print("VISION END")
