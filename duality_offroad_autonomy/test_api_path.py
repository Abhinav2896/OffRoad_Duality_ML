import urllib.request
import urllib.error
import json
import base64
import os
import uuid

PATH_URL = "http://localhost:8000/path"
REPORT_URL = "http://localhost:8000/report"
IMAGE_PATH = r"C:\Users\Abhinav\Desktop\Projects\OffRoad_Duality_ml\duality_offroad_autonomy\Data\Offroad_Segmentation_testImages\Color_Images\0000063.png"

def make_body(image_path):
    boundary = str(uuid.uuid4())
    data = []
    data.append(f'--{boundary}'.encode())
    data.append(f'Content-Disposition: form-data; name="file"; filename="test.png"'.encode())
    data.append('Content-Type: image/png'.encode())
    data.append(b'')
    with open(image_path, 'rb') as f:
        data.append(f.read())
    data.append(f'--{boundary}--'.encode())
    data.append(b'')
    body = b'\r\n'.join(data)
    return boundary, body

def post(url, image_path):
    boundary, body = make_body(image_path)
    req = urllib.request.Request(url, data=body)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req) as response:
        result = response.read()
        return json.loads(result)

def main():
    if not os.path.exists(IMAGE_PATH):
        print("Image not found")
        return
    try:
        path_data = post(PATH_URL, IMAGE_PATH)
        print("Path keys:", path_data.keys())
        assert "path_image" in path_data
        # Save image
        img_b64 = path_data["path_image"]
        with open("path_output.png", "wb") as f:
            f.write(base64.b64decode(img_b64))
        print("Saved path_output.png")

        report_data = post(REPORT_URL, IMAGE_PATH)
        print("Report keys:", report_data.keys())
        assert "status" in report_data and "summary" in report_data
        print("Status:", report_data["status"])
        print("Summary:", report_data["summary"])
    except urllib.error.HTTPError as e:
        print("HTTPError:", e.code, e.read().decode())
    except Exception as e:
        print("Exception:", e)

if __name__ == "__main__":
    main()
