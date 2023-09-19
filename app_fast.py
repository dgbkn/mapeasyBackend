from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import io
import uvicorn


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")



@app.post("/lap")
async def upload_image(file: UploadFile):
    try:
        # Read the uploaded image into memory
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image using OpenCV
        # Example processing: Convert to grayscale
        laplacian = cv2.Laplacian(image,cv2.CV_64F)

        # Encode the processed image as JPEG in memory
        processed_image_data = laplacian.tobytes()
        return FileResponse(io.BytesIO(processed_image_data), media_type='image/jpeg')

    except Exception as e:
        return {"error": str(e)}

@app.post("/canny")
async def upload_image_canny(file: UploadFile):
    try:
        # Read the uploaded image into memory
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image using OpenCV
        # Example processing: Convert to grayscale
        edges = cv2.Canny(image,250,250)
        # Encode the processed image as JPEG in memory
        processed_image_data = edges.tobytes()
        return FileResponse(io.BytesIO(processed_image_data), media_type='image/jpeg')

    except Exception as e:
        return {"error": str(e)}
    
    
    
@app.post("/sobel")
async def upload_image(file: UploadFile):
    try:
        # Read the uploaded image into memory
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image using OpenCV
        # Example processing: Convert to grayscale
            # filtered_image_y = cv2.filter2D(image, -1, sobel_y)
        sobel_x = np.array([[-1,0,1],
                    [ -2, 0 , 2],
                    [ -1,0,1]])
        filtered_image_x = cv2.filter2D(image, -1, sobel_x)

        # Encode the processed image as JPEG in memory
        processed_image_data = filtered_image_x.tobytes()
        return FileResponse(io.BytesIO(processed_image_data), media_type='image/jpeg')

    except Exception as e:
        return {"error": str(e)}
   


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
