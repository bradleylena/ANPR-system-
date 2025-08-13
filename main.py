import cv2
import pytesseract
import re

# Link to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Brad\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Load the image
image = cv2.imread('car.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Optional: improve contrast
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Edge detection
edged = cv2.Canny(gray, 30, 200)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

plate_img = None
for c in contours:
    approx = cv2.approxPolyDP(c, 10, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(c)
        plate_img = gray[y:y + h, x:x + w]
        break

if plate_img is not None:
    text = pytesseract.image_to_string(plate_img, config='--psm 8')
    clean_text = re.sub('[^A-Z0-9]', '', text.upper())
    print("Detected Plate:", clean_text)
    cv2.imshow("Detected Plate", plate_img)
else:
    print("No plate detected")

cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
