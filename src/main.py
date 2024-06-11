import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# YOLO 네트워크와 클래스 이름을 로드하는 함수
def load_yolo(weights_path, cfg_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# 이미지에서 텍스트를 탐지하는 함수
def detect_text(img, net, output_layers):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, indexes

# 이미지에서 텍스트를 추출하는 함수
def ocr_text_detection(img, boxes, indexes, lang):
    detected_texts = []
    with open("text1.txt", "w", encoding="utf-8") as f:
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                roi = img[y:y+h, x:x+w]
                try:
                    text = pytesseract.image_to_string(roi, lang=lang).strip()
                    detected_texts.append((text, (x, y, w, h)))
                    f.write(text + "\n")
                except Exception as e:
                    print(f"Error extracting text: {e}")
    return detected_texts

# 폰트 지정 처리하는 함수
def render_text_with_font(img, text, position, font_path, font_size):
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=(0, 0, 0))
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error rendering text '{text}' at {position} with font {font_path}: {e}")
    
    return img

# 이미지에 저장된 텍스트를 사용자가 정의한 폰트와 크기로 이미지를 생성하는 함수
def create_image_with_text(font_path, font_size):
    with open("text.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    max_width = max([len(line) for line in lines]) * font_size
    max_height = len(lines) * font_size

    img = np.ones((max_height, max_width, 3), np.uint8) * 255
    for i, line in enumerate(lines):
        img = render_text_with_font(img, line.strip(), (0, i * font_size), font_path, font_size)
    return img

def main():
    # 사용자 입력 받기
    weights_path = input("YOLO 가중치 파일 경로를 입력하세요: ")
    cfg_path = input("YOLO 구성 파일 경로를 입력하세요: ")
    names_path = input("YOLO 클래스 이름 파일 경로를 입력하세요: ")
    input_image_path = input("입력 이미지 파일 경로를 입력하세요: ")
    font_path = input("폰트 파일 경로를 입력하세요: ") 
    font_size = int(input("폰트 크기를 입력하세요: ")) 
    lang = input("텍스트 추출 언어를 입력하세요 (예: 'kor', 'eng', 'kor+eng'): ")

    # YOLO 네트워크 로드
    net, classes, output_layers = load_yolo(weights_path, cfg_path, names_path)
    
    # 입력 이미지 로드
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError("입력 이미지를 찾을 수 없습니다.")
    
    # 이미지에서 텍스트 탐지
    boxes, indexes = detect_text(img, net, output_layers)
    
    # 텍스트 추출
    detected_texts = ocr_text_detection(img, boxes, indexes, lang)
    
    # 텍스트가 있는 이미지 생성
    text_image = create_image_with_text(font_path, font_size)
    
    # 결과 이미지 저장
    cv2.imwrite('output_image.jpg', text_image)
    print("output_image.jpg 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()
