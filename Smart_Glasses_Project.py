"""
مشروع النظارات الذكية - نظام تحليل شكل الوجه واقتراح النظارات.
Smart Glasses Project - Face Shape Analysis and Glasses Recommendation System.

التبعيات (Dependencies):
- opencv-python (cv2)
- cvzone
- numpy
- pillow (PIL)
- arabic-reshaper
- python-bidi
"""
import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import tkinter as tk
import math
from collections import Counter
import os

# === مكتبات اللغة العربية (Arabic Support Libraries) ===
import arabic_reshaper
from bidi.algorithm import get_display

# === استيراد كلاس الخبير (Import Expert System) ===
from SmartExpert import SmartExpert

# ==========================================
# إعدادات الألوان (Color Theme)
# ==========================================
COLOR_BG_DARK = (15, 15, 20)
COLOR_BG_SIDEBAR = (25, 25, 32)
COLOR_BG_CARD = (35, 35, 45)
COLOR_BG_GLASS = (40, 40, 55)
COLOR_ACCENT_PRIMARY = (0, 180, 255)
COLOR_ACCENT_SECONDARY = (255, 180, 50)
COLOR_TEXT_PRIMARY = (245, 245, 245)
COLOR_TEXT_SECONDARY = (180, 180, 200)
COLOR_TEXT_MUTED = (120, 120, 140)
COLOR_SUCCESS = (50, 220, 100)
COLOR_WARNING = (255, 100, 100)
COLOR_BORDER = (60, 60, 80)

# ==========================================
# دوال مساعدة (رسم وكتابة)
# ==========================================
def overlay_image_alpha(img, img_overlay, x, y, scale=1.0):
    if img_overlay is None: return img
    
    if scale != 1.0:
        h, w = img_overlay.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_overlay = cv2.resize(img_overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o: return img

    if img_overlay.shape[2] == 4:
        alpha_mask = img_overlay[y1o:y2o, x1o:x2o, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(3):
            img[y1:y2, x1:x2, c] = (alpha_mask * img_overlay[y1o:y2o, x1o:x2o, c] + 
                                   alpha_inv * img[y1:y2, x1:x2, c])
    else:
        img[y1:y2, x1:x2] = img_overlay[y1o:y2o, x1o:x2o, :3]
    return img

def put_arabic_text(img, text, position, font_size=30, color=(255, 255, 255), align="right"):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    
    x, y = position
    if align == "center": x -= text_width // 2
    elif align == "right": x -= text_width
    
    draw.text((x+1, y+1), bidi_text, font=font, fill=(0,0,0))
    draw.text((x, y), bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=1, fill=False):
    x1, y1 = pt1
    x2, y2 = pt2
    if fill:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, -1)
        cv2.circle(overlay, (x1+radius, y1+radius), radius, color, -1)
        cv2.circle(overlay, (x2-radius, y1+radius), radius, color, -1)
        cv2.circle(overlay, (x1+radius, y2-radius), radius, color, -1)
        cv2.circle(overlay, (x2-radius, y2-radius), radius, color, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    else:
        cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
        cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
        cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
        cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_gradient_background(img, rect, color1, color2, direction='horizontal'):
    x1, y1, x2, y2 = rect
    if direction == 'horizontal':
        for i in range(x1, x2):
            ratio = (i - x1) / (x2 - x1)
            color = tuple(int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3))
            cv2.line(img, (i, y1), (i, y2), color, 1)
    else:
        for i in range(y1, y2):
            ratio = (i - y1) / (y2 - y1)
            color = tuple(int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3))
            cv2.line(img, (x1, i), (x2, i), color, 1)

# ==========================================
# المنطق الهندسي وتحديد شكل الوجه
# Geometric Logic & Face Shape Detection
# ==========================================
def calculate_angle(p1, p2, p3):
    """حساب الزاوية بين ثلاث نقاط."""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0: angle += 360
    if angle > 180: angle = 360 - angle
    return angle

def is_head_aligned(detector, face):
    """
    التحقق من أن الرأس ينظر للأمام مباشرة.
    Check if head is aligned (looking forward).
    """
    pt_nose = face[4]; pt_ear_L = face[234]; pt_ear_R = face[454]
    dist_L, _ = detector.findDistance(pt_nose, pt_ear_L)
    dist_R, _ = detector.findDistance(pt_nose, pt_ear_R)
    yaw_ratio = dist_L / (dist_R + 1e-5)
    return 0.60 < yaw_ratio < 1.60

def get_geometric_shape(detector, face):
    """
    تحليل المقاييس الهندسية للوجه وتحديد الشكل بناءً على القواعد.
    Analyze face geometric metrics and determine shape based on rules.
    """
    pt_cheek_L, pt_cheek_R = face[234], face[454]
    pt_jaw_L, pt_jaw_R = face[132], face[361]
    pt_head_top, pt_chin_bottom = face[10], face[152]
    pt_fore_L, pt_fore_R = face[103], face[332]
    chin_angle = calculate_angle(face[132], face[152], face[361])
    h_face, _ = detector.findDistance(pt_head_top, pt_chin_bottom)
    w_cheeks, _ = detector.findDistance(pt_cheek_L, pt_cheek_R)
    w_jaw, _ = detector.findDistance(pt_jaw_L, pt_jaw_R)
    w_forehead, _ = detector.findDistance(pt_fore_L, pt_fore_R)
    
    face_ratio = (h_face / w_cheeks) * 100
    jaw_ratio = (w_jaw / w_cheeks) * 100
    forehead_ratio = (w_forehead / w_cheeks) * 100
    
    shape_eng = "Oval"
    # الخوارزمية لتصنيف شكل الوجه
    # Face Shape Classification Algorithm
    if face_ratio >= 128: shape_eng = "Oblong"
    elif face_ratio <= 123:
        if chin_angle >= 89.2: shape_eng = "Square"
        else: shape_eng = "Round"
    else:
        if jaw_ratio > 94: shape_eng = "Oval"
        elif chin_angle < 84:
            if forehead_ratio > 70: shape_eng = "Heart"
            else: shape_eng = "Diamond"
        else: shape_eng = "Oval"
    data = {'shape': shape_eng, 'jaw_width': w_jaw, 'cheek_width': w_cheeks, 'angle': chin_angle}
    stats = (face_ratio, chin_angle, jaw_ratio, forehead_ratio)
    return data, shape_eng, stats

# ==========================================
# النظام الرئيسي 
# Main System Logic
# ==========================================
def start_system():
    window_name = "Smart Vision Pro"
    # إعداد النافذة بملء الشاشة
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0) 
    detector = FaceMeshDetector(maxFaces=1)
    expert_engine = SmartExpert()
    
    shape_ar_map = {
        "Oval": "بيضاوي", "Round": "دائري", "Square": "مربع",
        "Oblong": "مستطيل/طويل", "Heart": "قلب", "Diamond": "ماسي"
    }

    # تحميل الصور من مجلد الأصول
    # Load images from assets folder
    glasses_images = {}
    assets_dir = "assets"
    if not os.path.exists(assets_dir): os.makedirs(assets_dir, exist_ok=True)
    
    for key, val in expert_engine.glass_types.items():
        img_name = val['img']
        img_path = os.path.join(assets_dir, f"{img_name}.png")
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            glasses_images[img_name] = img
        else:
            print(f"Warning: Image not found {img_path}")

    shapes_buffer = [] 
    BUFFER_SIZE = 15

    while True:
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)

        # === [الحل] فرض دقة عالية للنافذة ===
        # Force high resolution for the window UI
        target_h = 720
        aspect_ratio = img.shape[1] / img.shape[0]
        target_w = int(target_h * aspect_ratio)
        img = cv2.resize(img, (target_w, target_h))
        # ====================================
        
        h, w, _ = img.shape
        sidebar_w = int(target_h * 0.6) # عرض الشريط نسبي للطول
        
        # إنشاء الصورة النهائية (Canvas)
        final_img = np.zeros((h, w + sidebar_w, 3), dtype='uint8')
        final_img[0:h, 0:w] = img
        
        draw_gradient_background(final_img, (w, 0, w+sidebar_w, h), COLOR_BG_SIDEBAR, COLOR_BG_DARK, 'vertical')

        img_mesh, faces = detector.findFaceMesh(img, draw=False)
        final_img[0:h, 0:w] = img_mesh

        # متغيرات الحالة
        display_shape = "..."
        display_rec_name = ""
        display_rec_desc = ""
        display_rec_img_key = None
        status_color = COLOR_ACCENT_SECONDARY
        warning_msg = ""
        stats = (0, 0, 0, 0)
        sidebar_center_x = w + (sidebar_w // 2)

        # التحليل
        if faces:
            face = faces[0]
            if is_head_aligned(detector, face):
                face_data, raw_shape, stats = get_geometric_shape(detector, face)
                shapes_buffer.append(raw_shape)
                if len(shapes_buffer) > BUFFER_SIZE: shapes_buffer.pop(0)
                
                if len(shapes_buffer) == BUFFER_SIZE:
                    most_common = Counter(shapes_buffer).most_common(1)[0][0]
                    face_data['shape'] = most_common
                    recs = expert_engine.recommend(face_data)
                    display_shape = shape_ar_map.get(most_common, most_common)
                    if recs:
                        display_rec_name = recs[0]['name']
                        display_rec_desc = recs[0].get('desc', 'تتناسب مع شكل وجهك')
                        display_rec_img_key = recs[0]['img']
                    status_color = COLOR_SUCCESS
                else:
                    progress = len(shapes_buffer) / BUFFER_SIZE * 100
                    display_shape = f"تحليل {progress:.0f}%"
                    status_color = COLOR_ACCENT_PRIMARY
            else:
                warning_msg = "انظر للأمام"
                status_color = COLOR_WARNING
                shapes_buffer = []

            # رسم الإطار
            x1, y1 = face[234][0], face[10][1] - 30
            x2, y2 = face[454][0], face[152][1] + 30
            overlay = img_mesh.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), status_color, 2)
            cv2.addWeighted(overlay, 0.3, final_img[0:h, 0:w], 0.7, 0, final_img[0:h, 0:w])

        # === بناء الواجهة ديناميكياً (بدون تداخل) ===
        
        card_x = w + 20
        card_w = sidebar_w - 40
        
        # 1. الهيدر
        header_h = 80
        draw_gradient_background(final_img, (w, 0, w+sidebar_w, header_h), (30, 30, 40), COLOR_BG_SIDEBAR, 'horizontal')
        cv2.line(final_img, (w, header_h), (w+sidebar_w, header_h), COLOR_ACCENT_PRIMARY, 2)
        final_img = put_arabic_text(final_img, "نظام التحليل البيومتري", (sidebar_center_x, 25), 22, COLOR_TEXT_PRIMARY, align="center")
        final_img = put_arabic_text(final_img, "Smart Vision Pro", (sidebar_center_x, 55), 12, COLOR_TEXT_MUTED, align="center")

        # 2. بطاقة الحالة (ثابتة في الأعلى)
        y_cursor = header_h + 20
        status_h = 100
        draw_rounded_rect(final_img, (card_x, y_cursor), (card_x + card_w, y_cursor + status_h), COLOR_BG_CARD, radius=12, fill=True)
        cv2.circle(final_img, (card_x + 30, y_cursor + 30), 8, status_color, -1)
        final_img = put_arabic_text(final_img, "الشكل:", (card_x + card_w - 20, y_cursor + 20), 14, COLOR_TEXT_SECONDARY, align="right")
        final_img = put_arabic_text(final_img, display_shape, (sidebar_center_x, y_cursor + 50), 24, status_color, align="center")
        
        y_cursor += status_h + 20

        # 3. بطاقة الإحصائيات (نثبتها في الأسفل ونحسب مكانها)
        footer_h = 40
        stats_h = 130
        y_stats = h - footer_h - stats_h - 20 # مكانها قبل الفوتر
        
        # رسم بطاقة الإحصائيات
        draw_rounded_rect(final_img, (card_x, y_stats), (card_x + card_w, y_stats + stats_h), COLOR_BG_CARD, radius=12, fill=True)
        final_img = put_arabic_text(final_img, "القياسات", (card_x + card_w - 15, y_stats + 15), 14, COLOR_TEXT_SECONDARY, align="right")
        
        labels = ["الطول", "الزاوية", "الفك", "الجبهة"]
        vals = stats
        grid_start_y = y_stats + 45
        for i, (lbl, v) in enumerate(zip(labels, vals)):
            row, col = i // 2, i % 2
            cx = card_x + 20 + col * (card_w // 2)
            cy = grid_start_y + row * 35
            final_img = put_arabic_text(final_img, f"{lbl}: {v:.0f}", (cx, cy), 13, COLOR_ACCENT_PRIMARY, align="left")

        # 4. بطاقة التوصية (تأخذ كل المساحة المتبقية في الوسط) !!
        # هذا هو الحل السحري لمنع التداخل
        rec_card_h = (y_stats - 20) - y_cursor 
        
        if rec_card_h > 150: # فقط إذا كانت هناك مساحة كافية
            draw_rounded_rect(final_img, (card_x, y_cursor), (card_x + card_w, y_cursor + rec_card_h), COLOR_BG_GLASS, radius=12, fill=True)
            final_img = put_arabic_text(final_img, "التوصية", (card_x + card_w - 20, y_cursor + 20), 16, COLOR_ACCENT_SECONDARY, align="right")
            
            if display_rec_name:
                # اسم النظارة
                final_img = put_arabic_text(final_img, display_rec_name, (sidebar_center_x, y_cursor + 50), 20, COLOR_TEXT_PRIMARY, align="center")
                
                # حساب مساحة الصورة
                img_area_y = y_cursor + 80
                img_area_h = rec_card_h - 110 # ترك مسافة للنصوص فوق وتحت
                
                if img_area_h > 50 and display_rec_img_key in glasses_images:
                    g_img = glasses_images[display_rec_img_key]
                    # تحجيم الصورة لتناسب المساحة المتاحة
                    scale = min((card_w - 60) / g_img.shape[1], img_area_h / g_img.shape[0])
                    new_w, new_h = int(g_img.shape[1] * scale), int(g_img.shape[0] * scale)
                    g_resized = cv2.resize(g_img, (new_w, new_h))
                    
                    g_y_pos = img_area_y + (img_area_h - new_h) // 2
                    g_x_pos = sidebar_center_x - (new_w // 2)
                    
                    # خلفية خفيفة للصورة
                    cv2.rectangle(final_img, (g_x_pos-5, g_y_pos-5), (g_x_pos+new_w+5, g_y_pos+new_h+5), (255,255,255), -1)
                    final_img = overlay_image_alpha(final_img, g_resized, g_x_pos, g_y_pos)

                final_img = put_arabic_text(final_img, "يناسب وجهك", (sidebar_center_x, y_cursor + rec_card_h - 30), 12, COLOR_SUCCESS, align="center")
            else:
                 final_img = put_arabic_text(final_img, "جاري التحليل...", (sidebar_center_x, y_cursor + rec_card_h // 2), 16, COLOR_TEXT_MUTED, align="center")

        # الفوتر والتحذيرات
        cv2.rectangle(final_img, (w, h - footer_h), (w + sidebar_w, h), (20, 20, 30), -1)
        final_img = put_arabic_text(final_img, "ESC: خروج", (sidebar_center_x, h - 15), 11, COLOR_TEXT_MUTED, align="center")

        if warning_msg:
             cv2.rectangle(final_img, (w//2-100, h//2-30), (w//2+100, h//2+30), COLOR_WARNING, -1)
             final_img = put_arabic_text(final_img, warning_msg, (w//2, h//2+10), 20, (255,255,255), align="center")

        cv2.imshow(window_name, final_img)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

def start_launcher():
    root = tk.Tk()
    root.title("Launcher")
    root.geometry("400x300")
    tk.Button(root, text="ابدأ النظام", font=("Arial", 20), command=lambda: [root.destroy(), start_system()]).pack(expand=True)
    root.mainloop()

if __name__ == "__main__":
    start_launcher()