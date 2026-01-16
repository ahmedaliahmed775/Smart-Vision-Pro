"""
أداة تحليل البيانات لاستخراج خصائص الوجه من مجموعة الصور.
Data analysis tool to extract face features from the dataset.
"""
import cv2
import os
import math
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# ==========================================
# إعدادات المسار (Path Configuration)
# ضع مسار الداتاسيت هنا - Update Dataset Path Here
# ==========================================
# تم تحديث المسار ليعود خطوة للخلف لأن الملف أصبح داخل مجلد فرعي
DATASET_PATH = os.path.join("..", "FaceShape Dataset", "testing_set")
# استخدم مسار نسبي لجعله يعمل على أي جهاز
# Use relative path to make it portable

SHAPES = ["Square", "Round", "Oval", "Heart", "Oblong", "Diamond"]
detector = FaceMeshDetector(maxFaces=1)

def calculate_angle(p1, p2, p3):
    """
    حساب الزاوية بين ثلاث نقاط.
    Calculate angle between three points.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0: angle += 360
    if angle > 180: angle = 360 - angle
    return angle

def analyze_shape_advanced(shape_name):
    """
    تحليل مجموعة صور لشكل وجه معين واستخراج المتوسطات.
    Analyze a set of images for a specific face shape and extract averages.
    """
    folder = os.path.join(DATASET_PATH, shape_name)
    if not os.path.exists(folder):
        print(f"Folder not found: {shape_name} | مسار غير موجود")
        return

    print(f"--- Analyzing: {shape_name} ---")
    
    r_face_len = []
    r_jaw_width = []
    r_forehead = []
    angles = []
    
    count = 0
    files = os.listdir(folder)
    
    for file in files:
        if count >= 60: break # نكتفي بـ 60 صورة للعينة - Limit to 60 images sample
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')): continue
        
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None: continue
        
        img, faces = detector.findFaceMesh(img, draw=False)
        if faces:
            f = faces[0]
            
            # النقاط الرئيسية في الوجه (Face Mesh Points)
            pt_cheek_L, pt_cheek_R = f[234], f[454]
            pt_jaw_L, pt_jaw_R = f[132], f[361]
            pt_fore_L, pt_fore_R = f[103], f[332]
            pt_top, pt_bottom = f[10], f[152]
            
            # حساب المسافات (Distances)
            w_cheeks, _ = detector.findDistance(pt_cheek_L, pt_cheek_R)
            w_jaw, _ = detector.findDistance(pt_jaw_L, pt_jaw_R)
            w_forehead, _ = detector.findDistance(pt_fore_L, pt_fore_R)
            h_face, _ = detector.findDistance(pt_top, pt_bottom)
            
            # حساب الزاوية (Chin Angle)
            ang = calculate_angle(f[132], f[152], f[361])
            
            # حساب النسب (Ratios)
            if w_cheeks != 0: # Avoid division by zero
                rf_len = (h_face / w_cheeks) * 100
                rf_jaw = (w_jaw / w_cheeks) * 100
                rf_fore = (w_forehead / w_cheeks) * 100
                
                r_face_len.append(rf_len)
                r_jaw_width.append(rf_jaw)
                r_forehead.append(rf_fore)
                angles.append(ang)
                
                count += 1
                print(f"Processed {count} images...", end='\r')

    if r_face_len:
        print(f"\nResults for: {shape_name.upper()}")
        print(f"  > Face Ratio (Height):    {np.mean(r_face_len):.1f}")
        print(f"  > Jaw Ratio (Width):      {np.mean(r_jaw_width):.1f}")
        print(f"  > Forehead Ratio:         {np.mean(r_forehead):.1f}")
        print(f"  > Chin Angle:             {np.mean(angles):.1f}")
        print("------------------------------------------------")

# تشغيل التحليل (Run Analysis)
if __name__ == "__main__":
    for s in SHAPES:
        analyze_shape_advanced(s)