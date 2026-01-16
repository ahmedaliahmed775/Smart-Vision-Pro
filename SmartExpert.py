"""
نظام الخبير الذكي لتوصية النظارات بناءً على شكل الوجه.
Smart Expert System for glasses recommendation based on face shape.
"""
import numpy as np

class SmartExpert:
    """
    كلاس الخبير المسؤول عن تحليل بيانات الوجه واقتراح النظارات المناسبة.
    The Expert class responsible for analyzing face data and suggesting suitable glasses.
    """
    def __init__(self):
        # القاموس يحتوي على: الاسم للعرض + اسم ملف الصورة
        # Dictionary containing: Display Name + Image Filename
        self.glass_types = {
            0:  {'name': 'Rectangle (مستطيل)', 'img': 'rectangle'},
            1:  {'name': 'Square (مربع)',       'img': 'square'},
            2:  {'name': 'Oval (بيضاوي)',      'img': 'oval'},
            3:  {'name': 'Round (دائري)',      'img': 'round'},
            4:  {'name': 'Aviator (طيار)',     'img': 'aviator'},
            5:  {'name': 'Cat Eye (عين القطة)', 'img': 'cateye'},
            7:  {'name': 'Clubmaster',          'img': 'clubmaster'},
            9:  {'name': 'Wayfarer',            'img': 'wayfarer'},
            12: {'name': 'Rimless (بدون إطار)', 'img': 'rimless'}
        }

    def recommend(self, face_data):
        """
        يقوم بحساب التوصيات بناءً على بيانات الوجه.
        Calculates recommendations based on face data.
        """
        # تهيئة متجه النتائج للنظارات المختلفة (بناءً على نظام نقاط)
        # Initialize score vector for different glasses (based on a scoring system)
        s_vector = np.zeros(31) 
        shape_name = face_data.get('shape', 'Oval').lower()
        chin_angle = face_data.get('angle', 120)
        
        # --- تطبيق قواعد النظام الخبير (Expert System Rules) ---
        if 'oval' in shape_name:
            s_vector[[0, 1, 4]] += 2.0; s_vector[[9]] += 1.5; s_vector[[12]] += 1.0; s_vector[[3]] += 0.5
        elif 'round' in shape_name:
            s_vector[[0, 1]] += 3.0; s_vector[[9]] += 2.0; s_vector[[5]] += 2.0; s_vector[[7]] += 1.5; s_vector[[2, 3]] -= 3.0
        elif 'square' in shape_name:
            s_vector[[3]] += 3.0; s_vector[[2, 4]] += 2.5; s_vector[[5]] += 1.0; s_vector[[12]] += 1.0; s_vector[[0, 1]] -= 3.0
        elif 'heart' in shape_name:
            s_vector[[9]] += 3.0; s_vector[[4]] += 2.0; s_vector[[7]] += 2.0; s_vector[[12]] += 1.5; s_vector[[2]] += 1.0
        elif 'diamond' in shape_name:
            s_vector[[5]] += 3.0; s_vector[[2]] += 2.0; s_vector[[12]] += 2.0; s_vector[[7]] += 1.5; s_vector[[0, 1]] -= 1.0
        elif 'oblong' in shape_name:
            s_vector[[4]] += 3.0; s_vector[[1]] += 2.0; s_vector[[9]] += 1.5; s_vector[[0]] -= 1.0

        # تعديل التوصيات بناءً على زوايا محددة للذقن
        # Adjust recommendations based on specific chin angles
        if chin_angle < 100:
            s_vector[[7]] += 1.0; s_vector[[1]] -= 1.0

        # استرجاع أفضل 3 توصيات
        # Retrieve top 3 recommendations
        top_indices = s_vector.argsort()[-3:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if idx in self.glass_types:
                recommendations.append(self.glass_types[idx])
        
        return recommendations