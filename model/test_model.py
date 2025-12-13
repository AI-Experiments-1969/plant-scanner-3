# نصيحة: test_model.py
# تحميل النموذج المدرب واختباره على صورة جديدة

import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 1. تحميل النموذج
model_path = "plant_disease_model.keras"
print(f"جاري تحميل النموذج من: {model_path}")
model = tf.keras.models.load_model(model_path)
print("✅ النموذج تم تحميله بنجاح")

# 2. تحضير صورة للاختبار (نستخدم صورة من dataset كمثال)
test_image_path = "../dataset/healthy/h1.jpg"  # جرب تغيير المسار إلى diseased/d1.jpg لاحقاً
print(f"جاري تحليل الصورة: {test_image_path}")

# 3. تحميل الصورة وتغيير حجمها ومعالجتها
img = Image.open(test_image_path)
img = img.resize((180, 180))  # نفس الأبعاد المستخدمة في التدريب
img_array = np.array(img) / 255.0  # تسوية القيم
img_array = np.expand_dims(img_array, axis=0)  # إضافة بعد الدفعة: (1, 180, 180, 3)

# 4. التنبؤ
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])  # تحويل النتائج إلى احتمالات

# 5. تفسير النتائج
class_names = ['diseased', 'healthy']  # يجب أن تكون بنفس ترتيب التدريب
print(f"\nنتائج التحليل للصورة '{os.path.basename(test_image_path)}':")
print(f"  - احتمالية أن تكون 'مريضة (diseased)': {score[0]:.4f}")
print(f"  - احتمالية أن تكون 'سليمة (healthy)': {score[1]:.4f}")

predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)
print(f"\n🎯 النتيجة: الصورة تصنف كـ '{predicted_class}' بنسبة ثقة {confidence:.2f}%")