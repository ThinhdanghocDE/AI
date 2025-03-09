import streamlit as st
import numpy as np
import tensorflow as tf

# Load mô hình Keras
MODEL_FILE = "houseprice.keras"  # Đảm bảo tệp này tồn tại trong thư mục làm việc
try:
    model = tf.keras.models.load_model(MODEL_FILE)
    st.success("✅ Mô hình đã tải thành công!")
except Exception as e:
    st.error(f"⚠ Lỗi khi tải mô hình: {e}")
    st.stop()

# Load danh sách features
FEATURE_FILE = "features.pkl"
try:
    import pickle
    with open(FEATURE_FILE, "rb") as f:
        features = pickle.load(f)
except Exception as e:
    st.error(f"⚠ Lỗi khi tải danh sách features: {e}")
    st.stop()

st.title("🏠 Dự đoán giá nhà")
st.write("Nhập thông tin bên dưới để dự đoán giá nhà:")

# Tạo giao diện nhập dữ liệu
data_input = []
for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    data_input.append(value)

# Khi bấm nút Predict
if st.button("Dự đoán giá"):
    data_array = np.array(data_input).reshape(1, -1)  # Chuyển dữ liệu thành mảng 2D
    try:
        prediction = model.predict(data_array)[0, 0]  # Lấy giá trị dự đoán
        st.success(f"🏡 Giá nhà dự đoán: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"⚠ Lỗi khi dự đoán: {e}")
