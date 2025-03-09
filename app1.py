import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load mô hình Keras
MODEL_FILE = "houseprice.keras"
model = tf.keras.models.load_model(MODEL_FILE)
st.success("✅ Mô hình đã tải thành công!")

# Load danh sách features từ file feature1.pkl
FEATURE_FILE = "features1.pkl"
with open(FEATURE_FILE, "rb") as f:
    features = pickle.load(f)

# Kiểm tra danh sách features
st.write(f"📌 Danh sách features: {features}")

st.title("🏠 Dự đoán giá nhà")
st.write("Nhập thông tin bên dưới để dự đoán giá nhà:")

# Tạo giao diện nhập dữ liệu dựa trên danh sách features
data_input = [st.number_input(f"{feature}", value=0.0) for feature in features]

# Khi bấm nút Predict
if st.button("Dự đoán giá"):
    data_array = np.array(data_input).reshape(1, -1)  # Chuyển dữ liệu thành mảng 2D

    # Kiểm tra số lượng input có khớp với mô hình không
    if data_array.shape[1] != len(features):
        st.error(f"⚠ Số lượng input ({data_array.shape[1]}) không khớp với số features ({len(features)}).")
    else:
        prediction = model.predict(data_array)[0, 0]  # Lấy giá trị dự đoán
        st.success(f"🏡 Giá nhà dự đoán: ${prediction:,.2f}")
