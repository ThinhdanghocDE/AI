import streamlit as st
import pickle
import numpy as np

# Load mô hình
with open("houseprice.pkl", "rb") as f:
    model = pickle.load(f)

# Load danh sách features
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

st.title("🏠 Dự đoán giá nhà")
st.write("Nhập thông tin bên dưới để dự đoán giá nhà:")

# Tạo giao diện nhập liệu theo các features
data_input = []
for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    data_input.append(value)

# Khi bấm nút Predict
if st.button("Dự đoán giá"):
    data_array = np.array(data_input).reshape(1, -1)  # Chuyển thành mảng numpy
    prediction = model.predict(data_array)
    st.success(f"Giá nhà dự đoán: ${prediction[0]:,.2f}")
