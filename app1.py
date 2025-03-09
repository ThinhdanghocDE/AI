import streamlit as st
import pickle
import numpy as np

# Load mô hình Scikit-learn
with open("houseprice.pkl", "rb") as f:
    model = pickle.load(f)

# Load danh sách features
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

st.title("🏠 Dự đoán giá nhà")
st.write("Nhập thông tin bên dưới để dự đoán giá nhà:")

# Tạo giao diện nhập dữ liệu
data_input = []
for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    data_input.append(value)

# Khi bấm nút Predict
if st.button("Dự đoán giá"):
    st.write(type(model))
    data_array = np.array(data_input).reshape(1, -1)  # Chuyển dữ liệu thành mảng 2D
    prediction = model.predict(data_array)  # Dự đoán giá nhà
    st.success(f"🏡 Giá nhà dự đoán: ${prediction[0]:,.2f}")
