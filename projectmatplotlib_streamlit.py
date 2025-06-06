import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(layout="wide")
st.title("🧠 Phân cụm KMeans với dữ liệu Minimart")

# Hàm cache chuyển DataFrame thành CSV
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Tải dữ liệu
uploaded_file = st.file_uploader("Tải lên file dữ liệu (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dữ liệu ban đầu")
    st.dataframe(df.head(10))

    # Chọn các cột để phân cụm
    cols = st.multiselect("Chọn các cột dùng cho phân cụm:", df.select_dtypes(include=['number']).columns.tolist(), default=['Distance', 'Grocery', 'Milk'])

    if len(cols) >= 2:
        X = df[cols].dropna()

        # Thêm phần gợi ý số cụm bằng biểu đồ Elbow
        with st.expander("📉 Gợi ý số cụm bằng phương pháp Elbow"):
            distortions = []
            K_range = range(1, 11)
            for i in K_range:
                km = KMeans(n_clusters=i, random_state=42)
                km.fit(X)
                distortions.append(km.inertia_)
    
            fig2, ax2 = plt.subplots()
            ax2.plot(K_range, distortions, 'bo-')
            ax2.set_xlabel('Số cụm (k)')
            ax2.set_ylabel('Inertia')
            ax2.set_title('Biểu đồ Elbow')
            st.pyplot(fig2)

        k = st.slider("Chọn số cụm (k):", 2, 10, 3)

        # Huấn luyện mô hình
        model = KMeans(n_clusters=k, random_state=42)
        clusters = model.fit_predict(X)
        df['cluster'] = clusters

        st.success(f"✅ Hoàn tất phân cụm với k = {k}")

        # Hiển thị thống kê từng cụm
        selected_cluster = st.selectbox("Chọn cụm để xem chi tiết:", sorted(df['cluster'].unique()))
        st.write(df[df['cluster'] == selected_cluster].describe())

        # Vẽ biểu đồ 3D nếu có đủ 3 cột
        if len(cols) >= 3:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]], 
                       c=df['cluster'], cmap='viridis', s=60)

            centers = model.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                       c='black', s=200, alpha=0.5, marker='s', label='Tâm cụm')

            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.set_zlabel(cols[2])
            ax.legend()
            st.pyplot(fig)
            
        # Tải dữ liệu sau khi phân cụm
        csv = convert_df(df)
        st.download_button(
            label="📥 Tải dữ liệu đã phân cụm",
            data=csv,
            file_name='clustered_data.csv',
            mime='text/csv',
        )
    else:
        st.warning("🔴 Cần chọn ít nhất 2 cột để phân cụm.")
