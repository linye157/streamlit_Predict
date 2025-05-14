import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from pathlib import Path

class DataProcessingModule:
    def __init__(self):
        self.train_data_path = Path("data/train_data.xlsx")
        self.test_data_path = Path("data/test_data.xlsx")
    
    def load_default_data(self):
        """Load the default training and testing data"""
        try:
            if self.train_data_path.exists():
                train_data = pd.read_excel(self.train_data_path)
                st.session_state.train_data = train_data
                st.success(f"训练数据加载成功: {train_data.shape[0]} 行, {train_data.shape[1]} 列")
            else:
                st.warning(f"未找到默认训练数据文件: {self.train_data_path}")
                
            if self.test_data_path.exists():
                test_data = pd.read_excel(self.test_data_path)
                st.session_state.test_data = test_data
                st.success(f"测试数据加载成功: {test_data.shape[0]} 行, {test_data.shape[1]} 列")
            else:
                st.warning(f"未找到默认测试数据文件: {self.test_data_path}")
                
            return True
        except Exception as e:
            st.error(f"加载数据时出错: {str(e)}")
            return False
    
    def upload_data(self):
        """Upload custom dataset"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("上传训练数据")
            train_file = st.file_uploader("选择训练数据文件", type=["xlsx", "csv"], key="train_uploader")
            if train_file is not None:
                try:
                    if train_file.name.endswith('.csv'):
                        train_data = pd.read_csv(train_file)
                    else:
                        train_data = pd.read_excel(train_file)
                    
                    st.session_state.train_data = train_data
                    st.success(f"训练数据上传成功: {train_data.shape[0]} 行, {train_data.shape[1]} 列")
                except Exception as e:
                    st.error(f"上传训练数据时出错: {str(e)}")
        
        with col2:
            st.subheader("上传测试数据")
            test_file = st.file_uploader("选择测试数据文件", type=["xlsx", "csv"], key="test_uploader")
            if test_file is not None:
                try:
                    if test_file.name.endswith('.csv'):
                        test_data = pd.read_csv(test_file)
                    else:
                        test_data = pd.read_excel(test_file)
                    
                    st.session_state.test_data = test_data
                    st.success(f"测试数据上传成功: {test_data.shape[0]} 行, {test_data.shape[1]} 列")
                except Exception as e:
                    st.error(f"上传测试数据时出错: {str(e)}")
    
    def download_data(self):
        """Download current dataset"""
        st.subheader("下载数据")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.train_data is not None:
                train_csv = self.convert_df_to_csv(st.session_state.train_data)
                train_excel = self.convert_df_to_excel(st.session_state.train_data)
                
                st.download_button(
                    label="下载训练数据 (CSV)",
                    data=train_csv,
                    file_name="train_data.csv",
                    mime="text/csv",
                )
                
                st.download_button(
                    label="下载训练数据 (Excel)",
                    data=train_excel,
                    file_name="train_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.warning("无训练数据可下载")
        
        with col2:
            if st.session_state.test_data is not None:
                test_csv = self.convert_df_to_csv(st.session_state.test_data)
                test_excel = self.convert_df_to_excel(st.session_state.test_data)
                
                st.download_button(
                    label="下载测试数据 (CSV)",
                    data=test_csv,
                    file_name="test_data.csv",
                    mime="text/csv",
                )
                
                st.download_button(
                    label="下载测试数据 (Excel)",
                    data=test_excel,
                    file_name="test_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.warning("无测试数据可下载")
    
    def data_preview(self):
        """Preview current dataset"""
        st.subheader("数据预览")
        tab1, tab2 = st.tabs(["训练数据", "测试数据"])
        
        with tab1:
            if st.session_state.train_data is not None:
                st.write(f"训练数据形状: {st.session_state.train_data.shape}")
                st.dataframe(st.session_state.train_data.head(10))
                
                st.subheader("特征统计信息")
                st.dataframe(st.session_state.train_data.describe())
                
                # Count missing values
                missing_values = st.session_state.train_data.isnull().sum()
                if missing_values.sum() > 0:
                    st.subheader("缺失值统计")
                    missing_df = pd.DataFrame({
                        '缺失值数量': missing_values,
                        '缺失值比例': missing_values / len(st.session_state.train_data) * 100
                    })
                    st.dataframe(missing_df[missing_df['缺失值数量'] > 0])
            else:
                st.info("请先加载训练数据")
        
        with tab2:
            if st.session_state.test_data is not None:
                st.write(f"测试数据形状: {st.session_state.test_data.shape}")
                st.dataframe(st.session_state.test_data.head(10))
                
                st.subheader("特征统计信息")
                st.dataframe(st.session_state.test_data.describe())
                
                # Count missing values
                missing_values = st.session_state.test_data.isnull().sum()
                if missing_values.sum() > 0:
                    st.subheader("缺失值统计")
                    missing_df = pd.DataFrame({
                        '缺失值数量': missing_values,
                        '缺失值比例': missing_values / len(st.session_state.test_data) * 100
                    })
                    st.dataframe(missing_df[missing_df['缺失值数量'] > 0])
            else:
                st.info("请先加载测试数据")
    
    def data_preprocessing(self):
        """Data preprocessing options"""
        st.subheader("数据预处理")
        
        if st.session_state.train_data is None and st.session_state.test_data is None:
            st.warning("请先加载数据")
            return
        
        preprocess_options = st.multiselect(
            "选择预处理方法",
            ["填充缺失值", "特征标准化", "特征归一化", "异常值处理"],
            default=[]
        )
        
        if "填充缺失值" in preprocess_options:
            st.subheader("填充缺失值")
            fill_method = st.selectbox(
                "选择填充方法",
                ["均值填充", "中位数填充", "众数填充", "固定值填充"],
                index=0
            )
            
            fixed_value = None
            if fill_method == "固定值填充":
                fixed_value = st.number_input("填充值", value=0.0)
            
            if st.button("应用缺失值填充"):
                if st.session_state.train_data is not None:
                    st.session_state.train_data = self.fill_missing_values(st.session_state.train_data, fill_method, fixed_value)
                if st.session_state.test_data is not None:
                    st.session_state.test_data = self.fill_missing_values(st.session_state.test_data, fill_method, fixed_value)
                st.success("缺失值填充完成")
        
        if "特征标准化" in preprocess_options:
            st.subheader("特征标准化")
            if st.button("应用标准化"):
                if st.session_state.train_data is not None and st.session_state.test_data is not None:
                    st.session_state.train_data, st.session_state.test_data = self.standardize_features(
                        st.session_state.train_data, st.session_state.test_data
                    )
                    st.success("特征标准化完成")
                else:
                    st.warning("需要同时加载训练和测试数据")
        
        if "特征归一化" in preprocess_options:
            st.subheader("特征归一化")
            if st.button("应用归一化"):
                if st.session_state.train_data is not None and st.session_state.test_data is not None:
                    st.session_state.train_data, st.session_state.test_data = self.normalize_features(
                        st.session_state.train_data, st.session_state.test_data
                    )
                    st.success("特征归一化完成")
                else:
                    st.warning("需要同时加载训练和测试数据")
        
        if "异常值处理" in preprocess_options:
            st.subheader("异常值处理")
            outlier_method = st.selectbox(
                "选择异常值处理方法",
                ["IQR方法", "Z-Score方法", "百分位数方法"],
                index=0
            )
            
            if st.button("检测并处理异常值"):
                if st.session_state.train_data is not None:
                    st.session_state.train_data = self.handle_outliers(st.session_state.train_data, outlier_method)
                    st.success("训练数据异常值处理完成")
                if st.session_state.test_data is not None:
                    st.session_state.test_data = self.handle_outliers(st.session_state.test_data, outlier_method)
                    st.success("测试数据异常值处理完成")
    
    def parameter_settings(self):
        """Parameter settings interface"""
        st.subheader("特征工程与参数设置")
        
        if st.session_state.train_data is None:
            st.warning("请先加载训练数据")
            return
        
        st.markdown("#### 目标变量设置")
        num_targets = st.number_input("目标变量数量", min_value=1, max_value=10, value=3)
        
        # Store in session state
        if 'num_targets' not in st.session_state or st.session_state.num_targets != num_targets:
            st.session_state.num_targets = num_targets
        
        st.markdown("#### 特征选择")
        if st.checkbox("启用特征选择"):
            feature_selection_method = st.selectbox(
                "特征选择方法",
                ["相关性分析", "递归特征消除", "特征重要性", "主成分分析(PCA)"],
                index=0
            )
            
            if feature_selection_method == "相关性分析":
                corr_threshold = st.slider("相关性阈值", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
                if st.button("应用特征选择"):
                    st.info("特征选择功能将在机器学习模块中应用")
            
            elif feature_selection_method == "递归特征消除":
                n_features = st.slider("选择特征数量", min_value=1, max_value=st.session_state.train_data.shape[1]-st.session_state.num_targets, value=10)
                if st.button("应用特征选择"):
                    st.info("特征选择功能将在机器学习模块中应用")
            
            elif feature_selection_method == "特征重要性":
                if st.button("应用特征选择"):
                    st.info("特征选择功能将在机器学习模块中应用")
            
            elif feature_selection_method == "主成分分析(PCA)":
                n_components = st.slider("主成分数量", min_value=1, max_value=st.session_state.train_data.shape[1]-st.session_state.num_targets, value=10)
                variance_ratio = st.slider("解释方差比例", min_value=0.5, max_value=0.99, value=0.95, step=0.01)
                if st.button("应用特征选择"):
                    st.info("特征选择功能将在机器学习模块中应用")
    
    def training_visualization(self):
        """Training process visualization interface"""
        st.subheader("训练过程可视化")
        st.info("训练过程可视化将在模型训练过程中显示")
    
    def error_analysis(self):
        """Error analysis interface"""
        st.subheader("训练误差分析")
        st.info("训练误差分析将在模型训练完成后显示")
        
    # Helper methods
    def convert_df_to_csv(self, df):
        """Convert dataframe to CSV for download"""
        return df.to_csv(index=False).encode('utf-8')
    
    def convert_df_to_excel(self, df):
        """Convert dataframe to Excel for download"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        processed_data = output.getvalue()
        return processed_data
    
    def fill_missing_values(self, df, method, fixed_value=None):
        """Fill missing values using specified method"""
        df_copy = df.copy()
        
        # Get numerical columns
        num_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
        
        for col in num_cols:
            if df_copy[col].isnull().sum() > 0:
                if method == "均值填充":
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                elif method == "中位数填充":
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
                elif method == "众数填充":
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
                elif method == "固定值填充" and fixed_value is not None:
                    df_copy[col].fillna(fixed_value, inplace=True)
        
        return df_copy
    
    def standardize_features(self, train_df, test_df):
        """Standardize features (z-score normalization)"""
        from sklearn.preprocessing import StandardScaler
        
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        
        # Get feature columns (exclude target columns)
        feature_cols = train_df_copy.columns[:-st.session_state.num_targets]
        target_cols = train_df_copy.columns[-st.session_state.num_targets:]
        
        # Apply StandardScaler
        scaler = StandardScaler()
        train_df_copy[feature_cols] = scaler.fit_transform(train_df_copy[feature_cols])
        test_df_copy[feature_cols] = scaler.transform(test_df_copy[feature_cols])
        
        # Store scaler in session state for later use
        if 'scalers' not in st.session_state:
            st.session_state.scalers = {}
        st.session_state.scalers['standard'] = scaler
        
        return train_df_copy, test_df_copy
    
    def normalize_features(self, train_df, test_df):
        """Normalize features (min-max scaling)"""
        from sklearn.preprocessing import MinMaxScaler
        
        train_df_copy = train_df.copy()
        test_df_copy = test_df.copy()
        
        # Get feature columns (exclude target columns)
        feature_cols = train_df_copy.columns[:-st.session_state.num_targets]
        target_cols = train_df_copy.columns[-st.session_state.num_targets:]
        
        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        train_df_copy[feature_cols] = scaler.fit_transform(train_df_copy[feature_cols])
        test_df_copy[feature_cols] = scaler.transform(test_df_copy[feature_cols])
        
        # Store scaler in session state for later use
        if 'scalers' not in st.session_state:
            st.session_state.scalers = {}
        st.session_state.scalers['minmax'] = scaler
        
        return train_df_copy, test_df_copy
    
    def handle_outliers(self, df, method):
        """Handle outliers using specified method"""
        df_copy = df.copy()
        
        # Get numerical columns
        num_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
        
        for col in num_cols[:-st.session_state.num_targets]:  # Skip target columns
            if method == "IQR方法":
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with bounds
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == "Z-Score方法":
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                z_scores = (df_copy[col] - mean) / std
                
                # Replace outliers (|z| > 3) with mean
                df_copy.loc[abs(z_scores) > 3, col] = mean
                
            elif method == "百分位数方法":
                lower_bound = df_copy[col].quantile(0.01)
                upper_bound = df_copy[col].quantile(0.99)
                
                # Replace outliers with bounds
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_copy
    
    def render(self):
        st.title("系统接口")
        
        # Tabs for different interface sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "数据接口", "数据预览", "数据预处理", "参数设置与调优", "训练过程与误差分析"
        ])
        
        with tab1:
            st.subheader("数据加载")
            load_option = st.radio(
                "选择数据加载方式",
                ["加载默认数据", "上传自定义数据"],
                horizontal=True
            )
            
            if load_option == "加载默认数据":
                if st.button("加载默认数据"):
                    self.load_default_data()
            else:
                self.upload_data()
            
            st.markdown("---")
            self.download_data()
        
        with tab2:
            self.data_preview()
        
        with tab3:
            self.data_preprocessing()
        
        with tab4:
            self.parameter_settings()
        
        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                self.training_visualization()
            with col2:
                self.error_analysis() 