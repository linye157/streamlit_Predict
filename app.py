import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import modules
from modules.data_processing import DataProcessingModule
from modules.machine_learning import MachineLearningModule
from modules.stacking_ensemble import StackingEnsembleModule
from modules.auto_ml import AutoMLModule
from modules.visualization import VisualizationModule
from modules.report import ReportModule

# Set page config
st.set_page_config(
    page_title="机器学习预测系统",
    page_icon="🧠",
    layout="wide"
)

# Session state initialization
if 'current_page' not in st.session_state:
    st.session_state.current_page = "首页"
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'breadcrumb' not in st.session_state:
    st.session_state.breadcrumb = ["首页"]

# Function to update breadcrumb
def update_breadcrumb(page_name):
    if page_name in st.session_state.breadcrumb:
        # If page already in breadcrumb, truncate to that page
        idx = st.session_state.breadcrumb.index(page_name)
        st.session_state.breadcrumb = st.session_state.breadcrumb[:idx+1]
    else:
        # Add new page to breadcrumb
        st.session_state.breadcrumb.append(page_name)
    st.session_state.current_page = page_name

# Sidebar navigation
with st.sidebar:
    st.title("机器学习预测系统")
    st.markdown("---")
    
    # Main tabs
    st.header("系统导航")
    system_choice = st.radio(
        "选择子系统",
        ["机器学习子系统", "用户交互子系统"],
        key="system_choice"
    )
    
    # Sub-tabs based on selection
    if system_choice == "机器学习子系统":
        module_choice = st.radio(
            "选择模块",
            ["系统接口", "机器学习", "机器学习Stacking集成", "自动化机器学习"],
            key="ml_module_choice"
        )
    else:
        module_choice = st.radio(
            "选择模块",
            ["可视化分析", "报表"],
            key="ui_module_choice"
        )
    
    # Update current page and breadcrumb
    if system_choice == "机器学习子系统":
        update_breadcrumb(f"机器学习子系统 > {module_choice}")
    else:
        update_breadcrumb(f"用户交互子系统 > {module_choice}")

# Display breadcrumb navigation
breadcrumb_str = " > ".join(st.session_state.breadcrumb)
st.markdown(f"**当前位置:** {breadcrumb_str}")
st.markdown("---")

# Main content area
if st.session_state.current_page == "首页":
    st.title("机器学习预测系统")
    st.markdown("""
    ## 欢迎使用机器学习预测系统
    
    请使用左侧边栏进行导航，选择您需要的功能模块。
    
    本系统包含以下功能：
    
    ### 机器学习子系统
    - **系统接口**：数据接口、参数设置与调优接口、训练过程可视化、训练误差分析
    - **机器学习**：线性(LR)模型、随机森林(RF)模型、GBR模型、XGBR模型、支持向量机(SVR)模型、人工神经网络(ANN)模型
    - **机器学习Stacking集成**：训练一级模型、训练二级模型、k折交叉验证
    - **自动化机器学习**：模型自动筛选、模型参数自动最优化设置、模型打包制作与输出
    
    ### 用户交互子系统
    - **可视化分析**：数据可视化、模型可视化、分析结果可视化
    - **报表**：报表项目订制、报表自动生成、报表多维度信息展示、报表多种格式下载、不同时间端报表对比分析、报表存储
    """)
    
    # Display data load status
    st.markdown("### 数据加载状态")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.train_data is not None:
            st.success("训练数据已加载")
        else:
            st.error("训练数据未加载")
    with col2:
        if st.session_state.test_data is not None:
            st.success("测试数据已加载")
        else:
            st.error("测试数据未加载")

# Render the appropriate module based on the selection
elif "机器学习子系统 > 系统接口" in st.session_state.current_page:
    DataProcessingModule().render()
elif "机器学习子系统 > 机器学习" in st.session_state.current_page:
    MachineLearningModule().render()
elif "机器学习子系统 > 机器学习Stacking集成" in st.session_state.current_page:
    StackingEnsembleModule().render()
elif "机器学习子系统 > 自动化机器学习" in st.session_state.current_page:
    AutoMLModule().render()
elif "用户交互子系统 > 可视化分析" in st.session_state.current_page:
    VisualizationModule().render()
elif "用户交互子系统 > 报表" in st.session_state.current_page:
    ReportModule().render() 