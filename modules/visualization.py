import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff

class VisualizationModule:
    def __init__(self):
        pass
    
    def data_visualization(self):
        """Visualize input data"""
        st.subheader("数据可视化")
        
        if st.session_state.train_data is None:
            st.warning("请先加载训练数据")
            return
        
        # Get data
        data = st.session_state.train_data
        
        # Set number of target variables
        if 'num_targets' not in st.session_state:
            st.session_state.num_targets = 3  # Default
        
        # Get feature and target columns
        feature_cols = data.columns[:-st.session_state.num_targets]
        target_cols = data.columns[-st.session_state.num_targets:]
        
        # Visualization options
        viz_type = st.selectbox(
            "选择可视化类型",
            ["特征分布", "特征相关性", "特征-目标关系", "降维可视化"],
            index=0
        )
        
        if viz_type == "特征分布":
            # Select features to visualize
            selected_features = st.multiselect(
                "选择要可视化的特征",
                feature_cols.tolist(),
                default=feature_cols[:5].tolist()
            )
            
            if not selected_features:
                st.warning("请至少选择一个特征")
                return
            
            # Distribution plot type
            dist_type = st.radio(
                "分布图类型",
                ["直方图", "核密度图", "箱线图"],
                horizontal=True
            )
            
            if dist_type == "直方图":
                # Create histograms
                for feature in selected_features:
                    fig = px.histogram(
                        data, 
                        x=feature,
                        title=f"{feature} 分布",
                        marginal="box",
                        color_discrete_sequence=['#3366CC']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif dist_type == "核密度图":
                # Create KDE plots
                for feature in selected_features:
                    fig = px.histogram(
                        data, 
                        x=feature,
                        title=f"{feature} 分布",
                        marginal="rug",
                        histnorm="probability density",
                        color_discrete_sequence=['#3366CC']
                    )
                    fig.update_traces(opacity=0.6)
                    
                    # Add KDE curve
                    kde_fig = ff.create_distplot(
                        [data[feature].dropna()], 
                        group_labels=[feature],
                        colors=['#FF6600'],
                        show_hist=False,
                        show_rug=False
                    )
                    
                    for trace in kde_fig.data:
                        fig.add_trace(trace)
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            elif dist_type == "箱线图":
                # Create box plots
                fig = px.box(
                    data,
                    y=selected_features,
                    title="特征箱线图",
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "特征相关性":
            # Correlation type
            corr_type = st.radio(
                "相关性类型",
                ["Pearson", "Spearman", "Kendall"],
                horizontal=True
            )
            
            # Get correlation matrix
            if corr_type == "Pearson":
                corr_matrix = data.corr(method='pearson')
            elif corr_type == "Spearman":
                corr_matrix = data.corr(method='spearman')
            else:  # Kendall
                corr_matrix = data.corr(method='kendall')
            
            # Correlation visualization type
            corr_viz = st.radio(
                "相关性可视化类型",
                ["热力图", "气泡图"],
                horizontal=True
            )
            
            if corr_viz == "热力图":
                # Create heatmap
                fig = px.imshow(
                    corr_matrix,
                    title=f"{corr_type} 相关性热力图",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # 气泡图
                # Create bubble chart
                # Convert correlation matrix to long format
                corr_data = corr_matrix.reset_index().melt(id_vars='index')
                corr_data.columns = ['Feature 1', 'Feature 2', 'Correlation']
                
                # Filter out self-correlations and duplicate pairs
                corr_data = corr_data[corr_data['Feature 1'] < corr_data['Feature 2']]
                
                # Create bubble chart
                fig = px.scatter(
                    corr_data,
                    x='Feature 1',
                    y='Feature 2',
                    size=corr_data['Correlation'].abs() * 100,
                    color='Correlation',
                    color_continuous_scale="RdBu_r",
                    hover_data=['Correlation'],
                    title=f"{corr_type} 相关性气泡图",
                    range_color=[-1, 1]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "特征-目标关系":
            # Select features
            selected_features = st.multiselect(
                "选择要分析的特征",
                feature_cols.tolist(),
                default=feature_cols[:3].tolist()
            )
            
            # Select target
            selected_target = st.selectbox(
                "选择目标变量",
                target_cols.tolist()
            )
            
            if not selected_features:
                st.warning("请至少选择一个特征")
                return
            
            # Visualization type
            relation_viz_type = st.radio(
                "关系可视化类型",
                ["散点图", "配对图", "箱体图"],
                horizontal=True
            )
            
            if relation_viz_type == "散点图":
                # Create scatter plots
                for feature in selected_features:
                    fig = px.scatter(
                        data,
                        x=feature,
                        y=selected_target,
                        title=f"{feature} vs {selected_target}",
                        trendline="ols",
                        trendline_color_override="red"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif relation_viz_type == "配对图":
                # Create pair plot for selected features and target
                selected_columns = selected_features + [selected_target]
                pair_data = data[selected_columns]
                
                # Create pairplot using plotly
                dims = [dict(label=col, values=pair_data[col]) for col in pair_data.columns]
                fig = go.Figure(go.Splom(
                    dimensions=dims,
                    showupperhalf=False,
                    marker=dict(
                        color=pair_data[selected_target],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=selected_target)
                    )
                ))
                
                fig.update_layout(
                    title="特征-目标配对图",
                    width=800,
                    height=800
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif relation_viz_type == "箱体图":
                # Option to display continuous features
                show_continuous = st.checkbox("显示连续型特征的箱体图", value=False)
                
                # Options for binning continuous features
                if show_continuous:
                    bin_continuous = st.checkbox("对连续型特征进行分箱", value=True)
                    num_bins = st.slider("分箱数量", min_value=3, max_value=20, value=5)
                
                # Create box plots for categorical features
                for feature in selected_features:
                    # Check if feature has less than 10 unique values
                    if data[feature].nunique() < 10:
                        fig = px.box(
                            data,
                            x=feature,
                            y=selected_target,
                            title=f"{selected_target} by {feature}",
                            color=feature
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    # For continuous features
                    elif show_continuous:
                        if bin_continuous:
                            # Create a copy to avoid modifying original data
                            plot_data = data.copy()
                            # Create bins for continuous feature
                            try:
                                # Remove NaN and infinite values for binning
                                valid_data = plot_data[~np.isnan(plot_data[feature]) & ~np.isinf(plot_data[feature])]
                                if len(valid_data) > num_bins:  # Ensure we have enough data points for the requested bins
                                    plot_data[f"{feature}_binned"] = pd.qcut(
                                        plot_data[feature], 
                                        q=num_bins, 
                                        duplicates='drop'
                                    ).astype(str)
                                    
                                    # Create box plot with binned feature
                                    fig = px.box(
                                        plot_data,
                                        x=f"{feature}_binned",
                                        y=selected_target,
                                        title=f"{selected_target} by {feature} (分箱)",
                                        color=f"{feature}_binned"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning(f"特征 {feature} 有效数据点不足，无法进行 {num_bins} 分箱")
                                    # Fall back to direct plot
                                    fig = px.box(
                                        data,
                                        x=feature,
                                        y=selected_target,
                                        title=f"{selected_target} by {feature}",
                                        color=feature
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"为特征 {feature} 创建分箱时出错: {str(e)}")
                                # Fall back to direct plot
                                fig = px.box(
                                    data,
                                    x=feature,
                                    y=selected_target,
                                    title=f"{selected_target} by {feature}",
                                    color=feature
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Direct box plot for continuous feature
                            fig = px.box(
                                data,
                                x=feature,
                                y=selected_target,
                                title=f"{selected_target} by {feature}",
                                color=feature
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"跳过 {feature} (连续型特征)")
        
        elif viz_type == "降维可视化":
            # Select dimensionality reduction technique
            dim_reduction = st.radio(
                "降维技术",
                ["PCA", "t-SNE"],
                horizontal=True
            )
            
            # Get features only
            X = data.iloc[:, :-st.session_state.num_targets]
            
            # Standardize data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Select target for coloring
            selected_target = st.selectbox(
                "选择目标变量 (用于着色)",
                target_cols.tolist()
            )
            
            if dim_reduction == "PCA":
                # Create PCA
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_scaled)
                
                # Create dataframe for plotting
                pca_df = pd.DataFrame({
                    'PC1': components[:, 0],
                    'PC2': components[:, 1],
                    'Target': data[selected_target]
                })
                
                # Create scatter plot
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Target',
                    title=f"PCA 降维 (着色: {selected_target})",
                    color_continuous_scale="Viridis"
                )
                
                # Calculate explained variance
                explained_variance = pca.explained_variance_ratio_ * 100
                fig.update_layout(
                    xaxis_title=f"PC1 ({explained_variance[0]:.2f}% 解释方差)",
                    yaxis_title=f"PC2 ({explained_variance[1]:.2f}% 解释方差)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature loadings
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                loading_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=X.columns)
                
                # Create loadings plot
                fig = px.scatter(
                    loading_df.reset_index(),
                    x='PC1',
                    y='PC2',
                    text='index',
                    title="PCA 特征权重",
                )
                
                fig.update_traces(textposition='top center')
                
                # Add arrows from origin to loadings
                for i, feature in enumerate(X.columns):
                    fig.add_shape(
                        type='line',
                        x0=0, y0=0,
                        x1=loadings[i, 0],
                        y1=loadings[i, 1],
                        line=dict(color='red', width=1)
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # t-SNE
                # Parameters
                perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30)
                learning_rate = st.slider("Learning Rate", min_value=10, max_value=1000, value=200)
                
                # Create t-SNE
                tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
                components = tsne.fit_transform(X_scaled)
                
                # Create dataframe for plotting
                tsne_df = pd.DataFrame({
                    'Dimension 1': components[:, 0],
                    'Dimension 2': components[:, 1],
                    'Target': data[selected_target]
                })
                
                # Create scatter plot
                fig = px.scatter(
                    tsne_df,
                    x='Dimension 1',
                    y='Dimension 2',
                    color='Target',
                    title=f"t-SNE 降维 (着色: {selected_target})",
                    color_continuous_scale="Viridis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def model_visualization(self):
        """Visualize model structures and feature importance"""
        st.subheader("模型可视化")
        
        if not st.session_state.models:
            st.warning("尚未训练任何模型")
            return
        
        # Select model to visualize
        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox("选择模型", model_names)
        
        if selected_model:
            model = st.session_state.models[selected_model]
            
            # Get model type
            model_type = type(model).__name__
            
            st.write(f"模型类型: {model_type}")
            
            # Feature importance visualization
            if hasattr(model, 'feature_importances_'):
                # Get feature names
                if 'model_results' in st.session_state and selected_model in st.session_state.model_results:
                    if 'feature_names' in st.session_state.model_results[selected_model]:
                        feature_names = st.session_state.model_results[selected_model]['feature_names']
                    else:
                        feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
                else:
                    feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
                
                # Create dataframe for plotting
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Display top 15 features
                top_n = min(15, len(importance_df))
                top_features = importance_df.head(top_n)
                
                # Create bar chart
                fig = px.bar(
                    top_features,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"特征重要性 (前 {top_n} 项)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("该模型不支持特征重要性可视化")
            
            # Model parameters
            st.subheader("模型参数")
            
            if hasattr(model, 'get_params'):
                params = model.get_params()
                st.json(params)
            else:
                st.info("无法获取模型参数")
    
    def result_visualization(self):
        """Visualize prediction results"""
        st.subheader("结果可视化")
        
        # Check if we have model results
        if 'model_results' in st.session_state and st.session_state.model_results:
            # Select model to visualize
            model_names = list(st.session_state.model_results.keys())
            selected_model = st.selectbox("选择模型", model_names, key="result_viz_model")
            
            if selected_model and selected_model in st.session_state.model_results:
                results = st.session_state.model_results[selected_model]
                
                # Create tabs for different visualizations
                viz_tab1, viz_tab2, viz_tab3 = st.tabs(["预测 vs 实际", "残差分析", "误差分布"])
                
                with viz_tab1:
                    # Scatter plot of predicted vs actual values
                    if 'predictions' in results:
                        pred_data = results['predictions']
                        
                        # Create scatter plot
                        fig = go.Figure()
                        
                        # Add train data
                        fig.add_trace(go.Scatter(
                            x=pred_data['train_actual'],
                            y=pred_data['train'],
                            mode='markers',
                            name='训练集',
                            marker=dict(color='blue', size=8, opacity=0.6)
                        ))
                        
                        # Add test data
                        fig.add_trace(go.Scatter(
                            x=pred_data['test_actual'],
                            y=pred_data['test'],
                            mode='markers',
                            name='测试集',
                            marker=dict(color='red', size=8, opacity=0.6)
                        ))
                        
                        # Add perfect prediction line
                        min_val = min(min(pred_data['train_actual']), min(pred_data['test_actual']))
                        max_val = max(max(pred_data['train_actual']), max(pred_data['test_actual']))
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='完美预测',
                            line=dict(color='black', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title='实际值 vs 预测值',
                            xaxis_title='实际值',
                            yaxis_title='预测值',
                            legend_title='数据集',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                with viz_tab2:
                    # Residual analysis
                    if 'predictions' in results:
                        pred_data = results['predictions']
                        
                        # Calculate residuals
                        train_residuals = np.array(pred_data['train']) - np.array(pred_data['train_actual'])
                        test_residuals = np.array(pred_data['test']) - np.array(pred_data['test_actual'])
                        
                        # Create residual plot
                        fig = go.Figure()
                        
                        # Add train data
                        fig.add_trace(go.Scatter(
                            x=pred_data['train_actual'],
                            y=train_residuals,
                            mode='markers',
                            name='训练集',
                            marker=dict(color='blue', size=8, opacity=0.6)
                        ))
                        
                        # Add test data
                        fig.add_trace(go.Scatter(
                            x=pred_data['test_actual'],
                            y=test_residuals,
                            mode='markers',
                            name='测试集',
                            marker=dict(color='red', size=8, opacity=0.6)
                        ))
                        
                        # Add zero line
                        fig.add_hline(y=0, line_dash="dash", line_color="black")
                        
                        fig.update_layout(
                            title='残差图',
                            xaxis_title='实际值',
                            yaxis_title='残差 (预测值 - 实际值)',
                            legend_title='数据集',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create Q-Q plot
                        import scipy.stats as stats
                        
                        # For train residuals
                        qq_x = np.array(stats.probplot(train_residuals, dist="norm")[0][0])
                        qq_y = np.array(stats.probplot(train_residuals, dist="norm")[0][1])
                        
                        fig = go.Figure()
                        
                        # Add train data
                        fig.add_trace(go.Scatter(
                            x=qq_x,
                            y=qq_y,
                            mode='markers',
                            name='训练集残差',
                            marker=dict(color='blue', size=8, opacity=0.6)
                        ))
                        
                        # Add reference line
                        min_val = min(qq_x)
                        max_val = max(qq_x)
                        slope, intercept = np.polyfit(qq_x, qq_y, 1)
                        
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[slope * min_val + intercept, slope * max_val + intercept],
                            mode='lines',
                            name='参考线',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title='残差 Q-Q 图',
                            xaxis_title='理论分位数',
                            yaxis_title='样本分位数',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with viz_tab3:
                    # Error distribution
                    if 'predictions' in results:
                        pred_data = results['predictions']
                        
                        # Calculate absolute errors
                        train_abs_errors = np.abs(np.array(pred_data['train']) - np.array(pred_data['train_actual']))
                        test_abs_errors = np.abs(np.array(pred_data['test']) - np.array(pred_data['test_actual']))
                        
                        # Create histograms
                        fig = make_subplots(rows=1, cols=2, subplot_titles=("训练集绝对误差", "测试集绝对误差"))
                        
                        # Add train histogram
                        fig.add_trace(
                            go.Histogram(x=train_abs_errors, name="训练集", marker_color='blue', opacity=0.7),
                            row=1, col=1
                        )
                        
                        # Add test histogram
                        fig.add_trace(
                            go.Histogram(x=test_abs_errors, name="测试集", marker_color='red', opacity=0.7),
                            row=1, col=2
                        )
                        
                        fig.update_layout(
                            title='绝对误差分布',
                            xaxis_title='绝对误差',
                            yaxis_title='频数',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Error statistics
                        train_error_stats = {
                            "平均绝对误差": np.mean(train_abs_errors),
                            "中位数绝对误差": np.median(train_abs_errors),
                            "最大绝对误差": np.max(train_abs_errors),
                            "75%分位数误差": np.percentile(train_abs_errors, 75),
                            "90%分位数误差": np.percentile(train_abs_errors, 90)
                        }
                        
                        test_error_stats = {
                            "平均绝对误差": np.mean(test_abs_errors),
                            "中位数绝对误差": np.median(test_abs_errors),
                            "最大绝对误差": np.max(test_abs_errors),
                            "75%分位数误差": np.percentile(test_abs_errors, 75),
                            "90%分位数误差": np.percentile(test_abs_errors, 90)
                        }
                        
                        # Display error statistics
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("训练集误差统计:")
                            st.dataframe(pd.DataFrame([train_error_stats]).T.rename(columns={0: "值"}))
                        
                        with col2:
                            st.write("测试集误差统计:")
                            st.dataframe(pd.DataFrame([test_error_stats]).T.rename(columns={0: "值"}))
        else:
            st.warning("尚未训练任何模型")
    
    def render(self):
        st.title("可视化分析")
        
        tab1, tab2, tab3 = st.tabs(["数据可视化", "模型可视化", "结果可视化"])
        
        with tab1:
            self.data_visualization()
        
        with tab2:
            self.model_visualization()
        
        with tab3:
            self.result_visualization() 