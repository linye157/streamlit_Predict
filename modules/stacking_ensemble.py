import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import os
from pathlib import Path
import time
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go

class StackingEnsembleModule:
    def __init__(self):
        # Initialize model dict
        if 'models' not in st.session_state:
            st.session_state.models = {}
        
        # Initialize stacking models
        if 'stacking_models' not in st.session_state:
            st.session_state.stacking_models = {}
        
        # Initialize model results
        if 'stacking_results' not in st.session_state:
            st.session_state.stacking_results = {}
    
    def prepare_data(self):
        """Prepare data for model training"""
        if st.session_state.train_data is None:
            st.error("请先加载训练数据")
            return None, None, None, None
        
        if 'num_targets' not in st.session_state:
            st.session_state.num_targets = 3  # Default to 3 targets
        
        # Get features and targets
        X = st.session_state.train_data.iloc[:, :-st.session_state.num_targets]
        y = st.session_state.train_data.iloc[:, -st.session_state.num_targets:]
        
        # Split data
        test_size = st.slider("测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="stacking_test_size")
        random_state = st.number_input("随机种子", min_value=0, max_value=1000, value=42, key="stacking_random_state")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        st.success(f"数据准备完成！训练集: {X_train.shape[0]} 行，测试集: {X_test.shape[0]} 行")
        
        return X_train, X_test, y_train, y_test
    
    def train_level_one_models(self, X_train, X_test, y_train, y_test, target_idx):
        """Train level one models for stacking"""
        st.subheader("训练一级模型")
        
        # Available base models
        base_models = {
            "线性回归 (LR)": LinearRegression(),
            "岭回归 (Ridge)": Ridge(),
            "Lasso回归": Lasso(),
            "随机森林 (RF)": RandomForestRegressor(n_estimators=100, random_state=42),
            "梯度提升 (GBR)": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBoost (XGBR)": XGBRegressor(n_estimators=100, random_state=42),
            "支持向量机 (SVR)": SVR()
        }
        
        # Select models to include in ensemble
        selected_models = st.multiselect(
            "选择基础模型", 
            list(base_models.keys()),
            default=["随机森林 (RF)", "梯度提升 (GBR)", "XGBoost (XGBR)"]
        )
        
        if len(selected_models) < 2:
            st.warning("请至少选择两个基础模型")
            return None
        
        if st.button("训练一级模型"):
            with st.spinner("正在训练一级模型..."):
                # Create dict to store level one models
                level_one_models = {}
                level_one_preds = {}
                level_one_results = {}
                
                # Create progress bar
                progress_bar = st.progress(0)
                total_models = len(selected_models)
                
                # Train each selected model
                for i, model_name in enumerate(selected_models):
                    # Get model
                    model = base_models[model_name]
                    
                    # Start time
                    start_time = time.time()
                    
                    # Train model
                    model.fit(X_train, y_train.iloc[:, target_idx])
                    
                    # End time
                    end_time = time.time()
                    
                    # Make predictions
                    train_preds = model.predict(X_train)
                    test_preds = model.predict(X_test)
                    
                    # Calculate metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, target_idx], train_preds))
                    test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, target_idx], test_preds))
                    train_r2 = r2_score(y_train.iloc[:, target_idx], train_preds)
                    test_r2 = r2_score(y_test.iloc[:, target_idx], test_preds)
                    train_mae = mean_absolute_error(y_train.iloc[:, target_idx], train_preds)
                    test_mae = mean_absolute_error(y_test.iloc[:, target_idx], test_preds)
                    
                    # Store model and results
                    level_one_models[model_name] = model
                    level_one_preds[model_name] = {
                        'train': train_preds,
                        'test': test_preds
                    }
                    level_one_results[model_name] = {
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'training_time': end_time - start_time
                    }
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_models)
                
                # Store level one models and results in session state
                stacking_key = f"Stacking_Target_{target_idx}"
                if stacking_key not in st.session_state.stacking_models:
                    st.session_state.stacking_models[stacking_key] = {}
                
                st.session_state.stacking_models[stacking_key]['level_one'] = level_one_models
                st.session_state.stacking_models[stacking_key]['level_one_preds'] = level_one_preds
                st.session_state.stacking_models[stacking_key]['level_one_results'] = level_one_results
                st.session_state.stacking_models[stacking_key]['target_idx'] = target_idx
                st.session_state.stacking_models[stacking_key]['selected_models'] = selected_models
                
                # Display results
                self.display_level_one_results(stacking_key)
                
                return stacking_key
    
    def display_level_one_results(self, stacking_key):
        """Display level one model results"""
        if stacking_key in st.session_state.stacking_models and 'level_one_results' in st.session_state.stacking_models[stacking_key]:
            level_one_results = st.session_state.stacking_models[stacking_key]['level_one_results']
            
            st.subheader("一级模型结果")
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, results in level_one_results.items():
                comparison_data.append({
                    '模型': model_name,
                    '训练 RMSE': results['train_rmse'],
                    '测试 RMSE': results['test_rmse'],
                    '训练 R²': results['train_r2'],
                    '测试 R²': results['test_r2'],
                    '训练 MAE': results['train_mae'],
                    '测试 MAE': results['test_mae'],
                    '训练时间': results['training_time']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.set_index('模型'))
            
            # Create bar chart for test metrics
            metrics = ['测试 RMSE', '测试 MAE', '测试 R²']
            for metric in metrics:
                # Sort by metric (ascending for error metrics, descending for R²)
                ascending = metric != '测试 R²'
                sorted_df = comparison_df.sort_values(metric, ascending=ascending)
                
                fig = px.bar(
                    sorted_df,
                    x='模型',
                    y=metric,
                    title=f'一级模型比较 - {metric}',
                    color='模型'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def train_level_two_model(self, X_train, X_test, y_train, y_test, stacking_key):
        """Train level two model for stacking"""
        if stacking_key not in st.session_state.stacking_models or 'level_one_preds' not in st.session_state.stacking_models[stacking_key]:
            st.error("请先训练一级模型")
            return
        
        st.subheader("训练二级模型")
        
        # Get target index
        target_idx = st.session_state.stacking_models[stacking_key]['target_idx']
        
        # Get level one models and predictions
        level_one_models = st.session_state.stacking_models[stacking_key]['level_one']
        level_one_preds = st.session_state.stacking_models[stacking_key]['level_one_preds']
        selected_models = st.session_state.stacking_models[stacking_key]['selected_models']
        
        # Available meta learners
        meta_learners = {
            "线性回归 (LR)": LinearRegression(),
            "岭回归 (Ridge)": Ridge(),
            "Lasso回归": Lasso(),
            "随机森林 (RF)": RandomForestRegressor(n_estimators=100, random_state=42),
            "梯度提升 (GBR)": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBoost (XGBR)": XGBRegressor(n_estimators=100, random_state=42)
        }
        
        # Select meta learner
        meta_learner_name = st.selectbox(
            "选择二级模型 (元学习器)",
            list(meta_learners.keys()),
            index=0
        )
        
        if st.button("训练二级模型"):
            with st.spinner("正在训练二级模型..."):
                # Prepare level one predictions for meta learner
                X_train_meta = np.column_stack([level_one_preds[model_name]['train'] for model_name in selected_models])
                X_test_meta = np.column_stack([level_one_preds[model_name]['test'] for model_name in selected_models])
                
                # Get meta learner
                meta_learner = meta_learners[meta_learner_name]
                
                # Start time
                start_time = time.time()
                
                # Train meta learner
                meta_learner.fit(X_train_meta, y_train.iloc[:, target_idx])
                
                # End time
                end_time = time.time()
                
                # Make predictions
                train_preds = meta_learner.predict(X_train_meta)
                test_preds = meta_learner.predict(X_test_meta)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, target_idx], train_preds))
                test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, target_idx], test_preds))
                train_r2 = r2_score(y_train.iloc[:, target_idx], train_preds)
                test_r2 = r2_score(y_test.iloc[:, target_idx], test_preds)
                train_mae = mean_absolute_error(y_train.iloc[:, target_idx], train_preds)
                test_mae = mean_absolute_error(y_test.iloc[:, target_idx], test_preds)
                
                # Store meta learner and results
                st.session_state.stacking_models[stacking_key]['meta_learner'] = meta_learner
                st.session_state.stacking_models[stacking_key]['meta_learner_name'] = meta_learner_name
                
                # Create stacking model object
                estimators = [(model_name, level_one_models[model_name]) for model_name in selected_models]
                stacking_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=meta_learner
                )
                
                # Store in session state
                st.session_state.models[stacking_key] = stacking_model
                
                # Store results
                st.session_state.stacking_results[stacking_key] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'training_time': end_time - start_time,
                    'target_idx': target_idx,
                    'meta_learner': meta_learner_name,
                    'base_models': selected_models,
                    'predictions': {
                        'train': train_preds.tolist(),
                        'test': test_preds.tolist(),
                        'train_actual': y_train.iloc[:, target_idx].values.tolist(),
                        'test_actual': y_test.iloc[:, target_idx].values.tolist()
                    }
                }
                
                # Display results
                self.display_stacking_results(stacking_key)
                
                # Save model
                self.save_stacking_model(stacking_key)
    
    def display_stacking_results(self, stacking_key):
        """Display stacking model results"""
        if stacking_key in st.session_state.stacking_results:
            results = st.session_state.stacking_results[stacking_key]
            
            st.subheader(f"Stacking 模型结果")
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("训练 RMSE", f"{results['train_rmse']:.4f}")
                st.metric("测试 RMSE", f"{results['test_rmse']:.4f}")
            
            with col2:
                st.metric("训练 R²", f"{results['train_r2']:.4f}")
                st.metric("测试 R²", f"{results['test_r2']:.4f}")
            
            with col3:
                st.metric("训练 MAE", f"{results['train_mae']:.4f}")
                st.metric("测试 MAE", f"{results['test_mae']:.4f}")
            
            with col4:
                st.metric("训练时间", f"{results['training_time']:.2f} 秒")
                st.metric("元学习器", results['meta_learner'])
            
            # Visualization of predictions vs actual
            st.subheader("预测结果可视化")
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add train data
            fig.add_trace(go.Scatter(
                x=results['predictions']['train_actual'],
                y=results['predictions']['train'],
                mode='markers',
                name='训练集',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))
            
            # Add test data
            fig.add_trace(go.Scatter(
                x=results['predictions']['test_actual'],
                y=results['predictions']['test'],
                mode='markers',
                name='测试集',
                marker=dict(color='red', size=8, opacity=0.6)
            ))
            
            # Add perfect prediction line
            min_val = min(min(results['predictions']['train_actual']), min(results['predictions']['test_actual']))
            max_val = max(max(results['predictions']['train_actual']), max(results['predictions']['test_actual']))
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
            
            # Compare with level one models
            if stacking_key in st.session_state.stacking_models and 'level_one_results' in st.session_state.stacking_models[stacking_key]:
                level_one_results = st.session_state.stacking_models[stacking_key]['level_one_results']
                
                st.subheader("Stacking vs 基础模型比较")
                
                # Create comparison dataframe
                comparison_data = []
                
                # Add stacking model
                comparison_data.append({
                    '模型': 'Stacking',
                    '训练 RMSE': results['train_rmse'],
                    '测试 RMSE': results['test_rmse'],
                    '训练 R²': results['train_r2'],
                    '测试 R²': results['test_r2'],
                    '训练 MAE': results['train_mae'],
                    '测试 MAE': results['test_mae']
                })
                
                # Add level one models
                for model_name, model_results in level_one_results.items():
                    comparison_data.append({
                        '模型': model_name,
                        '训练 RMSE': model_results['train_rmse'],
                        '测试 RMSE': model_results['test_rmse'],
                        '训练 R²': model_results['train_r2'],
                        '测试 R²': model_results['test_r2'],
                        '训练 MAE': model_results['train_mae'],
                        '测试 MAE': model_results['test_mae']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.set_index('模型'))
                
                # Create bar chart for test metrics
                metrics = ['测试 RMSE', '测试 MAE', '测试 R²']
                for metric in metrics:
                    # Sort by metric (ascending for error metrics, descending for R²)
                    ascending = metric != '测试 R²'
                    sorted_df = comparison_df.sort_values(metric, ascending=ascending)
                    
                    fig = px.bar(
                        sorted_df,
                        x='模型',
                        y=metric,
                        title=f'模型比较 - {metric}',
                        color='模型'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def cross_validation(self, X_train, y_train, stacking_key):
        """Perform k-fold cross validation"""
        if stacking_key not in st.session_state.stacking_models:
            st.error("请先训练Stacking模型")
            return
        
        st.subheader("K折交叉验证")
        
        # Get target index
        target_idx = st.session_state.stacking_models[stacking_key]['target_idx']
        
        # CV parameters
        n_splits = st.slider("折数", min_value=2, max_value=10, value=5, step=1)
        random_state = st.number_input("随机种子 (CV)", min_value=0, max_value=1000, value=42)
        
        if st.button("执行交叉验证"):
            with st.spinner("正在执行交叉验证..."):
                # Create KFold
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                
                # Get base models and meta learner
                if 'level_one' in st.session_state.stacking_models[stacking_key] and 'meta_learner' in st.session_state.stacking_models[stacking_key]:
                    base_models = st.session_state.stacking_models[stacking_key]['level_one']
                    meta_learner = st.session_state.stacking_models[stacking_key]['meta_learner']
                    selected_models = st.session_state.stacking_models[stacking_key]['selected_models']
                    
                    # Define estimators for StackingRegressor
                    estimators = [(model_name, base_models[model_name]) for model_name in selected_models]
                    
                    # Create stacking regressor
                    stacking_model = StackingRegressor(
                        estimators=estimators,
                        final_estimator=meta_learner,
                        cv=kf
                    )
                    
                    # Prepare data
                    X = X_train
                    y = y_train.iloc[:, target_idx]
                    
                    # Perform cross validation
                    cv_results = cross_val_score(stacking_model, X, y, cv=kf, scoring='neg_mean_squared_error')
                    
                    # Calculate metrics
                    cv_rmse = np.sqrt(-cv_results)
                    cv_rmse_mean = np.mean(cv_rmse)
                    cv_rmse_std = np.std(cv_rmse)
                    
                    # Display results
                    st.subheader("交叉验证结果")
                    
                    # Show fold-by-fold results
                    fold_results = pd.DataFrame({
                        "Fold": range(1, n_splits + 1),
                        "RMSE": cv_rmse
                    })
                    
                    st.dataframe(fold_results.set_index("Fold"))
                    
                    # Show summary
                    st.metric("平均 RMSE", f"{cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
                    
                    # Create bar chart for fold results
                    fig = px.bar(
                        fold_results,
                        x="Fold",
                        y="RMSE",
                        title="各折验证结果",
                        color="Fold"
                    )
                    
                    fig.add_hline(y=cv_rmse_mean, line_dash="dash", line_color="red", annotation_text="平均值")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Store CV results in session state
                    st.session_state.stacking_models[stacking_key]['cv_results'] = {
                        'fold_rmse': cv_rmse.tolist(),
                        'mean_rmse': cv_rmse_mean,
                        'std_rmse': cv_rmse_std,
                        'n_splits': n_splits
                    }
                else:
                    st.error("请先完成基础模型和元学习器的训练")
    
    def save_stacking_model(self, stacking_key):
        """Save stacking model to file"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save model
            model_path = os.path.join('models', f"{stacking_key}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(st.session_state.models[stacking_key], f)
            
            st.success(f"Stacking模型已保存至 {model_path}")
        except Exception as e:
            st.error(f"保存模型时出错: {str(e)}")
    
    def render(self):
        st.title("机器学习Stacking集成")
        
        if st.session_state.train_data is None:
            st.warning("请先加载训练数据")
            return
        
        tab1, tab2, tab3 = st.tabs(["训练一级模型", "训练二级模型", "K折交叉验证"])
        
        with tab1:
            # Prepare data
            st.subheader("准备数据")
            X_train, X_test, y_train, y_test = self.prepare_data()
            
            if X_train is not None:
                # Select target
                available_targets = y_train.columns.tolist()
                target_idx = st.selectbox(
                    "选择目标变量",
                    range(len(available_targets)),
                    format_func=lambda i: available_targets[i]
                )
                
                # Train level one models
                stacking_key = self.train_level_one_models(X_train, X_test, y_train, y_test, target_idx)
        
        with tab2:
            # Check if stacking key is available
            stacking_keys = list(st.session_state.stacking_models.keys()) if 'stacking_models' in st.session_state else []
            
            if len(stacking_keys) > 0:
                # Select stacking model
                selected_stacking_key = st.selectbox(
                    "选择已训练的一级模型集合",
                    stacking_keys,
                    format_func=lambda k: f"{k} - {len(st.session_state.stacking_models[k]['selected_models'])} 个基础模型"
                )
                
                # Prepare data if not already prepared
                if 'X_train' not in locals() or X_train is None:
                    X_train, X_test, y_train, y_test = self.prepare_data()
                
                if X_train is not None and selected_stacking_key:
                    # Train level two model
                    self.train_level_two_model(X_train, X_test, y_train, y_test, selected_stacking_key)
            else:
                st.info("请先在 '训练一级模型' 标签页训练一级模型")
        
        with tab3:
            # Check if stacking key is available
            stacking_keys = list(st.session_state.stacking_models.keys()) if 'stacking_models' in st.session_state else []
            
            if len(stacking_keys) > 0:
                # Check which models have meta learners
                stacking_keys_with_meta = [
                    k for k in stacking_keys 
                    if 'meta_learner' in st.session_state.stacking_models[k]
                ]
                
                if len(stacking_keys_with_meta) > 0:
                    # Select stacking model
                    selected_stacking_key = st.selectbox(
                        "选择已训练的Stacking模型",
                        stacking_keys_with_meta,
                        format_func=lambda k: f"{k} - 元学习器: {st.session_state.stacking_models[k]['meta_learner_name']}"
                    )
                    
                    # Prepare data if not already prepared
                    if 'X_train' not in locals() or X_train is None:
                        X_train, X_test, y_train, y_test = self.prepare_data()
                    
                    if X_train is not None and selected_stacking_key:
                        # Perform cross validation
                        self.cross_validation(X_train, y_train, selected_stacking_key)
                else:
                    st.info("请先在 '训练二级模型' 标签页训练二级模型")
            else:
                st.info("请先训练Stacking模型") 