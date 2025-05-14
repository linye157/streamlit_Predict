import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import os
import time
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import plotly.express as px
import plotly.graph_objects as go

class AutoMLModule:
    def __init__(self):
        # Initialize model dict
        if 'models' not in st.session_state:
            st.session_state.models = {}
        
        # Initialize auto ml results
        if 'auto_ml_results' not in st.session_state:
            st.session_state.auto_ml_results = {}
    
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
        test_size = st.slider("测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="auto_test_size")
        random_state = st.number_input("随机种子", min_value=0, max_value=1000, value=42, key="auto_random_state")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        st.success(f"数据准备完成！训练集: {X_train.shape[0]} 行，测试集: {X_test.shape[0]} 行")
        
        return X_train, X_test, y_train, y_test
    
    def auto_model_selection(self, X_train, X_test, y_train, y_test, target_idx):
        """Auto model selection"""
        st.subheader("模型自动筛选")
        
        # Available models
        models = {
            "线性回归 (LR)": LinearRegression(),
            "岭回归 (Ridge)": Ridge(),
            "Lasso回归": Lasso(),
            "随机森林 (RF)": RandomForestRegressor(n_estimators=100, random_state=42),
            "梯度提升 (GBR)": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBoost (XGBR)": XGBRegressor(n_estimators=100, random_state=42),
            "支持向量机 (SVR)": SVR(),
            "人工神经网络 (ANN)": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
        }
        
        # Select models to evaluate
        selected_models = st.multiselect(
            "选择要评估的模型", 
            list(models.keys()),
            default=list(models.keys())
        )
        
        if len(selected_models) == 0:
            st.warning("请至少选择一个模型")
            return
        
        if st.button("开始模型自动筛选"):
            with st.spinner("正在评估模型..."):
                # Create dict to store model results
                model_results = {}
                
                # Create progress bar
                progress_bar = st.progress(0)
                total_models = len(selected_models)
                
                # Train and evaluate each selected model
                for i, model_name in enumerate(selected_models):
                    # Get model
                    model = models[model_name]
                    
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
                    
                    # Store results
                    model_results[model_name] = {
                        'model': model,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'training_time': end_time - start_time,
                        'predictions': {
                            'train': train_preds,
                            'test': test_preds
                        }
                    }
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / total_models)
                
                # Store results in session state
                auto_ml_key = f"AutoML_Target_{target_idx}"
                st.session_state.auto_ml_results[auto_ml_key] = {
                    'model_results': model_results,
                    'target_idx': target_idx,
                    'feature_names': X_train.columns.tolist(),
                    'target_name': y_train.columns[target_idx],
                    'train_actual': y_train.iloc[:, target_idx].values.tolist(),
                    'test_actual': y_test.iloc[:, target_idx].values.tolist()
                }
                
                # Select best model
                best_model_name = self.select_best_model(auto_ml_key)
                
                # Display results
                self.display_model_selection_results(auto_ml_key, best_model_name)
                
                return auto_ml_key
    
    def select_best_model(self, auto_ml_key):
        """Select best model based on test RMSE"""
        model_results = st.session_state.auto_ml_results[auto_ml_key]['model_results']
        
        # Get test RMSE for each model
        test_rmse = {model_name: results['test_rmse'] for model_name, results in model_results.items()}
        
        # Find model with lowest test RMSE
        best_model_name = min(test_rmse, key=test_rmse.get)
        
        # Store best model name
        st.session_state.auto_ml_results[auto_ml_key]['best_model_name'] = best_model_name
        
        return best_model_name
    
    def display_model_selection_results(self, auto_ml_key, best_model_name=None):
        """Display model selection results"""
        if auto_ml_key in st.session_state.auto_ml_results:
            model_results = st.session_state.auto_ml_results[auto_ml_key]['model_results']
            
            if best_model_name is None and 'best_model_name' in st.session_state.auto_ml_results[auto_ml_key]:
                best_model_name = st.session_state.auto_ml_results[auto_ml_key]['best_model_name']
            
            st.subheader("模型评估结果")
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, results in model_results.items():
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
            
            # Highlight best model
            if best_model_name:
                st.success(f"最佳模型: {best_model_name}")
                
                # Get best model metrics
                best_model_metrics = model_results[best_model_name]
                
                # Display best model metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("训练 RMSE", f"{best_model_metrics['train_rmse']:.4f}")
                    st.metric("测试 RMSE", f"{best_model_metrics['test_rmse']:.4f}")
                
                with col2:
                    st.metric("训练 R²", f"{best_model_metrics['train_r2']:.4f}")
                    st.metric("测试 R²", f"{best_model_metrics['test_r2']:.4f}")
                
                with col3:
                    st.metric("训练 MAE", f"{best_model_metrics['train_mae']:.4f}")
                    st.metric("测试 MAE", f"{best_model_metrics['test_mae']:.4f}")
                
                with col4:
                    st.metric("训练时间", f"{best_model_metrics['training_time']:.2f} 秒")
            
            # Display comparison table
            st.dataframe(comparison_df.set_index('模型'))
            
            # Create bar charts for test metrics
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
                
                # Highlight best model
                if best_model_name:
                    # Add annotation for best model
                    best_idx = sorted_df[sorted_df['模型'] == best_model_name].index[0]
                    best_value = sorted_df.loc[best_idx, metric]
                    
                    fig.add_annotation(
                        x=best_model_name,
                        y=best_value,
                        text="最佳模型",
                        showarrow=True,
                        arrowhead=1
                    )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def auto_parameter_optimization(self, X_train, X_test, y_train, y_test, auto_ml_key):
        """Auto parameter optimization for the best model"""
        if auto_ml_key not in st.session_state.auto_ml_results:
            st.error("请先进行模型自动筛选")
            return
        
        if 'best_model_name' not in st.session_state.auto_ml_results[auto_ml_key]:
            st.error("无法找到最佳模型")
            return
        
        st.subheader("模型参数自动优化")
        
        # Get best model
        best_model_name = st.session_state.auto_ml_results[auto_ml_key]['best_model_name']
        best_model = st.session_state.auto_ml_results[auto_ml_key]['model_results'][best_model_name]['model']
        target_idx = st.session_state.auto_ml_results[auto_ml_key]['target_idx']
        
        st.info(f"为模型 '{best_model_name}' 进行参数优化")
        
        # Define parameter grid for different models
        param_grids = {
            "线性回归 (LR)": {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            "岭回归 (Ridge)": {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            },
            "Lasso回归": {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'selection': ['cyclic', 'random']
            },
            "随机森林 (RF)": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            "梯度提升 (GBR)": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "XGBoost (XGBR)": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            "支持向量机 (SVR)": {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto'],
                'epsilon': [0.01, 0.1, 0.2]
            },
            "人工神经网络 (ANN)": {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'solver': ['adam', 'sgd', 'lbfgs'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        # Define parameter grid for the selected model
        if best_model_name in param_grids:
            param_grid = param_grids[best_model_name]
            
            # Set search options
            search_method = st.radio(
                "搜索方法",
                ["网格搜索", "随机搜索"],
                index=1
            )
            
            cv_folds = st.slider("交叉验证折数", min_value=2, max_value=10, value=5)
            n_iter = st.slider("随机搜索迭代次数", min_value=10, max_value=100, value=20) if search_method == "随机搜索" else None
            
            if st.button("开始参数优化"):
                with st.spinner("正在优化参数..."):
                    # Start time
                    start_time = time.time()
                    
                    # Create search object
                    if search_method == "网格搜索":
                        search = GridSearchCV(
                            best_model,
                            param_grid,
                            scoring='neg_mean_squared_error',
                            cv=cv_folds,
                            n_jobs=-1,
                            verbose=1
                        )
                    else:  # 随机搜索
                        search = RandomizedSearchCV(
                            best_model,
                            param_distributions=param_grid,
                            n_iter=n_iter,
                            scoring='neg_mean_squared_error',
                            cv=cv_folds,
                            n_jobs=-1,
                            verbose=1,
                            random_state=42
                        )
                    
                    # Fit search object
                    search.fit(X_train, y_train.iloc[:, target_idx])
                    
                    # End time
                    end_time = time.time()
                    
                    # Get best model
                    optimized_model = search.best_estimator_
                    
                    # Make predictions
                    train_preds = optimized_model.predict(X_train)
                    test_preds = optimized_model.predict(X_test)
                    
                    # Calculate metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, target_idx], train_preds))
                    test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, target_idx], test_preds))
                    train_r2 = r2_score(y_train.iloc[:, target_idx], train_preds)
                    test_r2 = r2_score(y_test.iloc[:, target_idx], test_preds)
                    train_mae = mean_absolute_error(y_train.iloc[:, target_idx], train_preds)
                    test_mae = mean_absolute_error(y_test.iloc[:, target_idx], test_preds)
                    
                    # Store optimized model results
                    st.session_state.auto_ml_results[auto_ml_key]['optimized_model'] = optimized_model
                    st.session_state.auto_ml_results[auto_ml_key]['optimized_params'] = search.best_params_
                    st.session_state.auto_ml_results[auto_ml_key]['optimized_results'] = {
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'training_time': end_time - start_time,
                        'search_method': search_method,
                        'cv_folds': cv_folds,
                        'n_iter': n_iter if search_method == "随机搜索" else None
                    }
                    
                    # Also store in models
                    optimized_model_name = f"Optimized_{best_model_name}_Target_{target_idx}"
                    st.session_state.models[optimized_model_name] = optimized_model
                    
                    # Display optimization results
                    self.display_optimization_results(auto_ml_key)
        else:
            st.warning(f"暂不支持对 '{best_model_name}' 进行参数优化")
    
    def display_optimization_results(self, auto_ml_key):
        """Display optimization results"""
        if auto_ml_key in st.session_state.auto_ml_results and 'optimized_results' in st.session_state.auto_ml_results[auto_ml_key]:
            # Get results
            best_model_name = st.session_state.auto_ml_results[auto_ml_key]['best_model_name']
            original_results = st.session_state.auto_ml_results[auto_ml_key]['model_results'][best_model_name]
            optimized_results = st.session_state.auto_ml_results[auto_ml_key]['optimized_results']
            optimized_params = st.session_state.auto_ml_results[auto_ml_key]['optimized_params']
            
            st.subheader("优化结果")
            
            # Display optimized parameters
            st.write("最佳参数:")
            st.json(optimized_params)
            
            # Compare original vs optimized
            st.subheader("优化前后对比")
            
            # Create comparison dataframe
            comparison_data = [
                {
                    '模型': '优化前',
                    '训练 RMSE': original_results['train_rmse'],
                    '测试 RMSE': original_results['test_rmse'],
                    '训练 R²': original_results['train_r2'],
                    '测试 R²': original_results['test_r2'],
                    '训练 MAE': original_results['train_mae'],
                    '测试 MAE': original_results['test_mae']
                },
                {
                    '模型': '优化后',
                    '训练 RMSE': optimized_results['train_rmse'],
                    '测试 RMSE': optimized_results['test_rmse'],
                    '训练 R²': optimized_results['train_r2'],
                    '测试 R²': optimized_results['test_r2'],
                    '训练 MAE': optimized_results['train_mae'],
                    '测试 MAE': optimized_results['test_mae']
                }
            ]
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.set_index('模型'))
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                improvement_rmse = (original_results['test_rmse'] - optimized_results['test_rmse']) / original_results['test_rmse'] * 100
                st.metric("测试 RMSE 改善", f"{improvement_rmse:.2f}%")
            
            with col2:
                improvement_r2 = (optimized_results['test_r2'] - original_results['test_r2']) / original_results['test_r2'] * 100
                st.metric("测试 R² 改善", f"{improvement_r2:.2f}%")
            
            with col3:
                improvement_mae = (original_results['test_mae'] - optimized_results['test_mae']) / original_results['test_mae'] * 100
                st.metric("测试 MAE 改善", f"{improvement_mae:.2f}%")
            
            # Create bar chart for comparison
            metrics = ['测试 RMSE', '测试 R²', '测试 MAE']
            for metric in metrics:
                fig = px.bar(
                    comparison_df,
                    x='模型',
                    y=metric,
                    title=f'{metric} 对比',
                    color='模型'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def export_model(self, auto_ml_key):
        """Export trained model"""
        if auto_ml_key not in st.session_state.auto_ml_results:
            st.error("请先进行模型自动筛选")
            return
        
        st.subheader("模型导出")
        
        # Get model options
        model_options = []
        
        if 'best_model_name' in st.session_state.auto_ml_results[auto_ml_key]:
            model_options.append("最佳基础模型")
            
        if 'optimized_model' in st.session_state.auto_ml_results[auto_ml_key]:
            model_options.append("优化后模型")
        
        if len(model_options) == 0:
            st.error("没有可用的模型可导出")
            return
        
        # Select model to export
        model_choice = st.radio(
            "选择要导出的模型",
            model_options,
            index=len(model_options) - 1  # Default to optimized model if available
        )
        
        # Set file format
        file_format = st.selectbox(
            "导出格式",
            ["Pickle (.pkl)", "ONNX (.onnx)"],
            index=0
        )
        
        if st.button("导出模型"):
            with st.spinner("正在导出模型..."):
                # Create models directory if it doesn't exist
                os.makedirs('models', exist_ok=True)
                
                # Get model to export
                if model_choice == "最佳基础模型":
                    best_model_name = st.session_state.auto_ml_results[auto_ml_key]['best_model_name']
                    model = st.session_state.auto_ml_results[auto_ml_key]['model_results'][best_model_name]['model']
                    model_name = f"{best_model_name}_Target_{st.session_state.auto_ml_results[auto_ml_key]['target_idx']}"
                else:  # 优化后模型
                    model = st.session_state.auto_ml_results[auto_ml_key]['optimized_model']
                    best_model_name = st.session_state.auto_ml_results[auto_ml_key]['best_model_name']
                    model_name = f"Optimized_{best_model_name}_Target_{st.session_state.auto_ml_results[auto_ml_key]['target_idx']}"
                
                # Export model
                if file_format == "Pickle (.pkl)":
                    # Export as pickle
                    model_path = os.path.join('models', f"{model_name}.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    st.success(f"模型已导出至 {model_path}")
                
                # Create download button
                if file_format == "Pickle (.pkl)":
                    with open(model_path, 'rb') as f:
                        model_bytes = f.read()
                    
                    st.download_button(
                        label="下载模型",
                        data=model_bytes,
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream"
                    )
    
    def render(self):
        st.title("自动化机器学习")
        
        if st.session_state.train_data is None:
            st.warning("请先加载训练数据")
            return
        
        tab1, tab2, tab3 = st.tabs(["模型自动筛选", "模型参数自动优化", "模型导出"])
        
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
                
                # Auto model selection
                auto_ml_key = self.auto_model_selection(X_train, X_test, y_train, y_test, target_idx)
        
        with tab2:
            # Check if automl key is available
            auto_ml_keys = list(st.session_state.auto_ml_results.keys()) if 'auto_ml_results' in st.session_state else []
            
            if len(auto_ml_keys) > 0:
                # Select automl run
                selected_auto_ml_key = st.selectbox(
                    "选择已完成的模型筛选结果",
                    auto_ml_keys,
                    format_func=lambda k: f"{k} - 最佳模型: {st.session_state.auto_ml_results[k].get('best_model_name', '未知')}"
                )
                
                # Prepare data if not already prepared
                if 'X_train' not in locals() or X_train is None:
                    X_train, X_test, y_train, y_test = self.prepare_data()
                
                if X_train is not None and selected_auto_ml_key:
                    # Auto parameter optimization
                    self.auto_parameter_optimization(X_train, X_test, y_train, y_test, selected_auto_ml_key)
            else:
                st.info("请先在 '模型自动筛选' 标签页进行模型筛选")
        
        with tab3:
            # Check if automl key is available
            auto_ml_keys = list(st.session_state.auto_ml_results.keys()) if 'auto_ml_results' in st.session_state else []
            
            if len(auto_ml_keys) > 0:
                # Select automl run
                selected_auto_ml_key = st.selectbox(
                    "选择要导出的模型",
                    auto_ml_keys,
                    format_func=lambda k: f"{k} - 目标: {st.session_state.auto_ml_results[k].get('target_name', '未知')}",
                    key="export_selector"
                )
                
                if selected_auto_ml_key:
                    # Export model
                    self.export_model(selected_auto_ml_key)
            else:
                st.info("请先在 '模型自动筛选' 标签页进行模型筛选") 